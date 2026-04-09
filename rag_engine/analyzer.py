import re
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

from .prompts import get_analysis_prompt
from .schema import (
    COLUMN_NAME_HINTS,
    DATE_COLUMN,
    ORDER_ID_COLUMN,
    PRODUCT_ID_COLUMN,
    PRODUCT_TEXT_COLUMN,
)

MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "januar": 1,
    "february": 2,
    "feb": 2,
    "februar": 2,
    "march": 3,
    "mar": 3,
    "maerz": 3,
    "märz": 3,
    "april": 4,
    "may": 5,
    "mai": 5,
    "june": 6,
    "jun": 6,
    "juni": 6,
    "july": 7,
    "jul": 7,
    "juli": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "oktober": 10,
    "october": 10,
    "oct": 10,
    "okt": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
    "dezember": 12,
    "dez": 12,
}

TOKEN_NORMALIZATION = {
    "tomato": "tomato",
    "tomaten": "tomato",
    "tomates": "tomato",
    "tomate": "tomato",
    "pomodoro": "tomato",
    "pomodori": "tomato",
    "sauce": "sauce",
    "sauces": "sauce",
    "sugo": "sauce",
    "recipe": "recipe",
    "recipes": "recipe",
    "rezept": "recipe",
    "rezeptheft": "recipe",
    "can": "can",
    "cans": "can",
    "dose": "can",
    "dosen": "can",
}

CANONICAL_TO_VARIANTS: Dict[str, Set[str]] = {}
for _variant, _canonical in TOKEN_NORMALIZATION.items():
    CANONICAL_TO_VARIANTS.setdefault(_canonical, set()).add(_variant)

PRODUCT_QUERY_STOPWORDS = {
    "how", "many", "much", "what", "which", "who", "where", "when", "why",
    "did", "do", "does", "had", "has", "have", "were", "was", "we", "our",
    "there", "them", "they", "with", "without", "from", "into", "during",
    "for", "and", "or", "the", "a", "an", "in", "on", "of", "to",
    "wie", "viele", "viel", "welche", "welcher", "welches", "hatten", "hat",
    "haben", "wir", "uns", "mit", "ohne", "von", "im", "am", "an",
    "orders", "order", "bestellungen", "bestellung", "umsatz", "revenue",
    "product", "produkt", "products", "produkte", "article", "artikel",
    "sku", "id", "contained", "contains", "contain", "include", "included",
    "inklusive", "containing",
}

for _month_name in MONTH_NAME_TO_NUMBER:
    PRODUCT_QUERY_STOPWORDS.add(_month_name)
for _year in range(2020, 2031):
    PRODUCT_QUERY_STOPWORDS.add(str(_year))


class OllamaCsvRAG:
    def __init__(
        self,
        df: pd.DataFrame,
        model: str = "gemma4",
        debug: bool = False,
        embedding_model: str = "nomic-embed-text:latest",
    ):
        self.df = df
        self.llm = Ollama(model=model)
        self.query_chain = LLMChain(llm=self.llm, prompt=get_analysis_prompt())
        self.debug = debug
        self.embedding_model = embedding_model
        self._embedding_backend = None
        self._schema_documents: List[Dict[str, str]] = []
        self._schema_embeddings: Dict[str, np.ndarray] = {}
        self._answer_cache: Dict[str, str] = {}
        self._trace_cache: Dict[str, Dict[str, object]] = {}
        self._last_trace: Dict[str, object] = {}
        self._literal_match_cache: Dict[str, List[Dict[str, object]]] = {}
        self._product_phrase_cache: Dict[str, List[Dict[str, object]]] = {}
        self._product_embeddings: Dict[int, np.ndarray] = {}
        self._product_catalog: List[Dict[str, object]] = []
        self._product_inverted_index: Dict[str, Set[int]] = {}
        self._product_normalized_index: Dict[str, Set[int]] = {}

        self.column_name_hints = dict(COLUMN_NAME_HINTS)

        self.data_description = self._generate_schema()
        self._schema_documents = self._build_schema_documents()
        self._embedding_backend = self._load_embedding_backend()
        self._schema_embeddings = self._build_schema_embeddings()
        self._build_product_catalog()

    def _generate_schema(self) -> str:
        lines = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            hint = self.column_name_hints.get(col, "")
            lines.append(f"- {col} (Type: {dtype}) — Synonyms: {hint}")
        return "\n".join(lines)

    def _question_asks_order_count(self, question: str) -> bool:
        patterns = [
            r"\bhow many orders\b",
            r"\bin how many orders\b",
            r"\bnumber of orders\b",
            r"\bwie viele bestellungen\b",
            r"\bin wie vielen bestellungen\b",
            r"\banzahl der bestellungen\b",
        ]
        lowered = question.lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _question_asks_quantity_sold(self, question: str) -> bool:
        patterns = [
            r"\bhow many .* did .* sell\b",
            r"\bhow many .* sold\b",
            r"\bhow much quantity\b",
            r"\bquantity sold\b",
            r"\bwie viele .* verkauft\b",
            r"\bwie viel .* verkauft\b",
            r"\bmenge\b",
        ]
        lowered = question.lower()
        return any(re.search(pattern, lowered) for pattern in patterns) and not self._question_asks_order_count(question)

    def _parse_month_year_filter(self, question: str) -> Optional[pd.Series]:
        lowered = question.lower()
        month = None
        year = None

        for name, month_number in MONTH_NAME_TO_NUMBER.items():
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                month = month_number
                break

        year_match = re.search(r"\b(20\d{2})\b", lowered)
        if year_match:
            year = int(year_match.group(1))

        if month is None or year is None or DATE_COLUMN not in self.df.columns:
            return None
        return (self.df[DATE_COLUMN].dt.year == year) & (self.df[DATE_COLUMN].dt.month == month)

    def _shared_product_phrase(self, query_tokens: List[str], product_tokens: List[str]) -> str:
        shared = set(query_tokens) & set(product_tokens)
        if not shared:
            return ""

        best_run: List[str] = []
        current_run: List[str] = []
        for token in product_tokens:
            if token in shared:
                current_run.append(token)
                if len(current_run) > len(best_run):
                    best_run = current_run[:]
            else:
                current_run = []

        if best_run:
            return " ".join(best_run)
        return ""

    def _concept_variant_group(self, token: str) -> List[str]:
        canonical = self._normalize_search_token(token)
        variants = sorted(CANONICAL_TO_VARIANTS.get(canonical, {canonical}))
        return variants

    def _token_groups_for_candidate(
        self,
        question: str,
        candidate: Dict[str, object],
    ) -> List[List[str]]:
        candidate_tokens = set(candidate.get("normalized_tokens", []))
        groups: List[List[str]] = []
        seen = set()
        for token in self._meaningful_query_tokens(question):
            canonical = self._normalize_search_token(token)
            if canonical in seen:
                continue
            if canonical not in candidate_tokens:
                continue
            seen.add(canonical)
            groups.append(self._concept_variant_group(token))
        return groups

    def _keyword_product_candidates(self, question: str, top_k: int = 200) -> List[Dict[str, object]]:
        if not self._product_catalog:
            return []

        query_tokens = self._meaningful_query_tokens(question)
        normalized_query_tokens = self._normalized_meaningful_query_tokens(question)
        if not query_tokens:
            return []

        candidate_scores: Dict[int, float] = {}
        for token in normalized_query_tokens:
            matching_ids = self._product_normalized_index.get(token, set())
            if not matching_ids:
                continue
            token_weight = 1.0 / np.log1p(len(matching_ids) + 1.0)
            for product_id in matching_ids:
                candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + token_weight

        if not candidate_scores:
            return []

        for product_id in list(candidate_scores.keys()):
            product = self._product_catalog[product_id]
            product_text = str(product["normalized_text"]).lower()
            for ngram_size in (3, 2):
                for start in range(0, len(normalized_query_tokens) - ngram_size + 1):
                    phrase = " ".join(normalized_query_tokens[start:start + ngram_size])
                    if phrase in product_text:
                        candidate_scores[product_id] += 0.35 * ngram_size

        ranked_ids = sorted(
            candidate_scores,
            key=lambda product_id: (
                candidate_scores[product_id],
                self._product_catalog[product_id]["distinct_orders"],
                self._product_catalog[product_id]["row_count"],
            ),
            reverse=True,
        )[:top_k]

        candidates = []
        for product_id in ranked_ids:
            product = self._product_catalog[product_id]
            candidates.append(
                {
                    "product_id": product_id,
                    "text": product["text"],
                    "keyword_score": float(candidate_scores[product_id]),
                    "row_count": int(product["row_count"]),
                    "distinct_orders": int(product["distinct_orders"]),
                    "category": str(product["category"]),
                }
            )
        return candidates

    def _ensure_product_embeddings(self, product_ids: List[int]) -> None:
        missing_ids = [product_id for product_id in product_ids if product_id not in self._product_embeddings]
        if not missing_ids:
            return

        texts = [str(self._product_catalog[product_id]["text"]) for product_id in missing_ids]
        vectors = self._embed_documents(texts)
        for product_id, vector in zip(missing_ids, vectors):
            if vector is None:
                continue
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            self._product_embeddings[product_id] = vector

    def _semantic_product_candidates(
        self,
        question: str,
        candidate_ids: List[int],
        top_k: int = 15,
    ) -> List[Dict[str, object]]:
        query_embedding = self._embed_text(question)
        if query_embedding is None:
            return []

        self._ensure_product_embeddings(candidate_ids)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        scored_candidates = []
        for product_id in candidate_ids:
            product_embedding = self._product_embeddings.get(product_id)
            if product_embedding is None:
                continue
            denominator = query_norm * np.linalg.norm(product_embedding)
            if denominator == 0:
                continue
            semantic_score = float(np.dot(query_embedding, product_embedding) / denominator)
            product = self._product_catalog[product_id]
            scored_candidates.append(
                {
                    "product_id": product_id,
                    "text": product["text"],
                    "semantic_score": semantic_score,
                    "row_count": int(product["row_count"]),
                    "distinct_orders": int(product["distinct_orders"]),
                    "category": str(product["category"]),
                }
            )

        scored_candidates.sort(
            key=lambda item: (
                item["semantic_score"],
                item["distinct_orders"],
                item["row_count"],
            ),
            reverse=True,
        )
        return scored_candidates[:top_k]

    def _hybrid_product_candidates(self, question: str, top_k: int = 5) -> List[Dict[str, object]]:
        if not self._product_catalog:
            return []

        query_tokens = self._meaningful_query_tokens(question)
        normalized_query_tokens = self._normalized_meaningful_query_tokens(question)
        keyword_candidates = self._keyword_product_candidates(question, top_k=250)
        candidate_map: Dict[int, Dict[str, object]] = {
            int(candidate["product_id"]): dict(candidate) for candidate in keyword_candidates
        }

        semantic_universe = list(candidate_map.keys())
        if not semantic_universe:
            semantic_universe = [int(record["idx"]) for record in self._product_catalog[:250]]

        semantic_candidates = self._semantic_product_candidates(
            question,
            semantic_universe,
            top_k=min(30, len(semantic_universe)),
        )

        for candidate in semantic_candidates:
            product_id = int(candidate["product_id"])
            if product_id not in candidate_map:
                candidate_map[product_id] = {
                    "product_id": product_id,
                    "text": candidate["text"],
                    "keyword_score": 0.0,
                    "row_count": candidate["row_count"],
                    "distinct_orders": candidate["distinct_orders"],
                    "category": candidate.get("category", ""),
                }
            candidate_map[product_id]["semantic_score"] = candidate["semantic_score"]

        for candidate in candidate_map.values():
            candidate.setdefault("keyword_score", 0.0)
            candidate.setdefault("semantic_score", 0.0)
            product_tokens = self._tokenize_search_text(str(candidate["text"]))
            normalized_product_tokens = {self._normalize_search_token(token) for token in product_tokens}
            candidate["normalized_tokens"] = normalized_product_tokens
            shared_phrase = self._shared_product_phrase(query_tokens, product_tokens)
            shared_token_count = len(set(normalized_query_tokens) & normalized_product_tokens)
            tomato_bonus = 0.0
            if "tomato" in normalized_query_tokens and "tomato" in normalized_product_tokens:
                tomato_bonus += 0.6
            category = str(candidate.get("category", ""))
            packaging_bonus = 0.0
            if "can" in normalized_query_tokens and category == "Obst - Gem├╝se":
                packaging_bonus += 0.4
            semantic_penalty = 0.0
            if "sauce" not in normalized_query_tokens and ("sauce" in normalized_product_tokens or category == "Saucen"):
                semantic_penalty += 0.45
            if "recipe" not in normalized_query_tokens and ("recipe" in normalized_product_tokens or category == "Rezeptheft"):
                semantic_penalty += 0.5
            candidate["shared_phrase"] = shared_phrase
            candidate["shared_token_count"] = shared_token_count
            candidate["combined_score"] = (
                (0.45 * float(candidate["keyword_score"]))
                + (0.55 * max(float(candidate["semantic_score"]), 0.0))
                + (0.15 * shared_token_count)
                + tomato_bonus
                + packaging_bonus
                - semantic_penalty
            )

        ranked_candidates = sorted(
            candidate_map.values(),
            key=lambda item: (
                float(item["combined_score"]),
                int(item["shared_token_count"]),
                float(item["semantic_score"]),
                float(item["keyword_score"]),
                int(item["distinct_orders"]),
            ),
            reverse=True,
        )[:top_k]

        dominant_category = ""
        if ranked_candidates:
            category_counts: Dict[str, int] = {}
            for candidate in ranked_candidates[:3]:
                category = str(candidate.get("category", "")).strip()
                if not category:
                    continue
                category_counts[category] = category_counts.get(category, 0) + 1
            if category_counts:
                dominant_category = max(category_counts, key=category_counts.get)

        results = []
        produkt_values = self._stringify_series(PRODUCT_TEXT_COLUMN)
        for candidate in ranked_candidates:
            token_groups = self._token_groups_for_candidate(question, candidate)
            if token_groups:
                filter_mask = pd.Series(True, index=self.df.index)
                for group in token_groups:
                    group_mask = pd.Series(False, index=self.df.index)
                    for variant in group:
                        group_mask = group_mask | produkt_values.str.contains(variant, case=False, na=False, regex=False)
                    filter_mask = filter_mask & group_mask
            else:
                filter_mask = produkt_values.str.contains(str(candidate["text"]), case=False, na=False, regex=False)
            if dominant_category and "WG_NAME" in self.df.columns and str(candidate.get("category", "")).strip() == dominant_category:
                filter_mask = filter_mask & (self._stringify_series("WG_NAME") == dominant_category)
            filter_row_count = int(filter_mask.sum())
            filter_distinct_orders = int(self.df.loc[filter_mask, ORDER_ID_COLUMN].nunique()) if ORDER_ID_COLUMN in self.df.columns else filter_row_count
            filter_token = str(candidate["shared_phrase"]) if candidate["shared_phrase"] else str(candidate["text"])
            results.append(
                {
                    "token": filter_token,
                    "column": PRODUCT_TEXT_COLUMN,
                    "match_type": "contains",
                    "row_count": filter_row_count,
                    "distinct_orders": filter_distinct_orders,
                    "samples": [str(candidate["text"])],
                    "keyword_score": float(candidate["keyword_score"]),
                    "semantic_score": float(candidate["semantic_score"]),
                    "combined_score": float(candidate["combined_score"]),
                    "retrieval": "hybrid",
                    "token_groups": token_groups,
                    "category_filter": dominant_category if str(candidate.get("category", "")).strip() == dominant_category else "",
                    "category": str(candidate.get("category", "")),
                }
            )
        return results

    def _best_product_match(self, question: str) -> Optional[Dict[str, object]]:
        ranked_matches: List[Dict[str, object]] = []
        for token in self._extract_literal_candidates(question):
            for match in self._search_literal_token(token):
                column = str(match["column"])
                if column not in {PRODUCT_TEXT_COLUMN, PRODUCT_ID_COLUMN}:
                    continue
                ranked_matches.append(match)

        ranked_matches.extend(self._hybrid_product_candidates(question, top_k=5))

        if not ranked_matches:
            return None

        def match_rank(item: Dict[str, object]):
            token_groups = item.get("token_groups") or []
            category_filter = str(item.get("category_filter", "")).strip()
            return (
                str(item["column"]) == PRODUCT_ID_COLUMN and str(item["match_type"]) == "exact",
                bool(token_groups),
                len(token_groups),
                bool(category_filter),
                str(item["match_type"]) == "exact",
                str(item["token"]).count(" "),
                -int(item["row_count"]),
                int(item["distinct_orders"]),
            )

        ranked_matches.sort(key=match_rank, reverse=True)
        return ranked_matches[0]

    def _build_match_mask(self, match: Dict[str, object]) -> pd.Series:
        column = str(match["column"])
        values = self._stringify_series(column)
        token_groups = match.get("token_groups") or []
        if token_groups:
            mask = pd.Series(True, index=self.df.index)
            for group in token_groups:
                group_mask = pd.Series(False, index=self.df.index)
                for variant in group:
                    group_mask = group_mask | values.str.contains(str(variant), case=False, na=False, regex=False)
                mask = mask & group_mask
            category_filter = str(match.get("category_filter", "")).strip()
            if category_filter and "WG_NAME" in self.df.columns:
                mask = mask & (self._stringify_series("WG_NAME") == category_filter)
            return mask
        token = str(match["token"])
        if str(match["match_type"]) == "exact":
            return values.str.upper() == token.upper()
        return values.str.contains(token, case=False, na=False, regex=False)

    def _build_deterministic_code(
        self,
        match: Dict[str, object],
        metric: str,
        include_date_filter: bool,
        question: str,
    ) -> str:
        mask_expr = self._build_match_code_expr(match)
        lines = [f"product_filter = {mask_expr}"]
        if include_date_filter:
            lowered = question.lower()
            month = None
            year = None
            for name, month_number in MONTH_NAME_TO_NUMBER.items():
                if re.search(rf"\b{re.escape(name)}\b", lowered):
                    month = month_number
                    break
            year_match = re.search(r"\b(20\d{2})\b", lowered)
            if year_match:
                year = int(year_match.group(1))
            if month is not None and year is not None:
                lines.append(
                    f"date_filter = (df['{DATE_COLUMN}'].dt.year == {year}) & (df['{DATE_COLUMN}'].dt.month == {month})"
                )
                lines.append("mask = product_filter & date_filter")
            else:
                lines.append("mask = product_filter")
        else:
            lines.append("mask = product_filter")

        if metric == "order_count":
            lines.append(f"result = df.loc[mask, '{ORDER_ID_COLUMN}'].nunique()")
        else:
            lines.append("result = df.loc[mask, 'MENGE'].sum()")
        return "\n".join(lines)

    def _deterministic_product_analysis(self, question: str) -> Optional[Dict[str, object]]:
        product_match = self._best_product_match(question)
        if product_match is None:
            return None

        mask = self._build_match_mask(product_match)
        month_year_filter = self._parse_month_year_filter(question)
        if month_year_filter is not None:
            mask = mask & month_year_filter

        if self._question_asks_order_count(question):
            if ORDER_ID_COLUMN not in self.df.columns:
                return None
            result = int(self.df.loc[mask, ORDER_ID_COLUMN].nunique())
            metric = "order_count"
        elif self._question_asks_quantity_sold(question):
            if "MENGE" not in self.df.columns:
                return None
            result = float(self.df.loc[mask, "MENGE"].sum())
            metric = "quantity_sold"
        else:
            return None

        return {
            "metric": metric,
            "result": result,
            "match": product_match,
            "used_date_filter": month_year_filter is not None,
            "code": self._build_deterministic_code(
                product_match,
                metric,
                month_year_filter is not None,
                question,
            ),
        }

    def _build_schema_documents(self) -> List[Dict[str, str]]:
        documents = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            hint = self.column_name_hints.get(col, "")
            text = f"{col} {hint} dtype {dtype}"
            documents.append(
                {
                    "column": col,
                    "dtype": dtype,
                    "hint": hint,
                    "text": text,
                }
            )
        return documents

    def _tokenize_search_text(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())

    def _normalize_search_token(self, token: str) -> str:
        return TOKEN_NORMALIZATION.get(token.lower(), token.lower())

    def _meaningful_query_tokens(self, question: str) -> List[str]:
        tokens = []
        seen = set()
        for token in self._tokenize_search_text(question):
            if len(token) < 3:
                continue
            if token in PRODUCT_QUERY_STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    def _normalized_meaningful_query_tokens(self, question: str) -> List[str]:
        tokens = []
        seen = set()
        for token in self._meaningful_query_tokens(question):
            normalized = self._normalize_search_token(token)
            if normalized in seen:
                continue
            seen.add(normalized)
            tokens.append(normalized)
        return tokens

    def _build_product_catalog(self) -> None:
        if PRODUCT_TEXT_COLUMN not in self.df.columns:
            return

        produkt_series = self._stringify_series(PRODUCT_TEXT_COLUMN)
        non_empty_mask = produkt_series != ""
        if not non_empty_mask.any():
            return

        catalog_df = (
            self.df.loc[non_empty_mask]
            .assign(_produkt_value=produkt_series.loc[non_empty_mask].values)
            .groupby("_produkt_value", dropna=False)
            .agg(
                row_count=("_produkt_value", "size"),
                distinct_orders=(ORDER_ID_COLUMN, "nunique"),
                category=("WG_NAME", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
            )
            .reset_index()
            .rename(columns={"_produkt_value": "text"})
            .sort_values(["row_count", "distinct_orders", "text"], ascending=[False, False, True])
            .reset_index(drop=True)
        )

        self._product_catalog = []
        self._product_inverted_index = {}
        self._product_normalized_index = {}
        for idx, row in catalog_df.iterrows():
            text = str(row["text"]).strip()
            ordered_tokens = self._tokenize_search_text(text)
            tokens = set(ordered_tokens)
            normalized_tokens = {self._normalize_search_token(token) for token in tokens}
            record = {
                "idx": int(idx),
                "text": text,
                "row_count": int(row["row_count"]),
                "distinct_orders": int(row["distinct_orders"]),
                "category": str(row["category"]) if pd.notna(row["category"]) else "",
                "tokens": tokens,
                "normalized_tokens": normalized_tokens,
                "normalized_text": " ".join(self._normalize_search_token(token) for token in ordered_tokens),
            }
            self._product_catalog.append(record)
            for token in tokens:
                if len(token) < 3:
                    continue
                self._product_inverted_index.setdefault(token, set()).add(int(idx))
            for token in normalized_tokens:
                if len(token) < 3:
                    continue
                self._product_normalized_index.setdefault(token, set()).add(int(idx))

    def _load_embedding_backend(self):
        try:
            import ollama

            return ollama.Client()
        except Exception:
            try:
                from langchain_community.embeddings import OllamaEmbeddings

                return OllamaEmbeddings(model=self.embedding_model)
            except Exception:
                return None

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._embedding_backend is None:
            return None

        try:
            if hasattr(self._embedding_backend, "embed_query"):
                vector = self._embedding_backend.embed_query(text)
            else:
                embed_method = getattr(self._embedding_backend, "embed", None)
                if embed_method is not None:
                    response = embed_method(model=self.embedding_model, input=text)
                else:
                    response = self._embedding_backend.embeddings(
                        model=self.embedding_model,
                        prompt=text,
                    )
                if "embeddings" in response:
                    vector = response["embeddings"][0]
                else:
                    vector = response["embedding"]
            return np.array(vector, dtype=float)
        except Exception:
            return None

    def _embed_documents(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        if self._embedding_backend is None or not texts:
            return [None for _ in texts]

        try:
            if hasattr(self._embedding_backend, "embed_documents"):
                vectors = self._embedding_backend.embed_documents(texts)
                return [np.array(vector, dtype=float) for vector in vectors]
        except Exception:
            pass

        vectors: List[Optional[np.ndarray]] = []
        for text in texts:
            vectors.append(self._embed_text(text))
        return vectors

    def _build_schema_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings = {}
        for doc in self._schema_documents:
            vector = self._embed_text(doc["text"])
            if vector is not None and np.linalg.norm(vector) > 0:
                embeddings[doc["column"]] = vector
        return embeddings

    def _normalize_tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _keyword_score(self, question: str, document: Dict[str, str]) -> float:
        query_tokens = set(self._normalize_tokens(question))
        doc_tokens = set(self._normalize_tokens(document["text"]))
        if not query_tokens or not doc_tokens:
            return 0.0

        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            return 0.0

        return overlap / max(len(query_tokens), 1)

    def _semantic_score(
        self,
        question_embedding: Optional[np.ndarray],
        column: str,
    ) -> float:
        if question_embedding is None:
            return 0.0

        column_embedding = self._schema_embeddings.get(column)
        if column_embedding is None:
            return 0.0

        denominator = np.linalg.norm(question_embedding) * np.linalg.norm(column_embedding)
        if denominator == 0:
            return 0.0

        return float(np.dot(question_embedding, column_embedding) / denominator)

    def _extract_literal_candidates(self, question: str) -> List[str]:
        candidates = []
        seen = set()
        for token in self._extract_quoted_candidates(question):
            normalized = token.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(token)

        for token in self._extract_product_phrase_candidates(question):
            normalized = token.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(token)

        for token in re.findall(r"[A-Za-z0-9_-]+", question):
            if len(token) < 4:
                continue
            has_digit = any(char.isdigit() for char in token)
            has_alpha = any(char.isalpha() for char in token)
            if not has_digit:
                continue
            if not has_alpha and len(token) < 6:
                continue
            normalized = token.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(token)

        for token in self._extract_product_ngram_candidates(question):
            normalized = token.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(token)
        return candidates

    def _extract_quoted_candidates(self, question: str) -> List[str]:
        candidates = []
        for match in re.findall(r"['\"]([^'\"]{3,})['\"]", question):
            cleaned = self._clean_candidate_phrase(match)
            if cleaned:
                candidates.append(cleaned)
        return candidates

    def _extract_product_phrase_candidates(self, question: str) -> List[str]:
        marker_patterns = [
            r"\bproduct\s+id\b",
            r"\bproduct\b",
            r"\bprodukt\b",
            r"\bartikelnummer\b",
            r"\bartikel\b",
            r"\barticle\b",
            r"\bsku\b",
            r"\bart[_ ]?nr\b",
        ]
        combined_pattern = "|".join(marker_patterns)
        stopwords = {
            "in",
            "im",
            "on",
            "for",
            "from",
            "mit",
            "with",
            "had",
            "hat",
            "have",
            "has",
            "there",
            "them",
            "it",
            "orders",
            "order",
            "bestellungen",
            "april",
            "march",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }
        candidates = []

        patterns = [
            re.compile(
                rf"(?i)(?:{combined_pattern})\s+['\"]?(?P<value>[^?.!,;]+?)['\"]?(?=$|[?.!,;]|\b(?:in|im|mit|with|for|from|on|am|an|during|zwischen|between|hat|had|have|has)\b)"
            ),
            re.compile(
                rf"(?i)(?:{combined_pattern})\s+['\"]?(?P<value>[A-Za-z0-9][A-Za-z0-9 _/\-]{{2,80}})"
            ),
        ]

        for pattern in patterns:
            for match in pattern.finditer(question):
                cleaned = self._clean_candidate_phrase(match.group("value"))
                if not cleaned:
                    continue
                words = cleaned.lower().split()
                if words and all(word in stopwords for word in words):
                    continue
                candidates.append(cleaned)
        return candidates

    def _clean_candidate_phrase(self, value: str) -> str:
        cleaned = value.strip(" \t\n\r\"'`")
        cleaned = re.sub(r"^(?:the|a|an|den|die|das|dem|der|ein|eine)\s+", "", cleaned, flags=re.IGNORECASE)
        trailing_pattern = re.compile(
            r"\s+(?:in|im|mit|with|for|from|on|am|an|during|zwischen|between|hat|had|have|has|them|it)$",
            flags=re.IGNORECASE,
        )
        while True:
            updated = trailing_pattern.sub("", cleaned)
            if updated == cleaned:
                break
            cleaned = updated
        cleaned = cleaned.strip(" \t\n\r\"'`-,:;")
        if len(cleaned) < 3:
            return ""
        return cleaned

    def _extract_product_ngram_candidates(self, question: str) -> List[str]:
        if PRODUCT_TEXT_COLUMN not in self.df.columns:
            return []

        cache_key = question.strip().lower()
        cached = self._product_phrase_cache.get(cache_key)
        if cached is not None:
            return [str(item["token"]) for item in cached]

        stopwords = {
            "how", "many", "much", "what", "which", "who", "where", "when", "why",
            "did", "do", "does", "had", "has", "have", "were", "was", "we", "our",
            "there", "them", "they", "with", "without", "from", "into", "during",
            "for", "and", "or", "the", "a", "an", "in", "on", "of", "to",
            "wie", "viele", "viel", "welche", "welcher", "welches", "hatten", "hat",
            "haben", "wir", "uns", "mit", "ohne", "von", "im", "in", "am", "an",
            "orders", "order", "bestellungen", "bestellung", "umsatz", "revenue",
            "march", "april", "may", "june", "july", "august", "september",
            "october", "november", "december", "january", "february",
            "maerz", "märz", "januar", "februar", "april", "mai", "juni", "juli",
            "august", "september", "oktober", "november", "dezember",
            "2023", "2024", "2025", "2026", "2027",
            "product", "produkt", "article", "artikel", "sku", "id",
        }
        words = [
            word for word in re.findall(r"[A-Za-zÀ-ÿ0-9]+", question)
            if len(word) >= 3 and word.lower() not in stopwords
        ]
        if len(words) < 2:
            self._product_phrase_cache[cache_key] = []
            return []

        produkt_values = self._stringify_series(PRODUCT_TEXT_COLUMN)
        candidate_phrases = []
        seen = set()
        max_ngram = min(3, len(words))
        for size in range(max_ngram, 1, -1):
            for start in range(0, len(words) - size + 1):
                phrase = " ".join(words[start:start + size])
                normalized = phrase.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                mask = produkt_values.str.contains(phrase, case=False, na=False, regex=False)
                row_count = int(mask.sum())
                if row_count == 0:
                    continue
                distinct_orders = int(self.df.loc[mask, ORDER_ID_COLUMN].nunique()) if ORDER_ID_COLUMN in self.df.columns else row_count
                candidate_phrases.append(
                    {
                        "token": phrase,
                        "row_count": row_count,
                        "distinct_orders": distinct_orders,
                    }
                )

        candidate_phrases.sort(
            key=lambda item: (
                item["token"].count(" "),
                item["distinct_orders"],
                item["row_count"],
            ),
            reverse=True,
        )
        top_candidates = candidate_phrases[:3]
        self._product_phrase_cache[cache_key] = top_candidates
        return [str(item["token"]) for item in top_candidates]

    def _literal_exact_match_columns(self) -> List[str]:
        priority = [PRODUCT_ID_COLUMN, ORDER_ID_COLUMN, "RECHNUNG", "NUMMER"]
        return [col for col in priority if col in self.df.columns]

    def _literal_contains_match_columns(self) -> List[str]:
        priority = [PRODUCT_TEXT_COLUMN, "WG_NAME", "GROESSE", "FARBE"]
        return [col for col in priority if col in self.df.columns]

    def _stringify_series(self, column: str) -> pd.Series:
        return self.df[column].fillna("").astype(str).str.strip()

    def _search_literal_token(self, token: str) -> List[Dict[str, object]]:
        cache_key = token.upper()
        cached_matches = self._literal_match_cache.get(cache_key)
        if cached_matches is not None:
            return cached_matches

        matches: List[Dict[str, object]] = []

        for column in self._literal_exact_match_columns():
            values = self._stringify_series(column)
            mask = values.str.upper() == cache_key
            row_count = int(mask.sum())
            if row_count == 0:
                continue
            distinct_orders = (
                int(self.df.loc[mask, ORDER_ID_COLUMN].nunique())
                if ORDER_ID_COLUMN in self.df.columns
                else row_count
            )
            matches.append(
                {
                    "token": token,
                    "column": column,
                    "match_type": "exact",
                    "row_count": row_count,
                    "distinct_orders": distinct_orders,
                    "samples": values.loc[mask].drop_duplicates().head(3).tolist(),
                }
            )

        for column in self._literal_contains_match_columns():
            values = self._stringify_series(column)
            mask = values.str.contains(token, case=False, na=False, regex=False)
            row_count = int(mask.sum())
            if row_count == 0:
                continue
            distinct_orders = (
                int(self.df.loc[mask, ORDER_ID_COLUMN].nunique())
                if ORDER_ID_COLUMN in self.df.columns
                else row_count
            )
            matches.append(
                {
                    "token": token,
                    "column": column,
                    "match_type": "contains",
                    "row_count": row_count,
                    "distinct_orders": distinct_orders,
                    "samples": values.loc[mask].drop_duplicates().head(3).tolist(),
                }
            )

        self._literal_match_cache[cache_key] = matches
        return matches

    def _literal_match_columns(self, question: str) -> List[str]:
        columns = []
        seen = set()
        for token in self._extract_literal_candidates(question):
            for match in self._search_literal_token(token):
                column = str(match["column"])
                if column in seen:
                    continue
                seen.add(column)
                columns.append(column)
        for match in self._hybrid_product_candidates(question, top_k=3):
            column = str(match["column"])
            if column in seen:
                continue
            seen.add(column)
            columns.append(column)
        return columns

    def _rank_schema_matches(self, question: str, top_k: int = 5) -> List[Dict[str, float]]:
        question_embedding = self._embed_text(question)
        literal_match_columns = set(self._literal_match_columns(question))
        matches = []

        for document in self._schema_documents:
            keyword_score = self._keyword_score(question, document)
            semantic_score = self._semantic_score(question_embedding, document["column"])
            literal_boost = 1.0 if document["column"] in literal_match_columns else 0.0
            combined_score = (0.4 * keyword_score) + (0.6 * semantic_score) + literal_boost

            matches.append(
                {
                    "column": document["column"],
                    "dtype": document["dtype"],
                    "hint": document["hint"],
                    "keyword_score": keyword_score,
                    "semantic_score": semantic_score,
                    "literal_boost": literal_boost,
                    "combined_score": combined_score,
                }
            )

        matches.sort(
            key=lambda item: (
                item["combined_score"],
                item["literal_boost"],
                item["semantic_score"],
                item["keyword_score"],
            ),
            reverse=True,
        )
        return matches[:top_k]

    def _format_relevant_schema(self, matches: List[Dict[str, float]]) -> str:
        if not matches:
            return "- No ranked schema matches were available."

        lines = []
        for match in matches:
            lines.append(
                f"- {match['column']} (Type: {match['dtype']}) — {match['hint']} "
                f"[keyword={match['keyword_score']:.2f}, semantic={match['semantic_score']:.2f}, "
                f"literal={match['literal_boost']:.0f}]"
            )
        return "\n".join(lines)

    def _build_filter_hint(self, column: str, token: str, match_type: str) -> str:
        # token may be unused when a richer structured match is available.
        if match_type == "exact":
            return f"df['{column}'].astype(str).str.upper() == '{token.upper()}'"
        return (
            f"df['{column}'].astype(str).str.contains("
            f"'{token}', case=False, na=False, regex=False)"
        )

    def _build_match_code_expr(self, match: Dict[str, object]) -> str:
        column = str(match["column"])
        token_groups = match.get("token_groups") or []
        if token_groups:
            group_exprs = []
            for group in token_groups:
                variants = [
                    f"df['{column}'].astype(str).str.contains('{variant}', case=False, na=False, regex=False)"
                    for variant in group
                ]
                group_exprs.append("(" + " | ".join(variants) + ")")
            expr = " & ".join(group_exprs) if group_exprs else "pd.Series(True, index=df.index)"
            category_filter = str(match.get("category_filter", "")).strip()
            if category_filter:
                expr = f"({expr}) & (df['WG_NAME'].astype(str) == '{category_filter}')"
            return expr
        return self._build_filter_hint(column, str(match["token"]), str(match["match_type"]))

    def _build_literal_match_context(self, question: str) -> str:
        lines = []
        for token in self._extract_literal_candidates(question):
            matches = self._search_literal_token(token)
            if not matches:
                continue
            lines.append(f"- Token `{token}` was found in the data:")
            for match in matches[:4]:
                sample_values = ", ".join(f"`{sample}`" for sample in match["samples"])
                if not sample_values:
                    sample_values = "`<no sample>`"
                filter_hint = self._build_filter_hint(
                    str(match["column"]),
                    token,
                    str(match["match_type"]),
                )
                lines.append(
                    f"  - column `{match['column']}` via {match['match_type']} match; "
                    f"{match['row_count']} rows, {match['distinct_orders']} distinct orders; "
                    f"samples: {sample_values}; preferred filter: `{filter_hint}`"
                )
        hybrid_candidates = self._hybrid_product_candidates(question, top_k=3)
        if hybrid_candidates:
            lines.append("- Hybrid product candidates from keyword + semantic retrieval:")
            for match in hybrid_candidates:
                sample_values = ", ".join(f"`{sample}`" for sample in match["samples"])
                filter_hint = self._build_filter_hint(
                    str(match["column"]),
                    str(match["token"]),
                    str(match["match_type"]),
                )
                lines.append(
                    f"  - column `{match['column']}` via hybrid retrieval; "
                    f"filter phrase `{match['token']}`; keyword={match['keyword_score']:.2f}, "
                    f"semantic={match['semantic_score']:.2f}, combined={match['combined_score']:.2f}; "
                    f"{match['row_count']} rows, {match['distinct_orders']} distinct orders; "
                    f"samples: {sample_values}; category: `{match.get('category', '')}`; "
                    f"category filter: `{match.get('category_filter', '')}`; "
                    f"token groups: `{match.get('token_groups', [])}`; preferred filter: `{self._build_match_code_expr(match)}`"
                )
        return "\n".join(lines)

    def _build_schema_context(
        self,
        question: str,
        include_full_schema: bool = False,
        top_k: int = 5,
    ) -> str:
        ranked_matches = self._rank_schema_matches(question, top_k=top_k)
        relevant_schema = self._format_relevant_schema(ranked_matches)
        literal_match_context = self._build_literal_match_context(question)

        sections = [f"Most relevant schema fields:\n{relevant_schema}"]
        if literal_match_context:
            sections.append(f"Direct literal matches from the question:\n{literal_match_context}")
        if include_full_schema:
            sections.append(f"Complete schema:\n{self.data_description}")
        return "\n\n".join(sections)

    def _extract_code(self, output: str) -> str:
        match = re.search(r"```(?:python)?(.*?)```", output, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if "result =" in code:
                return code

            lines = code.split("\n")
            for index in range(len(lines) - 1, -1, -1):
                line = lines[index].strip()
                if line and not line.startswith("#"):
                    if not line.startswith("result =") and "=" not in line:
                        lines[index] = f"result = {line}"
                    break
            return "\n".join(lines)
        return "result = None"

    def _run_code(self, code: str):
        try:
            local_vars = {"df": self.df.copy(), "pd": pd, "np": np}
            exec(code, local_vars)
            return local_vars.get("result", None)
        except Exception as exc:
            import traceback

            error_details = traceback.format_exc()
            return (
                f"Execution Error: {exc}\n\nGenerated Code:\n{code}\n\n"
                f"Full Traceback:\n{error_details}"
            )

    def _result_has_execution_error(self, result) -> bool:
        return isinstance(result, str) and result.startswith("Execution Error:")

    def _result_needs_retry(self, result) -> bool:
        if self._result_has_execution_error(result):
            return True
        if result is None:
            return True
        if isinstance(result, str) and result.strip() in {"", "No result", "None"}:
            return True
        if isinstance(result, (pd.Series, pd.DataFrame)) and result.empty:
            return True
        return False

    def _generate_and_run(self, question: str, schema_context: str):
        llm_output = self.query_chain.run(
            {
                "question": question,
                "schema_context": schema_context,
            }
        )
        code = self._extract_code(llm_output)
        result = self._run_code(code)
        return llm_output, code, result

    def _format_number_german(self, value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return str(value)

    def _format_multivalue_result(self, result):
        if isinstance(result, pd.Series):
            formatted = result.apply(self._format_number_german).reset_index()
            formatted.columns = ["Kategorie", "Wert"]
            return formatted.to_markdown(index=True)
        if isinstance(result, pd.DataFrame):
            result = result.copy()
            for col in result.select_dtypes(include=[np.number]).columns:
                result[col] = result[col].apply(self._format_number_german)
            return result.to_markdown(index=True)
        return str(result)

    def _result_preview(self, result) -> str:
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if result.empty:
                return "No result"
            return self._format_multivalue_result(result)
        if result is None or (isinstance(result, str) and result.strip() in {"", "No result", "None"}):
            return "No result"
        if isinstance(result, (int, float, np.integer, np.floating)):
            return self._format_number_german(result)
        return str(result)

    def _build_trace(
        self,
        *,
        question: str,
        route: str,
        schema_context: Optional[str] = None,
        llm_output: Optional[str] = None,
        code: Optional[str] = None,
        result=None,
        deterministic_match: Optional[Dict[str, object]] = None,
        used_fallback: bool = False,
    ) -> Dict[str, object]:
        return {
            "question": question,
            "route": route,
            "schema_context": schema_context or "",
            "literal_candidates": self._extract_literal_candidates(question),
            "hybrid_candidates": self._hybrid_product_candidates(question, top_k=5),
            "deterministic_match": deterministic_match or {},
            "used_fallback": used_fallback,
            "llm_output": llm_output or "",
            "code": code or "",
            "result_preview": self._result_preview(result),
        }

    def get_last_trace(self) -> Dict[str, object]:
        return dict(self._last_trace)

    def ask_with_trace(
        self,
        question: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, Dict[str, object]]:
        answer = self.ask(question, progress_callback=progress_callback)
        return answer, self.get_last_trace()

    def ask(
        self,
        question: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        normalized_question = question.strip()
        if normalized_question in self._answer_cache:
            if progress_callback:
                progress_callback("Using cached answer.")
            self._last_trace = dict(self._trace_cache.get(normalized_question, {}))
            return self._answer_cache[normalized_question]

        deterministic_analysis = self._deterministic_product_analysis(normalized_question)
        if deterministic_analysis is not None:
            if progress_callback:
                progress_callback("Using deterministic product matching.")
            formatted_result = self._result_preview(deterministic_analysis["result"])
            answer_text = (
                f"The question asks {normalized_question}\n\n"
                f"The Answer is:\n\n"
                f"{formatted_result}\n\n"
            )
            self._last_trace = self._build_trace(
                question=normalized_question,
                route=str(deterministic_analysis["metric"]),
                schema_context=self._build_schema_context(normalized_question, include_full_schema=False),
                code=str(deterministic_analysis["code"]),
                result=deterministic_analysis["result"],
                deterministic_match=dict(deterministic_analysis["match"]),
            )
            self._answer_cache[normalized_question] = answer_text
            self._trace_cache[normalized_question] = dict(self._last_trace)
            return answer_text

        if progress_callback:
            progress_callback("Matching your question to the schema.")
        compact_schema_context = self._build_schema_context(
            normalized_question,
            include_full_schema=False,
        )

        if progress_callback:
            progress_callback("Generating pandas query with compact schema context.")
        llm_output, code, result = self._generate_and_run(
            normalized_question,
            compact_schema_context,
        )
        used_fallback = False

        if self._result_needs_retry(result):
            used_fallback = True
            if progress_callback:
                progress_callback("Retrying with the full schema for a more reliable match.")
            full_schema_context = self._build_schema_context(
                normalized_question,
                include_full_schema=True,
            )
            llm_output, code, result = self._generate_and_run(
                normalized_question,
                full_schema_context,
            )

        if self.debug:
            print("=" * 60)
            print("DEBUG MODE - RAG Processing Steps:")
            print("=" * 60)
            print(f"1. QUESTION: {normalized_question}")
            print(f"\n1b. COMPACT SCHEMA CONTEXT:\n{compact_schema_context}")
            print(f"\n1c. USED FALLBACK FULL SCHEMA: {used_fallback}")
            print(f"\n2. LLM RAW OUTPUT:\n{llm_output}")
            print(f"\n3. EXTRACTED CODE:\n{code}")
            print(f"\n4. RESULT:\n{result}")
            print("=" * 60)

        self._last_trace = self._build_trace(
            question=normalized_question,
            route="llm",
            schema_context=compact_schema_context,
            llm_output=llm_output,
            code=code,
            result=result,
            used_fallback=used_fallback,
        )

        if isinstance(result, (pd.Series, pd.DataFrame)) and not result.empty:
            formatted_result = self._format_multivalue_result(result)
        elif result is None or (isinstance(result, str) and result.strip() in {"", "No result", "None"}):
            formatted_result = "No result"
        elif isinstance(result, (pd.Series, pd.DataFrame)):
            formatted_result = "No result"
        else:
            formatted_result = (
                self._format_number_german(result)
                if isinstance(result, (int, float, np.integer, np.floating))
                else str(result)
            )

        answer_text = (
            f"The question asks {normalized_question}\n\n"
            f"The Answer is:\n\n"
            f"{formatted_result}\n\n"
        )

        self._answer_cache[normalized_question] = answer_text
        self._trace_cache[normalized_question] = dict(self._last_trace)
        return answer_text
