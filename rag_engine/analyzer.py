import re
import pandas as pd
import numpy as np
from langchain_core.runnables import RunnableSequence
from langchain_community.llms import Ollama
from .prompts import get_analysis_prompt

class OllamaCsvRAG:
    def __init__(self, df: pd.DataFrame, model: str = "qwen3:latest", debug: bool = False):
        self.df = df
        self.llm = Ollama(model=model)
        self.query_chain = get_analysis_prompt() | self.llm
        self.debug = debug
        
        #### ============ Below, define the schema of your table(s). With each column name (if needed, like our case, translate them, propose variations of the name, and explain what goes in each column) ============ ####
        self.column_name_hints = { 
        "AUFTRAG_NR": (
            "Order ID (Auftragsnummer) — A unique identifier for each customer order. "
            "Can be used to count total number of sales. Example: 63877537"
        ),
        "RECHNUNG": (
            "Invoice ID (Rechnung) — one or more invoices per order. "
            "Can be used to count total number of invoices. "
            "May be repeated if multiple invoices are generated for a single order. "
            "Example: 63877537-1"
        ),
        "WG_NAME": (
            "Product category (Warengruppe) — Describes the product group. "
            "Can be grouped to analyze revenue by category. should be normalized so it's not case sensitive, and if someone give all lowercase "
            "or all uppercase, it should still be the same category. Example: 'ELEKTRONIK', 'elektronik', 'Elektronik' all refer to the same category."
        ),
        "Netto_Umsatz": (
            "Price (Netto Umsatz, netto revenue, Umsätze) — The value of each item in EUR. "
            "Summing this gives total netto revenue. Numeric. Example: 29.95. When asked for the Umsatz, usually this netto umsatz ist meant."
        ),
        "Brutto_Umsatz": (
            "Brutto Umsatzt (Netto Price + Mehrwertsteuer) — Tax rate or amount applied to the Netto Umsatz that makes the Price the customer has actually paid. "
            "Mostly higher than Netto Umsatz. Unless asked specifically for the Brutto Umsatz, usually the Netto Umsatz is meant. "
            "Example: 35.64."
        ),
        "MENGE": (
            "Quantity — Number of units ordered for the item/Product. "
            "Useful for aggregating total items sold."
        ),
        "SOURCE": (
            "from which source did the customer came to our website and ordered. Source is a translation of the MEDIACODES we have."
            "Example: users who buy from Amazon, or Sovendus or other sources. "
        ),
        "ProduKt": (
            "Product -  A combination of Product ID + Product Description."
            "identifies the product by its name or description or both when needed. "
        ),
        "DATUM": (
            "Order creation date, order date — datetime column in format YYYY-MM-DD. "
            "Use `.dt.year == 2025` to filter by year. "
            "Use `.dt.to_period('M') == '2025-02'` for month-level filters like February 2025."
        ),
        "NUMMER": (
            "Customer ID — customer identifier. May not be unique across files."
        ),
        "Herkunft": (
            "through which means did the customer ordered. did they order through phone, email, internet, or other means. "
        ),
        "Retouren": (
            "Returned items, returned products - Items that were returned. they can be Brauchbar or Unbrauchbar. "
            "when the field is empty, it means the item was not returned. "
            "Brauchbar means the item was returned and can be used again. "
            "Unbrauchbar means the item was returned and cannot be used again. "
        ),
        "Land": (
            "Country - where the order was made from."
            "DE: Deutschland, Germany"
            "AT: Austria, Österreich"
            "CH: Switzerland,Swiss,Schweiz" 
            "FR: France, Frankreich "
        )
    }

        self.data_description = self._generate_schema()
    
    def _generate_schema(self) -> str:
        lines = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            hint = self.column_name_hints.get(col, "")
            lines.append(f"- {col} (Type: {dtype}) — Synonyms: {hint}")
        return "\n".join(lines)

    def _extract_code(self, output: str) -> str:
        match = re.search(r"```python(.*?)```", output, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Check if code already ends with a result assignment
            if "result =" in code:
                return code
            # If not, append result assignment to the last line that's not a comment
            lines = code.split('\n')
            # Find the last non-empty, non-comment line
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line and not line.startswith('#'):
                    # If it's not already an assignment, wrap it
                    if not line.startswith('result =') and '=' not in line:
                        lines[i] = f"result = {line}"
                    break
            return '\n'.join(lines)
        return "result = None"

    def _run_code(self, code: str):
        try:
            local_vars = {"df": self.df.copy(), "pd": pd, "np": np}
            exec(code, local_vars)
            result = local_vars.get("result", "No result")
            return result
        except Exception as e:
            # Enhanced error reporting for debugging
            import traceback
            error_details = traceback.format_exc()
            return f"Execution Error: {e}\n\nGenerated Code:\n{code}\n\nFull Traceback:\n{error_details}"

    def _format_number_german(self, value):
        """ In germany, numbers are formatted like this 123.456,00 . 
        To have the same formatting in our output, here function 
        transforms standard formatting to German formatting"""
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return str(value)

    def _format_multivalue_result(self, result):
        """ This function formats tables, when the results is not just single value,
        but a mutltitude of values."""
        if isinstance(result, pd.Series):
            # Format numbers and convert to DataFrame for markdown
            formatted = result.apply(self._format_number_german).reset_index()
            formatted.columns = ['Kategorie','Wert']
            return formatted.to_markdown(index=True)
        elif isinstance(result, pd.DataFrame):
            # Format numeric columns
            for col in result.select_dtypes(include=[np.number]).columns:
                result[col] = result[col].apply(self._format_number_german)
            return result.to_markdown(index=True)
        else:
            return str(result)

    def ask(self, question: str) -> str:
        """this is an aggregating function, that gets in the question from the user,
        translate it into a query (python code in this case), clean it up from excess 
        text, and run it over the data using run_code() and based on the resulting value,
        formats it as a single value result or a mutlivalue one and outputs it.
        """
        llm_output = self.query_chain.invoke({
            "question": question,
            "data_description": self.data_description
        })

        code = self._extract_code(llm_output)
        
        # Debug output
        if self.debug:
            print("=" * 60)
            print("DEBUG MODE - RAG Processing Steps:")
            print("=" * 60)
            print(f"1. QUESTION: {question}")
            print(f"\n2. LLM RAW OUTPUT:\n{llm_output}")
            print(f"\n3. EXTRACTED CODE:\n{code}")
            print("=" * 60)
        
        result = self._run_code(code)

        # Determine if single or multi-value
        if isinstance(result, (pd.Series, pd.DataFrame)):
            formatted_result = self._format_multivalue_result(result)
            answer_text = (
                f"The question asks {question.strip()}\n\n"
                f"The Answer is:\n\n"
                f"{formatted_result}\n\n"

            )
        else:
            formatted_result = self._format_number_german(result) if isinstance(result, (int, float, np.integer, np.floating)) else str(result)
            answer_text = (
                f"The question asks {question.strip()}\n\n"
                f"The Answer is:\n\n"
                f"{formatted_result}\n\n"

            )

        return answer_text
