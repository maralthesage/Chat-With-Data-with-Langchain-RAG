TARGET_COLUMNS = [
    "AUFTRAG_NR",
    "DATUM",
    "RECHNUNG",
    "ART_NR",
    "GROESSE",
    "NUMMER",
    "FARBE",
    "MENGE",
    "PREIS",
    "MWST",
    "WG_NAME",
    "EK",
    "BEZEICHNG",
]

NUMERIC_COLUMNS = ["MENGE", "PREIS", "MWST", "EK"]

PRODUCT_TEXT_COLUMN = "BEZEICHNG"
PRODUCT_ID_COLUMN = "ART_NR"
ORDER_ID_COLUMN = "AUFTRAG_NR"
DATE_COLUMN = "DATUM"

COLUMN_NAME_HINTS = {
    "AUFTRAG_NR": (
        "Order ID (Auftragsnummer) — unique identifier for each customer order. "
        "Use this to count distinct orders."
    ),
    "DATUM": (
        "Order date — datetime column in format YYYY-MM-DD. "
        "Use it for year, month, and date filters."
    ),
    "RECHNUNG": (
        "Invoice ID (Rechnung) — invoice number. One order can have one or more invoices."
    ),
    "ART_NR": (
        "Article number, product ID, SKU, item code — exact identifier for a product. "
        "Use exact matching for values such as 086L06P."
    ),
    "GROESSE": (
        "Product size or size variant."
    ),
    "NUMMER": (
        "Customer ID — customer identifier."
    ),
    "FARBE": (
        "Product color or color variant."
    ),
    "MENGE": (
        "Quantity — number of units ordered for the line item."
    ),
    "PREIS": (
        "Net sales price / line revenue — monetary value of the item line. "
        "If the user asks for revenue or Umsatz, this column is the main candidate."
    ),
    "MWST": (
        "VAT / value-added tax amount associated with the line item."
    ),
    "WG_NAME": (
        "Product category (Warengruppe) — descriptive product group or category."
    ),
    "EK": (
        "Purchase cost / Einstandspreis — internal cost of the item line."
    ),
    "BEZEICHNG": (
        "Product description / product name — free-text item description. "
        "Use this for product wording, names, and semantic or keyword product matching."
    ),
}
