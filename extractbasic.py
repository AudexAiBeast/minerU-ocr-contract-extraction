import streamlit as st
import json
import re
import ollama
import copy
import subprocess
import tempfile
import os

from pathlib import Path
from bs4 import BeautifulSoup

# ==========================================================
# CONFIG
# ==========================================================

MODEL = "granite3.3:latest"

SYSTEM_PROMPT = """
You are FreightIQ.

You are an expert logistics contract extraction engine.

STRICT RULES:

1  Ignore OCR noise
2  Ignore duplicate text
3  Ignore page headers
4  Ignore page footers
5  Ignore signatures
6  Ignore bank details
7  Ignore GST details
8  Ignore addresses
9  Ignore witness details
10 Ignore account details
11 Ignore legal boilerplate
12 Ignore long clause copying
13 Missing values = null
14 Return JSON only
15 Never invent values
16 Never extract entire paragraphs
17 Keep extracted clauses <= 20 words

CRITICAL DATE RULES:
- All date fields must contain a date value ONLY
- Date fields must NEVER contain sentences
- If no clear date exists for a field, set it to null

CRITICAL CLAUSE RULES:
- claim_settlement_clause  : timeline + limit only
- liability_clause         : amount/limit only
- termination_clause       : notice period only
- payment_clause           : payment days only
- tat_clause               : TAT days only

Extract only the following JSON structure:

{
  "contract_information": {
    "contract_title": "",
    "agreement_id": "",
    "contract_effective_from_date": "",
    "contract_end_date": "",
    "contract_valid_until": "",
    "contract_duration": "",
    "contract_signed_date": "",
    "contract_issued_on_date": ""
  },
  "parties": {
    "lsp_name": "",
    "shipper_name": ""
  },
  "commercial_terms": {
    "freight_type": "",
    "minimum_chargeable_weight": "",
    "docket_charge": "",
    "fuel_surcharge": "",
    "oda_charge": "",
    "liability_limit": "",
    "payment_frequency": "",
    "credit_period": ""
  },
  "transport": {
    "zones": []
  },
  "critical_clauses": {
    "claim_settlement_clause": "",
    "liability_clause": "",
    "termination_clause": "",
    "payment_clause": "",
    "tat_clause": ""
  }
}

Return JSON only.
"""

# ==========================================================
# MASTER SCHEMA
# ==========================================================

MASTER_SCHEMA = {
    "contract_information": {
        "contract_title": None,
        "agreement_id": None,
        "contract_effective_from_date": None,
        "contract_end_date": None,
        "contract_valid_until": None,
        "contract_duration": None,
        "contract_signed_date": None,
        "contract_issued_on_date": None,
    },
    "parties": {
        "lsp_name": None,
        "shipper_name": None
    },
    "commercial_terms": {
        "freight_type": None,
        "minimum_chargeable_weight": None,
        "docket_charge": None,
        "fuel_surcharge": None,
        "oda_charge": None,
        "liability_limit": None,
        "payment_frequency": None,
        "credit_period": None,
    },
    "transport": {
        "zones": []
    },
    "critical_clauses": {
        "claim_settlement_clause": None,
        "liability_clause": None,
        "termination_clause": None,
        "payment_clause": None,
        "tat_clause": None,
    },
}

# ==========================================================
# IGNORE TABLE PATTERNS
# ==========================================================

IGNORE_TABLE_PATTERNS = [
    "account details",
    "bank",
    "account holder",
    "ifsc",
    "branch",
    "billing address",
    "billing city",
    "billing state",
    "billing pin code",
    "gst number",
    "signature",
    "stamp",
    "witness",
    "address",
]

# ==========================================================
# DATE FIELDS
# ==========================================================

DATE_FIELDS = [
    "contract_effective_from_date",
    "contract_end_date",
    "contract_valid_until",
    "contract_signed_date",
    "contract_issued_on_date",
    "contract_duration",
]

DATE_MAX_LEN = 40

CLAUSE_INDICATOR_WORDS = re.compile(
    r"\b(shall|herein|party|customer|service|provider|agreement)\b",
    re.IGNORECASE,
)

# ==========================================================
# CLEAN MARKDOWN
# ==========================================================

def clean_markdown(text):

    cleaned = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        if re.search(
            r"^\s*-*\s*page\s+\d+\s*-*\s*$",
            line,
            re.IGNORECASE
        ):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

# ==========================================================
# NORMALIZE OCR DATES
# ==========================================================

def normalize_ocr_dates(text):

    text = re.sub(
        r"\$\s*\^\{.*?\}\s*\$",
        "",
        text
    )

    text = re.sub(
        r"(\d+)(st|nd|rd|th)\b",
        r"\1",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text

# ==========================================================
# EXTRACT DATES
# ==========================================================

def extract_dates(text):

    result = {
        "contract_effective_from_date": None,
        "contract_end_date": None,
        "contract_signed_date": None,
    }

    text = normalize_ocr_dates(text)

    effective_patterns = [
        r"effective\s+from\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"effective\s+date\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    ]

    expiry_patterns = [
        r"valid\s+(?:until|till)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"end\s+date\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    ]

    signed_patterns = [
        r"signed\s+on\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"execution\s+date\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    ]

    for p in effective_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            result["contract_effective_from_date"] = m.group(1)
            break

    for p in expiry_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            result["contract_end_date"] = m.group(1)
            break

    for p in signed_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            result["contract_signed_date"] = m.group(1)
            break

    return result

# ==========================================================
# EXTRACT DURATION
# ==========================================================

def extract_duration(text):

    patterns = [
        r"(\d+\s+year[s]?)",
        r"(\d+\s+month[s]?)",
    ]

    for p in patterns:

        m = re.search(
            p,
            text,
            re.IGNORECASE
        )

        if m:
            return m.group(1)

    return None

# ==========================================================
# SANITIZE OUTPUT
# ==========================================================

def sanitize_extracted(data):

    ci = data.get("contract_information", {})

    for field in DATE_FIELDS:

        val = ci.get(field)

        if val and isinstance(val, str):

            val = val.strip()

            if len(val) > DATE_MAX_LEN:
                ci[field] = None

            elif CLAUSE_INDICATOR_WORDS.search(val):
                ci[field] = None

    return data

# ==========================================================
# EXTRACT TABLES
# ==========================================================

def extract_html_tables(md):

    tables = []

    soup = BeautifulSoup(
        md,
        "html.parser"
    )

    for idx, table in enumerate(soup.find_all("table")):

        rows = []

        for tr in table.find_all("tr"):

            row = [
                cell.get_text(" ", strip=True)
                for cell in tr.find_all(["th", "td"])
            ]

            if row:
                rows.append(row)

        if rows:

            headers = rows[0]

            data = []

            for r in rows[1:]:

                while len(r) < len(headers):
                    r.append("")

                data.append(r)

            tables.append({
                "table_id": f"table_{idx+1}",
                "headers": headers,
                "rows": data,
            })

    return tables

# ==========================================================
# FILTER TABLES
# ==========================================================

def filter_tables(tables):

    final = []

    for t in tables:

        txt = " ".join(t["headers"])

        for row in t["rows"]:
            txt += " " + " ".join(row)

        txt = txt.lower()

        skip = any(
            pattern in txt
            for pattern in IGNORE_TABLE_PATTERNS
        )

        if not skip:
            final.append(t)

    return final

# ==========================================================
# TABLE EXTRACTION
# ==========================================================

def extract_table_business_data(
    tables,
    source_text
):

    result = {
        "minimum_chargeable_weight": None,
        "docket_charge": None,
        "fuel_surcharge": None,
        "oda_charge": None,
        "liability_limit": None,
        "payment_frequency": None,
        "credit_period": None,
    }

    patterns = {
        "fuel_surcharge": [
            r"fuel\s+surcharge[:\-\s]*(.+)",
            r"fsc[:\-\s]*(.+)"
        ],
        "docket_charge": [
            r"docket\s+charges?[:\-\s]*(.+)"
        ],
        "oda_charge": [
            r"oda\s+charges?[:\-\s]*(.+)"
        ],
        "credit_period": [
            r"credit\s+period[:\-\s]*(.+)"
        ],
    }

    all_lines = []

    for t in tables:

        for h in t["headers"]:
            all_lines.append(str(h))

        for row in t["rows"]:
            for cell in row:
                all_lines.append(str(cell))

    for line in all_lines:

        for field, plist in patterns.items():

            for p in plist:

                m = re.search(
                    p,
                    line,
                    re.IGNORECASE
                )

                if m:

                    value = m.group(1).strip()

                    if len(value) > 100:
                        continue

                    result[field] = value

    return result

# ==========================================================
# REMOVE TABLES
# ==========================================================

def remove_tables(md):

    soup = BeautifulSoup(
        md,
        "html.parser"
    )

    for t in soup.find_all("table"):
        t.decompose()

    return soup.get_text()

# ==========================================================
# SPLIT PARAGRAPHS
# ==========================================================

def split_paragraphs(text):

    paras = []

    current = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:

            if current:
                paras.append(" ".join(current))
                current = []

            continue

        current.append(line)

    if current:
        paras.append(" ".join(current))

    return paras

# ==========================================================
# CHUNK TEXT
# ==========================================================

def chunk_text(
    text,
    max_chars=1800
):

    words = text.split()

    chunks = []

    current = []

    size = 0

    for w in words:

        current.append(w)

        size += len(w)

        if size > max_chars:

            chunks.append(
                " ".join(current)
            )

            current = []

            size = 0

    if current:
        chunks.append(
            " ".join(current)
        )

    return chunks

# ==========================================================
# LLM ANALYSIS
# ==========================================================

def analyse_chunk(text):

    try:

        response = ollama.chat(
            model=MODEL,
            options={"temperature": 0},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": text
                },
            ],
        )

        content = response["message"]["content"]

        m = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if m:

            raw = m.group()

            raw = re.sub(
                r",\s*([\}\]])",
                r"\1",
                raw
            )

            return json.loads(raw)

    except Exception:
        return None

# ==========================================================
# MERGE
# ==========================================================

def recursive_merge(base, incoming):

    for k, v in incoming.items():

        if k not in base:
            base[k] = v
            continue

        if (
            isinstance(v, dict)
            and isinstance(base.get(k), dict)
        ):

            recursive_merge(
                base[k],
                v
            )

        elif (
            isinstance(v, list)
            and isinstance(base.get(k), list)
        ):

            existing = set(
                map(str, base[k])
            )

            for x in v:

                if str(x) not in existing:
                    base[k].append(x)

        else:

            if (
                base[k] in [None, ""]
                and v not in [None, ""]
            ):
                base[k] = v

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.set_page_config(
    page_title="FreightIQ Contract Extractor",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 FreightIQ Contract Extractor")

st.caption(
    "Upload logistics contract PDF"
)

uploaded = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded:

    with tempfile.TemporaryDirectory() as temp_dir:

        pdf_path = os.path.join(
            temp_dir,
            uploaded.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())

        mineru_output = os.path.join(
            temp_dir,
            "mineru_output"
        )

        os.makedirs(
            mineru_output,
            exist_ok=True
        )

        if st.button("Process Contract"):

            # ==================================================
            # RUN MINERU
            # ==================================================

            with st.spinner(
                "Running MinerU..."
            ):

                command = [
                    "mineru",
                    "-p",
                    pdf_path,
                    "-o",
                    mineru_output
                ]

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )

            if result.returncode != 0:

                st.error("MinerU failed")

                st.text(result.stderr)

                st.stop()

            # ==================================================
            # FIND MARKDOWN
            # ==================================================

            md_files = list(
                Path(mineru_output).rglob("*.md")
            )

            if not md_files:

                st.error(
                    "No markdown file generated"
                )

                st.stop()

            md_file = md_files[0]

            with open(
                md_file,
                "r",
                encoding="utf-8"
            ) as f:

                md = f.read()

            st.success(
                "MinerU extraction complete"
            )

            # ==================================================
            # CLEAN MARKDOWN
            # ==================================================

            md = clean_markdown(md)

            # ==================================================
            # TABLE EXTRACTION
            # ==================================================

            with st.spinner(
                "Extracting tables..."
            ):

                tables = extract_html_tables(md)

                tables = filter_tables(tables)

            # ==================================================
            # DATE EXTRACTION
            # ==================================================

            date_data = extract_dates(md)

            duration = extract_duration(md)

            # ==================================================
            # COMMERCIAL TERMS
            # ==================================================

            table_data = extract_table_business_data(
                tables=tables,
                source_text=md
            )

            # ==================================================
            # REMOVE TABLES
            # ==================================================

            text = remove_tables(md)

            paragraphs = split_paragraphs(text)

            final = copy.deepcopy(
                MASTER_SCHEMA
            )

            recursive_merge(
                final["contract_information"],
                date_data
            )

            if (
                duration
                and not final["contract_information"].get(
                    "contract_duration"
                )
            ):

                final["contract_information"][
                    "contract_duration"
                ] = duration

            recursive_merge(
                final["commercial_terms"],
                table_data
            )

            chunks = []

            for p in paragraphs:

                if len(p) < 50:
                    continue

                chunks.extend(
                    chunk_text(p)
                )

            # ==================================================
            # LLM PROCESSING
            # ==================================================

            st.subheader(
                "Processing contract..."
            )

            progress = st.progress(0)

            total = len(chunks)

            for i, c in enumerate(chunks):

                result = analyse_chunk(c)

                if result:

                    temp = copy.deepcopy(result)

                    if "commercial_terms" in temp:
                        del temp["commercial_terms"]

                    recursive_merge(
                        final,
                        temp
                    )

                progress.progress(
                    (i + 1) / total
                )

            progress.empty()

            # ==================================================
            # SANITIZE
            # ==================================================

            final = sanitize_extracted(final)

            # ==================================================
            # OUTPUT
            # ==================================================

            output = {
                "extracted_contract": final,
                "relevant_tables": tables
            }

            st.success(
                "✅ Extraction complete"
            )

            col1, col2 = st.columns([2, 1])

            with col1:

                st.subheader(
                    "Extracted Contract"
                )

                st.json(output)

            with col2:

                ci = final.get(
                    "contract_information",
                    {}
                )

                parties = final.get(
                    "parties",
                    {}
                )

                ct = final.get(
                    "commercial_terms",
                    {}
                )

                st.subheader("Summary")

                st.markdown("### Contract")

                st.write(
                    f"Effective: "
                    f"{ci.get('contract_effective_from_date') or '—'}"
                )

                st.write(
                    f"End Date: "
                    f"{ci.get('contract_end_date') or '—'}"
                )

                st.write(
                    f"Duration: "
                    f"{ci.get('contract_duration') or '—'}"
                )

                st.markdown("### Parties")

                st.write(
                    f"LSP: "
                    f"{parties.get('lsp_name') or '—'}"
                )

                st.write(
                    f"Shipper: "
                    f"{parties.get('shipper_name') or '—'}"
                )

                st.markdown(
                    "### Commercial Terms"
                )

                st.write(
                    f"Fuel Surcharge: "
                    f"{ct.get('fuel_surcharge') or '—'}"
                )

                st.write(
                    f"Credit Period: "
                    f"{ct.get('credit_period') or '—'}"
                )

                st.write(
                    f"ODA Charge: "
                    f"{ct.get('oda_charge') or '—'}"
                )

            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(
                    output,
                    indent=4
                ),
                file_name="contract_output.json",
                mime="application/json",
            )

            st.download_button(
                label="⬇️ Download Markdown",
                data=md,
                file_name="mineru_output.md",
                mime="text/markdown",
            )