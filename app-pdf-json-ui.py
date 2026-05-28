import streamlit as st
import json
import re
import ollama
import copy
import subprocess
import tempfile
import os
import pandas as pd

from pathlib import Path
from bs4 import BeautifulSoup
from dateutil import parser as dparser

# ==========================================================
# CONFIG
# ==========================================================

# Fast model for speed — swap up if you want accuracy over speed
# ┌──────────────────────────────────────────────────────────────┐
# │  MODEL OPTIONS (all fit in 16 GB VRAM)                      │
# │                                                              │
# │  FASTEST (recommended):                                      │
# │    qwen3:4b       ~3 GB   ollama pull qwen3:4b   ~80 tok/s  │
# │    gemma3:4b      ~3 GB   ollama pull gemma3:4b  ~90 tok/s  │
# │    phi4-mini      ~3 GB   ollama pull phi4-mini  ~75 tok/s  │
# │                                                              │
# │  BALANCED:                                                   │
# │    qwen3:8b       ~5 GB   ollama pull qwen3:8b   ~50 tok/s  │
# │    granite3.3     ~5 GB   already installed       ~45 tok/s  │
# │                                                              │
# │  BEST ACCURACY (slow):                                       │
# │    qwen3:14b      ~9 GB   ollama pull qwen3:14b  ~25 tok/s  │
# │    deepseek-r1:14b ~9 GB  ollama pull deepseek-r1:14b       │
# └──────────────────────────────────────────────────────────────┘
MODEL = "qwen3:4b"

# ==========================================================
# TWO FOCUSED PROMPTS (shorter = faster inference)
# ==========================================================

# Prompt 1: Used on ALL TABLES concatenated — extracts commercials, parties, dates
TABLE_PROMPT = """You are a logistics contract parser. Extract data from the table text below.

Rules:
- Return ONLY valid JSON, nothing else, no markdown fences
- null for any field not found
- Dates: DD-MMM-YYYY format only, never sentences
- Values: numbers/percentages only, no long sentences

Return this exact JSON:
{
  "lsp_name": null,
  "shipper_name": null,
  "contract_effective_from_date": null,
  "contract_end_date": null,
  "contract_duration": null,
  "freight_type": null,
  "minimum_chargeable_weight": null,
  "docket_charge": null,
  "fuel_surcharge": null,
  "oda_charge": null,
  "fov_charge": null,
  "liability_limit": null,
  "invoice_frequency": null,
  "credit_period": null,
  "penalty": null,
  "volumetric_method": null
}"""

# Prompt 2: Used on contract TEXT (clauses only) — 2-3 large chunks max
TEXT_PROMPT = """You are a logistics contract clause extractor.

Rules:
- Return ONLY valid JSON, nothing else, no markdown fences  
- null if not found in this text
- Keep each value to 20 words max — core term/limit only
- claim_settlement_clause: timeline + limit only (e.g. "30 days, Rs.10000 max")
- liability_clause: amount cap only (e.g. "Rs.500/kg max Rs.10000")
- termination_clause: notice period only (e.g. "1 month written notice")
- payment_clause: days only (e.g. "30 days from invoice")
- tat_clause: days only (e.g. "1-2 days intra-zone")
- parties: company names only

Return this exact JSON:
{
  "lsp_name": null,
  "shipper_name": null,
  "contract_title": null,
  "contract_effective_from_date": null,
  "contract_end_date": null,
  "contract_signed_date": null,
  "contract_duration": null,
  "claim_settlement_clause": null,
  "liability_clause": null,
  "termination_clause": null,
  "payment_clause": null,
  "tat_clause": null
}"""

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
        "shipper_name": None,
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
    "transport": {"zones": []},
    "critical_clauses": {
        "claim_settlement_clause": None,
        "liability_clause": None,
        "termination_clause": None,
        "payment_clause": None,
        "tat_clause": None,
    },
}

# ==========================================================
# IGNORE TABLE PATTERNS (bank/address noise)
# ==========================================================

IGNORE_TABLE_PATTERNS = [
    "account holder", "ifsc", "hdfc", "icici", "sbi", "axis bank",
    "branch address", "current account", "billing address",
    "billing city", "billing state", "billing pin",
    "gst number", "signature", "stamp duty", "witness",
]

DATE_FIELDS = [
    "contract_effective_from_date", "contract_end_date",
    "contract_valid_until", "contract_signed_date",
    "contract_issued_on_date", "contract_duration",
]

DATE_MAX_LEN = 40
CLAUSE_WORDS = re.compile(
    r"\b(shall|herein|party|customer|service|provider|agreement|whereas|pursuant)\b",
    re.IGNORECASE,
)

# ==========================================================
# HELPERS
# ==========================================================

def clean_markdown(text):
    return "\n".join(
        line.strip() for line in text.split("\n")
        if line.strip()
        and not re.search(r"^\s*-*\s*page\s+\d+\s*-*\s*$", line, re.IGNORECASE)
    )

def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def parse_llm_json(content):
    """Strip think tags, fences, then parse JSON."""
    content = strip_think_tags(content)
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        raw = re.sub(r",\s*([\}\]])", r"\1", m.group())
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None

def normalize_ocr_dates(text):
    text = re.sub(r"\$\s*\^\{.*?\}\s*\$", "", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text)

# ==========================================================
# EXTRACT HTML TABLES
# ==========================================================

def extract_html_tables(md):
    tables = []
    soup = BeautifulSoup(md, "html.parser")
    for idx, table in enumerate(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            row = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if row:
                rows.append(row)
        if rows:
            headers = rows[0]
            data = [r + [""] * (len(headers) - len(r)) for r in rows[1:]]
            tables.append({"table_id": f"table_{idx+1}", "headers": headers, "rows": data})
    return tables

def filter_tables(tables):
    result = []
    for t in tables:
        txt = (" ".join(t["headers"]) + " " + " ".join(" ".join(r) for r in t["rows"])).lower()
        if not any(p in txt for p in IGNORE_TABLE_PATTERNS):
            result.append(t)
    return result

def tables_to_text(tables):
    """Flatten all tables into one readable text block for the LLM."""
    lines = []
    for t in tables:
        lines.append("--- TABLE ---")
        lines.append(" | ".join(t["headers"]))
        for row in t["rows"]:
            lines.append(" | ".join(row))
    return "\n".join(lines)

# ==========================================================
# RATE MATRIX EXTRACTION (deterministic, no LLM)
# ==========================================================

def extract_rate_matrix(tables):
    """Find and parse the zone-rate grid table."""
    rate_matrix = {}
    for t in tables:
        headers = t["headers"]
        rows = t["rows"]
        h0 = headers[0].strip().lower() if headers else ""

        # Pattern A: header row contains "rate matrix" or similar
        is_rate_table = "rate matrix" in h0 or "rate" in h0

        # Pattern B: first data row starts with "Zone"
        first_row_is_zone = rows and rows[0][0].strip().lower() == "zone"

        # Pattern C: headers[0] == "Zone"
        header_is_zone = headers[0].strip().lower() == "zone" if headers else False

        if not (is_rate_table or first_row_is_zone or header_is_zone):
            continue

        if first_row_is_zone:
            col_headers = rows[0]
            data_rows = rows[1:]
        elif header_is_zone:
            col_headers = headers
            data_rows = rows
        else:
            # rate matrix title table: sub-row might have Zone header
            if rows and rows[0][0].strip().lower() == "zone":
                col_headers = rows[0]
                data_rows = rows[1:]
            else:
                continue

        dest_zones = [z.strip() for z in col_headers[1:]]
        for row in data_rows:
            if not row or not row[0].strip():
                continue
            src = row[0].strip()
            rate_matrix[src] = {}
            for i, dz in enumerate(dest_zones):
                rate_matrix[src][dz] = row[i+1].strip() if i+1 < len(row) else ""

    return rate_matrix

# ==========================================================
# REGEX COMMERCIAL FALLBACKS
# ==========================================================

def regex_commercials(tables, full_text):
    """
    Pure-regex extraction from table text + free text.
    Returns dict with keys: mcw, docket, fuel, oda, fov, liability,
                             penalty, invoice_frequency, credit_period,
                             volumetric
    """
    c = {}

    # ── Build a flat text from all table cells ──
    table_text = tables_to_text(tables)
    combined   = table_text + "\n" + full_text

    def _find(patterns, text=combined):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # Fuel surcharge — covers "Fuel Surge Charge", "Fuel Surcharge", "FSC"
    c["fuel"] = _find([
        r"Fuel\s+Sur(?:ge|charge)[^\n|]*?[|\s]+(\d+(?:\.\d+)?%)",
        r"Fuel\s+Sur\w+[^%\n]*?(\d+(?:\.\d+)?%)",
        r"FSC[:\s]+(\d+(?:\.\d+)?%)",
    ])

    # Docket charge
    c["docket"] = _find([
        r"Docket\s+Charges?[^\n|]*?[|\s]+(?:Rs\.?\s*)?(\d[\d,\.]*)",
        r"Docket\s+Charges?[:\s]+Rs\.?\s*(\d[\d,\.]*)",
    ])
    if c["docket"]:
        c["docket"] = f"Rs. {c['docket']}"

    # MCW
    c["mcw"] = _find([
        r"Minimum\s+(?:Chargeable\s+)?Weight[^\n|]*?[|\s]+(\d[\d,\.]*\s*(?:kg|Kg)?[^\n|]{0,20})",
        r"Minimum\s+Chargeable\s+Weight[:\s]+([^\n]{1,40}?)(?:\s+II\.|\s+Docket|$)",
    ])

    # ODA
    c["oda"] = _find([
        r"ODA\s+Charges?[^\n|]*?[|\s]+(?:Rs\.?\s*)?(\d[\d,\.]*)",
        r"ODA\s+Charges?[:\s]+Rs\.?\s*(\d[\d,\.]*)",
    ])
    if c["oda"]:
        c["oda"] = f"Rs. {c['oda']}"

    # FOV
    c["fov"] = _find([
        r"(?:FOV|Freight\s+on\s+Value)[^\n|]*?[|\s]+([\d\.]+%[^\n|]{0,30})",
        r"FOV[:\s]+([\d\.]+%[^\n]{0,40})",
    ])

    # Liability
    c["liability"] = _find([
        r"(?:Minimum\s+)?Liability[^\n|]*?[|\s]+(?:Rs\.?\s*)?([\d,]+[^\n|]{0,20})",
        r"liability.*?Rs\.?\s*([\d,/\- ]+)[^\n]{0,30}",
    ])
    if c["liability"]:
        c["liability"] = f"Rs. {c['liability']}" if not c["liability"].startswith("Rs") else c["liability"]

    # Penalty
    c["penalty"] = _find([
        r"Penalty[^\n|]*?[|\s]+(\d+%[^\n|]{0,30})",
        r"Penalty[:\s-]+(\d+%[^\n]{0,40})",
    ])

    # Invoice frequency
    c["invoice_frequency"] = _find([
        r"(?:Invoicing|Invoice)\s+Frequency[^\n|]*?[|\s]+([A-Za-z]+ly\b[^\n|]{0,20})",
        r"Invoice\s+Frequency[:\s]+([A-Za-z]+ly\b)",
    ])

    # Credit period
    c["credit_period"] = _find([
        r"Credit\s+Period[^\n|]*?[|\s]+(\d+\s*(?:days?|Days?)[^\n|]{0,20})",
        r"Credit\s+Period[:\s]+(\d+\s*(?:days?|Days?)[^\n]{0,30})",
        r"payment.*?(\d+\s*days?\s+from[^\n]{0,40})",
    ])

    # Volumetric
    c["volumetric"] = _find([
        r"(\d+\s*(?:CFT|cft)\s*=\s*\d+\s*(?:Kg|kg|KG))",
        r"(?:Volumetric|CFT)[^\n]*?(\d+\s*(?:Kg|kg)\s*(?:per|/)\s*CFT)",
    ])

    # Strip None values
    return {k: v for k, v in c.items() if v}

# ==========================================================
# REGEX DATE EXTRACTION
# ==========================================================

def regex_dates(text):
    text = normalize_ocr_dates(text)
    D = (r"(\d{1,2}[-/]\w{3,9}[-/]\d{2,4}"
         r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
         r"|\d{1,2}\s+\w{3,9}\s+\d{4}"
         r"|\w{3,9}\s+\d{1,2},?\s+\d{4})")
    result = {}
    pairs = [
        ("contract_effective_from_date", [
            r"effective\s+(?:from|date)[:\s]*" + D,
            r"commencement\s+date[:\s]*" + D,
        ]),
        ("contract_end_date", [
            r"end\s+date[:\s]*" + D,
            r"valid\s+(?:until|till|upto)[:\s]*" + D,
            r"expir\w+\s+(?:on|date)[:\s]*" + D,
        ]),
        ("contract_signed_date", [
            r"signed\s+(?:on|date)[:\s]*" + D,
            r"executed\s+(?:on|this)[:\s]*" + D,
            r"date\s+of\s+execution[:\s]*" + D,
        ]),
        ("contract_issued_on_date", [
            r"issued\s+(?:on|date)[:\s]*" + D,
        ]),
    ]
    for field, patterns in pairs:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result[field] = m.group(1).strip()
                break
    return result

def regex_duration(text):
    for pat in [r"(\d+\s+year[s]?)", r"(\d+\s+month[s]?)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None

# ==========================================================
# SANITIZE — strip sentences from date fields
# ==========================================================

def sanitize(data):
    ci = data.get("contract_information", {})
    for field in DATE_FIELDS:
        val = ci.get(field)
        if val and isinstance(val, str):
            val = val.strip()
            if len(val) > DATE_MAX_LEN or CLAUSE_WORDS.search(val):
                ci[field] = None
    return data

# ==========================================================
# LLM CALL — generic, single call
# ==========================================================

def llm_call(system_prompt, user_text):
    try:
        resp = ollama.chat(
            model=MODEL,
            options={"temperature": 0, "top_p": 1.0, "think": False},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text[:6000]},  # cap tokens
            ],
        )
        return parse_llm_json(resp["message"]["content"])
    except Exception:
        return None

# ==========================================================
# MERGE HELPERS
# ==========================================================

def merge_into(target_dict, source_dict):
    """Fill nulls in target from source (flat dict)."""
    for k, v in source_dict.items():
        if v not in [None, "", "null", "NULL"]:
            if not target_dict.get(k):
                target_dict[k] = v

def recursive_merge(base, incoming):
    for k, v in incoming.items():
        if k not in base:
            base[k] = v
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_merge(base[k], v)
        elif isinstance(v, list) and isinstance(base.get(k), list):
            existing = set(map(str, base[k]))
            for x in v:
                if str(x) not in existing:
                    base[k].append(x)
        else:
            if base[k] in [None, ""] and v not in [None, ""]:
                base[k] = v

# ==========================================================
# TEXT CHUNKER — large chunks, few passes
# ==========================================================

def remove_tables_from_md(md):
    soup = BeautifulSoup(md, "html.parser")
    for t in soup.find_all("table"):
        t.decompose()
    return soup.get_text()

def big_chunks(text, max_chars=5000, max_chunks=3):
    """Split text into at most max_chunks large pieces."""
    # Split on double newlines (paragraph boundaries)
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip() and len(p.strip()) >= 40]
    chunks, current, size = [], [], 0
    for p in paras:
        if size + len(p) > max_chars and current:
            chunks.append("\n\n".join(current))
            if len(chunks) >= max_chunks:
                break
            current, size = [], 0
        current.append(p)
        size += len(p)
    if current and len(chunks) < max_chunks:
        chunks.append("\n\n".join(current))
    return chunks

# ==========================================================
# RESOLVE — pick first non-null from multiple paths
# ==========================================================

def resolve(data, *paths):
    for path in paths:
        if isinstance(path, (list, tuple)):
            val = data
            for k in path:
                val = val.get(k) if isinstance(val, dict) else None
                if val is None:
                    break
        else:
            val = data.get(path) if isinstance(data, dict) else None
        if val not in [None, "", "null", "NULL"]:
            return str(val).strip()
    return ""

# ==========================================================
# BUILD UI FIELDS FROM FINAL MERGED DATA
# ==========================================================

def build_ui_fields(final, commercials):
    ci  = final.get("contract_information", {})
    pts = final.get("parties", {})
    cc  = final.get("critical_clauses", {})

    lsp     = resolve(pts, "lsp_name") or commercials.get("lsp_name", "")
    shipper = resolve(pts, "shipper_name") or commercials.get("shipper_name", "")
    title   = resolve(ci, "contract_title") or ""

    effective = resolve(ci, "contract_effective_from_date") or ""
    end_date  = resolve(ci, "contract_end_date") or ""
    duration  = resolve(ci, "contract_duration") or ""

    # Fallback: end = effective + 1 year
    if not end_date and effective:
        try:
            eff_dt = dparser.parse(effective, dayfirst=True)
            end_dt = eff_dt.replace(year=eff_dt.year + 1)
            end_date = end_dt.strftime("%d-%b-%Y") + " (est. +1yr)"
        except Exception:
            pass

    valid_until = resolve(ci, "contract_valid_until") or end_date
    signed      = resolve(ci, "contract_signed_date") or ""

    def _clause(key, *fallbacks):
        v = cc.get(key)
        if v and str(v) not in ["None", "null", "NULL", ""]:
            return str(v)
        for fb in fallbacks:
            r = resolve(final, fb) if isinstance(fb, str) else resolve(final, fb)
            if r:
                return r
        return ""

    termination_raw = _clause("termination_clause")
    termination = (
        f"Either party may terminate with {termination_raw} written notice."
        if termination_raw and "notice" not in termination_raw.lower()
        else termination_raw
    )

    return {
        "lsp":        lsp,
        "shipper":    shipper,
        "title":      title,
        "effective":  effective,
        "end_date":   end_date,
        "valid_until": valid_until,
        "signed":     signed,
        "duration":   duration,
        "clauses": {
            "Claim Settlement Clause": _clause("claim_settlement_clause"),
            "Liability Clause":        _clause("liability_clause"),
            "Termination Clause":      termination,
            "Payment Clause":          _clause("payment_clause"),
            "TAT Clause":              _clause("tat_clause"),
        },
    }

# ==========================================================
# STREAMLIT APP
# ==========================================================

st.set_page_config(page_title="FreightIQ", page_icon="🚚", layout="wide")

st.markdown("""
<style>
  .stTextInput > div > div > input    { background: #1e2330 !important; }
  .stTextArea  > div > div > textarea { background: #1e2330 !important; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚚 FreightIQ Contract Extractor")
st.caption(f"Model: `{MODEL}`")

uploaded = st.file_uploader("Upload Contract PDF", type=["pdf"])

if uploaded:
    with tempfile.TemporaryDirectory() as tmp:

        pdf_path = os.path.join(tmp, uploaded.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())

        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)

        if st.button("⚙️ Process Contract"):

            # ── Step 1: MinerU ────────────────────────────────────
            with st.spinner("📄 MinerU: converting PDF…"):
                proc = subprocess.run(
                    ["mineru", "-p", pdf_path, "-o", out_dir],
                    capture_output=True, text=True
                )

            if proc.returncode != 0:
                st.error("MinerU failed"); st.text(proc.stderr); st.stop()

            md_files = list(Path(out_dir).rglob("*.md"))
            if not md_files:
                st.error("No markdown output"); st.stop()

            with open(md_files[0], "r", encoding="utf-8") as f:
                md = clean_markdown(f.read())

            # ── Step 2: Extract + filter tables ──────────────────
            with st.spinner("📊 Parsing tables…"):
                all_tables  = extract_html_tables(md)
                tables      = filter_tables(all_tables)
                rate_matrix = extract_rate_matrix(tables)
                table_text  = tables_to_text(tables)

            # ── Step 3: Regex pass (instant, no LLM) ─────────────
            with st.spinner("🔍 Regex extraction…"):
                plain_text   = remove_tables_from_md(md)
                rx_comm      = regex_commercials(tables, plain_text)
                rx_dates     = regex_dates(plain_text)
                rx_duration  = regex_duration(plain_text)

            # ── Step 4: LLM pass 1 — ALL TABLES → one call ───────
            with st.spinner("🤖 LLM: reading tables (1/2)…"):
                lm_tables = llm_call(TABLE_PROMPT, table_text) or {}

            # ── Step 5: LLM pass 2 — contract text → 2-3 chunks ──
            chunks = big_chunks(plain_text, max_chars=5000, max_chunks=3)
            lm_text_results = []

            prog = st.progress(0)
            for i, chunk in enumerate(chunks):
                with st.spinner(f"🤖 LLM: reading contract text ({i+1}/{len(chunks)})…"):
                    r = llm_call(TEXT_PROMPT, chunk)
                    if r:
                        lm_text_results.append(r)
                prog.progress((i + 1) / len(chunks))
            prog.empty()

            # ── Step 6: Merge everything — priority: LLM > regex ──
            # Start with master schema
            final = copy.deepcopy(MASTER_SCHEMA)

            # Seed dates from regex
            recursive_merge(final["contract_information"], rx_dates)
            if rx_duration and not final["contract_information"]["contract_duration"]:
                final["contract_information"]["contract_duration"] = rx_duration

            # Merge LLM table result (flat dict → schema fields)
            ci_fields   = ["contract_effective_from_date", "contract_end_date",
                           "contract_duration", "contract_title"]
            pt_fields   = ["lsp_name", "shipper_name"]
            comm_fields = ["freight_type", "minimum_chargeable_weight", "docket_charge",
                           "fuel_surcharge", "oda_charge", "liability_limit",
                           "payment_frequency", "credit_period"]

            for fld in ci_fields:
                v = lm_tables.get(fld)
                if v and v not in ["null", "NULL", ""] and not final["contract_information"].get(fld):
                    final["contract_information"][fld] = v

            for fld in pt_fields:
                v = lm_tables.get(fld)
                if v and v not in ["null", "NULL", ""] and not final["parties"].get(fld):
                    final["parties"][fld] = v

            for fld in comm_fields:
                v = lm_tables.get(fld)
                if v and v not in ["null", "NULL", ""] and not final["commercial_terms"].get(fld):
                    final["commercial_terms"][fld] = v

            # Merge LLM text results (for clauses + parties + dates)
            for r in lm_text_results:
                for fld in ci_fields + ["contract_signed_date"]:
                    v = r.get(fld)
                    if v and v not in ["null", "NULL", ""] and not final["contract_information"].get(fld):
                        final["contract_information"][fld] = v
                for fld in pt_fields:
                    v = r.get(fld)
                    if v and v not in ["null", "NULL", ""] and not final["parties"].get(fld):
                        final["parties"][fld] = v
                for fld in ["claim_settlement_clause", "liability_clause",
                             "termination_clause", "payment_clause", "tat_clause"]:
                    v = r.get(fld)
                    if v and v not in ["null", "NULL", ""] and not final["critical_clauses"].get(fld):
                        final["critical_clauses"][fld] = v

            final = sanitize(final)

            # ── Step 7: Build commercials dict (LLM first, regex fallback) ──
            commercials = {}

            # LLM table values
            comm_map = {
                "mcw":               lm_tables.get("minimum_chargeable_weight"),
                "docket":            lm_tables.get("docket_charge"),
                "fuel":              lm_tables.get("fuel_surcharge"),
                "oda":               lm_tables.get("oda_charge"),
                "fov":               lm_tables.get("fov_charge"),
                "liability":         lm_tables.get("liability_limit"),
                "invoice_frequency": lm_tables.get("invoice_frequency"),
                "credit_period":     lm_tables.get("credit_period"),
                "penalty":           lm_tables.get("penalty"),
                "volumetric":        lm_tables.get("volumetric_method"),
                "lsp_name":          lm_tables.get("lsp_name"),
                "shipper_name":      lm_tables.get("shipper_name"),
            }
            for k, v in comm_map.items():
                if v and str(v) not in ["null", "NULL", ""]:
                    commercials[k] = str(v)

            # Regex fallback for anything still missing
            for k, v in rx_comm.items():
                if not commercials.get(k) and v:
                    commercials[k] = v

            # Fill parties from commercials if schema still empty
            if not final["parties"]["lsp_name"] and commercials.get("lsp_name"):
                final["parties"]["lsp_name"] = commercials["lsp_name"]
            if not final["parties"]["shipper_name"] and commercials.get("shipper_name"):
                final["parties"]["shipper_name"] = commercials["shipper_name"]

            output = {"extracted_contract": final, "relevant_tables": tables}

            st.success("✅ Done")

            with st.expander("📋 Raw JSON", expanded=False):
                st.json(output)

            # ── UI Fields ─────────────────────────────────────────
            ui = build_ui_fields(final, commercials)

            # ====================================================
            # CONTRACT HEADER
            # ====================================================

            st.divider()
            st.header("📄 Contract UI")
            if ui["title"]:
                st.caption(f"**{ui['title']}**")

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.text_input("LSP (Service Provider)", value=ui["lsp"] or "N/A")
            with c2: st.text_input("Shipper (Customer)",     value=ui["shipper"] or "N/A")
            with c3: st.text_input("Effective Date",         value=ui["effective"] or "N/A")
            with c4: st.text_input("End Date",               value=ui["end_date"] or "N/A")

            d1, d2, d3 = st.columns(3)
            with d1: st.text_input("Signed / Issued Date", value=ui["signed"] or "N/A")
            with d2: st.text_input("Valid Until",          value=ui["valid_until"] or "N/A")
            with d3: st.text_input("Contract Duration",    value=ui["duration"] or "N/A")

            # ====================================================
            # COMMERCIAL TERMS
            # ====================================================

            st.subheader("💰 Commercial Terms")
            t1, t2, t3, t4 = st.columns(4)
            with t1: st.text_input("Min Chargeable Weight", value=commercials.get("mcw") or "N/A")
            with t2: st.text_input("Docket Charge",         value=commercials.get("docket") or "N/A")
            with t3: st.text_input("Fuel Surcharge",        value=commercials.get("fuel") or "N/A")
            with t4: st.text_input("ODA Charge",            value=commercials.get("oda") or "N/A")

            t5, t6, t7, t8 = st.columns(4)
            with t5: st.text_input("FOV Charge",        value=commercials.get("fov") or "N/A")
            with t6: st.text_input("Liability Limit",   value=commercials.get("liability") or "N/A")
            with t7: st.text_input("Invoice Frequency", value=commercials.get("invoice_frequency") or "N/A")
            with t8: st.text_input("Credit Period",     value=commercials.get("credit_period") or "N/A")

            if commercials.get("penalty"):
                st.warning(f"⚠️ **Penalty:** {commercials['penalty']}")
            if commercials.get("volumetric"):
                st.info(f"📐 **Volumetric:** {commercials['volumetric']}")

            # ====================================================
            # RATE MATRIX
            # ====================================================

            st.subheader("🗺️ All India Rate Matrix (Surface – Per Kg)")
            if rate_matrix:
                all_dest = sorted({dz for v in rate_matrix.values() for dz in v})
                df_rate  = pd.DataFrame(
                    [{**{"From \\ To": src}, **{dz: dm.get(dz, "–") for dz in all_dest}}
                     for src, dm in rate_matrix.items()]
                ).set_index("From \\ To")
                st.dataframe(df_rate, use_container_width=True)

                st.subheader("📦 Lane Details")
                for src, dest_map in rate_matrix.items():
                    for dest, rate in dest_map.items():
                        if not rate or rate in ["–", "-", ""]:
                            continue
                        with st.expander(f"{src} → {dest}"):
                            lc1, lc2, lc3, lc4 = st.columns(4)
                            with lc1: st.text_input("Rate (₹/kg)", value=rate,                              key=f"r_{src}_{dest}")
                            with lc2: st.text_input("Min Weight",  value=commercials.get("mcw") or "N/A",  key=f"m_{src}_{dest}")
                            with lc3: st.text_input("Docket",      value=commercials.get("docket") or "N/A", key=f"d_{src}_{dest}")
                            with lc4: st.text_input("ODA",         value=commercials.get("oda") or "N/A",  key=f"o_{src}_{dest}")
            else:
                st.info("No rate matrix found in contract tables.")

            # ====================================================
            # CRITICAL CLAUSES
            # ====================================================

            st.subheader("⚖️ Critical Clauses")
            for heading, content in ui["clauses"].items():
                st.text_area(heading, value=content or "N/A", height=90)

            # ====================================================
            # DOWNLOADS
            # ====================================================

            st.divider()
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇️ Download JSON", data=json.dumps(output, indent=4),
                                   file_name="contract_output.json", mime="application/json")
            with dl2:
                st.download_button("⬇️ Download Markdown", data=md,
                                   file_name="mineru_output.md", mime="text/markdown")