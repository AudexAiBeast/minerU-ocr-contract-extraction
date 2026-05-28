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

# ==========================================================
# CONFIG — swap MODEL to any of the recommended models below
# ==========================================================

MODEL = "qwen3:14b"
# ┌─────────────────────────────────────────────────────────┐
# │  RECOMMENDED MODELS FOR 16 GB VRAM (best first)        │
# │                                                         │
# │  BEST OVERALL REASONING + JSON EXTRACTION:             │
# │    qwen3:14b          ~9 GB  ollama pull qwen3:14b      │
# │    (hybrid think/no-think, best structured output)     │
# │                                                         │
# │  BEST PURE REASONING (chain-of-thought):               │
# │    deepseek-r1:14b    ~9 GB  ollama pull deepseek-r1:14b│
# │    (writes <think> block before JSON, very accurate)   │
# │                                                         │
# │  BEST SPEED + QUALITY BALANCE:                         │
# │    gpt-oss:20b        ~12 GB ollama pull gpt-oss:20b    │
# │    (139 tok/s on RTX 4080, OpenAI open-source)        │
# │                                                         │
# │  BEST ANALYTICAL / STEM REASONING:                     │
# │    phi4:14b           ~9 GB  ollama pull phi4:14b       │
# │    (Microsoft, MATH benchmark 80.4%, dense knowledge)  │
# │                                                         │
# │  CURRENT (FAST, LIGHTWEIGHT):                          │
# │    granite3.3:latest  ~5 GB  already installed         │
# └─────────────────────────────────────────────────────────┘

# ==========================================================
# SYSTEM PROMPT — precision-engineered for contract extraction
# ==========================================================

SYSTEM_PROMPT = """You are FreightIQ, an expert logistics contract data extraction engine.

Your ONLY job is to extract structured data from contract text and return valid JSON.

═══════════════════════════════════════
OUTPUT RULES (non-negotiable)
═══════════════════════════════════════
1. Return ONLY a JSON object. No preamble, no explanation, no markdown fences.
2. If a field is not found in the text, set it to null. Never invent values.
3. Keep all extracted text to 20 words or fewer per field.
4. Never copy entire sentences or clauses verbatim.

═══════════════════════════════════════
IGNORE COMPLETELY (do not extract)
═══════════════════════════════════════
- Bank account numbers, IFSC codes, account holder names
- GST numbers, PAN numbers, CIN numbers
- Full street addresses
- Witness names, notary details
- Stamp duty details
- Signature blocks
- Legal boilerplate (indemnity, force majeure, governing law, arbitration text)
- Page headers and footers
- OCR noise and garbled characters

═══════════════════════════════════════
DATE FIELDS — STRICT FORMAT RULES
═══════════════════════════════════════
- Extract dates as: DD-MMM-YYYY or DD/MM/YYYY or "1st August 2025" style
- Date fields must contain ONLY a date — never a sentence
- If the date is written as "1st day of August 2025", extract: "01-Aug-2025"
- If no date found: null
- contract_duration: extract as "X year(s)" or "X month(s)" only

═══════════════════════════════════════
CLAUSE FIELDS — EXTRACT ONLY THE CORE LIMIT/TERM
═══════════════════════════════════════
- claim_settlement_clause  → settlement timeline + max limit only (e.g. "30 days, Rs.10,000 max")
- liability_clause         → liability cap only (e.g. "Rs.500/kg, max Rs.10,000")
- termination_clause       → notice period only (e.g. "1 month written notice")
- payment_clause           → payment days only (e.g. "30 days from invoice")
- tat_clause               → TAT days only (e.g. "1-2 days within zone")

═══════════════════════════════════════
TARGET JSON SCHEMA
═══════════════════════════════════════
{
  "contract_information": {
    "contract_title": null,
    "agreement_id": null,
    "contract_effective_from_date": null,
    "contract_end_date": null,
    "contract_valid_until": null,
    "contract_duration": null,
    "contract_signed_date": null,
    "contract_issued_on_date": null
  },
  "parties": {
    "lsp_name": null,
    "shipper_name": null
  },
  "commercial_terms": {
    "freight_type": null,
    "minimum_chargeable_weight": null,
    "docket_charge": null,
    "fuel_surcharge": null,
    "oda_charge": null,
    "liability_limit": null,
    "payment_frequency": null,
    "credit_period": null
  },
  "transport": {
    "zones": []
  },
  "critical_clauses": {
    "claim_settlement_clause": null,
    "liability_clause": null,
    "termination_clause": null,
    "payment_clause": null,
    "tat_clause": null
  }
}

Return only JSON. Nothing else.
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
    "transport": {
        "zones": [],
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
    "account holder",
    "ifsc",
    "hdfc",
    "branch address",
    "current account",
    "billing address",
    "billing city",
    "billing state",
    "billing pin",
    "gst number",
    "signature",
    "stamp duty",
    "witness",
]

# ==========================================================
# DATE FIELD VALIDATION
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

CLAUSE_WORDS = re.compile(
    r"\b(shall|herein|party|customer|service|provider|agreement|whereas|pursuant)\b",
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
        if re.search(r"^\s*-*\s*page\s+\d+\s*-*\s*$", line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

# ==========================================================
# NORMALIZE OCR DATES
# ==========================================================

def normalize_ocr_dates(text):
    text = re.sub(r"\$\s*\^\{.*?\}\s*\$", "", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text

# ==========================================================
# REGEX DATE EXTRACTION (fast-path, no LLM needed)
# ==========================================================

def extract_dates_regex(text):
    result = {
        "contract_effective_from_date": None,
        "contract_end_date": None,
        "contract_signed_date": None,
        "contract_issued_on_date": None,
    }
    text = normalize_ocr_dates(text)

    # Patterns: dd-Mon-yyyy, dd/mm/yyyy, dd Month yyyy, Month dd yyyy
    DATE_PAT = (
        r"(\d{1,2}[-/]\w{3,9}[-/]\d{2,4}"       # 01-Aug-2025 / 01/August/2025
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"        # 01/08/2025
        r"|\d{1,2}\s+\w{3,9}\s+\d{4}"            # 1 August 2025
        r"|\w{3,9}\s+\d{1,2},?\s+\d{4})"         # August 1, 2025
    )

    pairs = [
        ("contract_effective_from_date",
         [r"effective\s+(?:from|date)[:\s]*" + DATE_PAT,
          r"commencement\s+date[:\s]*" + DATE_PAT,
          r"agreement.*?(?:from|commencing)\s+" + DATE_PAT]),
        ("contract_end_date",
         [r"end\s+date[:\s]*" + DATE_PAT,
          r"valid\s+(?:until|till|upto)[:\s]*" + DATE_PAT,
          r"expir\w+\s+(?:on|date)[:\s]*" + DATE_PAT,
          r"(?:to|till|until)\s+(?:the\s+)?" + DATE_PAT]),
        ("contract_signed_date",
         [r"signed\s+(?:on|date)[:\s]*" + DATE_PAT,
          r"executed\s+(?:on|this)[:\s]*" + DATE_PAT,
          r"date\s+of\s+execution[:\s]*" + DATE_PAT]),
        ("contract_issued_on_date",
         [r"issued\s+(?:on|date)[:\s]*" + DATE_PAT,
          r"issu\w+\s+date[:\s]*" + DATE_PAT]),
    ]

    for field, patterns in pairs:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result[field] = m.group(1).strip()
                break

    return result

# ==========================================================
# REGEX DURATION EXTRACTION
# ==========================================================

def extract_duration(text):
    for pat in [r"(\d+\s+year[s]?)", r"(\d+\s+month[s]?)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None

# ==========================================================
# SANITIZE LLM OUTPUT — strip sentences from date fields
# ==========================================================

def sanitize_extracted(data):
    ci = data.get("contract_information", {})
    for field in DATE_FIELDS:
        val = ci.get(field)
        if val and isinstance(val, str):
            val = val.strip()
            if len(val) > DATE_MAX_LEN or CLAUSE_WORDS.search(val):
                ci[field] = None
    return data

# ==========================================================
# STRIP THINK TAGS (for DeepSeek R1 / Qwen3 thinking mode)
# ==========================================================

def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ==========================================================
# EXTRACT HTML TABLES FROM MINERU MARKDOWN
# ==========================================================

def extract_html_tables(md):
    tables = []
    soup = BeautifulSoup(md, "html.parser")
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
# FILTER TABLES — remove bank/address/stamp noise
# ==========================================================

def filter_tables(tables):
    final = []
    for t in tables:
        txt = " ".join(t["headers"])
        for row in t["rows"]:
            txt += " " + " ".join(row)
        txt = txt.lower()
        if not any(p in txt for p in IGNORE_TABLE_PATTERNS):
            final.append(t)
    return final

# ==========================================================
# PARSE TABLES → rate matrix + commercials + payment info
# ==========================================================

def parse_tables(tables):
    """
    Returns:
      rate_matrix  : { src_zone: { dest_zone: rate_str } }
      commercials  : { mcw, docket, fov, fuel, liability, oda, penalty,
                       invoice_frequency, credit_period }
      volumetric   : str
      payment_info : { invoicing_basis, frequency, credit_period }
    """
    rate_matrix = {}
    commercials = {}
    volumetric = ""
    payment_info = {}

    def _is_axis_header(cell):
        """Detect first-column labels used in rate-card tables."""
        if not cell:
            return False
        label = re.sub(r"\s+", " ", str(cell).strip().lower())
        label = label.replace("\\", " ").replace("/", " ")
        return bool(re.search(r"\b(zone|from|to|from to|from to zone|origin|source)\b", label))

    for table in tables:
        headers = table.get("headers", [])
        rows    = table.get("rows", [])
        if not headers:
            continue
        h0 = headers[0].strip()
        full_text = h0 + " " + " ".join(" ".join(r) for r in rows)

        # ── Rate matrix ───────────────────────────────────────
        if "rate matrix" in h0.lower() or "rate" in h0.lower():
            # First row of rows may be the actual column headers
            col_row = None
            data_rows = rows
            if rows and rows[0] and _is_axis_header(rows[0][0]) and len(rows[0]) > 1:
                col_row = rows[0]
                data_rows = rows[1:]
            elif len(headers) > 1 and _is_axis_header(headers[0]):
                col_row = headers
                data_rows = rows

            if col_row:
                dest_zones = col_row[1:]
                for row in data_rows:
                    if not row or not row[0].strip():
                        continue
                    src = row[0].strip()
                    rate_matrix[src] = {}
                    for i, dz in enumerate(dest_zones):
                        rate_matrix[src][dz.strip()] = row[i+1].strip() if i+1 < len(row) else ""
            continue

        # ── Structured charge table (rows: [#, Description, Unit, Value, Confirmation]) ──
        # Handles tables like: A=Basic Freight, B=Fuel Surge Charge, C=Docket, etc.
        has_desc_col = any("description" in str(h).lower() for h in headers)
        has_charge_rows = any(
            any(kw in " ".join(str(c) for c in row).lower()
                for kw in ["fuel", "docket", "oda", "liability", "minimum weight",
                           "minimum charges", "fov", "penalty", "basic freight"])
            for row in rows
        )
        if has_desc_col or has_charge_rows:
            for row in rows:
                row_text = " ".join(str(c) for c in row).lower()
                # Find the value cell — prefer cells that look like amounts/percents
                vals = [str(c).strip() for c in row if str(c).strip()]

                def _row_val():
                    # Pick the cell that looks most like a value (%, Rs, number)
                    for cell in reversed(vals):
                        if re.search(r"\d", cell) and len(cell) <= 60:
                            return cell
                    return vals[-1] if vals else ""

                if "fuel surge" in row_text or ("fuel" in row_text and "surcharge" in row_text):
                    if not commercials.get("fuel"):
                        # Extract percentage from any cell in this row
                        pct = next((re.search(r"(\d+(?:\.\d+)?%)", str(c)).group(1)
                                    for c in row if re.search(r"\d+%", str(c))), None)
                        if pct:
                            commercials["fuel"] = pct
                        else:
                            commercials["fuel"] = _row_val()

                elif "docket" in row_text and "charge" in row_text:
                    if not commercials.get("docket"):
                        amt = next((re.search(r"(?:rs\.?\s*)?(\d[\d,\.]*)", str(c), re.I).group(0)
                                    for c in row if re.search(r"\d", str(c)) and "docket" not in str(c).lower()), None)
                        if amt:
                            commercials["docket"] = amt.strip()

                elif "oda" in row_text:
                    if not commercials.get("oda"):
                        commercials["oda"] = _row_val()

                elif "fov" in row_text or "freight on value" in row_text:
                    if not commercials.get("fov"):
                        commercials["fov"] = _row_val()

                elif "minimum weight" in row_text or ("minimum" in row_text and "weight" in row_text):
                    if not commercials.get("mcw"):
                        commercials["mcw"] = _row_val()

                elif "minimum charges" in row_text:
                    if not commercials.get("min_charges"):
                        commercials["min_charges"] = _row_val()

                elif "liability" in row_text and "limit" in row_text:
                    if not commercials.get("liability"):
                        commercials["liability"] = _row_val()

                elif "penalty" in row_text:
                    if not commercials.get("penalty"):
                        commercials["penalty"] = _row_val()

        # ── Commercials / charges free-text block ─────────────
        if any(kw in full_text.lower() for kw in
               ["minimum chargeable", "docket charge", "fuel surcharge",
                "oda charge", "liability", "penalty", "fov"]):

            def _get(pattern):
                m = re.search(pattern, full_text, re.IGNORECASE)
                return m.group(1).strip() if m else ""

            mcw = _get(r"Minimum Chargeable Weight[:\s]*([^\n]+?)(?:\s+II\.|\s+Docket|$)")
            if mcw and not commercials.get("mcw"):
                commercials["mcw"] = mcw[:80]

            docket = _get(r"Docket Charges?[:\s]*Rs\.?\s*(\d[\d,\.]*)")
            if docket and not commercials.get("docket"):
                commercials["docket"] = f"Rs. {docket}"

            fov = _get(r"FOV[^:]*[:\s]*([\d\.]+%[^I]{0,40}?)(?:\s+IV\.|\s+Fuel|$)")
            if fov and not commercials.get("fov"):
                commercials["fov"] = fov

            # Broader fuel regex: catches "Fuel Surge Charge ... 10%" anywhere in block
            if not commercials.get("fuel"):
                fuel = _get(r"Fuel\s+Sur(?:ge|charge)[^:]*[:\-–\s]*(?:will be\s*)?([\d]+(?:\.\d+)?%[^\n]{0,40}?)(?:\s+V\.|\s+Min|$)")
                if not fuel:
                    fuel = _get(r"Fuel\s+Sur\w+.*?(\d+(?:\.\d+)?%)")
                if fuel:
                    commercials["fuel"] = fuel[:60]

            liability = _get(r"Minimum Liability\s*Rs\.?\s*([\d,/\- ]+[^\n]{0,30}?)(?:\s+VI\.|\s+ODA|$)")
            if liability and not commercials.get("liability"):
                commercials["liability"] = f"Rs. {liability[:60]}"

            oda = _get(r"ODA Charges?[^R\d]*(?:Rs\.?\s*)?([\d,]+)[^\n]{0,30}")
            if oda and not commercials.get("oda"):
                commercials["oda"] = f"Rs. {oda}"

            penalty = _get(r"Penalty[:\s-]*([\d]+%[^\n]{0,40}?)")
            if penalty and not commercials.get("penalty"):
                commercials["penalty"] = penalty[:60]
            continue

        # ── Volumetric / CFT table ────────────────────────────
        if "cft" in full_text.lower() and "volumetric" in full_text.lower():
            for row in rows:
                line = " ".join(row)
                if "cft" in line.lower() and "kg" in line.lower():
                    volumetric = line.strip()
                    break
            continue

        # ── Payment terms table (#, Details, Terms, Confirmation) ──
        if len(headers) >= 3 and "Details" in headers and "Terms" in headers:
            for row in rows:
                if len(row) < 3:
                    continue
                detail = row[1].strip().lower()
                term   = row[2].strip()
                if "invoicing basis" in detail:
                    payment_info["invoicing_basis"] = term
                elif "invoicing frequency" in detail or "invoice frequency" in detail:
                    payment_info["frequency"] = term
                    commercials["invoice_frequency"] = term
                elif "credit period" in detail:
                    payment_info["credit_period"] = term
                    commercials["credit_period"] = term
            continue

    return rate_matrix, commercials, volumetric, payment_info

# ==========================================================
# SMART RESOLVER — pull values from anywhere in LLM blob
# ==========================================================

def resolve(data, *paths):
    """Try key paths in order, return first non-null string."""
    for path in paths:
        if isinstance(path, (list, tuple)):
            val = data
            for k in path:
                val = val.get(k) if isinstance(val, dict) else None
                if val is None:
                    break
        else:
            val = data.get(path)
        if val not in [None, "", "null", "NULL"]:
            return str(val).strip()
    return ""

# ==========================================================
# EXTRACT UI FIELDS FROM FULL JSON BLOB
# ==========================================================

def extract_ui_fields(output):
    data = output.get("extracted_contract", {})
    ci   = data.get("contract_information", {})
    pts  = data.get("parties", {})
    cc   = data.get("critical_clauses", {})

    lsp = (
        resolve(data, ["ServiceProviderDetails", "Name"])
        or resolve(pts, "lsp_name")
        or resolve(data, "Second_Party")
        or ""
    )
    shipper = (
        resolve(data, ["CustomerDetails", "Name"])
        or resolve(pts, "shipper_name")
        or resolve(data, "First_Party")
        or resolve(data, "Purchased_by")
        or ""
    )
    effective = (
        resolve(ci,   "contract_effective_from_date")
        or resolve(data, ["Agreement_Duration", "Start_Date"])
        or resolve(data, "Agreement_Effective_Date")
        or resolve(data, "EffectiveDate")
        or ""
    )
    end_date = (
        resolve(ci,   "contract_end_date")
        or resolve(data, ["Agreement_Duration", "End_Date"])
        or resolve(data, "DateOfExecution")
        or ""
    )

    # ── Fallback: if no end date, compute effective_date + 1 year ──
    if not end_date and effective:
        try:
            from dateutil import parser as dparser
            eff_dt = dparser.parse(effective, dayfirst=True)
            end_dt = eff_dt.replace(year=eff_dt.year + 1)
            end_date = end_dt.strftime("%d-%b-%Y") + " (est. +1yr)"
        except Exception:
            pass

    valid_until = resolve(ci, "contract_valid_until") or end_date
    signed      = resolve(ci, "contract_signed_date") or resolve(data, "Issued_Date") or ""
    duration    = resolve(ci, "contract_duration") or ""
    title       = (
        resolve(ci,   "contract_title")
        or resolve(data, "AgreementType")
        or resolve(data, "Property_Description")
        or ""
    )

    def _clause(schema_key, *fallbacks):
        v = cc.get(schema_key)
        if v and str(v) not in ["None", "null", "NULL", ""]:
            return str(v)
        for fb in fallbacks:
            r = resolve(data, fb) if isinstance(fb, str) else resolve(data, fb)
            if r:
                return r
        return ""

    termination_raw = _clause("termination_clause",
                              ["Termination", "Convenience_Termination", "Notice_Period"])
    termination = (
        f"Either party may terminate with {termination_raw} written notice (registered post)."
        if termination_raw and "notice" not in termination_raw.lower()
        else termination_raw
    )

    clauses = {
        "Claim Settlement Clause":
            _clause("claim_settlement_clause",
                    ["Consideration", "Risk_Charges"],
                    ["Clause_11", "a"]),
        "Liability Clause":
            _clause("liability_clause",
                    ["clause_9", "description"],
                    ["Clause_11", "d"]),
        "Termination Clause": termination,
        "Payment Clause":
            _clause("payment_clause",
                    ["5.0_INVOICING_&_PAYMENT_TERMS", "b", "Payment_terms"],
                    ["i", "payment_term"]),
        "TAT Clause":
            _clause("tat_clause",
                    ["Transit_Time", "TAT"],
                    ["Annexure-I", "a"]),
    }

    return {
        "lsp":          lsp,
        "shipper":      shipper,
        "effective":    effective,
        "end_date":     end_date,
        "valid_until":  valid_until,
        "signed":       signed,
        "duration":     duration,
        "title":        title,
        "clauses":      clauses,
    }

# ==========================================================
# REMOVE TABLES FROM MARKDOWN TEXT
# ==========================================================

def remove_tables(md):
    soup = BeautifulSoup(md, "html.parser")
    for t in soup.find_all("table"):
        t.decompose()
    return soup.get_text()

# ==========================================================
# SPLIT PARAGRAPHS
# ==========================================================

def split_paragraphs(text):
    paras, current = [], []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current:
                paras.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paras.append(" ".join(current))
    return paras

# ==========================================================
# CHUNK TEXT
# ==========================================================

def chunk_text(text, max_chars=1800):
    words = text.split()
    chunks, current, size = [], [], 0
    for w in words:
        current.append(w)
        size += len(w)
        if size > max_chars:
            chunks.append(" ".join(current))
            current, size = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks

# ==========================================================
# LLM ANALYSIS (single chunk)
# ==========================================================

def analyse_chunk(text):
    try:
        response = ollama.chat(
            model=MODEL,
            options={
                "temperature": 0,
                "top_p": 1.0,
                # For Qwen3: disable thinking mode for speed
                # Remove this if using deepseek-r1 (it needs think tokens)
                "think": False,
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
        )
        content = response["message"]["content"]

        # Strip <think>...</think> blocks (DeepSeek R1 / Qwen3 thinking mode)
        content = strip_think_tags(content)

        # Strip markdown code fences if model wraps JSON
        content = re.sub(r"```(?:json)?", "", content).strip()

        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            raw = m.group()
            raw = re.sub(r",\s*([\}\]])", r"\1", raw)
            return json.loads(raw)

    except Exception:
        return None

# ==========================================================
# RECURSIVE MERGE
# ==========================================================

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
# STREAMLIT APP
# ==========================================================

st.set_page_config(
    page_title="FreightIQ Contract Extractor",
    page_icon="🚚",
    layout="wide",
)

st.markdown("""
<style>
  .stTextInput > div > div > input   { background: #1e2330 !important; }
  .stTextArea  > div > div > textarea{ background: #1e2330 !important; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚚 FreightIQ Contract Extractor")
st.caption(f"Model: `{MODEL}` · Upload a logistics contract PDF to extract structured data")

uploaded = st.file_uploader("Upload Contract PDF", type=["pdf"])

if uploaded:

    with tempfile.TemporaryDirectory() as temp_dir:

        pdf_path = os.path.join(temp_dir, uploaded.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())

        mineru_output = os.path.join(temp_dir, "mineru_output")
        os.makedirs(mineru_output, exist_ok=True)

        if st.button("⚙️ Process Contract"):

            # ── MinerU ──────────────────────────────────────────
            with st.spinner("Running MinerU PDF extraction..."):
                cmd = ["mineru", "-p", pdf_path, "-o", mineru_output]
                proc = subprocess.run(cmd, capture_output=True, text=True)

            if proc.returncode != 0:
                st.error("MinerU failed")
                st.text(proc.stderr)
                st.stop()

            md_files = list(Path(mineru_output).rglob("*.md"))
            if not md_files:
                st.error("No markdown generated by MinerU")
                st.stop()

            with open(md_files[0], "r", encoding="utf-8") as f:
                md = f.read()

            md = clean_markdown(md)
            st.success("MinerU extraction complete")

            # ── Tables ──────────────────────────────────────────
            with st.spinner("Parsing tables..."):
                tables_raw = extract_html_tables(md)
                tables     = filter_tables(tables_raw)

            # ── Regex fast-path for dates & commercials ─────────
            date_data = extract_dates_regex(md)
            duration  = extract_duration(md)
            rate_matrix, commercials, volumetric, payment_info = parse_tables(tables)

            # ── LLM chunked extraction ──────────────────────────
            text      = remove_tables(md)
            paragraphs = split_paragraphs(text)
            final     = copy.deepcopy(MASTER_SCHEMA)

            # Seed schema with regex results
            recursive_merge(final["contract_information"], date_data)
            if duration and not final["contract_information"].get("contract_duration"):
                final["contract_information"]["contract_duration"] = duration

            chunks = [
                chunk
                for p in paragraphs
                if len(p) >= 50
                for chunk in chunk_text(p)
            ]

            st.subheader("🔍 Running LLM extraction...")
            progress = st.progress(0)
            total    = len(chunks)

            for i, c in enumerate(chunks):
                result = analyse_chunk(c)
                if result:
                    # Don't let LLM overwrite commercial_terms —
                    # table regex is more reliable for those
                    temp = copy.deepcopy(result)
                    temp.pop("commercial_terms", None)
                    recursive_merge(final, temp)
                progress.progress((i + 1) / total)

            progress.empty()

            final = sanitize_extracted(final)

            output = {
                "extracted_contract": final,
                "relevant_tables":    tables,
            }

            st.success("✅ Extraction complete")

            # ── Raw JSON expander ────────────────────────────────
            with st.expander("📋 Raw Extracted JSON", expanded=False):
                st.json(output)

            # ====================================================
            # DERIVE UI FIELDS
            # ====================================================

            fields = extract_ui_fields(output)

            # Merge payment_info into commercials
            for k in ("invoice_frequency", "credit_period"):
                if not commercials.get(k) and payment_info.get(k.replace("invoice_frequency","frequency")):
                    commercials[k] = payment_info[k.replace("invoice_frequency","frequency")]

            # Fallback credit period from LLM output
            if not commercials.get("credit_period"):
                cp = resolve(
                    final,
                    ["5.0_INVOICING_&_PAYMENT_TERMS", "b", "Payment_terms"],
                    ["i", "payment_term"],
                )
                if cp:
                    commercials["credit_period"] = cp

            # ====================================================
            # CONTRACT HEADER
            # ====================================================

            st.divider()
            st.header("📄 Contract UI")

            if fields["title"]:
                st.caption(f"**{fields['title']}**")

            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1:
                st.text_input("LSP (Service Provider)",  value=fields["lsp"] or "N/A")
            with r1c2:
                st.text_input("Shipper (Customer)",      value=fields["shipper"] or "N/A")
            with r1c3:
                st.text_input("Effective Date",          value=fields["effective"] or "N/A")
            with r1c4:
                st.text_input("End Date",                value=fields["end_date"] or "N/A")

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                st.text_input("Signed / Issued Date",    value=fields["signed"] or "N/A")
            with r2c2:
                st.text_input("Valid Until",             value=fields["valid_until"] or "N/A")
            with r2c3:
                st.text_input("Contract Duration",       value=fields["duration"] or "N/A")

            # ====================================================
            # COMMERCIAL TERMS
            # ====================================================

            st.subheader("💰 Commercial Terms")

            ct1, ct2, ct3, ct4 = st.columns(4)
            with ct1:
                st.text_input("Min Chargeable Weight", value=commercials.get("mcw") or "N/A")
            with ct2:
                st.text_input("Docket Charge",         value=commercials.get("docket") or "N/A")
            with ct3:
                st.text_input("Fuel Surcharge",        value=commercials.get("fuel") or "N/A")
            with ct4:
                st.text_input("ODA Charge",            value=commercials.get("oda") or "N/A")

            ct5, ct6, ct7, ct8 = st.columns(4)
            with ct5:
                st.text_input("FOV Charge",            value=commercials.get("fov") or "N/A")
            with ct6:
                st.text_input("Liability Limit",       value=commercials.get("liability") or "N/A")
            with ct7:
                st.text_input("Invoice Frequency",     value=commercials.get("invoice_frequency") or "N/A")
            with ct8:
                st.text_input("Credit Period",         value=commercials.get("credit_period") or "N/A")

            if commercials.get("penalty"):
                st.warning(f"⚠️ **Penalty Clause:** {commercials['penalty']}")

            # ====================================================
            # RATE MATRIX
            # ====================================================

            st.subheader("🗺️ All India Rate Matrix (Surface – Per Kg)")

            if rate_matrix:
                all_dest = sorted({dz for v in rate_matrix.values() for dz in v})
                matrix_rows = []
                for src, dest_map in rate_matrix.items():
                    row = {"From \\ To": src}
                    for dz in all_dest:
                        row[dz] = dest_map.get(dz, "–")
                    matrix_rows.append(row)

                df_rate = pd.DataFrame(matrix_rows).set_index("From \\ To")
                st.dataframe(df_rate, use_container_width=True)

                # Per-lane expanders
                st.subheader("📦 Lane Details")
                for src, dest_map in rate_matrix.items():
                    for dest, rate in dest_map.items():
                        if not rate or rate in ["–", "-"]:
                            continue
                        with st.expander(f"{src} → {dest}"):
                            lc1, lc2, lc3, lc4 = st.columns(4)
                            with lc1:
                                st.text_input("Rate (₹/kg)",  value=rate,
                                              key=f"r_{src}_{dest}")
                            with lc2:
                                st.text_input("Min Weight",   value=commercials.get("mcw") or "N/A",
                                              key=f"mcw_{src}_{dest}")
                            with lc3:
                                st.text_input("Docket",       value=commercials.get("docket") or "N/A",
                                              key=f"dkt_{src}_{dest}")
                            with lc4:
                                st.text_input("ODA",          value=commercials.get("oda") or "N/A",
                                              key=f"oda_{src}_{dest}")
            else:
                st.info("No rate matrix found in contract tables.")

            # ====================================================
            # VOLUMETRIC
            # ====================================================

            if volumetric:
                st.subheader("📐 Volumetric / CFT Calculation")
                st.info(f"Conversion: {volumetric}")

            # ====================================================
            # CRITICAL CLAUSES
            # ====================================================

            st.subheader("⚖️ Critical Clauses")
            for heading, content in fields["clauses"].items():
                st.text_area(
                    heading,
                    value=content or "Not found in contract",
                    height=90,
                )

            # ====================================================
            # DOWNLOADS
            # ====================================================

            st.divider()
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(output, indent=4),
                    file_name="contract_output.json",
                    mime="application/json",
                )
            with dl2:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=md,
                    file_name="mineru_output.md",
                    mime="text/markdown",
                )