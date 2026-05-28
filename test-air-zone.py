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

MODEL = "qwen3:4b"

# ==========================================================
# PROMPTS
# ==========================================================

TABLE_PROMPT = """You are a logistics contract parser. Extract data from the table text below.

Rules:
- Return ONLY valid JSON
- null for missing fields

Return:
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
}
"""

TEXT_PROMPT = """You are a logistics contract clause extractor.

Rules:
- Return ONLY valid JSON
- null if not found

Return:
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
}
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
        "matrices": {}
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
# IGNORE TABLES
# ==========================================================

IGNORE_TABLE_PATTERNS = [
    "account holder",
    "ifsc",
    "bank",
    "billing address",
    "gst number",
    "signature",
]

# ==========================================================
# HELPERS
# ==========================================================

def clean_markdown(text):
    return "\n".join(
        line.strip()
        for line in text.split("\n")
        if line.strip()
    )

def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def parse_llm_json(content):

    content = strip_think_tags(content)

    content = re.sub(r"```json|```", "", content).strip()

    m = re.search(r"\{.*\}", content, re.DOTALL)

    if not m:
        return {}

    try:
        return json.loads(m.group())
    except:
        return {}

# ==========================================================
# TABLE EXTRACTION
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

            data = [
                r + [""] * (len(headers) - len(r))
                for r in rows[1:]
            ]

            tables.append({
                "table_id": f"table_{idx+1}",
                "headers": headers,
                "rows": data
            })

    return tables

def filter_tables(tables):

    filtered = []

    for t in tables:

        txt = (
            " ".join(t["headers"]) +
            " " +
            " ".join(" ".join(r) for r in t["rows"])
        ).lower()

        if not any(p in txt for p in IGNORE_TABLE_PATTERNS):
            filtered.append(t)

    return filtered

def tables_to_text(tables):

    lines = []

    for t in tables:

        lines.append("---- TABLE ----")

        lines.append(" | ".join(t["headers"]))

        for row in t["rows"]:
            lines.append(" | ".join(row))

    return "\n".join(lines)

# ==========================================================
# MULTI-MODE RATE MATRIX EXTRACTION
# ==========================================================

def detect_rate_mode(text):

    txt = text.lower()

    if any(x in txt for x in [
        "air express",
        "air mode",
        "air freight",
        "air rates",
        "air cargo",
        "by air"
    ]):
        return "Air"

    if any(x in txt for x in [
        "surface",
        "road",
        "ground",
        "surface express",
        "by surface"
    ]):
        return "Surface"

    if any(x in txt for x in [
        "express",
        "priority"
    ]):
        return "Express"

    return "Unknown"

def extract_rate_matrices(tables):

    matrices = {}

    for t in tables:

        headers = t["headers"]
        rows = t["rows"]

        full_text = (
            " ".join(headers) + " " +
            " ".join(" ".join(r) for r in rows)
        )

        mode = detect_rate_mode(full_text)

        header_is_zone = (
            headers and
            headers[0].strip().lower() == "zone"
        )

        first_row_is_zone = (
            rows and
            rows[0] and
            rows[0][0].strip().lower() == "zone"
        )

        is_rate_table = (
            "rate matrix" in full_text.lower()
            or "rate" in full_text.lower()
            or header_is_zone
            or first_row_is_zone
        )

        if not is_rate_table:
            continue

        if first_row_is_zone:

            col_headers = rows[0]
            data_rows = rows[1:]

        elif header_is_zone:

            col_headers = headers
            data_rows = rows

        else:
            continue

        if mode not in matrices:
            matrices[mode] = {}

        dest_zones = [z.strip() for z in col_headers[1:]]

        for row in data_rows:

            if not row:
                continue

            src = row[0].strip()

            if not src:
                continue

            if src not in matrices[mode]:
                matrices[mode][src] = {}

            for i, dz in enumerate(dest_zones):

                value = ""

                if i + 1 < len(row):
                    value = row[i + 1].strip()

                matrices[mode][src][dz] = value

    return matrices

# ==========================================================
# COMMERCIAL REGEX
# ==========================================================

def regex_commercials(tables, text):

    combined = tables_to_text(tables) + "\n" + text

    def find(patterns):

        for pat in patterns:

            m = re.search(pat, combined, re.IGNORECASE)

            if m:
                return m.group(1).strip()

        return None

    return {
        "fuel": find([
            r"Fuel.*?(\d+(?:\.\d+)?%)"
        ]),
        "docket": find([
            r"Docket.*?(\d+)"
        ]),
        "mcw": find([
            r"Minimum.*?Weight.*?(\d+\s*kg)"
        ]),
        "oda": find([
            r"ODA.*?(\d+)"
        ]),
    }

# ==========================================================
# LLM
# ==========================================================

def llm_call(prompt, text):

    try:

        resp = ollama.chat(
            model=MODEL,
            options={
                "temperature": 0,
                "top_p": 1,
                "think": False
            },
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": text[:6000]
                }
            ]
        )

        return parse_llm_json(
            resp["message"]["content"]
        )

    except:
        return {}

# ==========================================================
# TEXT HELPERS
# ==========================================================

def remove_tables_from_md(md):

    soup = BeautifulSoup(md, "html.parser")

    for t in soup.find_all("table"):
        t.decompose()

    return soup.get_text()

def big_chunks(text, max_chars=5000):

    paras = [
        p.strip()
        for p in re.split(r"\n{2,}", text)
        if p.strip()
    ]

    chunks = []

    current = []

    size = 0

    for p in paras:

        if size + len(p) > max_chars:

            chunks.append("\n\n".join(current))

            current = []

            size = 0

        current.append(p)

        size += len(p)

    if current:
        chunks.append("\n\n".join(current))

    return chunks[:3]

# ==========================================================
# STREAMLIT
# ==========================================================

st.set_page_config(
    page_title="FreightIQ",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 FreightIQ Contract Extractor")

st.caption(f"Model: {MODEL}")

uploaded = st.file_uploader(
    "Upload Contract PDF",
    type=["pdf"]
)

if uploaded:

    with tempfile.TemporaryDirectory() as tmp:

        pdf_path = os.path.join(tmp, uploaded.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())

        out_dir = os.path.join(tmp, "out")

        os.makedirs(out_dir, exist_ok=True)

        if st.button("⚙️ Process Contract"):

            # ==================================================
            # MINERU
            # ==================================================

            with st.spinner("Running MinerU..."):

                proc = subprocess.run(
                    [
                        "mineru",
                        "-p",
                        pdf_path,
                        "-o",
                        out_dir
                    ],
                    capture_output=True,
                    text=True
                )

            if proc.returncode != 0:

                st.error(proc.stderr)

                st.stop()

            md_files = list(
                Path(out_dir).rglob("*.md")
            )

            if not md_files:

                st.error("No markdown generated")

                st.stop()

            with open(md_files[0], "r", encoding="utf-8") as f:
                md = clean_markdown(f.read())

            # ==================================================
            # TABLES
            # ==================================================

            tables = extract_html_tables(md)

            tables = filter_tables(tables)

            table_text = tables_to_text(tables)

            rate_matrices = extract_rate_matrices(tables)

            # ==================================================
            # TEXT
            # ==================================================

            plain_text = remove_tables_from_md(md)

            commercials = regex_commercials(
                tables,
                plain_text
            )

            # ==================================================
            # LLM TABLE
            # ==================================================

            with st.spinner("LLM reading tables..."):

                lm_tables = llm_call(
                    TABLE_PROMPT,
                    table_text
                )

            # ==================================================
            # LLM TEXT
            # ==================================================

            chunks = big_chunks(plain_text)

            text_results = []

            for idx, chunk in enumerate(chunks):

                with st.spinner(f"Reading clauses {idx+1}/{len(chunks)}"):

                    r = llm_call(
                        TEXT_PROMPT,
                        chunk
                    )

                    if r:
                        text_results.append(r)

            # ==================================================
            # MERGE
            # ==================================================

            final = copy.deepcopy(MASTER_SCHEMA)

            for k, v in lm_tables.items():

                if k in final["commercial_terms"]:
                    final["commercial_terms"][k] = v

                if k in final["parties"]:
                    final["parties"][k] = v

                if k in final["contract_information"]:
                    final["contract_information"][k] = v

            for r in text_results:

                for k, v in r.items():

                    if k in final["critical_clauses"]:
                        final["critical_clauses"][k] = v

                    if k in final["parties"]:
                        final["parties"][k] = v

                    if k in final["contract_information"]:
                        final["contract_information"][k] = v

            final["transport"]["matrices"] = rate_matrices

            output = {
                "extracted_contract": final,
                "relevant_tables": tables
            }

            # ==================================================
            # JSON
            # ==================================================

            st.success("Done")

            with st.expander("Raw JSON"):
                st.json(output)

            # ==================================================
            # CONTRACT INFO
            # ==================================================

            st.header("📄 Contract")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.text_input(
                    "LSP",
                    value=final["parties"]["lsp_name"] or "N/A"
                )

            with c2:
                st.text_input(
                    "Shipper",
                    value=final["parties"]["shipper_name"] or "N/A"
                )

            with c3:
                st.text_input(
                    "Effective",
                    value=final["contract_information"]["contract_effective_from_date"] or "N/A"
                )

            with c4:
                st.text_input(
                    "End Date",
                    value=final["contract_information"]["contract_end_date"] or "N/A"
                )

            # ==================================================
            # COMMERCIALS
            # ==================================================

            st.subheader("💰 Commercial Terms")

            cc1, cc2, cc3, cc4 = st.columns(4)

            with cc1:
                st.text_input(
                    "Fuel",
                    value=commercials.get("fuel") or "N/A"
                )

            with cc2:
                st.text_input(
                    "Docket",
                    value=commercials.get("docket") or "N/A"
                )

            with cc3:
                st.text_input(
                    "MCW",
                    value=commercials.get("mcw") or "N/A"
                )

            with cc4:
                st.text_input(
                    "ODA",
                    value=commercials.get("oda") or "N/A"
                )

            # ==================================================
            # RATE MATRICES
            # ==================================================

            st.subheader("🗺️ Lane Rate Matrices")

            if rate_matrices:

                available_modes = list(rate_matrices.keys())

                selected_mode = st.selectbox(
                    "Select Transport Mode",
                    available_modes
                )

                selected_matrix = rate_matrices[selected_mode]

                all_dest = sorted({
                    dz
                    for v in selected_matrix.values()
                    for dz in v
                })

                df_rate = pd.DataFrame(
                    [
                        {
                            **{"From \\ To": src},
                            **{
                                dz: dm.get(dz, "–")
                                for dz in all_dest
                            }
                        }
                        for src, dm in selected_matrix.items()
                    ]
                ).set_index("From \\ To")

                st.dataframe(
                    df_rate,
                    use_container_width=True
                )

                # ==============================================
                # FILTERS
                # ==============================================

                st.subheader("📦 Lane Filters")

                f1, f2 = st.columns(2)

                with f1:

                    src_filter = st.selectbox(
                        "Source Zone",
                        ["All"] + sorted(selected_matrix.keys())
                    )

                all_destinations = sorted({
                    d
                    for x in selected_matrix.values()
                    for d in x.keys()
                })

                with f2:

                    dest_filter = st.selectbox(
                        "Destination Zone",
                        ["All"] + all_destinations
                    )

                # ==============================================
                # LANE DETAILS
                # ==============================================

                st.subheader(
                    f"🚚 {selected_mode} Lane Details"
                )

                for src, dest_map in selected_matrix.items():

                    if src_filter != "All" and src != src_filter:
                        continue

                    for dest, rate in dest_map.items():

                        if dest_filter != "All" and dest != dest_filter:
                            continue

                        if not rate or rate in ["–", "-", ""]:
                            continue

                        with st.expander(f"{src} → {dest}"):

                            c1, c2, c3, c4 = st.columns(4)

                            with c1:
                                st.text_input(
                                    "Rate",
                                    value=rate,
                                    key=f"{selected_mode}_{src}_{dest}_rate"
                                )

                            with c2:
                                st.text_input(
                                    "MCW",
                                    value=commercials.get("mcw") or "N/A",
                                    key=f"{selected_mode}_{src}_{dest}_mcw"
                                )

                            with c3:
                                st.text_input(
                                    "Fuel",
                                    value=commercials.get("fuel") or "N/A",
                                    key=f"{selected_mode}_{src}_{dest}_fuel"
                                )

                            with c4:
                                st.text_input(
                                    "ODA",
                                    value=commercials.get("oda") or "N/A",
                                    key=f"{selected_mode}_{src}_{dest}_oda"
                                )

            else:

                st.info("No lane matrices found")

            # ==================================================
            # CLAUSES
            # ==================================================

            st.subheader("⚖️ Critical Clauses")

            for k, v in final["critical_clauses"].items():

                st.text_area(
                    k,
                    value=v or "N/A",
                    height=100
                )

            # ==================================================
            # DOWNLOADS
            # ==================================================

            st.divider()

            d1, d2 = st.columns(2)

            with d1:

                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(output, indent=4),
                    file_name="contract_output.json",
                    mime="application/json"
                )

            with d2:

                st.download_button(
                    "⬇️ Download Markdown",
                    data=md,
                    file_name="mineru_output.md",
                    mime="text/markdown"
                )