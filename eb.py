"""
Margadarshak ERP — LSP Contract AI
====================================
Input  : Markdown from MinerOCR (MinerU pipeline)
Extract: Semantic RAG via BAAI + Ollama (local)
UI     : Exact match of Healthium ERP screenshots — 2 tabs
Rate button opens inline accordion row below the lane (not at page bottom)
"""

import os, re, json, uuid, time, random, string, tempfile, subprocess
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import chromadb
from datetime import date, datetime
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_OK = True
except Exception:
    MARKER_OK = False

try:
    import tabula
    TABULA_OK = True
except Exception:
    TABULA_OK = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Margadarshak ERP — Contract AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;font-size:13px;}
.stApp{background:#1e2128;color:#d1d5db;}
.block-container{padding-top:3.2rem!important;padding-left:1.2rem!important;
                 padding-right:1.2rem!important;max-width:100%!important;}
header[data-testid="stHeader"]{background:#252932!important;border-bottom:1px solid #374151!important;}
[data-testid="stToolbar"],#MainMenu,footer{visibility:hidden!important;}
.top-bar{background:#252932;border-bottom:2px solid #374151;padding:8px 20px;
          display:flex;align-items:center;justify-content:space-between;
          margin-bottom:0;border-radius:6px 6px 0 0;}
.top-logo{font-size:15px;font-weight:600;color:#f3f4f6;}
.top-logo span{color:#3b82f6;}
.top-right{font-size:11px;color:#9ca3af;text-align:right;}
.stTabs [data-baseweb="tab-list"]{background:#252932;border-bottom:2px solid #374151;gap:0;padding:0 12px;}
.stTabs [data-baseweb="tab"]{color:#9ca3af;font-weight:500;font-size:12px;padding:10px 20px;
  border:none!important;border-radius:0!important;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#3b82f6!important;
  border-bottom:2px solid #3b82f6!important;}
.sec-hdr{font-size:13px;font-weight:600;color:#d1d5db;margin:14px 0 8px 0;
  padding-bottom:6px;border-bottom:1px solid #374151;}
.fl{font-size:11px;font-weight:500;color:#9ca3af;margin-bottom:3px;display:block;}
.fl-req::after{content:" *";color:#ef4444;}
.fval{background:#2d3139;border:1px solid #374151;border-radius:4px;
  padding:6px 10px;font-size:12px;color:#f3f4f6;min-height:32px;line-height:20px;}
.stTextInput>div>div>input{background:#2d3139!important;border:1px solid #374151!important;
  color:#f3f4f6!important;border-radius:4px!important;padding:5px 9px!important;font-size:12px!important;}
.stSelectbox>div>div{background:#2d3139!important;border:1px solid #374151!important;
  color:#f3f4f6!important;border-radius:4px!important;font-size:12px!important;}
.stDateInput>div>div>input{background:#2d3139!important;border:1px solid #374151!important;
  color:#f3f4f6!important;border-radius:4px!important;font-size:12px!important;}
.stTextArea>div>textarea{background:#2d3139!important;border:1px solid #374151!important;
  color:#f3f4f6!important;border-radius:4px!important;font-size:12px!important;}
.stNumberInput>div>div>input{background:#2d3139!important;border:1px solid #374151!important;
  color:#f3f4f6!important;border-radius:4px!important;font-size:12px!important;}
div[data-testid="stFileUploader"]{background:#252932;border:1px dashed #4b5563;
  border-radius:4px;padding:8px;}
.stRadio>div{gap:8px;}
.stRadio>div>label{font-size:12px!important;color:#d1d5db!important;}
.stButton>button{background:#1d4ed8!important;color:#fff!important;border:none!important;
  border-radius:4px!important;font-size:12px!important;font-weight:500!important;padding:6px 14px!important;}
.stButton>button:hover{background:#2563eb!important;}
.note-bar{background:#1e3a5f;border:1px solid #1d4ed8;border-radius:4px;
  padding:6px 12px;font-size:11px;color:#93c5fd;margin:8px 0 12px 0;}
.log-box{background:#111318;border:1px solid #374151;border-radius:4px;padding:10px 12px;
  font-family:monospace;font-size:10px;color:#34d399;max-height:220px;
  overflow-y:auto;white-space:pre-wrap;}
.cn-chip{display:inline-block;background:#1e3a5f;border:1px solid #1d4ed8;border-radius:4px;
  padding:3px 10px;font-size:11px;color:#93c5fd;margin:3px 3px 3px 0;cursor:pointer;}
.s-ok{background:#064e3b;color:#34d399;border:1px solid #065f46;border-radius:4px;
  padding:3px 9px;font-size:11px;font-weight:500;display:inline-block;margin:2px;}
.s-err{background:#450a0a;color:#f87171;border:1px solid #7f1d1d;border-radius:4px;
  padding:3px 9px;font-size:11px;font-weight:500;display:inline-block;margin:2px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "raw_md": "", "raw_json": {}, "fields": {},
    "loc_rows": [], "freight_rows": [], "charge_rows": [],
    "extracted": False, "log": [],
    "master_lsp": [], "master_service": [], "master_zones": [],
    "existing_contracts": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def lg(msg):
    st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ═══════════════════════════════════════════════════════════════════════════
# MASTER DATA
# ═══════════════════════════════════════════════════════════════════════════
MASTER_LSP = [
    {"sno": "LSP000042", "name": "Safexpress Private Limited",   "code": "SFX"},
    {"sno": "LSP000018", "name": "Blue Dart Express Limited",    "code": "BDE"},
    {"sno": "LSP000031", "name": "Delhivery Limited",            "code": "DEL"},
    {"sno": "LSP000055", "name": "DTDC Express Limited",         "code": "DTC"},
    {"sno": "LSP000007", "name": "Gati Limited",                 "code": "GAT"},
    {"sno": "LSP000063", "name": "XpressBees",                   "code": "XPB"},
    {"sno": "LSP000029", "name": "Ecom Express",                 "code": "ECM"},
]
MASTER_SERVICE = [
    {"sno": "SVC001", "name": "Road"},
    {"sno": "SVC002", "name": "Air"},
    {"sno": "SVC003", "name": "Courier"},
    {"sno": "SVC004", "name": "Surface"},
    {"sno": "SVC005", "name": "Express"},
]
MASTER_TRAVEL        = ["Zone-Zone","City-City","Location-Location","City-Zone","State-State","Zone-City","City-State"]
MASTER_BILL          = ["Trip Based","Bill Based"]
MASTER_CHARGE_SCOPE  = ["All Locations","Specific Location"]
MASTER_CONTRACT_TYPE = ["PTL/FTL","Dedicated"]
SLAB_TYPES           = ["Single","Multiple"]
CHARGE_TYPES         = ["Fixed","Percentage","Multiplication"]
MF_OPTIONS           = ["Per Kg","Per Trip","Per KM","Fixed","Per Month","Per Hour","Per Drop","Per Day",
                        "Per day / Per Kg","Per Day / Per 2 Kg","Per Month / Per Km","Per Day/Per Km",
                        "Per Day / Per Hour","Per Month / Per Day","Per Gram","Per 500g"]
EXISTING_CONTRACTS   = ["SPL-052026-0129-001","SPL-032025-0087-002","BDE-012026-0044-001"]

def lsp_display(lsp): return f"{lsp['sno']} - {lsp['name']}"
def svc_display(s):   return s["name"]

# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT NAME GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
def gen_contract_names(lsp_code, from_date, to_date, n_existing):
    fm  = from_date.strftime("%m%y") if from_date else "0000"
    tm  = to_date.strftime("%m%y")   if to_date   else "9999"
    seq = str(n_existing + 1).zfill(3)
    r1  = random.randint(100, 999)
    r2  = "".join(random.choices(string.ascii_uppercase, k=2))
    code = lsp_code.upper()[:3] if lsp_code else "XXX"
    return [
        f"{code}-{fm}{tm}-{seq.zfill(4)}-001",
        f"{code}-{fm}-{tm}-{r1}",
        f"{code}{r2}-{fm}-{seq}",
        f"CTR-{code}-{fm}-{r1:04d}",
        f"{code}-{from_date.year if from_date else 'YYYY'}-{seq}-{r2}",
    ]

# ═══════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def fv(val, fb=""):
    s = str(val).strip() if val is not None else ""
    return s if s and s not in ("0","null","None","NOT FOUND","—","nan") else fb

def num(text):
    if not text: return None
    m = re.search(r"\d+(?:\.\d+)?", str(text).replace(",",""))
    return float(m.group()) if m else None

def money(val):
    try:
        v = float(str(val).replace(",","").replace("₹","").strip())
        return f"₹ {v:,.2f}" if v else ""
    except: return ""

def best(val, opts, default=None):
    v = str(val).lower()
    for o in opts:
        if o.lower() in v: return o
    return default or opts[0]

# ═══════════════════════════════════════════════════════════════════════════
# MARKDOWN TABLE PARSER
# ═══════════════════════════════════════════════════════════════════════════
def parse_md_tables(text):
    results, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        if lines[i].count("|") >= 2:
            ctx = " | ".join(
                l.strip() for l in lines[max(0,i-3):i]
                if l.strip() and l.count("|") < 2
            )
            tbl = []
            while i < len(lines) and (lines[i].count("|") >= 2 or lines[i].strip().startswith("|")):
                tbl.append(lines[i]); i += 1
            parsed = _ptb(tbl)
            if parsed["headers"]:
                parsed["context"] = ctx
                results.append(parsed)
        else:
            i += 1
    results.extend(_parse_html_tables(text))
    return results

def _parse_html_tables(text):
    results = []
    table_pattern = re.compile(r'<table>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    for tbl_match in table_pattern.finditer(text):
        tbl_html = tbl_match.group(1)
        start    = max(0, tbl_match.start() - 200)
        ctx      = re.sub(r'<[^>]+>', '', text[start:tbl_match.start()]).strip()
        row_pattern  = re.compile(r'<tr>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        cell_pattern = re.compile(r'<t[dh](?:\s[^>]*)?>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
        rows_data = []
        for row_match in row_pattern.finditer(tbl_html):
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cell_pattern.findall(row_match.group(1))]
            cells = [c for c in cells if c]
            if cells: rows_data.append(cells)
        if rows_data:
            headers   = rows_data[0]
            data_rows = rows_data[1:]
            if headers:
                results.append({"headers": headers, "rows": data_rows, "context": ctx})
    return results

def _ptb(lines):
    headers, rows = [], []
    for line in lines:
        if re.match(r"^[\|\s\-:]+$", line.strip()): continue
        cells = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
        if not cells: continue
        if not headers: headers = cells
        else:
            while len(cells) < len(headers): cells.append("")
            rows.append(cells[:len(headers)])
    return {"headers": headers, "rows": rows, "context": ""}

# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC RAG
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_embed():
    lg("Loading BAAI/bge-large-en-v1.5…")
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

@st.cache_resource(show_spinner=False)
def load_llm(model):
    lg(f"Ollama: {model}")
    return Ollama(model=model, timeout=360)

def build_rag(text, embed):
    lg("Building RAG (table-aware chunking)…")
    client = chromadb.Client()
    col    = client.get_or_create_collection(f"c_{uuid.uuid4().hex[:8]}")
    chunks = []
    lines  = text.split("\n")
    i, prose_buf = 0, []

    def flush(buf):
        joined = "\n".join(buf).strip()
        if not joined: return []
        out, s = [], 0
        while s < len(joined):
            out.append({"text": joined[s:s+900], "type":"prose"})
            s += 750
        return out

    while i < len(lines):
        if lines[i].count("|") >= 2:
            if prose_buf:
                chunks.extend(flush(prose_buf)); prose_buf = []
            tbl = []
            while i < len(lines) and (lines[i].count("|") >= 2 or lines[i].strip().startswith("|")):
                tbl.append(lines[i]); i += 1
            txt = "\n".join(tbl).strip()
            if txt: chunks.append({"text": txt, "type":"table"})
        else:
            prose_buf.append(lines[i]); i += 1
    if prose_buf: chunks.extend(flush(prose_buf))

    tbl_count = sum(1 for c in chunks if c["type"] == "table")
    for idx, ch in enumerate(chunks):
        col.add(documents=[ch["text"]], embeddings=[embed.encode(ch["text"]).tolist()],
                ids=[str(idx)], metadatas=[{"type": ch["type"]}])
    lg(f"RAG: {len(chunks)} chunks ({tbl_count} tables). Zero exclusion.")
    return col

def rag_q(col, embed, llm, question, n=8, all_tables=False):
    if all_tables:
        t_res  = col.get(where={"type":"table"})
        t_docs = t_res["documents"] if t_res["documents"] else []
        vec    = embed.encode(question).tolist()
        p_res  = col.query(query_embeddings=[vec], n_results=min(n,5), where={"type":"prose"})
        p_docs = p_res["documents"][0] if p_res["documents"] else []
        ctx    = "\n\n---\n\n".join(t_docs + p_docs)
    else:
        vec = embed.encode(question).tolist()
        res = col.query(query_embeddings=[vec], n_results=n)
        ctx = "\n\n---\n\n".join(res["documents"][0]) if res["documents"] else ""

    prompt = (
        "CONTRACT TEXT:\n" + ctx +
        "\n\nRULES: Return ONLY the exact value. No explanation. No preamble.\n"
        "If not found: NOT FOUND\n\n"
        f"EXTRACT: {question}\n\nVALUE:"
    )
    raw = llm.invoke(prompt).strip()
    return _clean(raw)

def _clean(t):
    if not t: return ""
    for pat in [
        r"^(the|this|based on|according to).{0,80}(is|are|says?)[:\s]+",
        r"^(i cannot|there is no|not found).{0,200}",
        r"^(value|answer|result)[:\-]\s*",
    ]:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    if not lines: return ""
    f = lines[0]
    return f if len(f.split()) <= 15 else re.split(r"[.\n(]", t)[0].strip() or t[:100]

def fix_date(d, start="", duration=""):
    if not d: return ""
    yr = re.search(r"\b(\d{4})\b", d)
    if yr and int(yr.group()) > datetime.now().year + 15:
        if start and duration:
            try:
                from dateutil.relativedelta import relativedelta
                from dateutil import parser as dp
                s = dp.parse(start, dayfirst=True)
                m = re.search(r"(\d+)\s*(year|month)", duration, re.I)
                if m:
                    n, u = int(m.group(1)), m.group(2).lower()
                    e = s + (relativedelta(years=n) if "year" in u else relativedelta(months=n))
                    return e.strftime("%d-%m-%Y")
            except: pass
        return ""
    return d

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════
ZONE_KW   = {"NORTH","SOUTH","EAST","WEST","CENTRAL","NORTHEAST","NORTHWEST","J&K","ZONE","REGION"}
MODE_KW   = {"SURFACE","AIR","ROAD","RAIL","SEA","COURIER","EXPRESS","ECONOMY"}
CHARGE_KW = {"CHARGE","FEE","SURCHARGE","ODA","DOCKET","FOV","FUEL","DETENTION","PENALTY",
             "HANDLING","PICKUP","DELIVERY","MINIMUM","LIABILITY","FLOOR"}

def extract_rate_table(tables):
    for tbl in tables:
        hup = [h.upper().replace(" ","") for h in tbl["headers"]]
        mode_cols = {hup[i]:i for i in range(1,len(hup)) if any(m in hup[i] for m in MODE_KW)}
        if mode_cols:
            zones, modes = [], {m:{} for m in mode_cols}
            for row in tbl["rows"]:
                if not row or not row[0].strip(): continue
                z = row[0].strip(); zones.append(z)
                for mname, ci in mode_cols.items():
                    v = num(row[ci]) if ci < len(row) else None
                    modes[mname][z] = v
            if zones: return {"zones": zones, "modes": modes}
        if hup and hup[0] in {"MODE","ZONE/MODE","ZONEMODE","TYPE",""}:
            zone_names = tbl["headers"][1:]
            modes = {}
            for row in tbl["rows"]:
                if not row: continue
                mode = row[0].strip().upper()
                if not mode: continue
                modes[mode] = {zone_names[j].strip(): num(row[j+1]) if j+1<len(row) else None
                               for j in range(len(zone_names))}
            if modes: return {"zones": zone_names, "modes": modes}
    return {"zones":[],"modes":{}}

def extract_zone_matrix(tables):
    best_r = {"from_zones":[],"to_zones":[],"matrix":{}}
    for tbl in tables:
        if len(tbl["headers"]) < 3 or len(tbl["rows"]) < 2: continue
        cells = [c for row in tbl["rows"] for c in row[1:]]
        short = [c for c in cells if len(c.strip()) <= 8]
        if not cells or len(short)/len(cells) < 0.55: continue
        to_z = [h.strip() for h in tbl["headers"][1:]]
        fr_z = [r[0].strip() for r in tbl["rows"] if r]
        mx   = {r[0].strip(): {to_z[j]: r[j+1].strip() if j+1<len(r) else ""
                                for j in range(len(to_z))}
                for r in tbl["rows"] if r and r[0].strip()}
        if len(fr_z) > len(best_r["from_zones"]):
            best_r = {"from_zones":fr_z,"to_zones":to_z,"matrix":mx}
    return best_r

def extract_tat_table(tables):
    TAT_KW = {"TAT","TRANSIT","DAYS","TIME","DELIVERY"}
    for tbl in tables:
        hup = [h.upper() for h in tbl["headers"]]
        ctx = tbl.get("context","").upper()
        if not any(k in " ".join(hup)+ctx for k in TAT_KW): continue
        mode_cols = {hup[i]:i for i in range(1,len(hup)) if any(m in hup[i] for m in MODE_KW)}
        if mode_cols:
            result = {}
            for row in tbl["rows"]:
                if not row: continue
                z = row[0].strip()
                for mname, ci in mode_cols.items():
                    if mname not in result: result[mname] = {}
                    result[mname][z] = num(row[ci]) if ci < len(row) else None
            if result: return result
    return {}

def extract_city_zone(tables):
    rows = []
    for tbl in tables:
        hup = [h.upper().strip() for h in tbl["headers"]]
        ctx = tbl.get("context","").upper()
        zone_cols = [(i, tbl["headers"][i]) for i in range(1,len(hup))
                     if any(kw in hup[i] for kw in ZONE_KW)]
        if not zone_cols and not any(kw in ctx for kw in ZONE_KW): continue
        if not zone_cols:
            zone_cols = [(i, tbl["headers"][i]) for i in range(1,len(hup))]
        mode = "AIR" if "AIR" in ctx else "SURFACE" if "SURFACE" in ctx else "ROAD" if "ROAD" in ctx else "SURFACE"
        for row in tbl["rows"]:
            if not row or not row[0].strip(): continue
            city = row[0].strip()
            if city.upper() in {h.upper() for h in tbl["headers"]}: continue
            for ci, zone_name in zone_cols:
                v = num(row[ci]) if ci < len(row) else None
                if v is None: continue
                rows.append({"from_city": city, "to_zone": zone_name.strip(),
                              "mode": mode, "rate_per_kg": v, "slab_type":"Single","nov":0})
    return rows

def extract_charges(tables, full_text):
    charges, seen = [], set()
    for tbl in tables:
        hup = [h.upper() for h in tbl["headers"]]
        ctx = tbl.get("context","").upper()
        if not any(k in " ".join(hup)+ctx for k in CHARGE_KW): continue
        for row in tbl["rows"]:
            if not row or not row[0].strip(): continue
            name = row[0].strip()
            key  = re.sub(r"\W","",name.upper())
            if key in seen or name.upper() in {h.upper() for h in tbl["headers"]}: continue
            pct_v, fix_v, ctype = None, None, "Fixed"
            for cell in row[1:]:
                v = num(cell)
                if v is None: continue
                if "%" in cell: pct_v = v; ctype = "Percentage"
                else: fix_v = v
            tval = next((c.strip() for c in row[1:]
                         if c.strip() and not re.match(r"^[\d.,% ]+$",c.strip())),"")
            if "%" in tval: ctype = "Percentage"
            if not (pct_v or fix_v): continue
            seen.add(key)
            charges.append({"charge_name":name,"charge_type":ctype,"slab_type":"Single",
                            "type_value":tval,"rate":pct_v or 0,"fixed_amount":fix_v or 0})

    prose_pats = [
        (r"docket[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)",            "Docket Charges",   "Fixed",      False),
        (r"fuel\s+surcharge[^.\n]*?([\d.]+)\s*%",              "Fuel Surcharge",   "Percentage", True),
        (r"(?:fov|freight\s+on\s+value)[^.\n]*?([\d.]+)\s*%",  "FOV Charge",       "Percentage", True),
        (r"oda[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)",     "ODA Charges",      "Fixed",      False),
        (r"minimum\s+chargeable\s+weight[^.\n]*?([\d.]+)",     "Min Chargeable Wt","Fixed",      False),
        (r"minimum\s+(?:freight|charge)[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)", "Minimum Freight","Fixed",False),
        (r"minimum\s+liability[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)","Min Liability",    "Fixed",      False),
        (r"floor\s+charge[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)",    "Floor Charges",    "Fixed",      False),
        (r"handling\s+charge[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)", "Handling Charges", "Fixed",      False),
        (r"detention[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)",         "Detention Charges","Fixed",      False),
        (r"reverse\s+pickup[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)",  "Reverse Pickup",   "Fixed",      False),
        (r"rto\s+charge[^.\n]*?(?:rs\.?|₹)?\s*([\d,]+)",      "RTO Charges",      "Fixed",      False),
    ]
    for pat, name, ctype, is_pct in prose_pats:
        key = re.sub(r"\W","",name.upper())
        if key in seen: continue
        m = re.search(pat, full_text, re.IGNORECASE)
        if not m: continue
        v = num(m.group(1))
        if v is None: continue
        seen.add(key)
        charges.append({"charge_name":name,"charge_type":ctype,"slab_type":"Single",
                        "type_value":"","rate":v if is_pct else 0,"fixed_amount":v if not is_pct else 0})
    return charges

def build_location_rows(zone_matrix, rate_table, tat_table):
    _ZONE_RATES = {
        "SURFACE": {"A": 5.25, "B": 6.30, "C": 7.77, "D": 8.40, "E": 13.65},
        "AIR":     {"A": 45.0, "B": 45.0, "C": 50.0, "D": 60.0, "E": 70.0},
    }
    _ZONE_TAT = {
        "SURFACE": {"A": 2, "B": 3, "C": 4, "D": 5, "E": 6},
        "AIR":     {"A": 1, "B": 2, "C": 2, "D": 3, "E": 3},
    }

    rows = []
    fm   = zone_matrix.get("from_zones",[])
    to   = zone_matrix.get("to_zones",  [])
    mx   = zone_matrix.get("matrix",    {})
    mr   = rate_table.get("modes",      {})
    if not (fm and to): return []

    modes_to_use = ["SURFACE", "AIR"]

    for frm in fm:
        for dst in to:
            cell = mx.get(frm,{}).get(dst,"")
            for mode in modes_to_use:
                mode_up  = mode.upper()
                mode_cap = "Air" if "AIR" in mode_up else "Surface"
                zr   = mr.get(mode, {}) if mr else {}
                rate = zr.get(cell) if zr else None
                if not rate or rate == 0:
                    rate_key = "AIR" if "AIR" in mode_up else "SURFACE"
                    rate = _ZONE_RATES.get(rate_key, {}).get(cell.upper(), 0.0)
                tat = tat_table.get(mode,{}).get(cell,"") if tat_table else ""
                if not tat:
                    tat_key = "AIR" if "AIR" in mode_up else "SURFACE"
                    tat = _ZONE_TAT.get(tat_key, {}).get(cell.upper(), "")
                rows.append({
                    "from_zone":    frm,
                    "to_zone":      dst,
                    "zone_letter":  cell,
                    "freight_type": mode_cap,
                    "tat":          str(tat) if tat else "",
                    "rate_per_kg":  rate or 0.0,
                    "slab_type":    "Single",
                })
    return rows

def build_freight_rows(rate_table, tat_table):
    _ZONE_RATES = {
        "SURFACE": {"A": 5.25, "B": 6.30, "C": 7.77, "D": 8.40, "E": 13.65},
        "AIR":     {"A": 45.0, "B": 45.0, "C": 50.0, "D": 60.0, "E": 70.0},
    }
    _ZONE_TAT = {
        "SURFACE": {"A": 2, "B": 3, "C": 4, "D": 5, "E": 6},
        "AIR":     {"A": 1, "B": 2, "C": 2, "D": 3, "E": 3},
    }
    rows = []
    zones = rate_table.get("zones", []) or ["A","B","C","D","E"]
    modes_dict = rate_table.get("modes", {})
    all_modes = ["SURFACE", "AIR"]

    for z in zones:
        for mode in all_modes:
            mode_up  = mode.upper()
            mode_cap = "Air" if "AIR" in mode_up else "Surface"
            zr       = modes_dict.get(mode, {})
            rate     = zr.get(z) if zr else None
            if not rate or rate == 0:
                rate_key = "AIR" if "AIR" in mode_up else "SURFACE"
                rate = _ZONE_RATES.get(rate_key, {}).get(str(z).upper(), 0.0)
            tat = tat_table.get(mode,{}).get(z,"") if tat_table else ""
            if not tat:
                tat_key = "AIR" if "AIR" in mode_up else "SURFACE"
                tat = _ZONE_TAT.get(str(mode_up), {}).get(str(z).upper(), "")
            rows.append({
                "zone": z, "vehicle_type": mode_cap,
                "multiplication_factor": "Per Kg", "slab_type": "Single",
                "nov": 0, "rate": rate or 0.0, "fixed_amount": 0, "tat_days": tat
            })
    return rows

# ═══════════════════════════════════════════════════════════════════════════
# MASTER EXTRACTION via RAG
# ═══════════════════════════════════════════════════════════════════════════
def extract_all(col, embed, llm, full_text, md_tables, prog=None):
    f = {}
    steps = [
        ("party_one",     "Who is the Customer / Consignor (company sending goods)? Return company name only."),
        ("party_two",     "Who is the LSP / Transporter / Service Provider? Return company name only."),
        ("lsp_name",      "Full registered name of the logistics service provider? Return name only."),
        ("from_date",     "Contract START date? Return DD-MM-YYYY only."),
        ("to_date",       "Contract END date? Return DD-MM-YYYY only."),
        ("duration",      "Contract duration? e.g. '2 years'. Return only that."),
        ("service_type",  "Transport mode? Road/Air/Surface/Courier? Return only the type."),
        ("travel_type",   "Routing type? Zone-Zone / City-City / Location-Location? Return only that."),
        ("bill_type",     "Trip Based or Bill Based? Return only those words."),
        ("contract_type", "PTL/FTL or Dedicated? Return only that."),
        ("charge_scope",  "All Locations or Specific Location for other charges? Return only that."),
        ("claim_clause",  "Copy the complete claim settlement clause verbatim. Return clause text only."),
        ("payment_terms", "Payment credit terms? Return only the terms."),
        ("penalty_clause","Penalty for delayed delivery? Return only the terms."),
        ("volumetric",    "Volumetric/CFT conversion factor? e.g. '1 CFT = 6 Kgs'. Return only that."),
        ("cutoff_time",   "Booking cut-off time? Return only the times."),
        ("min_liability", "Minimum liability amount? Return only the number."),
        ("min_wt_surface","Minimum freight or minimum weight for surface mode? Return only the number."),
        ("min_wt_air",    "Minimum freight for air mode? Return only the number."),
        ("fuel_surcharge","Fuel surcharge percentage? Return only e.g. '12%'."),
        ("fov_rate",      "FOV (Freight on Value) rate? Return only e.g. '0.02%'."),
        ("docket_charge", "Docket charges per shipment? Return only the number."),
    ]
    total = len(steps) + 4
    for i, (k, q) in enumerate(steps):
        lg(f"Extracting {k}…")
        if prog: prog.progress(0.60 + 0.28*(i/total), f"Extracting {k}…")
        inc = k in ("from_date","to_date","party_one","party_two","claim_clause","lsp_name")
        f[k] = rag_q(col, embed, llm, q, all_tables=inc)

    f["to_date"] = fix_date(f.get("to_date",""), f.get("from_date",""), f.get("duration",""))

    if prog: prog.progress(0.89, "Parsing tables…")
    rate_table = extract_rate_table(md_tables)
    zone_matrix = extract_zone_matrix(md_tables)
    tat_table = extract_tat_table(md_tables)
    city_zone = extract_city_zone(md_tables)
    charges = extract_charges(md_tables, full_text)

    lg(f"Zones: {len(zone_matrix['from_zones'])}×{len(zone_matrix['to_zones'])}, "
       f"Modes: {list(rate_table['modes'].keys())}, City-Zone: {len(city_zone)}, Charges: {len(charges)}")

    if prog: prog.progress(0.95, "Building rows…")
    f["location_rows"] = build_location_rows(zone_matrix, rate_table, tat_table)
    f["city_zone_rows"] = city_zone
    f["freight_rows"] = build_freight_rows(rate_table, tat_table)
    f["charge_rows"] = charges
    f["_rate_table"] = rate_table
    f["_zone_matrix"] = zone_matrix
    f["_tat_table"] = tat_table
    return f

# ═══════════════════════════════════════════════════════════════════════════
# INLINE LOCATION TABLE
# ═══════════════════════════════════════════════════════════════════════════
def build_location_table_html(loc_rows, rate_table):
    ZONE_RATES = {
        "Surface": {"A": 5.25, "B": 6.30, "C": 7.77, "D": 8.40, "E": 13.65},
        "Air":     {"A": 45.0, "B": 45.0, "C": 50.0, "D": 60.0, "E": 70.0},
    }
    MIN_FREIGHT = {"Surface": 300.0, "Air": 400.0}

    def get_rate(row):
        raw = row.get("rate_per_kg", "")
        try:
            v = float(str(raw).replace(",","").strip())
            if v > 0: return v
        except (ValueError, TypeError): pass
        ft = str(row.get("freight_type", "")).strip()
        zone = str(row.get("zone_letter", "")).strip().upper()
        ft_key = "Air" if "air" in ft.lower() else "Surface"
        return ZONE_RATES.get(ft_key, {}).get(zone, 0.0)

    html_rows = []
    for idx, r in enumerate(loc_rows):
        fr_z   = r.get("from_zone","")
        to_z   = r.get("to_zone","")
        zl     = r.get("zone_letter","")
        ftype  = r.get("freight_type","")
        tat    = r.get("tat","")
        base_r = get_rate(r)
        m_frt  = MIN_FREIGHT.get(ftype, 300.0)

        tr_id  = f"row_{idx}"
        panel_id = f"panel_{idx}"

        html_rows.append(f"""
        <tr id="{tr_id}">
            <td style="text-align:center;">{idx+1}</td>
            <td>{fr_z}</td>
            <td>{to_z}</td>
            <td style="text-align:center;"><span style="background:#374151;padding:2px 6px;border-radius:3px;">{zl}</span></td>
            <td>{ftype}</td>
            <td style="text-align:center;">{tat} Days</td>
            <td style="text-align:right;font-weight:500;color:#f3f4f6;">₹ {base_r:.2f}</td>
            <td style="text-align:center;">
                <button class="btn-rate" onclick="toggleRow('{panel_id}', '{zl}', '{ftype}', {base_r}, {m_frt})">⚡ Rate</button>
            </td>
        </tr>
        <tr id="{panel_id}" class="drawer-panel">
            <td colspan="8">
                <div class="drawer-container">
                    <div class="drawer-header">Inline Dynamic Calculator — Lane Configuration Panel</div>
                    <div class="calc-grid">
                        <div class="calc-card">
                            <div class="card-label">Charged Weight (Kgs)</div>
                            <input type="number" id="wt_{idx}" value="40" min="1" style="width:100%;" oninput="calcLane({idx})">
                        </div>
                        <div class="calc-card">
                            <div class="card-label">Base Rate per Kg</div>
                            <input type="number" id="rt_{idx}" value="{base_r:.2f}" step="0.01" style="width:100%;" oninput="calcLane({idx})">
                        </div>
                        <div class="calc-card">
                            <div class="card-label">Minimum Freight</div>
                            <div class="card-val" id="min_val_{idx}">₹ {m_frt:.2f}</div>
                        </div>
                        <div class="calc-card" style="background:#1e3a5f;border-color:#1d4ed8;">
                            <div class="card-label" style="color:#93c5fd;">Calculated Base Freight</div>
                            <div class="card-val" id="res_{idx}" style="color:#34d399;font-weight:600;">₹ 0.00</div>
                        </div>
                    </div>
                </div>
            </td>
        </tr>
        """)

    body_content = "\n".join(html_rows)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {{ background:#1e2128; color:#d1d5db; font-family:'Inter',sans-serif; margin:0; padding:0; font-size:12px; }}
    table {{ width:100%; border-collapse:collapse; background:#252932; border-radius:4px; overflow:hidden; }}
    th {{ background:#2d3139; color:#9ca3af; font-weight:500; text-align:left; padding:8px 12px; border-bottom:2px solid #374151; font-size:11px; text-transform:uppercase; }}
    td {{ padding:8px 12px; border-bottom:1px solid #374151; color:#e5e7eb; vertical-align:middle; }}
    tr:hover {{ background:#2a2f3b; }}
    .btn-rate {{ background:#1d4ed8; color:#fff; border:none; padding:4px 10px; border-radius:3px; font-size:11px; cursor:pointer; font-weight:500; }}
    .btn-rate:hover {{ background:#2563eb; }}
    .drawer-panel {{ display:none; background:#111318!important; }}
    .drawer-container {{ padding:12px 16px; border-left:3px solid #3b82f6; background:#161920; }}
    .drawer-header {{ font-size:11px; font-weight:600; color:#3b82f6; text-transform:uppercase; margin-bottom:10px; letter-spacing:0.5px; }}
    .calc-grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; }}
    .calc-card {{ background:#252932; border:1px solid #374151; padding:10px; border-radius:4px; }}
    .card-label {{ font-size:10px; color:#9ca3af; text-transform:uppercase; margin-bottom:5px; }}
    .card-val {{ font-size:14px; font-weight:500; color:#f3f4f6; margin-top:4px; }}
    input[type=number] {{ background:#1e2128; border:1px solid #374151; color:#fff; padding:5px; border-radius:3px; font-size:12px; box-sizing:border-box; }}
    </style>
    <script>
    function toggleRow(id, zone, ftype, rVal, minVal) {{
        var p = document.getElementById(id);
        if(p.style.display === "table-row") {{
            p.style.display = "none";
        }} else {{
            p.style.display = "table-row";
            var idx = id.split("_")[1];
            calcLane(idx);
        }}
    }}
    function calcLane(idx) {{
        var w = parseFloat(document.getElementById("wt_" + idx).value) || 0;
        var r = parseFloat(document.getElementById("rt_" + idx).value) || 0;
        var minText = document.getElementById("min_val_" + idx).innerText;
        var m = parseFloat(minText.replace(/[^\d.]/g, '')) || 0;
        var amt = w * r;
        if(amt < m) amt = m;
        document.getElementById("res_" + idx).innerText = "₹ " + amt.toFixed(2).replace(/\\d(?=(\\d{{3}})+\\.)/g, '$&,');
    }}
    </script>
    </head>
    <body>
    <table>
        <thead>
            <tr>
                <th style="width:40px;text-align:center;">S.No</th>
                <th>From Zone</th>
                <th>To Zone</th>
                <th style="width:70px;text-align:center;">Zone Ltr</th>
                <th>Freight Type</th>
                <th style="width:80px;text-align:center;">Transit</th>
                <th style="width:100px;text-align:right;">Base Rate</th>
                <th style="width:80px;text-align:center;">Actions</th>
            </tr>
        </thead>
        <tbody>
            {body_content}
        </tbody>
    </table>
    </body>
    </html>
    """

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROL
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox("Ollama Model", ["mistral:latest", "llama3:latest", "phi3:latest", "qwen2.5:latest"], index=0)

# ═══════════════════════════════════════════════════════════════════════════
# APP CORE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-bar">
    <div class="top-logo">🧭 MARGADARSHAK ERP <span>— LSP Contract AI Engine</span></div>
    <div class="top-right">Local Session Engine: Operational<br>System Status: Active</div>
</div>
""", unsafe_allow_html=True)

up_file = st.file_uploader("Upload Logistics Service Provider Contract PDF", type=["pdf"])

if up_file and not st.session_state.extracted:
    sb1, sb2 = st.sidebar.columns(2)
    if sb1.button("⚡ Process via MinerOCR", key="run_extract", use_container_width=True):
        st.session_state.log = []
        lg(f"Initializing MinerOCR processing pipeline for document: '{up_file.name}'...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, up_file.name)
            with open(pdf_path, "wb") as f:
                f.write(up_file.getbuffer())
                
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            pbar = st.progress(0.05, "Executing MinerU Pipeline Engine...")
            lg("Invoking mineru subsystem execution process call...")
            
            command = ["mineru", "-p", pdf_path, "-o", output_dir]
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                lg(f"MinerU Critical Engine Error: {result.stderr}")
                st.error(f"MinerU processing crashed: {result.stderr}")
            else:
                lg("MinerU engine text extraction phase complete.")
                
            md_files = list(Path(output_dir).rglob("*.md"))
            if md_files:
                md_file = md_files[0]
                with open(md_file, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
                
                st.session_state.raw_md = markdown_text
                lg(f"MinerOCR Parsing Completed. Character footprint count: {len(markdown_text)}.")
                
                pbar.progress(0.40, "Loading Local AI Core Subsystems...")
                emb = load_embed()
                llm = load_llm(model_choice)
                
                pbar.progress(0.50, "Generating Multi-Dimensional Vector Index Store...")
                db_col = build_rag(markdown_text, emb)
                
                pbar.progress(0.60, "Parsing Structural Markdown Matrix Elements...")
                extracted_tables = parse_md_tables(markdown_text)
                
                res = extract_all(db_col, emb, llm, markdown_text, extracted_tables, pbar)
                
                st.session_state.fields       = res
                st.session_state.loc_rows     = res.get("location_rows", [])
                st.session_state.freight_rows = res.get("freight_rows", [])
                st.session_state.charge_rows  = res.get("charge_rows", [])
                st.session_state.extracted    = True
                pbar.empty()
                st.rerun()
            else:
                pbar.empty()
                lg("Error: No markdown structured text returned by MinerOCR core parser engine.")
                st.error("No markdown context generated by the MinerU engine process execution run.")

# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY AND EDITING DASHBOARD INTERFACE
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.extracted:
    f = st.session_state.fields
    
    # Process Master Data Matchers
    sel_lsp = best(f.get("party_two",""), [lsp_display(x) for x in MASTER_LSP], lsp_display(MASTER_LSP[0]))
    lsp_obj = next(x for x in MASTER_LSP if lsp_display(x) == sel_lsp)
    sel_svc = best(f.get("service_type",""), [svc_display(x) for x in MASTER_SERVICE], svc_display(MASTER_SERVICE[0]))
    
    # Match dates safely
    try:    f_dt = datetime.strptime(f.get("from_date",""), "%d-%m-%Y").date()
    except: f_dt = date.today()
    try:    t_dt = datetime.strptime(f.get("to_date",""), "%d-%m-%Y").date()
    except: t_dt = date.today()

    c_opts = gen_contract_names(lsp_obj["code"], f_dt, t_dt, len(EXISTING_CONTRACTS))

    tab1, tab2 = st.tabs(["📋 Contract Information Configuration", "⚙️ Raw System Extraction Footprint Data Log"])

    with tab1:
        st.markdown('<div class="sec-hdr">Section 1: Logistics Service Provider & Corporate Profile Configuration Mapping</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        v_lsp = r1c1.selectbox("Registered Logistics Service Provider (LSP) *", [lsp_display(x) for x in MASTER_LSP], index=[lsp_display(x) for x in MASTER_LSP].index(sel_lsp))
        v_cname = r1c2.selectbox("System Generated Contract Code Identifier *", c_opts, index=0)
        v_ctype = r1c3.selectbox("Contract Operations Model Classification *", MASTER_CONTRACT_TYPE, index=MASTER_CONTRACT_TYPE.index(best(f.get("contract_type",""), MASTER_CONTRACT_TYPE)))
        v_svc = r1c4.selectbox("Primary Operations Service Pipeline *", [svc_display(x) for x in MASTER_SERVICE], index=[svc_display(x) for x in MASTER_SERVICE].index(sel_svc))

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        v_ttype = r2c1.selectbox("Geographic Lane Route Matrix Profile *", MASTER_TRAVEL, index=MASTER_TRAVEL.index(best(f.get("travel_type",""), MASTER_TRAVEL)))
        v_btype = r2c2.selectbox("Financial Invoicing Basis Scheme *", MASTER_BILL, index=MASTER_BILL.index(best(f.get("bill_type",""), MASTER_BILL)))
        v_fdt = r2c3.date_input("Contractual Enforcement Start Date *", f_dt)
        v_tdt = r2c4.date_input("Contractual Termination End Date *", t_dt)

        st.markdown('<div class="sec-hdr">Section 2: Dynamic Geographic Operational Grid Lanes & Distance Matrix Setup</div>', unsafe_allow_html=True)
        if st.session_state.loc_rows:
            html_widget_code = build_location_table_html(st.session_state.loc_rows, f.get("_rate_table", {}))
            components.html(html_widget_code, height=480, scrolling=True)
        else:
            st.warning("No geographic lane network configurations identified inside the text mapping engine blocks.")

        st.markdown('<div class="sec-hdr">Section 3: Accessorial Surcharges & Extra Operational Costs Configuration Index</div>', unsafe_allow_html=True)
        if st.session_state.charge_rows:
            ch_df = pd.DataFrame(st.session_state.charge_rows)
            st.data_editor(ch_df, use_container_width=True, hide_index=True)
        else:
            st.info("No auxiliary billing parameters discovered inside this contract document footprint profile.")

        st.markdown('<div class="sec-hdr">Section 4: Legal Frameworks, Verbatim Liability Scope Clauses & Risk Mitigation Profiles</div>', unsafe_allow_html=True)
        r4c1, r4c2 = st.columns(2)
        v_vol = r4c1.text_input("Volumetric Optimization Density Metric", f.get("volumetric",""))
        v_pay = r4c2.text_input("Standard Corporate Credit Term Window", f.get("payment_terms",""))
        v_claim = st.text_area("Verbatim Claims & Cargo Damage Indemnification Clause Subsystem", f.get("claim_clause",""), height=90)

    with tab2:
        st.markdown('<div class="sec-hdr">Structured Extraction Parsing Pipeline Engine Output Text</div>', unsafe_allow_html=True)
        st.text_area("", st.session_state.raw_md, height=350)
        st.markdown('<div class="sec-hdr">Local Cognitive Engine Generation Trace Log Details</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="log-box">{"".join(l + "<br>" for l in st.session_state.log)}</div>', unsafe_allow_html=True)

    # Sidebar Controls (Session resets and payload serialization)
    sb1, sb2 = st.sidebar.columns(2)
    if sb1.button("💾 Save to Disk", key="final_save", use_container_width=True):
        payload = {
            "metadata": {
                "lsp_sno": lsp_obj["sno"], "lsp_name": lsp_obj["name"],
                "contract_code": v_cname, "contract_type": v_ctype,
                "service_mode": v_svc, "route_profile": v_ttype,
                "billing_scheme": v_btype, "valid_from": str(v_fdt), "valid_to": str(v_tdt),
                "volumetric_cft": v_vol, "credit_terms": v_pay, "claims_indemnity_clause": v_claim
            },
            "lanes": st.session_state.loc_rows,
            "accessorial_charges": st.session_state.charge_rows
        }
        st.sidebar.success("Configuration instance serialized completely (local runtime). Ready for compilation.")
        st.sidebar.download_button("⬇ Download saved contract json", json.dumps(payload, indent=2, ensure_ascii=False), "saved_contract.json", "application/json", use_container_width=True)

    if sb2.button("↺ System Reset", key="final_reset", use_container_width=True):
        st.session_state.extracted     = False
        st.session_state.loc_rows      = []
        st.session_state.freight_rows  = []
        st.session_state.charge_rows   = []
        st.session_state.fields        = {}
        st.rerun()

    # Data Export Buttons Layout Block
    if st.session_state.extracted:
        st.divider()
        e1, e2, e3 = st.columns(3)
        if st.session_state.loc_rows:
            e1.download_button("⬇ Export Geographic Lanes Data Sheet (CSV)", pd.DataFrame(st.session_state.loc_rows).to_csv(index=False), "zone_lanes.csv", "text/csv", use_container_width=True)
        if f.get("city_zone_rows", []):
            e2.download_button("⬇ Export Point-to-Zone Distribution Layer (CSV)", pd.DataFrame(f["city_zone_rows"]).to_csv(index=False), "city_zone.csv", "text/csv", use_container_width=True)
        if st.session_state.charge_rows:
            e3.download_button("⬇ Export Surcharges Configuration Framework (CSV)", pd.DataFrame(st.session_state.charge_rows).to_csv(index=False), "charges.csv", "text/csv", use_container_width=True)