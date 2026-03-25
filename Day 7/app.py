"""
app.py — Day 7: Invoice Compliance Pipeline
4-step pipeline: classify → look up rate → calculate → route

Run:     streamlit run app.py
Install: pip install streamlit openai python-dotenv
"""

import json
import os
import re
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
api_key    = os.getenv("AZURE_OPENAI_API_KEY", "")
chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Day 7 — Invoice Pipeline", page_icon="🔀", layout="wide")
st.title("🔀 Day 7 — Invoice Compliance Pipeline")
st.caption("Enter an invoice · Run the 4-node pipeline · See every decision explained")

if not all([endpoint, api_key, chat_model]):
    st.error("Missing .env credentials. Copy .env.example → .env and fill in your values.")
    st.stop()

client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-08-01-preview")

# ── GST rates lookup table ────────────────────────────────────────────────────
# Activity 3: add new transaction types here
RATES = {
    "B2B Intra-State":    {"rate": 18.0, "split": "CGST 9% + SGST 9%", "section": "CGST Act §9"},
    "B2B Inter-State":    {"rate": 18.0, "split": "IGST 18%",           "section": "IGST Act §5"},
    "Export of Services": {"rate": 0.0,  "split": "Zero-rated (LUT)",   "section": "IGST Act §16(1)"},
    "Import RCM":         {"rate": 18.0, "split": "RCM — recipient pays","section": "CGST Act §9(3)"},
    "Exempt":             {"rate": 0.0,  "split": "Exempt",              "section": "CGST Schedule III"},
    "Works Contract":     {"rate": 18.0, "split": "IGST 18%",           "section": "Notification 11/2017"},
}

# ── Sample invoices ────────────────────────────────────────────────────────────
# Activity 1: add your own invoice samples to this dict
SAMPLES = {
    "Select an invoice...": ("", 0),
    "TechSolutions — B2B Software (Mumbai→Bengaluru)": (
        "TechSolutions Pvt Ltd (Mumbai, GSTIN 27AAACT1234A1Z1) raised invoice #TS-892 "
        "for ₹8,50,000 to DataCore Ltd (Bengaluru, GSTIN 29AABCD5678B1Z2) for software "
        "implementation services. Q3 FY2025.", 850000),
    "GlobalServ — IT Export to USA (LUT filed)": (
        "GlobalServ India Pvt Ltd exports IT consulting to Acme Corp USA. "
        "Invoice USD 15,000 (~₹12,50,000). LUT filed for FY2024-25 (Ref: LUT/2024-25/0042). "
        "Payment in USD via wire transfer.", 1250000),
    "Advocate K.P. Sharma — Legal Services (RCM)": (
        "Advocate K.P. Sharma (individual) provides litigation services to "
        "Infovista Systems Ltd (GSTIN 33AABCI9876C1Z3), Chennai. Fee: ₹75,000. October 2024. "
        "No GST registration on advocate. RCM applies.", 75000),
    "BuildRight — Govt Road Contract (Works Contract)": (
        "Government of Karnataka awards road construction contract ₹45,00,000 "
        "to BuildRight Infra Pvt Ltd under PMGSY scheme. Rural road, FY2025.", 4500000),
    "Ambiguous — Incomplete invoice": (
        "Our company did some consulting for a client last month. Amount around 12 lakhs. "
        "Need to check tax implications. - Amit", 1200000),
    "Minimal — No details": (
        "Please check GST on recent payment. - Finance team", 0),
}

# ── Helper: one LLM call ─────────────────────────────────────────────────────
def chat(prompt):
    response = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content

# ── The 4 pipeline nodes ──────────────────────────────────────────────────────

def classify(state):
    """Node 1 — LLM classifies the invoice text."""
    prompt = (
        f"Classify this invoice and return ONLY valid JSON:\n\n{state['invoice_text']}\n\n"
        f"JSON format: {{\"transaction_type\": \"B2B Intra-State|B2B Inter-State|Export of Services"
        f"|Import RCM|Exempt|Works Contract|Unknown\", \"confidence\": 0.0-1.0, \"notes\": \"brief reason\"}}"
    )
    raw = chat(prompt)
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        result  = json.loads(cleaned)
        state.update({
            "transaction_type": result.get("transaction_type", "Unknown"),
            "confidence":       float(result.get("confidence", 0.5)),
            "classify_notes":   result.get("notes", ""),
        })
    except Exception:
        state.update({"transaction_type": "Unknown", "confidence": 0.3, "classify_notes": "Parse error"})
    return state

def lookup_rate(state):
    """Node 2 — dict lookup, zero LLM calls."""
    info = RATES.get(state["transaction_type"], {"rate": None, "split": "Unknown", "section": "—"})
    state.update({"rate": info["rate"], "rate_split": info["split"], "section": info["section"]})
    return state

def calculate(state):
    """Node 3 — pure Python math, zero LLM calls."""
    if state["amount"] and state["rate"] is not None:
        tax   = round(state["amount"] * state["rate"] / 100, 2)
        total = round(state["amount"] + tax, 2)
        state.update({"tax_amount": tax, "total_amount": total})
    else:
        state.update({"tax_amount": None, "total_amount": None})
    return state

def route(state, threshold):
    """Node 4 — if/else routing, no LLM."""
    reasons = []
    if state["confidence"] < threshold:
        reasons.append(f"low confidence ({state['confidence']:.2f})")
    if state["transaction_type"] == "Unknown":
        reasons.append("transaction type unresolved")
    if not state["amount"]:
        reasons.append("invoice amount missing")
    if state["rate"] is None:
        reasons.append("GST rate not found")

    state["route"]          = "human_review" if reasons else "db_insert"
    state["review_reasons"] = reasons
    return state

def run_pipeline(invoice_text, amount, threshold):
    """Wire all 4 nodes together."""
    state = {"invoice_text": invoice_text, "amount": amount}
    state = classify(state)
    state = lookup_rate(state)
    state = calculate(state)
    state = route(state, threshold)
    return state

# ── UI ────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Step 1 — Invoice Input")

    sample_choice = st.selectbox("Choose a sample invoice:", list(SAMPLES.keys()))
    sample_text, sample_amount = SAMPLES[sample_choice]

    invoice_text = st.text_area("Invoice text:", value=sample_text, height=160)
    amount       = st.number_input("Invoice amount (₹):", min_value=0, value=sample_amount, step=10000)

    # Activity 2: change this slider and observe how routing changes
    threshold = st.slider("Confidence threshold for DB insert:", 0.50, 0.95, 0.78, 0.01,
                          help="Invoices below this confidence score go to human review")

    run = st.button("▶  Run Pipeline", disabled=not invoice_text.strip(), use_container_width=True)

with right:
    st.subheader("Step 2 — Pipeline Results")

    if run and invoice_text.strip():
        with st.spinner("Running pipeline..."):
            state = run_pipeline(invoice_text.strip(), amount, threshold)
        st.session_state.state = state

    if "state" not in st.session_state:
        st.info("Choose an invoice on the left and click Run Pipeline.")
        st.stop()

    state = st.session_state.state

    # Node results
    node_colors = {"1": "#00897B", "2": "#4F46E5", "3": "#D97706", "4": "#059669"}

    def node_row(num, name, value, color):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;padding:0.5rem 0.8rem;'
            f'background:white;border-radius:8px;border-left:4px solid {color};margin:4px 0;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.06)">'
            f'<span style="background:{color};color:white;border-radius:50%;width:24px;height:24px;'
            f'display:flex;align-items:center;justify-content:center;font-weight:bold;'
            f'font-size:0.8rem;flex-shrink:0">{num}</span>'
            f'<span style="font-weight:600;color:#334E68;min-width:90px">{name}</span>'
            f'<span style="font-family:monospace;font-size:0.9rem;color:#0F1F3D">{value}</span>'
            f'</div>', unsafe_allow_html=True)

    node_row("1", "Classify", f"{state.get('transaction_type','—')}  ·  conf: {state.get('confidence',0):.2f}", node_colors["1"])
    node_row("2", "Rate",     f"{state.get('rate','—')}%  ·  {state.get('rate_split','—')}  ·  {state.get('section','—')}", node_colors["2"])

    tax_str = f"₹{state['tax_amount']:,.0f}  ·  Total: ₹{state['total_amount']:,.0f}" if state.get("tax_amount") else "—"
    node_row("3", "Calculate", tax_str, node_colors["3"])

    route_icon = "✅  DB Insert" if state["route"] == "db_insert" else "⚠️  Human Review"
    node_row("4", "Route", route_icon, node_colors["4"])

    if state["route"] == "human_review" and state["review_reasons"]:
        st.caption("Review reasons: " + " · ".join(state["review_reasons"]))

    if state.get("classify_notes"):
        st.caption(f"Classifier note: {state['classify_notes']}")

    # Full state dict
    with st.expander("🗂️ Full state dict (your audit trail)"):
        display = {k: v for k, v in state.items() if k != "invoice_text"}
        st.json(display)
