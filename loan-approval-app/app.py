"""Landing-first luxury editorial - loan approval predictor (port 8502)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001").rstrip("/")

# Matplotlib palette uses tuples only.
BASE = (0.0196, 0.0196, 0.0196)
PANEL = (0.055, 0.051, 0.075)
PURPLE = (0.616, 0.553, 0.945)
SAGE = (0.62, 0.682, 0.61)
CREAM = (0.992, 0.984, 0.968)
MUTED_T = (0.79, 0.78, 0.76)
GRID_T = (0.992, 0.984, 0.968, 0.10)
HAIR_T = (1.0, 1.0, 1.0, 0.10)

RUN_COMMANDS = """pip install -r requirements.txt
python dataset.py
python train_model.py
uvicorn main:app --reload --port 8001
streamlit run app.py --server.port 8502"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,300;1,9..144,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
:root {
  --base: #050505;
  --cream: #FDFBF7;
  --body: #C9C7C2;
  --muted: #A39FA8;
  --purple: #9D8DF1;
  --sage: #9EAE9C;
  --serif: "Fraunces", Didot, "Bodoni MT", Georgia, serif;
  --sans: "Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
html, body { background: #050505; }
[data-testid="stAppViewContainer"] {
  background: #050505;
  color: #C9C7C2;
  min-height: 100dvh;
  font-family: var(--sans);
}
[data-testid="stHeader"] { background: #05050500; }
.orb { position: fixed; pointer-events: none; z-index: 0; border-radius: 999px; }
.orb-a {
  bottom: -14rem; right: -8rem; width: 38rem; height: 38rem;
  background: radial-gradient(circle, #9D8DF133 0%, #9D8DF112 45%, #05050500 70%);
}
.orb-b {
  top: -10rem; left: -10rem; width: 30rem; height: 30rem;
  background: radial-gradient(circle, #9EAE9C29 0%, #9EAE9C0F 45%, #05050500 70%);
}
.grain {
  position: fixed; inset: 0; z-index: 60; pointer-events: none; opacity: 0.03;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E");
}
.block-container { max-width: 78rem; padding-top: 2.6rem; padding-bottom: 3rem; }
.serif-hero {
  font-family: var(--serif); font-weight: 400; letter-spacing: -0.03em;
  line-height: 0.98; color: #FDFBF7;
}
.serif-hero em { font-style: italic; font-weight: 300; }
.split-hero { display: flex; gap: 2rem; align-items: stretch; }
.lede-left { color: #C9C7C2; font-family: var(--sans); font-weight: 300; line-height: 1.7; font-size: 0.95rem; max-width: 32rem; }
.eyebrow-left {
  display: inline-flex; align-items: center; gap: 0.55rem;
  border: 1px solid #FFFFFF1A; border-radius: 999px; padding: 0.38rem 0.95rem;
  font-family: var(--sans); font-size: 0.64rem; font-weight: 700;
  letter-spacing: 0.22em; text-transform: uppercase; color: #FDFBF7; background: #FFFFFF0A;
}
.eyebrow-sq { width: 0.42rem; height: 0.42rem; background: #9EAE9C; display: inline-block; }
.kicker-row { font-family: var(--sans); font-size: 0.66rem; letter-spacing: 0.24em; text-transform: uppercase; color: #A39FA8; font-weight: 700; }
.rule { height: 1px; background: #FFFFFF1A; border: 0; margin: 1rem 0; }
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid #FFFFFF1A !important;
  border-radius: 1.4rem !important;
  background: #0D0B13 !important;
  padding: 0.375rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border: 1px solid #FFFFFF12 !important;
  border-radius: 1.05rem !important;
  background: #131118 !important;
  padding: 1.3rem 1.4rem !important;
}
.field-numeral { font-family: var(--serif); font-size: 2rem; line-height: 1; color: #FDFBF7; }
.field-name { font-family: var(--serif); font-style: italic; font-size: 1.35rem; color: #FDFBF7; margin-top: 0.3rem; }
.field-blurb { color: #C9C7C2; font-family: var(--sans); font-weight: 300; line-height: 1.65; font-size: 0.88rem; }
.lozenge {
  display: inline-block; background: #FFFFFF0A; border: 1px solid #FFFFFF1A;
  border-radius: 0.55rem; padding: 0.3rem 0.7rem; font-family: var(--sans);
  font-size: 0.72rem; letter-spacing: 0.06em; color: #FDFBF7; font-weight: 600;
}
.step-cell { border-left: 1px solid #FFFFFF1A; padding-left: 1.1rem; }
.step-cell-first { border-left: 0; padding-left: 0; }
.verdict-approved { font-family: var(--serif); font-size: clamp(2.4rem,4vw,3.6rem); color: #9EAE9C; line-height: 1; letter-spacing: -0.02em; }
.verdict-declined { font-family: var(--serif); font-size: clamp(2.4rem,4vw,3.6rem); color: #9D8DF1; line-height: 1; letter-spacing: -0.02em; font-style: italic; }
.cta-row { display: flex; gap: 0.8rem; justify-content: flex-start; flex-wrap: wrap; margin-top: 1.5rem; }
.cta-solid {
  display: inline-flex; align-items: center; border-radius: 0.8rem;
  padding: 0.8rem 1.4rem; border: 1px solid #FFFFFF24; background: #9EAE9C;
  color: #050505 !important; font-family: var(--sans); font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.74rem; text-decoration: none !important;
}
.cta-line {
  display: inline-flex; align-items: center; border-radius: 0.8rem;
  padding: 0.8rem 1.4rem; border: 1px solid #FFFFFF24; background: #FFFFFF0A;
  color: #FDFBF7 !important; font-family: var(--sans); font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.74rem; text-decoration: none !important;
}
.stButton > button {
  border-radius: 0.8rem !important; padding: 0.85rem 1.4rem !important;
  border: 1px solid #FFFFFF24 !important; background: #9EAE9C !important;
  color: #050505 !important; font-family: var(--sans) !important; font-weight: 700 !important;
  letter-spacing: 0.1em !important; text-transform: uppercase !important; font-size: 0.78rem !important;
  width: 100%;
}
.stNumberInput label, .stSlider label, [data-testid="stWidgetLabel"] {
  color: #FDFBF7 !important; font-family: var(--sans) !important;
  font-size: 0.68rem !important; letter-spacing: 0.18em !important;
  text-transform: uppercase !important; font-weight: 700 !important;
}
@media (max-width: 768px) {
  .block-container { width: 100%; max-width: 100%; padding-left: 1rem; padding-right: 1rem; }
  [data-testid="stHorizontalBlock"] { flex-direction: column; }
  [data-testid="stHorizontalBlock"] > div { width: 100% !important; }
  .cta-row { flex-direction: column; align-items: stretch; }
  .step-cell { border-left: 0; border-top: 1px solid #FFFFFF1A; padding-left: 0; padding-top: 1rem; }
  .step-cell-first { border-top: 0; padding-top: 0; }
}
"""


def call_predict(income, credit_score, loan_amount, employment_years):
    resp = requests.post(
        f"{BACKEND_URL}/predict",
        json={
            "income": float(income),
            "credit_score": float(credit_score),
            "loan_amount": float(loan_amount),
            "employment_years": float(employment_years),
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"prediction": int(data["prediction"]), "probability": float(data["probability"])}


def confidence_figure(probability, approved):
    fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=140)
    fig.patch.set_facecolor(BASE)
    ax.set_facecolor(BASE)
    color = SAGE if approved else PURPLE
    ax.barh(["confidence"], [probability], height=0.42, color=color)
    ax.barh(["confidence"], [1.0 - probability], left=[probability], height=0.42, color=PANEL)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability", color=MUTED_T, fontsize=9)
    ax.set_title("Model confidence", color=CREAM, fontsize=11, pad=10)
    ax.tick_params(colors=CREAM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(HAIR_T)
    ax.grid(True, axis="x", color=GRID_T, linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)
st.markdown('<div class="grain"></div>', unsafe_allow_html=True)
st.markdown('<div class="orb orb-a"></div><div class="orb orb-b"></div>', unsafe_allow_html=True)

# HERO: split editorial, left-aligned. Distinct from centered customer hero.
hero_left, hero_right = st.columns([1.12, 0.88], gap="large")
with hero_left:
    st.markdown(
        '<span class="eyebrow-left"><span class="eyebrow-sq"></span>Loan Approval Predictor</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="serif-hero" style="font-size:clamp(3rem,6vw,5rem);margin:0.9rem 0 0.8rem;">'
        "Decide credit<br/><em>before the bank does.</em></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="lede-left">Enter income, credit score, loan size, and tenure. '
        "The RandomForest model scores approval odds in milliseconds.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cta-row">'
        '<a class="cta-solid" href="#demo">Check eligibility</a>'
        '<a class="cta-line" href="#how-to-run">How to run</a>'
        "</div>",
        unsafe_allow_html=True,
    )
with hero_right:
    with st.container(border=True):
        st.markdown('<span class="kicker-row">At a glance</span>', unsafe_allow_html=True)
        st.markdown("<hr class='rule' />", unsafe_allow_html=True)
        st.markdown('<div class="verdict-approved">APPROVED</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="field-blurb">Sample verdict for income 60000, score 720, loan 15000, tenure 5 years.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="lozenge">4 fields &middot; 1 verdict &middot; live</span>', unsafe_allow_html=True)

# FEATURES: four applicant fields, one card each.
st.markdown('<span class="kicker-row">Features</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-hero" style="font-size:clamp(1.9rem,3.6vw,2.8rem);margin:0.5rem 0;">'
    "Four signals shape <em>every verdict.</em></h2>",
    unsafe_allow_html=True,
)
f1, f2, f3, f4 = st.columns(4, gap="medium")
with f1:
    with st.container(border=True):
        st.markdown('<div class="field-numeral">01</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-name">Income</div>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">Annual earnings power. Higher income lifts approval odds.</p>', unsafe_allow_html=True)
        st.markdown('<span class="lozenge">income &middot; USD</span>', unsafe_allow_html=True)
with f2:
    with st.container(border=True):
        st.markdown('<div class="field-numeral">02</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-name">Credit score</div>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">Repayment history condensed to a single trusted number.</p>', unsafe_allow_html=True)
        st.markdown('<span class="lozenge">credit_score &middot; 300-850</span>', unsafe_allow_html=True)
with f3:
    with st.container(border=True):
        st.markdown('<div class="field-numeral">03</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-name">Loan amount</div>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">Requested principal. Larger asks face stricter scoring.</p>', unsafe_allow_html=True)
        st.markdown('<span class="lozenge">loan_amount &middot; USD</span>', unsafe_allow_html=True)
with f4:
    with st.container(border=True):
        st.markdown('<div class="field-numeral">04</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-name">Tenure</div>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">Years employed. Stability signals reliable repayment.</p>', unsafe_allow_html=True)
        st.markdown('<span class="lozenge">employment_years &middot; yrs</span>', unsafe_allow_html=True)

# HOW IT WORKS: single timeline card with three cells.
st.markdown('<span class="kicker-row">How it works</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-hero" style="font-size:clamp(1.9rem,3.6vw,2.8rem);margin:0.5rem 0;">'
    "From form to <em>verdict in seconds.</em></h2>",
    unsafe_allow_html=True,
)
with st.container(border=True):
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        st.markdown('<div class="step-cell-first">', unsafe_allow_html=True)
        st.markdown('<span class="kicker-row">Step 01</span>', unsafe_allow_html=True)
        st.markdown('<div class="field-name">Enter details</div>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">Type four numbers into the demo form below.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="step-cell">', unsafe_allow_html=True)
        st.markdown('<span class="kicker-row">Step 02</span>', unsafe_allow_html=True)
        st.markdown('<div class="field-name">Score risk</div>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">POST /predict scales inputs and runs RandomForest.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="step-cell">', unsafe_allow_html=True)
        st.markdown('<span class="kicker-row">Step 03</span>', unsafe_allow_html=True)
        st.markdown('<div class="field-name">Read verdict</div>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">APPROVED or DECLINED lands with a confidence bar.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# HOW TO RUN: two side-by-side cards.
st.markdown('<span class="kicker-row" id="how-to-run">How to run</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-hero" style="font-size:clamp(1.9rem,3.6vw,2.8rem);margin:0.5rem 0;">'
    "Data to <em>live verdict.</em></h2>",
    unsafe_allow_html=True,
)
run_left, run_right = st.columns(2, gap="medium")
with run_left:
    with st.container(border=True):
        st.markdown('<span class="kicker-row">Terminal</span>', unsafe_allow_html=True)
        st.markdown("<hr class='rule' />", unsafe_allow_html=True)
        st.code(RUN_COMMANDS, language="bash")
with run_right:
    with st.container(border=True):
        st.markdown('<span class="kicker-row">Contract</span>', unsafe_allow_html=True)
        st.markdown("<hr class='rule' />", unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">GET /health returns status ok on port 8001.</p>', unsafe_allow_html=True)
        st.markdown('<p class="field-blurb">POST /predict takes income, credit_score, loan_amount, employment_years.</p>', unsafe_allow_html=True)
        st.markdown('<span class="lozenge">API 8001 &middot; UI 8502</span>', unsafe_allow_html=True)

# DEMO: full-width form, then verdict + confidence side by side.
st.markdown('<div id="demo"></div>', unsafe_allow_html=True)
st.markdown('<span class="kicker-row">Live demo</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-hero" style="font-size:clamp(1.9rem,3.6vw,2.8rem);margin:0.5rem 0;">'
    "Try an applicant <em>right now.</em></h2>",
    unsafe_allow_html=True,
)
with st.container(border=True):
    st.markdown('<span class="kicker-row">Applicant form</span>', unsafe_allow_html=True)
    st.markdown("<hr class='rule' />", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        income = st.number_input("income", min_value=5000.0, max_value=500000.0, value=60000.0, step=1000.0)
    with c2:
        credit_score = st.number_input("credit_score", min_value=300.0, max_value=850.0, value=720.0, step=1.0)
    with c3:
        loan_amount = st.number_input("loan_amount", min_value=1000.0, max_value=200000.0, value=15000.0, step=500.0)
    with c4:
        employment_years = st.number_input("employment_years", min_value=0.0, max_value=40.0, value=5.0, step=1.0)
    b1, b2, b3 = st.columns([1, 1, 1], gap="medium")
    with b2:
        submitted = st.button("Check approval", use_container_width=True)
    if submitted:
        try:
            st.session_state["loan_result"] = call_predict(income, credit_score, loan_amount, employment_years)
        except requests.exceptions.RequestException:
            st.warning("The loan service is down. Start the API on port 8001, then retry. Your entries are kept.")
        except (ValueError, KeyError):
            st.error("Unexpected model response. Please retry in a moment.")

res_left, res_right = st.columns(2, gap="medium")
with res_left:
    with st.container(border=True):
        st.markdown('<span class="kicker-row">Verdict</span>', unsafe_allow_html=True)
        st.markdown("<hr class='rule' />", unsafe_allow_html=True)
        result = st.session_state.get("loan_result")
        if result is not None:
            approved = int(result.get("prediction", 0)) == 1
            prob = float(result.get("probability", 0.0))
            if approved:
                st.markdown('<div class="verdict-approved">APPROVED</div>', unsafe_allow_html=True)
                st.markdown('<p class="field-blurb">This applicant clears the bar. Confidence shown on the right.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-declined">DECLINED</div>', unsafe_allow_html=True)
                st.markdown('<p class="field-blurb">This applicant falls short. Try raising income or score.</p>', unsafe_allow_html=True)
            st.markdown('<span class="lozenge">probability %.3f</span>' % prob, unsafe_allow_html=True)
        else:
            st.markdown('<div class="field-numeral">Awaiting check</div>', unsafe_allow_html=True)
            st.markdown('<div class="field-name">Press Check approval</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="field-blurb">Fill the four fields above, then press the sage button. '
                "Your APPROVED or DECLINED verdict will land here.</p>",
                unsafe_allow_html=True,
            )
            st.markdown('<span class="lozenge">no verdict yet</span>', unsafe_allow_html=True)
with res_right:
    with st.container(border=True):
        st.markdown('<span class="kicker-row">Confidence</span>', unsafe_allow_html=True)
        st.markdown("<hr class='rule' />", unsafe_allow_html=True)
        result2 = st.session_state.get("loan_result")
        if result2 is not None:
            approved2 = int(result2.get("prediction", 0)) == 1
            prob2 = float(result2.get("probability", 0.0))
            st.pyplot(confidence_figure(prob2, approved2), use_container_width=True)
            st.markdown('<span class="lozenge">sage approved &middot; purple declined</span>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="field-numeral">No score yet</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="field-blurb">Run one check to paint the confidence bar. '
                "Sage means APPROVED, purple means DECLINED.</p>",
                unsafe_allow_html=True,
            )
            st.markdown('<span class="lozenge">run a check first</span>', unsafe_allow_html=True)
