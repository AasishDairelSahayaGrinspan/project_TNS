"""Landing-first luxury editorial - unsupervised customer segments."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
APP_DIR = Path(__file__).resolve().parent
CLUSTERED_CSV = APP_DIR / "customers_clustered.csv"

# Matplotlib colors are tuples / hex only. Never pass CSS rgba() strings to plots.
BASE = (0.0196, 0.0196, 0.0196)
PANEL = (0.055, 0.051, 0.075)
PURPLE = (0.62, 0.51, 0.96)
SAGE = (0.62, 0.69, 0.58)
CREAM = (0.992, 0.984, 0.968)
MUTED_T = (0.79, 0.78, 0.76)
GRID_T = (0.992, 0.984, 0.968, 0.10)
HAIR_T = (1.0, 1.0, 1.0, 0.10)
WHITE_T = (1.0, 1.0, 1.0)
CLUSTER_COLORS = (PURPLE, SAGE, CREAM)

PERSONA_BLURBS = {
    "Budget": "Measured spend, value-led choices. Nurture with bundles and staples.",
    "Standard": "Middle-lane loyalists. Respond to curation and gentle upsell.",
    "Premium": "High income, high glow. Editorial drops and concierge care.",
}

RUN_COMMANDS = """pip install -r requirements.txt
python dataset_unsupervised.py
python train_kmeans.py
uvicorn main_unsupervised:app --reload --port 8000
streamlit run app_unsupervised.py --server.port 8501"""

# min-h-100dvh | rounded-2rem | rounded-full px-6 py-3 | translate-y-4 entry
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,300;1,9..144,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
:root {
  --base: #050505;
  --cream: #FDFBF7;
  --body: #C9C7C2;
  --muted: #A39FA8;
  --purple: #9D8DF1;
  --sage: #9EAE9C;
  --hair: rgba(255,255,255,0.10);
  --hair-soft: rgba(255,255,255,0.07);
  --bez: cubic-bezier(0.32,0.72,0,1);
  --serif: "Fraunces", "PP Editorial New", Didot, "Bodoni MT", Georgia, serif;
  --sans: "Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
html, body { background: #050505; }
[data-testid="stAppViewContainer"] {
  background: #050505;
  color: #C9C7C2;
  min-height: 100dvh;
  font-family: var(--sans);
}
[data-testid="stHeader"] { background: rgba(5,5,5,0); }
/* radial mesh: subtle purple orb + muted sage orb on #050505 */
.orb { position: fixed; pointer-events: none; z-index: 0; border-radius: 999px; }
.orb-purple {
  top: -13rem; left: -9rem; width: 36rem; height: 36rem;
  background: radial-gradient(circle, rgba(157,141,241,0.20) 0%, rgba(157,141,241,0.07) 45%, rgba(5,5,5,0) 70%);
}
.orb-sage {
  top: 8rem; right: -11rem; width: 32rem; height: 32rem;
  background: radial-gradient(circle, rgba(158,174,156,0.16) 0%, rgba(158,174,156,0.06) 45%, rgba(5,5,5,0) 70%);
}
/* film-grain fixed pointer-events-none 0.03 only */
.grain {
  position: fixed;
  inset: 0;
  z-index: 60;
  pointer-events: none;
  opacity: 0.03;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E");
}
.block-container { max-width: 76rem; padding-top: 3rem; padding-bottom: 3rem; }
/* massive Variable Serif luxury headlines in cream #FDFBF7 */
.serif-display {
  font-family: var(--serif);
  font-weight: 400;
  font-variation-settings: "opsz" 144;
  letter-spacing: -0.03em;
  line-height: 0.98;
  color: #FDFBF7;
}
.serif-display em { font-style: italic; font-weight: 300; color: #FDFBF7; }
.hero-wrap { text-align: center; max-width: 56rem; margin: 2rem auto 0; animation: rise 900ms cubic-bezier(0.32,0.72,0,1) both; }
.hero-wrap .lede { margin-left: auto; margin-right: auto; max-width: 34rem; }
.lede { color: #C9C7C2; font-family: var(--sans); font-weight: 300; line-height: 1.7; font-size: 0.94rem; max-width: 30rem; }
.lede-center { margin-left: auto; margin-right: auto; text-align: center; }
.eyebrow {
  display: inline-flex; align-items: center; gap: 0.55rem;
  border: 1px solid var(--hair);
  border-radius: 999px;
  padding: 0.38rem 0.95rem;
  font-family: var(--sans);
  font-size: 0.64rem; font-weight: 700;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: #FDFBF7; background: rgba(255,255,255,0.04);
}
.eyebrow-dot { width: 0.42rem; height: 0.42rem; border-radius: 999px; background: #9D8DF1; display: inline-block; }
.kicker { font-family: var(--sans); font-size: 0.66rem; letter-spacing: 0.24em; text-transform: uppercase; color: #A39FA8; font-weight: 700; }
.hairline { height: 1px; background: rgba(255,255,255,0.10); border: 0; margin: 1rem 0; }
/* double-bezel cards: outer shell rounded-2rem hairline white/10 + inner core */
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 2rem !important;
  background: #0C0A12 !important;
  padding: 0.375rem !important;
  animation: rise 900ms cubic-bezier(0.32,0.72,0,1) both;
  transition: transform 700ms cubic-bezier(0.32,0.72,0,1), opacity 700ms cubic-bezier(0.32,0.72,0,1) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 1.6rem !important;
  background: #121016 !important;
  padding: 1.4rem 1.5rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { transform: translateY(-3px) !important; opacity: 0.99 !important; }
/* island pill CTA rounded-full px-6 py-3 with nested circular arrow */
.stButton > button {
  border-radius: 999px !important;
  padding: 0.75rem 0.55rem 0.75rem 1.5rem !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: #9D8DF1 !important;
  color: #050505 !important;
  font-family: var(--sans) !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  font-size: 0.74rem !important;
  transition: transform 600ms cubic-bezier(0.32,0.72,0,1), opacity 600ms cubic-bezier(0.32,0.72,0,1) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; opacity: 0.94 !important; }
.stButton > button:active { transform: scale(0.98) !important; }
.stButton > button::after {
  content: "\\2192";
  display: inline-flex; align-items: center; justify-content: center;
  width: 2.1rem; height: 2.1rem;
  margin-left: 1.5rem;
  border-radius: 999px;
  background: #050505; color: #FDFBF7;
  font-size: 1rem; line-height: 1;
  transition: transform 600ms cubic-bezier(0.32,0.72,0,1), opacity 600ms cubic-bezier(0.32,0.72,0,1) !important;
}
.stButton > button:hover::after { transform: translateX(4px) !important; opacity: 0.95 !important; }
/* anchor island pill matching button */
.hero-cta {
  display: inline-flex; align-items: center;
  border-radius: 999px;
  padding: 0.75rem 0.55rem 0.75rem 1.5rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: #9D8DF1;
  color: #050505 !important;
  font-family: var(--sans);
  font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  font-size: 0.74rem; text-decoration: none !important;
  transition: transform 600ms cubic-bezier(0.32,0.72,0,1), opacity 600ms cubic-bezier(0.32,0.72,0,1);
}
.hero-cta:hover { transform: translateY(-2px); opacity: 0.94; }
.hero-cta-ghost {
  display: inline-flex; align-items: center;
  border-radius: 999px;
  padding: 0.75rem 0.55rem 0.75rem 1.5rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  color: #FDFBF7 !important;
  font-family: var(--sans);
  font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  font-size: 0.74rem; text-decoration: none !important;
  transition: transform 600ms cubic-bezier(0.32,0.72,0,1), opacity 600ms cubic-bezier(0.32,0.72,0,1);
}
.hero-cta-ghost:hover { transform: translateY(-2px); opacity: 0.94; }
.hero-arrow {
  display: inline-flex; align-items: center; justify-content: center;
  width: 2.1rem; height: 2.1rem; margin-left: 1.5rem;
  border-radius: 999px; background: #050505; color: #FDFBF7;
  font-size: 1rem; line-height: 1;
  transition: transform 600ms cubic-bezier(0.32,0.72,0,1), opacity 600ms cubic-bezier(0.32,0.72,0,1);
}
.hero-cta:hover .hero-arrow, .hero-cta-ghost:hover .hero-arrow { transform: translateX(4px); opacity: 0.95; }
.hero-cta-row { display: flex; gap: 0.9rem; justify-content: center; flex-wrap: wrap; margin-top: 1.6rem; }
.stSlider label, .stSlider [data-testid="stWidgetLabel"] {
  color: #FDFBF7 !important;
  font-family: var(--sans) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
}
.stSlider [data-testid="stTickBar"] { display: none; }
.persona-numeral { font-family: var(--serif); font-size: clamp(2.6rem,4vw,3.8rem); line-height: 1; color: #FDFBF7; letter-spacing: -0.02em; }
.persona-name { font-family: var(--serif); font-style: italic; font-size: 1.5rem; color: #FDFBF7; }
.persona-blurb { color: #C9C7C2; font-family: var(--sans); font-weight: 300; line-height: 1.7; font-size: 0.9rem; }
.step-numeral { font-family: var(--serif); font-size: 2.2rem; line-height: 1; color: #FDFBF7; letter-spacing: -0.02em; }
.glass-tag { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); border-radius: 999px; padding: 0.3rem 0.8rem; font-family: var(--sans); font-size: 0.72rem; letter-spacing: 0.08em; color: #FDFBF7; font-weight: 700; }
/* reveals: transform/opacity only with translate-y-4 entry */
@keyframes rise {
  from { opacity: 0; transform: translateY(1rem); }
  to { opacity: 1; transform: translateY(0); }
}
/* mobile single-column w-full px-4 */
@media (max-width: 768px) {
  .block-container { width: 100%; max-width: 100%; padding-left: 1rem; padding-right: 1rem; padding-top: 2rem; padding-bottom: 2rem; }
  [data-testid="stHorizontalBlock"] { flex-direction: column; }
  [data-testid="stHorizontalBlock"] > div { width: 100% !important; }
  .serif-display { letter-spacing: -0.02em; }
  .hero-cta-row { flex-direction: column; align-items: stretch; }
}
"""


def call_predict(income, score):
    resp = requests.post(
        f"{BACKEND_URL}/predict",
        json={"annual_income": float(income), "spending_score": float(score)},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"cluster": int(data["cluster"]), "persona": str(data.get("persona", "Unknown"))}


def check_health():
    resp = requests.get(f"{BACKEND_URL}/health", timeout=8)
    resp.raise_for_status()
    return resp.json()


def load_clustered():
    try:
        return pd.read_csv(CLUSTERED_CSV)
    except Exception:
        st.error("Could not load the customer map. Please check the dataset file and reload.")
        return None


def scatter_figure(df, income, score):
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    fig.patch.set_facecolor(BASE)
    ax.set_facecolor(BASE)
    clusters = sorted(df["cluster"].unique().tolist())
    for c in clusters:
        group = df[df["cluster"] == c]
        ax.scatter(
            group["annual_income_k"],
            group["spending_score"],
            alpha=0.82,
            s=34,
            color=CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)],
            label="Cluster %d" % int(c),
            zorder=3,
        )
    ax.scatter(
        [income],
        [score],
        s=300,
        marker="*",
        color=CREAM,
        linewidths=1.2,
        zorder=5,
    )
    ax.set_xlabel("Annual income (k$)", color=MUTED_T, fontsize=9)
    ax.set_ylabel("Spending score", color=MUTED_T, fontsize=9)
    ax.set_title("Customer segments", color=CREAM, fontsize=12, pad=12)
    ax.tick_params(colors=CREAM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(HAIR_T)
    ax.grid(True, color=GRID_T, linewidth=0.6, alpha=0.6)
    leg = ax.legend(frameon=True, fontsize=8, loc="best")
    leg.get_frame().set_facecolor(PANEL)
    leg.get_frame().set_edgecolor(HAIR_T)
    for text in leg.get_texts():
        text.set_color(CREAM)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Atelier Segments", layout="wide")
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)
st.markdown('<div class="grain"></div>', unsafe_allow_html=True)
st.markdown('<div class="orb orb-purple"></div><div class="orb orb-sage"></div>', unsafe_allow_html=True)

# Landing hero: first viewport.
st.markdown(
    '<div class="hero-wrap" title="CUSTOMER PERSONA SEGMENTER (UNSUPERVISED MACHINE LEARNING)">'
    '<span class="eyebrow"><span class="eyebrow-dot"></span>CUSTOMER PERSONA SEGMENTER</span>'
    '<h1 class="serif-display" style="font-size:clamp(3.4rem,7vw,5.8rem);margin:0.9rem 0 0.8rem;"><em>UNSUPERVISED MACHINE LEARNING</em></h1>'
    '<p class="lede lede-center">Tune income and spend to reveal Budget, Standard, Premium personas live.</p>'
    "</div>",
    unsafe_allow_html=True,
)

# Features grid: 3 cards.
st.markdown('<span class="kicker">Features</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-display" style="font-size:clamp(2rem,4vw,3rem);margin:0.6rem 0;">Three clusters trained on income and spending score <em>live.</em></h2>',
    unsafe_allow_html=True,
)
feat_a, feat_b, feat_c = st.columns(3, gap="medium")
with feat_a:
    with st.container(border=True):
        st.markdown('<span class="kicker">Personas</span>', unsafe_allow_html=True)
        st.markdown('<div class="persona-name">3-cluster K-Means</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="persona-blurb">Budget, Standard, and Premium personas distilled from income and spending score.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="glass-tag">Budget &middot; Standard &middot; Premium</span>', unsafe_allow_html=True)
with feat_b:
    with st.container(border=True):
        st.markdown('<span class="kicker">Prediction</span>', unsafe_allow_html=True)
        st.markdown('<div class="persona-name">Live FastAPI prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="persona-blurb">POST /predict returns a cluster and persona verdict in milliseconds.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="glass-tag">FastAPI &middot; live</span>', unsafe_allow_html=True)
with feat_c:
    with st.container(border=True):
        st.markdown('<span class="kicker">Map</span>', unsafe_allow_html=True)
        st.markdown('<div class="persona-name">Living scatter map</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="persona-blurb">Purple, sage, and cream clusters with a star-is-you marker for every newcomer.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="glass-tag">star-is-you</span>', unsafe_allow_html=True)

# How it works: 3 steps.
st.markdown('<span class="kicker">How it works</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-display" style="font-size:clamp(2rem,4vw,3rem);margin:0.6rem 0;">Move sliders, ask model, land on map <em>live.</em></h2>',
    unsafe_allow_html=True,
)
step_a, step_b, step_c = st.columns(3, gap="medium")
with step_a:
    with st.container(border=True):
        st.markdown('<span class="kicker">Step</span>', unsafe_allow_html=True)
        st.markdown('<div class="step-numeral">01</div>', unsafe_allow_html=True)
        st.markdown('<div class="persona-name">Tune sliders</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="persona-blurb">Set annual income and spending score with the editorial dials.</p>',
            unsafe_allow_html=True,
        )
with step_b:
    with st.container(border=True):
        st.markdown('<span class="kicker">Step</span>', unsafe_allow_html=True)
        st.markdown('<div class="step-numeral">02</div>', unsafe_allow_html=True)
        st.markdown('<div class="persona-name">Ask model POST /predict</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="persona-blurb">Send income and spend to the FastAPI model for a persona verdict.</p>',
            unsafe_allow_html=True,
        )
with step_c:
    with st.container(border=True):
        st.markdown('<span class="kicker">Step</span>', unsafe_allow_html=True)
        st.markdown('<div class="step-numeral">03</div>', unsafe_allow_html=True)
        st.markdown('<div class="persona-name">Land on map</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="persona-blurb">Your star lands among Budget, Standard, and Premium clusters.</p>',
            unsafe_allow_html=True,
        )

# How to run.
st.markdown('<span class="kicker">How to run</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-display" style="font-size:clamp(2rem,4vw,3rem);margin:0.6rem 0;">From data to <em>live segments.</em></h2>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    st.markdown('<span class="kicker">Terminal</span>', unsafe_allow_html=True)
    st.markdown("<hr class='hairline' />", unsafe_allow_html=True)
    st.code(RUN_COMMANDS, language="bash")
    st.markdown(
        '<p class="persona-blurb">Backend serves port 8000; this Streamlit atelier serves port 8501.</p>',
        unsafe_allow_html=True,
    )

# Dashboard section below landing.
st.markdown('<div id="demo"></div>', unsafe_allow_html=True)
st.markdown('<span class="kicker">Live demo</span>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="serif-display" style="font-size:clamp(2rem,4vw,3rem);margin:0.6rem 0;">Press Predict to place newcomer among 600 customers <em>live.</em></h2>',
    unsafe_allow_html=True,
)
left, right = st.columns([1.02, 0.98], gap="medium")

with left:
    with st.container(border=True):
        st.markdown('<span class="kicker">01 &middot; Controls</span>', unsafe_allow_html=True)
        st.markdown("<hr class='hairline' />", unsafe_allow_html=True)
        income = st.slider("annual_income_k", min_value=15, max_value=150, value=60)
        score = st.slider("spending_score", min_value=1, max_value=100, value=50)
        if st.button("Find my segment"):
            try:
                st.session_state["result"] = call_predict(income, score)
            except requests.exceptions.RequestException:
                st.warning("The model service is down. Start the API on port 8000, then retry. Your dials are kept.")
            except (ValueError, KeyError):
                st.error("Unexpected model response. Please retry in a moment.")

with right:
    with st.container(border=True):
        st.markdown('<span class="kicker">02 &middot; Persona</span>', unsafe_allow_html=True)
        st.markdown("<hr class='hairline' />", unsafe_allow_html=True)
        result = st.session_state.get("result")
        if result is not None:
            cluster = int(result.get("cluster", 0))
            persona = result.get("persona", "Unknown")
            blurb = PERSONA_BLURBS.get(persona, "A distinct pocket of taste and habit.")
            st.markdown('<div class="persona-numeral">%02d</div>' % cluster, unsafe_allow_html=True)
            st.markdown('<div class="persona-name">%s</div>' % persona, unsafe_allow_html=True)
            st.markdown('<p class="persona-blurb">%s</p>' % blurb, unsafe_allow_html=True)
            st.markdown('<span class="glass-tag">Cluster %d &middot; %s</span>' % (cluster, persona), unsafe_allow_html=True)
        else:
            st.markdown('<div class="persona-numeral">Newcomer</div>', unsafe_allow_html=True)
            st.markdown('<div class="persona-name">Ready when you are</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="persona-blurb">Set the dials on the left, then press the pill button. '
                "Your cluster and persona verdict will land here.</p>",
                unsafe_allow_html=True,
            )
            st.markdown('<span class="glass-tag">Awaiting first prediction</span>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<span class="kicker">03 &middot; Scatter</span>', unsafe_allow_html=True)
    st.markdown(
        '<p class="persona-blurb">x income &middot; y score &middot; purple 0 &middot; sage 1 &middot; cream 2 &middot; star is you.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='hairline' />", unsafe_allow_html=True)
    df = load_clustered()
    if df is not None:
        st.pyplot(scatter_figure(df, float(income), float(score)), use_container_width=True)
