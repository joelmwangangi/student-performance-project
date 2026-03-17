import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
    text-align: center;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    margin-top: 0.7rem;
    font-weight: 300;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(167,139,250,0.25);
}

/* Prediction card */
.pred-card {
    border-radius: 18px;
    padding: 2.2rem;
    text-align: center;
    margin: 1rem 0;
    border: 2px solid;
    backdrop-filter: blur(10px);
}
.pred-card.risk    { background: rgba(239,68,68,0.12);  border-color: rgba(239,68,68,0.5);  }
.pred-card.average { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.5); }
.pred-card.high    { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.5); }

.pred-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.3rem 0;
}
.pred-label.risk    { color: #f87171; }
.pred-label.average { color: #fbbf24; }
.pred-label.high    { color: #34d399; }
.pred-emoji { font-size: 3.5rem; }
.pred-desc {
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* Probability bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.5rem 0;
}
.prob-label { color: rgba(255,255,255,0.75); font-size: 0.88rem; min-width: 120px; }
.prob-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }
.prob-val { color: white; font-size: 0.88rem; font-weight: 600; min-width: 42px; text-align: right; }

/* Info metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-lbl { color: rgba(255,255,255,0.5); font-size: 0.78rem; margin-top: 0.2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] .stMarkdown p { color: rgba(255,255,255,0.7) !important; }

/* Inputs */
.stSelectbox > div > div, .stSlider > div {
    background: rgba(255,255,255,0.05) !important;
}
label { color: rgba(255,255,255,0.8) !important; font-size: 0.88rem !important; }
.stSlider [data-testid="stTickBar"] { color: rgba(255,255,255,0.4) !important; }

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* Alert boxes */
.tip-box {
    background: rgba(99,102,241,0.1);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0;
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler from disk."""
    model_path  = "model/student_ann_model.pkl"
    scaler_path = "model/scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ── Feature configuration ─────────────────────────────────────────────────────
FEATURE_COLS = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
    'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
    'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
    'G1', 'G2'
]

# Label encoding maps — same order used during training
ENCODE_MAP = {
    'school':     {'GP': 0, 'MS': 1},
    'sex':        {'F': 0, 'M': 1},
    'address':    {'R': 0, 'U': 1},
    'famsize':    {'GT3': 0, 'LE3': 1},
    'Pstatus':    {'A': 0, 'T': 1},
    'Mjob':       {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'Fjob':       {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'reason':     {'course': 0, 'home': 1, 'other': 2, 'reputation': 3},
    'guardian':   {'father': 0, 'mother': 1, 'other': 2},
    'schoolsup':  {'no': 0, 'yes': 1},
    'famsup':     {'no': 0, 'yes': 1},
    'paid':       {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'nursery':    {'no': 0, 'yes': 1},
    'higher':     {'no': 0, 'yes': 1},
    'internet':   {'no': 0, 'yes': 1},
    'romantic':   {'no': 0, 'yes': 1},
}

CLASS_INFO = {
    0: {"label": "At Risk",        "emoji": "🔴", "color": "risk",    "hex": "#f87171",
        "desc": "Student is at risk of failing and may need immediate academic intervention."},
    1: {"label": "Average",        "emoji": "🟡", "color": "average", "hex": "#fbbf24",
        "desc": "Student is performing at an average level with room for improvement."},
    2: {"label": "High Performer", "emoji": "🟢", "color": "high",    "hex": "#34d399",
        "desc": "Student is performing excellently and on track for academic success."},
}


def encode_input(raw: dict) -> np.ndarray:
    """Encode raw form inputs into a numeric array for the model."""
    encoded = {}
    for feat in FEATURE_COLS:
        val = raw[feat]
        if feat in ENCODE_MAP:
            encoded[feat] = ENCODE_MAP[feat][val]
        else:
            encoded[feat] = int(val)
    return np.array([[encoded[f] for f in FEATURE_COLS]], dtype=np.float32)


def predict(model, scaler, raw_input: dict):
    """Run prediction and return class index + probabilities."""
    arr     = encode_input(raw_input)
    arr_s   = scaler.transform(arr)
    probs   = model.predict_proba(arr_s)[0]
    pred    = int(np.argmax(probs))
    return pred, probs


# ── Sidebar — student info form ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem'>
            <span style='font-size:2.5rem'>🎓</span>
            <div style='font-family:Syne,sans-serif; font-weight:800;
                        font-size:1.3rem; color:white; margin-top:0.3rem'>
                EduPredict AI
            </div>
            <div style='color:rgba(255,255,255,0.45); font-size:0.78rem; margin-top:0.2rem'>
                Student Risk Classifier
            </div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📋 Student Profile</div>', unsafe_allow_html=True)

        # ── Personal Info ──
        with st.expander("👤 Personal Information", expanded=True):
            school   = st.selectbox("School",          ["GP", "MS"], help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
            sex      = st.selectbox("Sex",             ["F", "M"])
            age      = st.slider("Age",                15, 22, 17)
            address  = st.selectbox("Address Type",   ["U", "R"], help="U = Urban, R = Rural")
            famsize  = st.selectbox("Family Size",    ["GT3", "LE3"], help="GT3 = >3 members, LE3 = ≤3 members")
            Pstatus  = st.selectbox("Parent Status",  ["T", "A"], help="T = Together, A = Apart")
            guardian = st.selectbox("Guardian",       ["mother", "father", "other"])

        # ── Family & Education ──
        with st.expander("🏠 Family & Education", expanded=True):
            Medu  = st.slider("Mother's Education",  0, 4, 2,
                              help="0=none, 1=primary, 2=5th–9th, 3=secondary, 4=higher")
            Fedu  = st.slider("Father's Education",  0, 4, 2,
                              help="0=none, 1=primary, 2=5th–9th, 3=secondary, 4=higher")
            Mjob  = st.selectbox("Mother's Job",     ["at_home", "health", "other", "services", "teacher"])
            Fjob  = st.selectbox("Father's Job",     ["at_home", "health", "other", "services", "teacher"])
            famsup  = st.selectbox("Family Support",   ["yes", "no"])
            famrel  = st.slider("Family Relationship Quality", 1, 5, 4,
                                help="1=very bad … 5=excellent")

        # ── School & Academic ──
        with st.expander("📚 School & Academic", expanded=True):
            reason    = st.selectbox("Reason to Choose School", ["course", "home", "reputation", "other"])
            traveltime = st.slider("Travel Time to School", 1, 4, 1,
                                   help="1=<15 min, 2=15–30 min, 3=30–60 min, 4=>60 min")
            studytime  = st.slider("Weekly Study Time",    1, 4, 2,
                                   help="1=<2h, 2=2–5h, 3=5–10h, 4=>10h")
            failures   = st.slider("Past Class Failures",  0, 3, 0)
            schoolsup  = st.selectbox("School Extra Support", ["yes", "no"])
            paid       = st.selectbox("Extra Paid Classes",   ["yes", "no"])
            absences   = st.slider("Number of Absences",  0, 75, 4)
            higher     = st.selectbox("Wants Higher Education", ["yes", "no"])
            nursery    = st.selectbox("Attended Nursery",       ["yes", "no"])
            activities = st.selectbox("Extra-Curricular Activities", ["yes", "no"])
            internet   = st.selectbox("Internet Access at Home",     ["yes", "no"])

        # ── Social & Lifestyle ──
        with st.expander("🎮 Social & Lifestyle", expanded=False):
            romantic = st.selectbox("In a Romantic Relationship", ["no", "yes"])
            freetime = st.slider("Free Time After School", 1, 5, 3, help="1=very low … 5=very high")
            goout    = st.slider("Going Out with Friends", 1, 5, 3, help="1=very low … 5=very high")
            Dalc     = st.slider("Workday Alcohol Consumption", 1, 5, 1, help="1=very low … 5=very high")
            Walc     = st.slider("Weekend Alcohol Consumption", 1, 5, 2, help="1=very low … 5=very high")
            health   = st.slider("Current Health Status", 1, 5, 3, help="1=very bad … 5=very good")

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 Grades (Mid-Term)</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            G1 = st.number_input("G1 — 1st Period", 0, 20, 10)
        with col2:
            G2 = st.number_input("G2 — 2nd Period", 0, 20, 10)

        st.markdown('<br>', unsafe_allow_html=True)
        predict_btn = st.button("🔮  PREDICT PERFORMANCE", use_container_width=True)

    raw = dict(
        school=school, sex=sex, age=age, address=address, famsize=famsize,
        Pstatus=Pstatus, Medu=Medu, Fedu=Fedu, Mjob=Mjob, Fjob=Fjob,
        reason=reason, guardian=guardian, traveltime=traveltime,
        studytime=studytime, failures=failures, schoolsup=schoolsup,
        famsup=famsup, paid=paid, activities=activities, nursery=nursery,
        higher=higher, internet=internet, romantic=romantic, famrel=famrel,
        freetime=freetime, goout=goout, Dalc=Dalc, Walc=Walc,
        health=health, absences=absences, G1=G1, G2=G2,
    )
    return raw, predict_btn


# ── Radar chart helper ────────────────────────────────────────────────────────
def radar_chart(raw):
    categories = ['Study Time', 'Family\nRelation', 'Health', 'Free Time',
                  'G1 Score', 'G2 Score']
    values = [
        raw['studytime'] / 4,
        raw['famrel']    / 5,
        raw['health']    / 5,
        raw['freetime']  / 5,
        raw['G1']        / 20,
        raw['G2']        / 20,
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#a78bfa')
    ax.fill(angles, values_plot, alpha=0.25, color='#6366f1')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8, color='white')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], color='gray', size=7)
    ax.yaxis.grid(True, color='rgba(255,255,255,0.1)', linestyle='--', linewidth=0.5)
    ax.xaxis.grid(True, color='rgba(255,255,255,0.15)', linewidth=0.5)
    ax.spines['polar'].set_color('rgba(255,255,255,0.1)')
    ax.set_ylim(0, 1)

    plt.tight_layout(pad=0.5)
    return fig


# ── Risk factors summary ──────────────────────────────────────────────────────
def get_risk_factors(raw):
    positives, negatives = [], []
    if raw['studytime'] >= 3:  positives.append("✅ Studies 5+ hours weekly")
    if raw['studytime'] <= 1:  negatives.append("⚠️ Studies less than 2 hours weekly")
    if raw['failures'] == 0:   positives.append("✅ No past failures")
    if raw['failures'] >= 2:   negatives.append("⚠️ Multiple past failures")
    if raw['G1'] >= 12:        positives.append(f"✅ Strong 1st period grade ({raw['G1']}/20)")
    if raw['G1'] < 8:          negatives.append(f"⚠️ Low 1st period grade ({raw['G1']}/20)")
    if raw['G2'] >= 12:        positives.append(f"✅ Strong 2nd period grade ({raw['G2']}/20)")
    if raw['G2'] < 8:          negatives.append(f"⚠️ Low 2nd period grade ({raw['G2']}/20)")
    if raw['higher'] == 'yes': positives.append("✅ Motivated for higher education")
    if raw['internet'] == 'yes': positives.append("✅ Has internet access")
    if raw['absences'] > 15:   negatives.append(f"⚠️ High absences ({raw['absences']} days)")
    if raw['absences'] <= 3:   positives.append("✅ Very few absences")
    if raw['Dalc'] >= 4:       negatives.append("⚠️ High workday alcohol consumption")
    if raw['health'] >= 4:     positives.append("✅ Good health status")
    if raw['health'] <= 2:     negatives.append("⚠️ Poor health status")
    if raw['Medu'] >= 3:       positives.append("✅ Mother has secondary/higher education")
    if raw['famrel'] >= 4:     positives.append("✅ Good family relationships")
    if raw['famrel'] <= 2:     negatives.append("⚠️ Poor family relationships")
    return positives, negatives


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
    model, scaler = load_model_and_scaler()
    raw, predict_btn = render_sidebar()

    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">EduPredict AI</div>
        <div class="hero-sub">
            Deep Learning–Powered Student Performance Prediction &nbsp;|&nbsp;
            ANN Classifier &nbsp;|&nbsp; 3-Class Risk Assessment
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model not found warning
    if model is None:
        st.warning("""
        ⚠️ **Model file not found.**

        Please run `train_model.py` first to train and save the model, then relaunch this app.

        ```bash
        python train_model.py
        streamlit run app.py
        ```
        """)
        st.stop()

    # ── Overview metrics ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">2,395</div>
            <div class="metric-lbl">Training Records</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">32</div>
            <div class="metric-lbl">Input Features</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">91.9%</div>
            <div class="metric-lbl">Model Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">4</div>
            <div class="metric-lbl">Hidden Layers</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">3</div>
            <div class="metric-lbl">Risk Classes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔮  Prediction", "📊  Analysis", "ℹ️  About"])

    # ════════════ TAB 1: Prediction ════════════
    with tab1:
        if predict_btn or st.session_state.get("has_predicted"):
            st.session_state["has_predicted"] = True
            pred_class, probs = predict(model, scaler, raw)
            info = CLASS_INFO[pred_class]
            st.session_state["last_pred"]  = pred_class
            st.session_state["last_probs"] = probs
            st.session_state["last_raw"]   = raw

            col_main, col_side = st.columns([1.1, 1], gap="large")

            # ── Prediction result ──
            with col_main:
                st.markdown(f"""
                <div class="pred-card {info['color']}">
                    <div class="pred-emoji">{info['emoji']}</div>
                    <div style="color:rgba(255,255,255,0.55); font-size:0.85rem;
                                text-transform:uppercase; letter-spacing:0.1em; margin-top:0.5rem">
                        Predicted Risk Level
                    </div>
                    <div class="pred-label {info['color']}">{info['label']}</div>
                    <div class="pred-desc">{info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                st.markdown('<div class="section-header" style="margin-top:1.5rem">Class Probabilities</div>',
                            unsafe_allow_html=True)
                bar_colors = {"At Risk": "#f87171", "Average": "#fbbf24", "High Performer": "#34d399"}
                for i, (lbl, hex_col) in enumerate(bar_colors.items()):
                    pct = probs[i] * 100
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-label">{lbl}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill"
                                 style="width:{pct:.1f}%; background:{hex_col}"></div>
                        </div>
                        <div class="prob-val">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence note
                confidence = max(probs) * 100
                conf_color = "#34d399" if confidence > 75 else "#fbbf24" if confidence > 55 else "#f87171"
                st.markdown(f"""
                <div class="tip-box" style="margin-top:1rem">
                    🎯 Model confidence: <strong style="color:{conf_color}">{confidence:.1f}%</strong>
                    &nbsp;—&nbsp;
                    {"High confidence prediction" if confidence > 75
                     else "Moderate confidence — review borderline factors"
                     if confidence > 55 else "Low confidence — this is a borderline case"}
                </div>
                """, unsafe_allow_html=True)

            # ── Radar + Risk factors ──
            with col_side:
                st.markdown('<div class="section-header">Student Profile Radar</div>',
                            unsafe_allow_html=True)
                st.pyplot(radar_chart(raw), use_container_width=True)

                st.markdown('<div class="section-header" style="margin-top:1rem">Risk Factor Summary</div>',
                            unsafe_allow_html=True)
                positives, negatives = get_risk_factors(raw)
                if positives:
                    for p in positives[:4]:
                        st.markdown(f'<div class="tip-box" style="border-color:#34d399">{p}</div>',
                                    unsafe_allow_html=True)
                if negatives:
                    for n in negatives[:4]:
                        st.markdown(f'<div class="tip-box" style="border-color:#f87171">{n}</div>',
                                    unsafe_allow_html=True)

        else:
            # Placeholder before first prediction
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem;
                        background:rgba(255,255,255,0.03);
                        border:1px dashed rgba(167,139,250,0.3);
                        border-radius:18px; margin-top:1rem">
                <div style="font-size:4rem">🔮</div>
                <div style="font-family:Syne,sans-serif; font-size:1.4rem;
                             font-weight:700; color:rgba(255,255,255,0.7);
                             margin-top:1rem">
                    Ready to Predict
                </div>
                <div style="color:rgba(255,255,255,0.4); margin-top:0.5rem">
                    Fill in the student details in the sidebar<br>
                    and click <strong style="color:#a78bfa">PREDICT PERFORMANCE</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════ TAB 2: Analysis ════════════
    with tab2:
        if not st.session_state.get("has_predicted"):
            st.info("Run a prediction first to see the analysis.")
        else:
            raw   = st.session_state["last_raw"]
            probs = st.session_state["last_probs"]
            pred  = st.session_state["last_pred"]

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown('<div class="section-header">Probability Distribution</div>',
                            unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                labels = ['At Risk', 'Average', 'High Performer']
                colors = ['#f87171', '#fbbf24', '#34d399']
                bars = ax.bar(labels, probs * 100, color=colors,
                              edgecolor='none', width=0.55)
                for bar, v in zip(bars, probs):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f'{v*100:.1f}%', ha='center',
                            color='white', fontsize=10, fontweight='bold')
                ax.set_ylim(0, 115)
                ax.tick_params(colors='white', labelsize=9)
                ax.spines[:].set_color('rgba(255,255,255,0.1)')
                ax.yaxis.grid(True, color='rgba(255,255,255,0.08)', linewidth=0.5)
                ax.set_ylabel('Probability (%)', color='rgba(255,255,255,0.6)', fontsize=9)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">Key Academic Indicators</div>',
                            unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                fig2.patch.set_facecolor('none')
                ax2.set_facecolor('none')

                indicators   = ['G1', 'G2', 'Study\nTime×5', 'Health×4', 'Failures\n(inv)']
                scale_values = [
                    raw['G1'],
                    raw['G2'],
                    raw['studytime'] * 5,
                    raw['health'] * 4,
                    (3 - raw['failures']) * 6,
                ]
                ind_colors = ['#60a5fa', '#a78bfa', '#34d399', '#fbbf24', '#f87171']
                ax2.barh(indicators, scale_values, color=ind_colors,
                         edgecolor='none', height=0.55)
                ax2.set_xlim(0, 22)
                ax2.tick_params(colors='white', labelsize=9)
                ax2.spines[:].set_color('rgba(255,255,255,0.1)')
                ax2.xaxis.grid(True, color='rgba(255,255,255,0.08)', linewidth=0.5)
                for i, v in enumerate(scale_values):
                    ax2.text(v + 0.2, i, str(round(v, 1)), va='center',
                             color='white', fontsize=9)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig2, use_container_width=True)

            # Input summary table
            st.markdown('<div class="section-header" style="margin-top:1.5rem">Full Input Summary</div>',
                        unsafe_allow_html=True)
            summary_data = {
                "Feature": list(raw.keys()),
                "Value":   list(raw.values()),
            }
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(
                df_summary.set_index("Feature"),
                use_container_width=True,
                height=350
            )

    # ════════════ TAB 3: About ════════════
    with tab3:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("""
            <div class="section-header">About This App</div>
            <div style="color:rgba(255,255,255,0.7); line-height:1.8; font-size:0.92rem">
            <p>
            <strong style="color:#a78bfa">EduPredict AI</strong> is a deep learning–powered
            student performance predictor built as part of a final year project on enhancing
            student performance prediction using deep learning architectures.
            </p>
            <p>
            The system classifies students into three risk levels based on 32 academic, social,
            and demographic features. Early identification allows educators to intervene
            before a student falls behind.
            </p>
            </div>

            <div class="section-header" style="margin-top:1.5rem">Model Architecture</div>
            <div style="color:rgba(255,255,255,0.7); font-size:0.88rem; line-height:1.9">
                <code style="color:#a78bfa">Input (32)</code> →
                <code style="color:#60a5fa">Dense(256) + BN + Dropout(0.3)</code> →<br>
                <code style="color:#60a5fa">Dense(128) + BN + Dropout(0.3)</code> →<br>
                <code style="color:#60a5fa">Dense(64)  + BN + Dropout(0.2)</code> →<br>
                <code style="color:#60a5fa">Dense(32)  + Dropout(0.2)</code> →<br>
                <code style="color:#34d399">Output(3, Softmax)</code>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="section-header">Risk Classes Explained</div>
            """, unsafe_allow_html=True)

            for cls, info in CLASS_INFO.items():
                boundary = "G3 < 10" if cls == 0 else "10 ≤ G3 < 14" if cls == 1 else "G3 ≥ 14"
                st.markdown(f"""
                <div class="pred-card {info['color']}" style="padding:1.2rem; margin:0.6rem 0">
                    <div style="display:flex; align-items:center; gap:0.8rem">
                        <span style="font-size:1.8rem">{info['emoji']}</span>
                        <div style="text-align:left">
                            <div class="pred-label {info['color']}"
                                 style="font-size:1.1rem">{info['label']}</div>
                            <div class="pred-desc" style="margin:0">
                                Grade boundary: <strong>{boundary}</strong><br>{info['desc']}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="section-header" style="margin-top:1.2rem">Performance Summary</div>
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val">91.9%</div>
                    <div class="metric-lbl">Accuracy</div></div>
                <div class="metric-card"><div class="metric-val">92.0%</div>
                    <div class="metric-lbl">F1-Score</div></div>
                <div class="metric-card"><div class="metric-val">100</div>
                    <div class="metric-lbl">Epochs</div></div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
