import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


try:
    import plotly.graph_objects as go
except ImportError:
    install("plotly")
    import plotly.graph_objects as go

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    install("scikit-learn")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

import streamlit as st

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #f0f4ff;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 750px; }

/* Top banner */
.banner {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.banner h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.banner p {
    margin: 0.4rem 0 0;
    opacity: 0.8;
    font-size: 0.92rem;
}

/* Section labels */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e0e7ff;
}

/* Result boxes */
.result-risk    { background:#fee2e2; border:2px solid #fca5a5; border-radius:14px; padding:1.5rem; text-align:center; }
.result-average { background:#fef9c3; border:2px solid #fcd34d; border-radius:14px; padding:1.5rem; text-align:center; }
.result-high    { background:#dcfce7; border:2px solid #6ee7b7; border-radius:14px; padding:1.5rem; text-align:center; }

.result-emoji { font-size: 2.5rem; }
.result-title { font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; }
.result-title.risk    { color: #dc2626; }
.result-title.average { color: #d97706; }
.result-title.high    { color: #16a34a; }
.result-desc { font-size: 0.88rem; color: #555; margin-top: 0.3rem; }

/* Prob bars */
.bar-row { display:flex; align-items:center; gap:0.8rem; margin:0.4rem 0; }
.bar-name { font-size:0.85rem; color:#444; min-width:130px; }
.bar-bg   { flex:1; background:#e5e7eb; border-radius:999px; height:10px; overflow:hidden; }
.bar-fill { height:100%; border-radius:999px; }
.bar-pct  { font-size:0.85rem; font-weight:600; color:#333; min-width:40px; text-align:right; }

/* Tip rows */
.tip { padding:0.5rem 0.7rem; border-radius:8px; font-size:0.85rem; margin:0.35rem 0; }
.tip-good { background:#dcfce7; color:#15803d; }
.tip-warn { background:#fee2e2; color:#b91c1c; }

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    opacity: 0.92 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Train model once ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    base = os.path.dirname(os.path.abspath(__file__))
    df   = pd.read_csv(os.path.join(base, "STUDENT_PERFORMANCE_EXPANDED.csv"))

    yes_no = ["schoolsup","famsup","paid","activities",
              "nursery","higher","internet","romantic"]
    for col in yes_no:
        df[col] = (df[col] == "yes").astype(int)

    cats = ["school","sex","address","famsize","Pstatus",
            "Mjob","Fjob","reason","guardian"]
    for col in cats:
        df[col] = pd.factorize(df[col])[0]

    df["risk"] = df["G3"].apply(lambda g: 0 if g < 10 else 1 if g < 14 else 2)

    FEATS = ["G1","G2","failures","studytime","absences",
             "Medu","Fedu","higher","internet","famrel","health"]

    X = df[FEATS].astype(float)
    y = df["risk"]

    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xs = sc.fit_transform(Xtr)

    m = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu", solver="adam",
        max_iter=200, random_state=42,
        early_stopping=False, verbose=False,
    )
    m.fit(Xs, ytr)
    return m, sc


FEATS = ["G1","G2","failures","studytime","absences",
         "Medu","Fedu","higher","internet","famrel","health"]

CLASSES = {
    0: ("At Risk",        "🔴", "risk",    "#ef4444",
        "This student may need immediate academic support."),
    1: ("Average",        "🟡", "average", "#f59e0b",
        "Performing adequately but has room to improve."),
    2: ("High Performer", "🟢", "high",    "#22c55e",
        "On track for excellent academic achievement."),
}


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="banner">
        <h1>🎓 EduPredict AI</h1>
        <p>Student Academic Risk Predictor — powered by deep learning</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, scaler = train_model()

    # ── Grades ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📊 Grades</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    G1       = c1.number_input("1st Period Grade (G1)", 0, 20, 10)
    G2       = c2.number_input("2nd Period Grade (G2)", 0, 20, 10)
    failures = c3.number_input("Past Failures", 0, 3, 0)

    # ── Study habits ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📚 Study Habits</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    studytime = c4.number_input("Study Time (1–4)", 1, 4, 2,
                                help="1 = under 2 hrs  |  2 = 2–5 hrs  |  3 = 5–10 hrs  |  4 = over 10 hrs")
    absences  = c5.number_input("Absences", 0, 75, 4)
    schoolsup = c6.selectbox("School Extra Support", ["No", "Yes"])

    # ── Family ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">👨‍👩‍👧 Family</div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    Medu   = c7.number_input("Mother Education (0–4)", 0, 4, 2,
                             help="0 = none  |  1 = primary  |  2 = middle  |  3 = secondary  |  4 = higher")
    Fedu   = c8.number_input("Father Education (0–4)", 0, 4, 2,
                             help="0 = none  |  1 = primary  |  2 = middle  |  3 = secondary  |  4 = higher")
    famrel = c9.number_input("Family Relations (1–5)", 1, 5, 4,
                             help="1 = very bad  |  5 = excellent")

    # ── Wellbeing ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💚 Wellbeing</div>', unsafe_allow_html=True)
    c10, c11, c12 = st.columns(3)
    health   = c10.number_input("Health (1–5)", 1, 5, 3,
                                help="1 = very bad  |  5 = very good")
    higher   = c11.selectbox("Wants Higher Education", ["Yes", "No"])
    internet = c12.selectbox("Internet at Home", ["Yes", "No"])

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🔮  Predict Risk Level"):
        inp = {
            "G1": float(G1), "G2": float(G2),
            "failures": float(failures), "studytime": float(studytime),
            "absences": float(absences),
            "Medu": float(Medu), "Fedu": float(Fedu),
            "higher":    1.0 if higher   == "Yes" else 0.0,
            "internet":  1.0 if internet == "Yes" else 0.0,
            "famrel": float(famrel), "health": float(health),
        }
        arr   = np.array([[inp[f] for f in FEATS]], dtype=np.float32)
        probs = model.predict_proba(scaler.transform(arr))[0]
        pred  = int(np.argmax(probs))
        label, emoji, css, color, desc = CLASSES[pred]

        st.markdown("---")

        # Result
        st.markdown(f"""
        <div class="result-{css}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-title {css}">{label}</div>
            <div class="result-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("**Confidence breakdown**")
        for i, (n, hx) in enumerate(zip(
            ["At Risk", "Average", "High Performer"],
            ["#ef4444", "#f59e0b", "#22c55e"]
        )):
            pct = probs[i] * 100
            st.markdown(f"""
            <div class="bar-row">
                <div class="bar-name">{n}</div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{pct:.1f}%;background:{hx}"></div>
                </div>
                <div class="bar-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Tips
        st.markdown("**Key factors**")
        if G1 >= 12 or G2 >= 12:
            st.markdown('<div class="tip tip-good">✅ Strong mid-term grades.</div>',
                        unsafe_allow_html=True)
        if G1 < 8 or G2 < 8:
            st.markdown('<div class="tip tip-warn">⚠️ Low mid-term grades — biggest risk factor.</div>',
                        unsafe_allow_html=True)
        if failures >= 2:
            st.markdown(f'<div class="tip tip-warn">⚠️ {failures} past failures increase risk significantly.</div>',
                        unsafe_allow_html=True)
        if failures == 0:
            st.markdown('<div class="tip tip-good">✅ No past failures.</div>',
                        unsafe_allow_html=True)
        if absences > 15:
            st.markdown(f'<div class="tip tip-warn">⚠️ {absences} absences is high — attendance matters.</div>',
                        unsafe_allow_html=True)
        if studytime >= 3:
            st.markdown('<div class="tip tip-good">✅ Good study habits.</div>',
                        unsafe_allow_html=True)
        if higher == "Yes":
            st.markdown('<div class="tip tip-good">✅ Aspires to higher education — positive motivator.</div>',
                        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
