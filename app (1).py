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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f0c29; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 820px; }

.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.hero p { color: rgba(255,255,255,0.5); margin-top: 0.5rem; font-size: 0.95rem; }

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem; font-weight: 700;
    color: #a78bfa; text-transform: uppercase;
    letter-spacing: 0.12em; margin-bottom: 1.2rem;
}

.result-box {
    border-radius: 16px; padding: 2rem;
    text-align: center; margin: 1.5rem 0; border: 2px solid;
}
.result-box.risk    { background: rgba(239,68,68,0.1);  border-color: rgba(239,68,68,0.4);  }
.result-box.average { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.4); }
.result-box.high    { background: rgba(52,211,153,0.1); border-color: rgba(52,211,153,0.4); }

.result-emoji { font-size: 3rem; }
.result-tag {
    font-size: 0.75rem; color: rgba(255,255,255,0.5);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.6rem;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800; margin: 0.2rem 0;
}
.result-label.risk    { color: #f87171; }
.result-label.average { color: #fbbf24; }
.result-label.high    { color: #34d399; }
.result-desc { color: rgba(255,255,255,0.55); font-size: 0.88rem; }

.prob-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.45rem 0; }
.prob-name { color: rgba(255,255,255,0.7); font-size: 0.85rem; min-width: 120px; }
.prob-track { flex: 1; background: rgba(255,255,255,0.07); border-radius: 999px; height: 9px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 999px; }
.prob-pct   { color: white; font-size: 0.85rem; font-weight: 600; min-width: 40px; text-align: right; }

label { color: rgba(255,255,255,0.75) !important; font-size: 0.88rem !important; }
.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; }

div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 0.7rem 2rem !important; width: 100% !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    margin-top: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Train model (cached — runs once) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    base = os.path.dirname(os.path.abspath(__file__))
    df   = pd.read_csv(os.path.join(base, "STUDENT_PERFORMANCE_EXPANDED.csv"))

    # Simple binary encoding for categoricals
    binary = {
        "school": {"GP": 0, "MS": 1},
        "sex":    {"F": 0, "M": 1},
        "address":{"U": 0, "R": 1},
        "famsize":{"GT3": 0, "LE3": 1},
        "Pstatus":{"T": 0, "A": 1},
        "Mjob":   {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
        "Fjob":   {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
        "reason": {"course": 0, "home": 1, "other": 2, "reputation": 3},
        "guardian":{"father": 0, "mother": 1, "other": 2},
    }
    yes_no = ["schoolsup","famsup","paid","activities","nursery",
              "higher","internet","romantic"]

    for col, mapping in binary.items():
        df[col] = df[col].map(mapping)
    for col in yes_no:
        df[col] = (df[col] == "yes").astype(int)

    def label(g):
        return 0 if g < 10 else 1 if g < 14 else 2

    df["risk"] = df["G3"].apply(label)

    FEATURES = ["G1","G2","failures","studytime","absences",
                "Medu","Fedu","higher","internet","schoolsup",
                "famrel","health","goout","Dalc","Walc"]

    X = df[FEATURES].astype(float)
    y = df["risk"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        early_stopping=False,
        verbose=False,
    )
    model.fit(Xs, y_train)
    return model, scaler


FEATURES = ["G1","G2","failures","studytime","absences",
            "Medu","Fedu","higher","internet","schoolsup",
            "famrel","health","goout","Dalc","Walc"]

CLASS_INFO = {
    0: {"label":"At Risk",        "emoji":"🔴","color":"risk",
        "desc":"This student may need immediate academic support and intervention."},
    1: {"label":"Average",        "emoji":"🟡","color":"average",
        "desc":"This student is performing adequately but has room to improve."},
    2: {"label":"High Performer", "emoji":"🟢","color":"high",
        "desc":"This student is on track for excellent academic achievement."},
}


def make_prob_chart(probs):
    labels = ["At Risk", "Average", "High Performer"]
    colors = ["#f87171", "#fbbf24", "#34d399"]
    fig = go.Figure(go.Bar(
        x=labels,
        y=[p * 100 for p in probs],
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.8)"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=260,
        yaxis=dict(range=[0, 120],
                   gridcolor="rgba(255,255,255,0.07)",
                   tickfont=dict(color="rgba(255,255,255,0.4)", size=10)),
        xaxis=dict(tickfont=dict(color="white", size=11)),
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>EduPredict AI</h1>
        <p>Enter student details below to predict their academic risk level</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Setting up model — please wait..."):
        model, scaler = train_model()

    # ── Input form ────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📝 Grades & Academic Record</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    G1       = c1.number_input("G1 — 1st Period Grade", 0, 20, 10,
                               help="Grade from 0 to 20")
    G2       = c2.number_input("G2 — 2nd Period Grade", 0, 20, 10,
                               help="Grade from 0 to 20")
    failures = c3.number_input("Past Class Failures", 0, 3, 0,
                               help="Number of times failed a class before")

    c4, c5, c6 = st.columns(3)
    studytime = c4.number_input("Weekly Study Time", 1, 4, 2,
                                help="1 = under 2 hrs   2 = 2-5 hrs   3 = 5-10 hrs   4 = over 10 hrs")
    absences  = c5.number_input("Number of Absences", 0, 75, 4,
                                help="Total school absences")
    schoolsup = c6.selectbox("School Extra Support", ["No", "Yes"],
                             help="Does the school provide extra educational support?")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">👨‍👩‍👧 Family Background</div>',
                unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    Medu   = c7.number_input("Mother's Education Level", 0, 4, 2,
                             help="0 = none   1 = primary   2 = middle school   3 = secondary   4 = higher")
    Fedu   = c8.number_input("Father's Education Level", 0, 4, 2,
                             help="0 = none   1 = primary   2 = middle school   3 = secondary   4 = higher")
    famrel = c9.number_input("Family Relationship Quality", 1, 5, 4,
                             help="1 = very bad   5 = excellent")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🌐 Lifestyle & Wellbeing</div>',
                unsafe_allow_html=True)

    c10, c11, c12 = st.columns(3)
    health  = c10.number_input("Health Status", 1, 5, 3,
                               help="1 = very bad   5 = very good")
    goout   = c11.number_input("Going Out with Friends", 1, 5, 3,
                               help="1 = very low   5 = very high")
    Dalc    = c12.number_input("Workday Alcohol Use", 1, 5, 1,
                               help="1 = very low   5 = very high")

    c13, c14, c15 = st.columns(3)
    Walc    = c13.number_input("Weekend Alcohol Use", 1, 5, 2,
                               help="1 = very low   5 = very high")
    higher  = c14.selectbox("Wants Higher Education", ["Yes", "No"])
    internet= c15.selectbox("Internet Access at Home", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    predict_btn = st.button("🔮  PREDICT RISK LEVEL")

    # ── Result ────────────────────────────────────────────────────────────────
    if predict_btn:
        input_vals = {
            "G1":        float(G1),
            "G2":        float(G2),
            "failures":  float(failures),
            "studytime": float(studytime),
            "absences":  float(absences),
            "Medu":      float(Medu),
            "Fedu":      float(Fedu),
            "higher":    1.0 if higher  == "Yes" else 0.0,
            "internet":  1.0 if internet == "Yes" else 0.0,
            "schoolsup": 1.0 if schoolsup == "Yes" else 0.0,
            "famrel":    float(famrel),
            "health":    float(health),
            "goout":     float(goout),
            "Dalc":      float(Dalc),
            "Walc":      float(Walc),
        }

        arr   = np.array([[input_vals[f] for f in FEATURES]], dtype=np.float32)
        arr_s = scaler.transform(arr)
        probs = model.predict_proba(arr_s)[0]
        pred  = int(np.argmax(probs))
        info  = CLASS_INFO[pred]

        # Result card
        st.markdown(f"""
        <div class="result-box {info['color']}">
            <div class="result-emoji">{info['emoji']}</div>
            <div class="result-tag">Predicted Risk Level</div>
            <div class="result-label {info['color']}">{info['label']}</div>
            <div class="result-desc">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown('<div class="card"><div class="card-title">Confidence Breakdown</div>',
                    unsafe_allow_html=True)
        for i, (lbl, hx) in enumerate(zip(
            ["At Risk", "Average", "High Performer"],
            ["#f87171", "#fbbf24", "#34d399"]
        )):
            pct = probs[i] * 100
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-name">{lbl}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{pct:.1f}%; background:{hx}"></div>
                </div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Bar chart
        st.plotly_chart(make_prob_chart(probs), use_container_width=True)

        # Key factor tips
        tips = []
        if G1 < 8 or G2 < 8:
            tips.append(("⚠️", "#f87171", "Low mid-term grades are the strongest risk factor."))
        if failures >= 2:
            tips.append(("⚠️", "#f87171", f"{failures} past failures significantly increases risk."))
        if absences > 15:
            tips.append(("⚠️", "#f87171", f"{absences} absences — high absenteeism hurts performance."))
        if studytime >= 3:
            tips.append(("✅", "#34d399", "Good study habits — studying 5+ hours weekly helps greatly."))
        if higher == "Yes":
            tips.append(("✅", "#34d399", "Aspiring to higher education is a positive motivator."))
        if Dalc >= 4:
            tips.append(("⚠️", "#fbbf24", "High weekday alcohol consumption is a risk factor."))
        if famrel >= 4:
            tips.append(("✅", "#34d399", "Strong family relationships support academic performance."))

        if tips:
            st.markdown('<div class="card"><div class="card-title">Key Factors</div>',
                        unsafe_allow_html=True)
            for icon, color, text in tips:
                st.markdown(
                    f'<div style="padding:0.5rem 0; border-left:3px solid {color}; '
                    f'padding-left:0.8rem; margin:0.4rem 0; '
                    f'color:rgba(255,255,255,0.75); font-size:0.88rem">'
                    f'{icon} {text}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;
                    background:rgba(255,255,255,0.02);
                    border:1px dashed rgba(167,139,250,0.25);
                    border-radius:16px; margin-top:1rem">
            <div style="font-size:3rem">🎓</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem;
                         font-weight:700; color:rgba(255,255,255,0.6);
                         margin-top:0.8rem">
                Fill in the form above and click Predict
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
