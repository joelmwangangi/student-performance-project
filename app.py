import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
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
    margin: 0; line-height: 1.1;
}
.hero-sub { color: rgba(255,255,255,0.65); font-size: 1.05rem; margin-top: 0.7rem; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(167,139,250,0.25);
}

.pred-card {
    border-radius: 18px; padding: 2.2rem;
    text-align: center; margin: 1rem 0; border: 2px solid;
}
.pred-card.risk    { background: rgba(239,68,68,0.12);  border-color: rgba(239,68,68,0.5);  }
.pred-card.average { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.5); }
.pred-card.high    { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.5); }
.pred-label { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; margin:0.3rem 0; }
.pred-label.risk    { color: #f87171; }
.pred-label.average { color: #fbbf24; }
.pred-label.high    { color: #34d399; }
.pred-emoji { font-size: 3.5rem; }
.pred-desc  { color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 0.5rem; }

.prob-row   { display:flex; align-items:center; gap:0.8rem; margin:0.5rem 0; }
.prob-label { color:rgba(255,255,255,0.75); font-size:0.88rem; min-width:130px; }
.prob-bar-bg { flex:1; background:rgba(255,255,255,0.08); border-radius:999px; height:10px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:999px; }
.prob-val   { color:white; font-size:0.88rem; font-weight:600; min-width:42px; text-align:right; }

.metric-row { display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:80px;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:14px; padding:1rem 1.2rem; text-align:center;
}
.metric-val { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#a78bfa; }
.metric-lbl { color:rgba(255,255,255,0.5); font-size:0.78rem; margin-top:0.2rem; }

.tip-box {
    background:rgba(99,102,241,0.1); border-left:3px solid #6366f1;
    border-radius:8px; padding:0.9rem 1.1rem; margin:0.8rem 0;
    color:rgba(255,255,255,0.75); font-size:0.88rem;
}

[data-testid="stSidebar"] { background: rgba(15,12,41,0.95) !important; border-right: 1px solid rgba(99,102,241,0.2); }
label { color: rgba(255,255,255,0.8) !important; font-size: 0.88rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 0.75rem 2rem !important; width: 100% !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
}
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.05); border-radius:12px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"]      { color:rgba(255,255,255,0.6) !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; border-radius:10px !important; }
.stTabs [aria-selected="true"]    { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:white !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    mp = os.path.join(base, "model", "student_ann_model.pkl")
    sp = os.path.join(base, "model", "scaler.pkl")
    if not os.path.exists(mp) or not os.path.exists(sp):
        return None, None
    with open(mp, "rb") as f: model = pickle.load(f)
    with open(sp, "rb") as f: scaler = pickle.load(f)
    return model, scaler


# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'school','sex','age','address','famsize','Pstatus',
    'Medu','Fedu','Mjob','Fjob','reason','guardian',
    'traveltime','studytime','failures','schoolsup','famsup',
    'paid','activities','nursery','higher','internet','romantic',
    'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2'
]
ENCODE_MAP = {
    'school':     {'GP':0,'MS':1},
    'sex':        {'F':0,'M':1},
    'address':    {'R':0,'U':1},
    'famsize':    {'GT3':0,'LE3':1},
    'Pstatus':    {'A':0,'T':1},
    'Mjob':       {'at_home':0,'health':1,'other':2,'services':3,'teacher':4},
    'Fjob':       {'at_home':0,'health':1,'other':2,'services':3,'teacher':4},
    'reason':     {'course':0,'home':1,'other':2,'reputation':3},
    'guardian':   {'father':0,'mother':1,'other':2},
    'schoolsup':  {'no':0,'yes':1},
    'famsup':     {'no':0,'yes':1},
    'paid':       {'no':0,'yes':1},
    'activities': {'no':0,'yes':1},
    'nursery':    {'no':0,'yes':1},
    'higher':     {'no':0,'yes':1},
    'internet':   {'no':0,'yes':1},
    'romantic':   {'no':0,'yes':1},
}
CLASS_INFO = {
    0: {"label":"At Risk",        "emoji":"🔴","color":"risk",   "hex":"#f87171",
        "desc":"Student is at risk of failing and may need immediate academic intervention."},
    1: {"label":"Average",        "emoji":"🟡","color":"average","hex":"#fbbf24",
        "desc":"Student is performing at an average level with room for improvement."},
    2: {"label":"High Performer", "emoji":"🟢","color":"high",   "hex":"#34d399",
        "desc":"Student is performing excellently and on track for academic success."},
}


def encode_and_predict(model, scaler, raw):
    encoded = []
    for feat in FEATURE_COLS:
        val = raw[feat]
        encoded.append(ENCODE_MAP[feat][val] if feat in ENCODE_MAP else int(val))
    arr = np.array([encoded], dtype=np.float32)
    arr_s = scaler.transform(arr)
    probs = model.predict_proba(arr_s)[0]
    return int(np.argmax(probs)), probs


def get_risk_factors(raw):
    pos, neg = [], []
    if raw['studytime'] >= 3:   pos.append("✅ Studies 5+ hours weekly")
    if raw['studytime'] <= 1:   neg.append("⚠️ Studies less than 2 hours weekly")
    if raw['failures'] == 0:    pos.append("✅ No past class failures")
    if raw['failures'] >= 2:    neg.append("⚠️ Multiple past failures")
    if raw['G1'] >= 12:         pos.append(f"✅ Strong 1st period grade ({raw['G1']}/20)")
    if raw['G1'] < 8:           neg.append(f"⚠️ Low 1st period grade ({raw['G1']}/20)")
    if raw['G2'] >= 12:         pos.append(f"✅ Strong 2nd period grade ({raw['G2']}/20)")
    if raw['G2'] < 8:           neg.append(f"⚠️ Low 2nd period grade ({raw['G2']}/20)")
    if raw['higher'] == 'yes':  pos.append("✅ Motivated for higher education")
    if raw['internet'] == 'yes':pos.append("✅ Has internet access at home")
    if raw['absences'] > 15:    neg.append(f"⚠️ High absences ({raw['absences']} days)")
    if raw['absences'] <= 3:    pos.append("✅ Very few absences")
    if raw['Dalc'] >= 4:        neg.append("⚠️ High workday alcohol consumption")
    if raw['health'] >= 4:      pos.append("✅ Good health status")
    if raw['health'] <= 2:      neg.append("⚠️ Poor health status")
    if raw['Medu'] >= 3:        pos.append("✅ Mother has secondary/higher education")
    if raw['famrel'] >= 4:      pos.append("✅ Good family relationships")
    if raw['famrel'] <= 2:      neg.append("⚠️ Poor family relationships")
    return pos, neg


# ── Plotly charts ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(255,255,255,0.8)', family='DM Sans'),
    margin=dict(l=10, r=10, t=30, b=10),
)

def radar_chart(raw):
    cats   = ['Study Time','Family Relation','Health','Free Time','G1 Score','G2 Score']
    vals   = [raw['studytime']/4, raw['famrel']/5, raw['health']/5,
              raw['freetime']/5, raw['G1']/20, raw['G2']/20]
    vals  += vals[:1]
    cats  += cats[:1]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(99,102,241,0.2)',
        line=dict(color='#a78bfa', width=2),
        marker=dict(color='#a78bfa', size=6)
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1],
                            gridcolor='rgba(255,255,255,0.1)',
                            tickfont=dict(size=8, color='rgba(255,255,255,0.4)')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                             tickfont=dict(size=9, color='white'))
        ), height=320
    )
    return fig


def prob_bar_chart(probs):
    labels = ['At Risk', 'Average', 'High Performer']
    colors = ['#f87171', '#fbbf24', '#34d399']
    fig = go.Figure(go.Bar(
        x=labels, y=[p*100 for p in probs],
        marker_color=colors,
        text=[f'{p*100:.1f}%' for p in probs],
        textposition='outside',
        textfont=dict(color='white', size=12)
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        yaxis=dict(range=[0,115], gridcolor='rgba(255,255,255,0.08)',
                   tickfont=dict(color='rgba(255,255,255,0.5)')),
        xaxis=dict(tickfont=dict(color='white', size=11)),
        height=320, title=dict(text='Class Probabilities (%)', font=dict(size=13))
    )
    return fig


def indicators_chart(raw):
    names = ['G1 Grade', 'G2 Grade', 'Study Time ×5', 'Health ×4', 'No-Failure Score']
    vals  = [raw['G1'], raw['G2'], raw['studytime']*5,
             raw['health']*4, (3-raw['failures'])*6]
    colors = ['#60a5fa','#a78bfa','#34d399','#fbbf24','#f87171']
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker_color=colors,
        text=[str(round(v,1)) for v in vals],
        textposition='outside',
        textfont=dict(color='white', size=10)
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        xaxis=dict(range=[0,24], gridcolor='rgba(255,255,255,0.08)',
                   tickfont=dict(color='rgba(255,255,255,0.5)')),
        yaxis=dict(tickfont=dict(color='white', size=10)),
        height=320, title=dict(text='Key Academic Indicators', font=dict(size=13))
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 0.5rem'>
            <span style='font-size:2.5rem'>🎓</span>
            <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;color:white;margin-top:0.3rem'>EduPredict AI</div>
            <div style='color:rgba(255,255,255,0.45);font-size:0.78rem'>Student Risk Classifier</div>
        </div><hr>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📋 Student Profile</div>', unsafe_allow_html=True)

        with st.expander("👤 Personal Information", expanded=True):
            school  = st.selectbox("School", ["GP","MS"], help="GP=Gabriel Pereira, MS=Mousinho da Silveira")
            sex     = st.selectbox("Sex", ["F","M"])
            age     = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address", ["U","R"], help="U=Urban, R=Rural")
            famsize = st.selectbox("Family Size", ["GT3","LE3"], help="GT3=>3 members, LE3=≤3")
            Pstatus = st.selectbox("Parent Status", ["T","A"], help="T=Together, A=Apart")
            guardian= st.selectbox("Guardian", ["mother","father","other"])

        with st.expander("🏠 Family & Education", expanded=True):
            Medu  = st.slider("Mother's Education", 0, 4, 2, help="0=none … 4=higher")
            Fedu  = st.slider("Father's Education", 0, 4, 2, help="0=none … 4=higher")
            Mjob  = st.selectbox("Mother's Job", ["at_home","health","other","services","teacher"])
            Fjob  = st.selectbox("Father's Job", ["at_home","health","other","services","teacher"])
            famsup= st.selectbox("Family Educational Support", ["yes","no"])
            famrel= st.slider("Family Relationship Quality", 1, 5, 4, help="1=very bad … 5=excellent")

        with st.expander("📚 School & Academic", expanded=True):
            reason    = st.selectbox("Reason to Choose School", ["course","home","reputation","other"])
            traveltime= st.slider("Travel Time to School", 1, 4, 1, help="1=<15min … 4=>60min")
            studytime = st.slider("Weekly Study Time", 1, 4, 2, help="1=<2h … 4=>10h")
            failures  = st.slider("Past Class Failures", 0, 3, 0)
            schoolsup = st.selectbox("School Extra Support", ["yes","no"])
            paid      = st.selectbox("Extra Paid Classes", ["yes","no"])
            absences  = st.slider("Number of Absences", 0, 75, 4)
            higher    = st.selectbox("Wants Higher Education", ["yes","no"])
            nursery   = st.selectbox("Attended Nursery", ["yes","no"])
            activities= st.selectbox("Extra-Curricular Activities", ["yes","no"])
            internet  = st.selectbox("Internet Access at Home", ["yes","no"])

        with st.expander("🎮 Social & Lifestyle"):
            romantic= st.selectbox("In a Romantic Relationship", ["no","yes"])
            freetime= st.slider("Free Time After School", 1, 5, 3, help="1=very low … 5=very high")
            goout   = st.slider("Going Out with Friends", 1, 5, 3)
            Dalc    = st.slider("Workday Alcohol Consumption", 1, 5, 1)
            Walc    = st.slider("Weekend Alcohol Consumption", 1, 5, 2)
            health  = st.slider("Current Health Status", 1, 5, 3, help="1=very bad … 5=very good")

        st.markdown('<hr><div class="section-header">📝 Mid-Term Grades</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        G1 = c1.number_input("G1 — 1st Period", 0, 20, 10)
        G2 = c2.number_input("G2 — 2nd Period", 0, 20, 10)

        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("🔮  PREDICT PERFORMANCE", use_container_width=True)

    return dict(
        school=school, sex=sex, age=age, address=address, famsize=famsize,
        Pstatus=Pstatus, Medu=Medu, Fedu=Fedu, Mjob=Mjob, Fjob=Fjob,
        reason=reason, guardian=guardian, traveltime=traveltime,
        studytime=studytime, failures=failures, schoolsup=schoolsup,
        famsup=famsup, paid=paid, activities=activities, nursery=nursery,
        higher=higher, internet=internet, romantic=romantic, famrel=famrel,
        freetime=freetime, goout=goout, Dalc=Dalc, Walc=Walc,
        health=health, absences=absences, G1=G1, G2=G2,
    ), btn


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, scaler = load_model()
    raw, predict_btn = sidebar()

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">EduPredict AI</div>
        <div class="hero-sub">
            Deep Learning–Powered Student Performance Prediction &nbsp;|&nbsp;
            ANN Classifier &nbsp;|&nbsp; 3-Class Risk Assessment
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("""
        **Model files not found.**

        Make sure `model/student_ann_model.pkl` and `model/scaler.pkl` exist in your repository.

        Run locally first:
        ```bash
        python train_model.py
        ```
        Then commit the `model/` folder to GitHub before deploying.
        """)
        st.stop()

    st.markdown("""
    <div class="metric-row">
        <div class="metric-card"><div class="metric-val">2,395</div><div class="metric-lbl">Training Records</div></div>
        <div class="metric-card"><div class="metric-val">32</div><div class="metric-lbl">Input Features</div></div>
        <div class="metric-card"><div class="metric-val">91.9%</div><div class="metric-lbl">Model Accuracy</div></div>
        <div class="metric-card"><div class="metric-val">4</div><div class="metric-lbl">Hidden Layers</div></div>
        <div class="metric-card"><div class="metric-val">100</div><div class="metric-lbl">Epochs Trained</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔮  Prediction", "📊  Analysis", "ℹ️  About"])

    # ── TAB 1: Prediction ────────────────────────────────────────────────────
    with tab1:
        if predict_btn or st.session_state.get("has_predicted"):
            st.session_state["has_predicted"] = True
            pred, probs = encode_and_predict(model, scaler, raw)
            st.session_state.update(last_pred=pred, last_probs=probs, last_raw=raw)
            info = CLASS_INFO[pred]

            col1, col2 = st.columns([1.1, 1], gap="large")

            with col1:
                st.markdown(f"""
                <div class="pred-card {info['color']}">
                    <div class="pred-emoji">{info['emoji']}</div>
                    <div style="color:rgba(255,255,255,0.55);font-size:0.85rem;
                                text-transform:uppercase;letter-spacing:0.1em;margin-top:0.5rem">
                        Predicted Risk Level
                    </div>
                    <div class="pred-label {info['color']}">{info['label']}</div>
                    <div class="pred-desc">{info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-header" style="margin-top:1.5rem">Class Probabilities</div>',
                            unsafe_allow_html=True)
                for i, (lbl, hex_c) in enumerate(zip(
                        ['At Risk','Average','High Performer'],
                        ['#f87171','#fbbf24','#34d399'])):
                    pct = probs[i] * 100
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-label">{lbl}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{hex_c}"></div>
                        </div>
                        <div class="prob-val">{pct:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                conf = max(probs) * 100
                conf_col = "#34d399" if conf>75 else "#fbbf24" if conf>55 else "#f87171"
                st.markdown(f"""
                <div class="tip-box" style="margin-top:1rem">
                    🎯 Model confidence: <strong style="color:{conf_col}">{conf:.1f}%</strong>
                    &nbsp;—&nbsp;
                    {"High confidence prediction" if conf>75
                     else "Moderate confidence — review borderline factors" if conf>55
                     else "Low confidence — borderline case, review carefully"}
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="section-header">Student Profile Radar</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(radar_chart(raw), use_container_width=True)

                st.markdown('<div class="section-header">Risk Factor Summary</div>',
                            unsafe_allow_html=True)
                pos, neg = get_risk_factors(raw)
                for p in pos[:4]:
                    st.markdown(f'<div class="tip-box" style="border-color:#34d399">{p}</div>',
                                unsafe_allow_html=True)
                for n in neg[:4]:
                    st.markdown(f'<div class="tip-box" style="border-color:#f87171">{n}</div>',
                                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;
                        background:rgba(255,255,255,0.03);
                        border:1px dashed rgba(167,139,250,0.3);
                        border-radius:18px;margin-top:1rem">
                <div style="font-size:4rem">🔮</div>
                <div style="font-family:Syne,sans-serif;font-size:1.4rem;
                             font-weight:700;color:rgba(255,255,255,0.7);margin-top:1rem">
                    Ready to Predict
                </div>
                <div style="color:rgba(255,255,255,0.4);margin-top:0.5rem">
                    Fill in the student details in the sidebar<br>
                    and click <strong style="color:#a78bfa">PREDICT PERFORMANCE</strong>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 2: Analysis ───────────────────────────────────────────────────────
    with tab2:
        if not st.session_state.get("has_predicted"):
            st.info("Run a prediction first to see the analysis.")
        else:
            raw   = st.session_state["last_raw"]
            probs = st.session_state["last_probs"]

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.plotly_chart(prob_bar_chart(probs), use_container_width=True)
            with c2:
                st.plotly_chart(indicators_chart(raw), use_container_width=True)

            st.markdown('<div class="section-header" style="margin-top:1rem">Full Input Summary</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame({"Feature": list(raw.keys()), "Value": list(raw.values())}).set_index("Feature"),
                use_container_width=True, height=400
            )

    # ── TAB 3: About ──────────────────────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("""
            <div class="section-header">About This App</div>
            <div style="color:rgba(255,255,255,0.7);line-height:1.8;font-size:0.92rem">
            <p><strong style="color:#a78bfa">EduPredict AI</strong> is a deep learning–powered
            student performance predictor. The system classifies students into three risk levels
            based on 32 academic, social, and demographic features, allowing educators to
            intervene before a student falls behind.</p>
            </div>
            <div class="section-header" style="margin-top:1.5rem">Model Architecture</div>
            <div style="color:rgba(255,255,255,0.7);font-size:0.88rem;line-height:2">
                <code style="color:#a78bfa">Input (32 features)</code><br>
                → <code style="color:#60a5fa">Dense(256) + BatchNorm + Dropout(0.3)</code><br>
                → <code style="color:#60a5fa">Dense(128) + BatchNorm + Dropout(0.3)</code><br>
                → <code style="color:#60a5fa">Dense(64)  + BatchNorm + Dropout(0.2)</code><br>
                → <code style="color:#60a5fa">Dense(32)  + Dropout(0.2)</code><br>
                → <code style="color:#34d399">Output(3) — Softmax</code>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-header">Risk Classes</div>', unsafe_allow_html=True)
            for cls, info in CLASS_INFO.items():
                boundary = "G3 < 10" if cls==0 else "10 ≤ G3 < 14" if cls==1 else "G3 ≥ 14"
                st.markdown(f"""
                <div class="pred-card {info['color']}" style="padding:1.2rem;margin:0.6rem 0">
                    <div style="display:flex;align-items:center;gap:0.8rem">
                        <span style="font-size:1.8rem">{info['emoji']}</span>
                        <div style="text-align:left">
                            <div class="pred-label {info['color']}" style="font-size:1.1rem">{info['label']}</div>
                            <div class="pred-desc" style="margin:0">
                                Grade: <strong>{boundary}</strong><br>{info['desc']}
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="section-header" style="margin-top:1rem">Performance</div>
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val">91.9%</div><div class="metric-lbl">Accuracy</div></div>
                <div class="metric-card"><div class="metric-val">92.0%</div><div class="metric-lbl">F1-Score</div></div>
                <div class="metric-card"><div class="metric-val">100</div><div class="metric-lbl">Epochs</div></div>
            </div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.8rem;margin-top:1rem">
                👩‍💻 Margaret Mukima Murungaru &nbsp;|&nbsp; CT204/109398/22<br>
                Supervisor: Mr. Kibaara
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
