import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================== PAGE CONFIG ==================

st.set_page_config(
    page_title="Mental Health AI",
    page_icon="🧠",
    layout="wide"
)

# ================== STYLE ==================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

* { font-family: 'Sora', sans-serif; }

.stApp { background: #0b0c10; color: #e8e9f0; }

header[data-testid="stHeader"] { background: transparent; }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #13141c;
    padding: 8px; border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px;
    color: #6b6d80; font-size: 14px; font-weight: 500;
    padding: 10px 28px; border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7b61ff, #4f8aff) !important;
    color: white !important;
}

.stSlider label { color: #c4c6d6 !important; font-size: 14px !important; }

div.stButton > button {
    background: linear-gradient(135deg, #7b61ff, #4f8aff);
    color: white; font-size: 15px; font-weight: 600;
    border-radius: 12px; padding: 14px 40px;
    border: none; width: 100%; transition: opacity .2s;
}
div.stButton > button:hover { opacity: 0.88; }

.stTextArea textarea {
    background: #1a1b26 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #e8e9f0 !important;
    font-size: 15px !important; line-height: 1.7 !important;
}
.stTextArea textarea:focus { border-color: rgba(123,97,255,0.5) !important; }

.result-card {
    background: #13141c; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 28px 24px; text-align: center;
}
.result-card h3 {
    font-size: 13px; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 12px; font-weight: 500;
}
.result-card .pct {
    font-family: 'DM Serif Display', serif;
    font-size: 52px; font-weight: 400; line-height: 1; margin-bottom: 8px;
}
.result-card .bar-track {
    height: 6px; background: rgba(255,255,255,0.08);
    border-radius: 999px; margin-top: 14px; overflow: hidden;
}
.result-card .bar-fill { height: 100%; border-radius: 999px; }

.card-anxiety    { border-top: 3px solid #f5a623; }
.card-anxiety    h3, .card-anxiety    .pct { color: #f5a623; }
.card-anxiety    .bar-fill { background: #f5a623; }

.card-depression { border-top: 3px solid #7b61ff; }
.card-depression h3, .card-depression .pct { color: #7b61ff; }
.card-depression .bar-fill { background: #7b61ff; }

.card-stress     { border-top: 3px solid #34d399; }
.card-stress     h3, .card-stress     .pct { color: #34d399; }
.card-stress     .bar-fill { background: #34d399; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 38px; font-style: italic;
    line-height: 1.2; color: #e8e9f0; margin-bottom: 8px;
}
.hero-sub {
    color: #6b6d80; font-size: 15px;
    line-height: 1.6; margin-bottom: 32px;
}
.section-label {
    font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: #6b6d80; margin-bottom: 6px;
}
.disclaimer {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 14px 18px;
    font-size: 12px; color: #6b6d80;
    line-height: 1.6; margin-top: 24px;
}
.badge-primary {
    display: inline-block; padding: 6px 18px;
    border-radius: 999px; font-size: 13px;
    font-weight: 600; letter-spacing: 0.04em; margin-bottom: 20px;
}
.badge-anxiety    { background: rgba(245,166,35,0.12); color: #f5a623; border: 1px solid rgba(245,166,35,0.25); }
.badge-depression { background: rgba(123,97,255,0.12); color: #7b61ff; border: 1px solid rgba(123,97,255,0.25); }
.badge-stress     { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
</style>
""", unsafe_allow_html=True)


# ================== LOAD MODELS ==================

@st.cache_resource
def load_survey_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(
         "mental_model.h5", 
          compile=False,
          custom_objects={'InputLayer': tf.keras.layers.InputLayer}
    )

    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

@st.cache_resource
def load_nlp_model():
    model     = load_model("mental_health_nlp_model.h5", compile=False)
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    le        = pickle.load(open("label_encoder.pkl", "rb"))
    return model, tokenizer, le


# ================== HEADER ==================

st.markdown('<p class="section-label">AI-Powered</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">Mental Health Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Two ways to understand how you\'re feeling — answer a quick survey or describe your thoughts freely.</p>', unsafe_allow_html=True)


# ================== TABS ==================

tab1, tab2 = st.tabs(["📋  Survey Assessment", "💬  Text Analysis (NLP)"])


# ══════════════════════════════════════════════
#  TAB 1 — SURVEY
# ══════════════════════════════════════════════

with tab1:
    st.markdown("#### Rate each statement from 0 (never) to 4 (almost always)")
    st.markdown("")

    questions = [
        "I found it hard to wind down",
        "I was aware of dryness of my mouth",
        "I couldn't seem to experience any positive feeling at all",
        "I experienced breathing difficulty",
        "I found it difficult to work up the initiative to do things",
        "I tended to over-react to situations",
        "I experienced trembling",
        "I felt that I was using a lot of nervous energy",
        "I was worried about situations in which I might panic",
        "I felt that I had nothing to look forward to",
        "I found myself getting agitated",
        "I found it difficult to relax",
        "I felt down-hearted and blue",
        "I was intolerant of anything that kept me from getting on",
        "I felt I was close to panic",
        "I was unable to become enthusiastic",
        "I felt I wasn't worth much as a person",
        "I felt that I was rather touchy",
        "I was aware of the action of my heart",
        "I felt scared without any good reason",
        "I felt that life was meaningless",
        "I found it hard to calm down",
        "I felt nervous",
        "I felt sad and depressed",
        "I found myself getting impatient",
        "I felt that I was rather emotional",
        "I felt restless",
        "I had difficulty concentrating",
        "I felt lonely",
        "I found it difficult to relax",
        "I felt hopeless",
        "I felt worried about many things",
        "I felt that I had no energy",
        "I felt tense",
        "I felt tired for no reason",
        "I felt uneasy",
        "I felt worthless",
        "I felt anxious",
        "I felt discouraged",
        "I felt stressed",
        "I felt overwhelmed",
        "I felt emotionally exhausted",
    ]

    answers = []
    cols_q = st.columns(2)
    for i, q in enumerate(questions):
        with cols_q[i % 2]:
            val = st.slider(q, 0, 4, 0, key=f"survey_{i}")
            answers.append(val)

    st.markdown("")
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        survey_btn = st.button("Analyse My Answers", key="survey_btn")

    if survey_btn:
        try:
            survey_model, scaler = load_survey_model()
            data = np.array(answers).reshape(1, -1)
            data = scaler.transform(data)
            pred = survey_model.predict(data)[0]

            depression = int(pred[0] * 100)
            anxiety    = int(pred[1] * 100)
            stress     = int(pred[2] * 100)

            labels  = ["Depression", "Anxiety", "Stress"]
            values  = [depression, anxiety, stress]
            primary = labels[np.argmax(values)]

            st.markdown("")
            st.markdown(f'<div style="text-align:center"><span class="badge-primary badge-{primary.lower()}">Primary: {primary}</span></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            for col, label, pct in zip([c1, c2, c3], labels, values):
                col.markdown(f"""
                <div class="result-card card-{label.lower()}">
                    <h3>{label}</h3>
                    <div class="pct">{pct}%</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="disclaimer"><strong>Note:</strong> This assessment is for informational purposes only and does not constitute a medical diagnosis.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load survey model: {e}")


# ══════════════════════════════════════════════
#  TAB 2 — NLP
# ══════════════════════════════════════════════

with tab2:
    st.markdown("#### Write freely about how you're feeling")
    st.markdown('<p style="color:#6b6d80;font-size:14px;margin-bottom:16px">Our model analyses the emotional tone of your words.</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">Try an example</p>', unsafe_allow_html=True)
    ex_col1, ex_col2, ex_col3, _ = st.columns([1, 1, 1, 2])

    examples = {
        "ex_anx": "I feel nervous and anxious all the time.I keep worrying about things that might happen. My heart races and I feel uneasy for no reason.",
        "ex_dep": "I don't see the point in anything anymore. I used to love things but now I feel empty. Getting out of bed feels impossible and nothing brings me joy.",
        "ex_str": "I have so many deadlines and responsibilities piling up. I feel overwhelmed and exhausted from all the pressure and I can't keep up with everything.",
    }

    if ex_col1.button("😰 Anxiety",    key="ex_anx"): st.session_state["nlp_input"] = examples["ex_anx"]
    if ex_col2.button("😞 Depression", key="ex_dep"): st.session_state["nlp_input"] = examples["ex_dep"]
    if ex_col3.button("😤 Stress",     key="ex_str"): st.session_state["nlp_input"] = examples["ex_str"]

    user_text = st.text_area(
        "Your thoughts",
        value=st.session_state.get("nlp_input", ""),
        height=160,
        placeholder="Describe what's been on your mind lately…",
        label_visibility="collapsed",
        key="nlp_textarea"
    )

    st.markdown(f'<p style="color:#6b6d80;font-size:12px;text-align:right">{len(user_text)} characters</p>', unsafe_allow_html=True)

    col_btn2 = st.columns([1, 2, 1])
    with col_btn2[1]:
        nlp_btn = st.button("Analyse Text", key="nlp_btn")

    if nlp_btn:
        if len(user_text.strip()) < 20:
            st.warning("Please write at least a sentence so the model has enough to analyse.")
        else:
            try:
                nlp_model, tokenizer, le = load_nlp_model()

                MAX_LEN = 150

                def clean_text(text):
                    text = str(text).lower()
                    text = re.sub(r'[^a-zA-Z\s]', '', text)
                    return re.sub(r'\s+', ' ', text).strip()

                seq    = tokenizer.texts_to_sequences([clean_text(user_text)])
                padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
                proba  = nlp_model.predict(padded, verbose=0)[0]

                proba = proba / proba.sum()

                # le.classes_ = ['anxiety', 'depression', 'stress']
                classes      = list(le.classes_)                          # ['anxiety','depression','stress']
                display      = [c.capitalize() for c in classes]          # ['Anxiety','Depression','Stress']
                pct_values   = [round(float(p) * 100, 1) for p in proba]

                diff = round(100.0 - sum(pct_values), 1)
                pct_values[-1] = round(pct_values[-1] + diff, 1)

                primary   = display[int(np.argmax(pct_values))]
                badge_cls = f"badge-{primary.lower()}"

                st.markdown("")
                st.markdown(f'<div style="text-align:center"><span class="badge-primary {badge_cls}">Primary Indicator: {primary}</span></div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                for col, label, pct in zip([c1, c2, c3], display, pct_values):
                    col.markdown(f"""
                    <div class="result-card card-{label.lower()}">
                        <h3>{label}</h3>
                        <div class="pct">{pct}%</div>
                        <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="disclaimer"><strong>Note:</strong> This tool analyses text patterns only and is <strong>not</strong> a medical diagnosis. If you\'re struggling, please reach out to a qualified mental health professional.</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
