from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- APP CONFIG -----------------------------
st.set_page_config(
    page_title="FraudGuard | Intelligence Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data" / "creditcard.csv",
    ROOT / "creditcard.csv",
]
MODEL_CANDIDATES = [
    ROOT / "models" / "extra_trees_model.joblib",
    ROOT / "models" / "extra_trees_model.pkl",
    ROOT / "extra_trees_model.joblib",
    ROOT / "extra_trees_model.pkl",
]
THRESHOLD_DEFAULT = 0.30

# ----------------------------- DESIGN SYSTEM -----------------------------
st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #070b16;
  --panel: rgba(17, 27, 50, .72);
  --line: rgba(139, 166, 220, .20);
  --text: #edf4ff;
  --muted: #93a5c6;
  --cyan: #5ee7f5;
  --violet: #a78bfa;
  --mint: #54efbd;
  --rose: #ff6f91;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  color: var(--text);
  background:
    radial-gradient(ellipse at 8% 0%, rgba(37, 99, 235, .18), transparent 32%),
    radial-gradient(ellipse at 92% 12%, rgba(139, 92, 246, .15), transparent 28%),
    radial-gradient(ellipse at 65% 100%, rgba(20, 184, 166, .08), transparent 35%),
    var(--bg);
}
[data-testid="stHeader"] { background: rgba(7, 11, 22, .88); }
[data-testid="stAppViewBlockContainer"] {
  max-width: 1600px;
  padding-top: 1.05rem;
  padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15, 24, 46, .98), rgba(8, 13, 27, .99));
  border-right: 1px solid rgba(139, 166, 220, .17);
}
[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
h1 { font-size: clamp(1.8rem, 3vw, 2.6rem) !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.12rem !important; }
p, label, span { }
.small-muted { color: var(--muted); font-size: .88rem; }
.eyebrow {
  color: var(--cyan); font-size: .68rem; letter-spacing: .19em;
  font-weight: 800; text-transform: uppercase; margin-bottom: .55rem;
}
.hero {
  position: relative; overflow: hidden; padding: 25px 30px 23px;
  border: 1px solid rgba(94, 231, 245, .28); border-radius: 22px;
  background: linear-gradient(115deg, rgba(20, 43, 80, .82), rgba(32, 32, 68, .65) 60%, rgba(67, 35, 94, .52));
  box-shadow: 0 18px 55px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.06);
  backdrop-filter: blur(18px); margin-bottom: 22px;
}
.hero:after {
  content: ""; position: absolute; width: 230px; height: 230px; right: -80px; top: -110px;
  border-radius: 50%; background: rgba(94, 231, 245, .10); filter: blur(12px);
}
.hero h1 { margin: 0 0 9px 0 !important; line-height: 1.12; }
.hero p { color: #c4d4f2; max-width: 850px; margin: 0; line-height: 1.7; }
.hero-meta { margin-top: 15px; color: #b7c7e7; font-size: .78rem; }
.pill {
  display: inline-block; border: 1px solid rgba(84,239,189,.32);
  background: rgba(84,239,189,.09); color: #82f6cd; border-radius: 99px;
  padding: 5px 10px; font-size: .72rem; font-weight: 700; margin-right: 6px;
}
.kpi {
  min-height: 126px; padding: 18px 18px 16px; border-radius: 17px;
  background: linear-gradient(145deg, rgba(24, 38, 68, .82), rgba(12, 19, 37, .82));
  border: 1px solid var(--line); backdrop-filter: blur(16px);
  box-shadow: 0 12px 32px rgba(0,0,0,.16); position: relative; overflow: hidden;
}
.kpi:before {
  content: ""; position: absolute; top: 0; left: 16px; right: 16px; height: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--violet), transparent);
}
.kpi-label { color: #9eb0d1; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.kpi-value { color: #f7faff; font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.45rem, 2vw, 2rem); font-weight: 700; margin-top: 9px; letter-spacing: -.04em; }
.kpi-note { color: #91a5c8; font-size: .74rem; margin-top: 7px; }
.glass {
  border: 1px solid var(--line); border-radius: 18px;
  background: linear-gradient(145deg, rgba(20, 31, 56, .78), rgba(10, 17, 33, .78));
  padding: 18px 20px; box-shadow: 0 12px 30px rgba(0,0,0,.12);
  backdrop-filter: blur(14px);
}
.section-kicker { color: var(--cyan); text-transform: uppercase; letter-spacing: .16em; font-size: .68rem; font-weight: 800; }
div[data-testid="stMetric"] {
  border: 1px solid var(--line); border-radius: 15px; padding: 14px 16px;
  background: rgba(18, 29, 52, .68);
}
div[data-testid="stMetricLabel"] { color: #9eb0d1; }
div[data-testid="stMetricValue"] { color: #f7faff; font-family: 'Space Grotesk', sans-serif; }
.stButton > button, .stDownloadButton > button {
  border-radius: 11px; border: 1px solid rgba(94,231,245,.36);
  color: #f5fbff; background: linear-gradient(110deg, #1769d2, #6246d9);
  font-weight: 700; min-height: 2.65rem; transition: .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--cyan); color: white; box-shadow: 0 0 22px rgba(94,231,245,.18);
  transform: translateY(-1px);
}
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
  background: rgba(12, 21, 41, .85); border-color: var(--line); border-radius: 10px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
hr { border-color: rgba(139,166,220,.16) !important; }
[data-testid="stAlert"] { border-radius: 13px; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
@media (max-width: 800px) {
  [data-testid="stAppViewBlockContainer"] { padding: .7rem 1rem 2rem; }
  .hero { padding: 20px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------- HELPERS -----------------------------
def money(value):
    return f"${value:,.2f}"

def fmt_int(value):
    return f"{int(value):,}"

def glass_kpi(label, value, note):
    st.markdown(
        f"""<div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def plotly_dark(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=35, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#dce8ff", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
        hoverlabel=dict(bgcolor="#111b32", bordercolor="#5ee7f5", font_color="#f5f8ff"),
    )
    fig.update_xaxes(gridcolor="rgba(139,166,220,.12)", zerolinecolor="rgba(139,166,220,.12)")
    fig.update_yaxes(gridcolor="rgba(139,166,220,.12)", zerolinecolor="rgba(139,166,220,.12)")
    return fig

@st.cache_data(show_spinner=False)
def load_data(path_string):
    df = pd.read_csv(path_string)
    required = {"Class", "Amount"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must include at least the 'Amount' and 'Class' columns.")
    return df

@st.cache_resource(show_spinner="Preparing the Extra Trees model…")
def get_model_and_evaluation(data_path):
    """Load a compatible saved model if available; otherwise train the project model once."""
    df = load_data(data_path)
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Class"].astype(int)

    # Recreate a stratified split; fit only on the training partition.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    loaded_saved_model = False
    if model_path is not None:
        try:
            model = joblib.load(model_path)
            if hasattr(model, "predict_proba") and hasattr(model, "feature_names_in_"):
                if list(model.feature_names_in_) == feature_cols:
                    loaded_saved_model = True
                else:
                    model = None
            else:
                model = None
        except Exception:
            model = None
    else:
        model = None

    if model is None:
        model = ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    test_ap = average_precision_score(y_test, test_prob)
    test_roc = roc_auc_score(y_test, test_prob)
    return {
        "df": df,
        "model": model,
        "features": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "test_prob": test_prob,
        "test_ap": test_ap,
        "test_roc": test_roc,
        "loaded_saved_model": loaded_saved_model,
        "model_path": str(model_path) if model_path else None,
    }

def local_feature_sensitivity(model, input_df, features, reference_df, top_n=10):
    """One-feature-at-a-time score changes versus dataset medians; not a causal explanation."""
    baseline_row = reference_df[features].median(numeric_only=True)
    original_score = float(model.predict_proba(input_df[features])[:, 1][0])
    rows = []
    for feature in features:
        changed = input_df[features].copy()
        changed.loc[changed.index[0], feature] = float(baseline_row.get(feature, 0.0))
        changed_score = float(model.predict_proba(changed)[:, 1][0])
        rows.append({
            "Feature": feature,
            "Current value": float(input_df.iloc[0][feature]),
            "Reference (median)": float(baseline_row.get(feature, 0.0)),
            "Score change if replaced": changed_score - original_score,
            "Absolute change": abs(changed_score - original_score),
        })
    return original_score, pd.DataFrame(rows).sort_values("Absolute change", ascending=False).head(top_n)


def render_local_explanation(model, input_df, features, reference_df, key_suffix):
    """Render an approximate local sensitivity view for one scored transaction."""
    with st.expander("Why did the model give this score?", expanded=True):
        st.caption(
            "Each feature is replaced individually with its dataset median and the fraud score is recalculated. "
            "This is a sensitivity check, not SHAP or proof of causation; correlated features and interactions matter."
        )
        try:
            _, sensitivity = local_feature_sensitivity(model, input_df, features, reference_df, top_n=10)
            plot_df = sensitivity.sort_values("Score change if replaced")
            fig = px.bar(
                plot_df, x="Score change if replaced", y="Feature", orientation="h",
                color="Score change if replaced",
                color_continuous_scale=["#43c6f4", "#a78bfa", "#ff6f91"],
                title="Features with the largest score changes",
                hover_data=["Current value", "Reference (median)"],
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#dce8ff", opacity=.7)
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Change in fraud score", yaxis_title="")
            st.plotly_chart(plotly_dark(fig, max(320, len(plot_df) * 34)), width="stretch",
                            config={"displayModeBar": False}, key=f"local_explanation_{key_suffix}")
            st.dataframe(sensitivity[["Feature", "Current value", "Reference (median)", "Score change if replaced"]],
                         width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"Could not generate this explanation for the current model/input: {exc}")


def threshold_metrics(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return predicted, {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                       "precision": precision, "recall": recall, "f1": f1}

def section_header(kicker, title, subtitle=None):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 18px">
          <div style="font-size:2.1rem">🛡️</div>
          <div style="font-family:'Space Grotesk';font-size:1.55rem;font-weight:700;color:#f5f8ff">FraudGuard</div>
          <div style="color:#5ee7f5;font-size:.68rem;letter-spacing:.18em;font-weight:800;margin-top:5px">FINANCIAL SECURITY AI</div>
        </div>
        <hr>
        <div class="section-kicker" style="margin:18px 0 8px">Workspace</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Overview", "Fraud Detection", "Model Evaluation", "Model Comparison", "Dataset Explorer", "Insights & Features", "About Project"],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">System status</div>', unsafe_allow_html=True)

# ----------------------------- DATA + MODEL -----------------------------
data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if data_path is None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Setup required</div>
          <h1>🛡️ FraudGuard Analytics</h1>
          <p>Place the Kaggle dataset at <code>data/creditcard.csv</code> inside the same project folder as this app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("Dataset not found. Expected `data/creditcard.csv` or `creditcard.csv` next to `app.py`.")
    st.code("credit-card-fraud-detection/\n├── app.py\n├── data/\n│   └── creditcard.csv\n└── requirements.txt", language="text")
    st.stop()

try:
    bundle = get_model_and_evaluation(str(data_path))
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.stop()

df = bundle["df"]
model = bundle["model"]
features = bundle["features"]
fraud_count = int((df["Class"] == 1).sum())
legit_count = int((df["Class"] == 0).sum())
fraud_rate = fraud_count / len(df) * 100

with st.sidebar:
    st.markdown(
        f"""
        <div class="glass" style="padding:14px;margin-top:12px">
          <div style="color:#9eb0d1;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em">Active estimator</div>
          <div style="font-size:1rem;font-weight:700;margin:7px 0">Extra Trees</div>
          <div style="color:#54efbd;font-size:.8rem">● Ready for inference</div>
          <hr>
          <div style="color:#9eb0d1;font-size:.75rem">Dataset rows</div>
          <div style="font-size:1.2rem;font-weight:700">{len(df):,}</div>
          <div style="color:#9eb0d1;font-size:.75rem;margin-top:8px">Fraud rate</div>
          <div style="font-size:1.1rem;font-weight:700">{fraud_rate:.3f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Compact top header, consistent on every page.
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Machine learning &nbsp; • &nbsp; anomaly detection &nbsp; • &nbsp; financial security</div>
      <h1>🛡️ FraudGuard <span style="color:#5ee7f5">Analytics</span></h1>
      <p>Explore transaction patterns, inspect model performance, and screen example transactions in an interactive fraud analytics workspace.</p>
      <div class="hero-meta"><span class="pill">● MODEL READY</span><span class="pill">EXTRA TREES CLASSIFIER</span><span class="pill">EDUCATIONAL DEMO</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- OVERVIEW -----------------------------
if page == "Overview":
    section_header("Executive snapshot", "Fraud monitoring overview", "A high-level view of the dataset and held-out model performance.")
    cols = st.columns(4, gap="medium")
    with cols[0]: glass_kpi("Transactions", fmt_int(len(df)), "Rows in loaded dataset")
    with cols[1]: glass_kpi("Fraud cases", fmt_int(fraud_count), f"{fraud_rate:.3f}% of all transactions")
    with cols[2]: glass_kpi("Average precision", f"{bundle['test_ap']:.3f}", "Held-out test split")
    with cols[3]: glass_kpi("ROC AUC", f"{bundle['test_roc']:.3f}", "Held-out test split")

    st.write("")
    left, right = st.columns([1.05, 1.35], gap="large")
    with left:
        section_header("Dataset composition", "Legitimate vs fraud")
        class_df = pd.DataFrame({"Transaction type": ["Legitimate", "Fraud"], "Count": [legit_count, fraud_count]})
        fig = px.pie(class_df, names="Transaction type", values="Count", hole=.72,
                     color="Transaction type", color_discrete_map={"Legitimate":"#42d6c5","Fraud":"#ff6f91"})
        fig.update_traces(textinfo="percent", textfont_size=13, marker=dict(line=dict(color="#0a1020", width=3)))
        fig.add_annotation(text=f"{len(df):,}<br><span style='font-size:11px'>transactions</span>",
                           x=.5, y=.5, showarrow=False, font=dict(size=22, color="#f4f7ff"))
        st.plotly_chart(plotly_dark(fig, 330), width="stretch", config={"displayModeBar": False})
    with right:
        section_header("Transaction behaviour", "Amount distribution", "Median and tail behaviour can differ; use the log scale to inspect the long tail.")
        log_amount = st.checkbox("Use logarithmic amount axis", value=True, key="overview_log")
        amount_df = df[["Amount", "Class"]].copy()
        amount_df["Transaction type"] = amount_df["Class"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.histogram(amount_df.sample(min(30000, len(amount_df)), random_state=42),
                           x="Amount", color="Transaction type", nbins=65, barmode="overlay",
                           opacity=.72, color_discrete_map={"Legitimate":"#43c6f4","Fraud":"#ff6f91"})
        if log_amount:
            fig.update_xaxes(type="log")
        fig.update_layout(xaxis_title="Transaction amount", yaxis_title="Number of transactions")
        st.plotly_chart(plotly_dark(fig, 330), width="stretch", config={"displayModeBar": False})

    a, b, c = st.columns(3, gap="medium")
    with a:
        st.markdown("**Model snapshot**")
        st.metric("Fraud recall at 0.30", f"{threshold_metrics(bundle['y_test'], bundle['test_prob'], .30)[1]['recall']:.1%}")
        st.caption("Share of fraud cases caught at the demonstration threshold.")
    with b:
        st.markdown("**False-alert count**")
        st.metric("False positives at 0.30", fmt_int(threshold_metrics(bundle['y_test'], bundle['test_prob'], .30)[1]["fp"]))
        st.caption("Legitimate transactions flagged as fraud on the held-out split.")
    with c:
        st.markdown("**Interpretation**")
        st.markdown("Fraud is rare in this dataset. Precision, recall, and average precision are more informative than accuracy alone.")

# ----------------------------- FRAUD DETECTION -----------------------------
elif page == "Fraud Detection":
    section_header(
        "Interactive screening",
        "Fraud detection lab",
        "Score a labeled historical transaction or explore a custom transaction using the fitted Extra Trees model.",
    )
    st.info("Educational demonstration only. The score is the model's output, not a guaranteed real-world probability.")

    mode = st.radio(
        "Choose your workflow",
        ["Historical transaction", "Manual feature entry"],
        horizontal=True,
        key="fraud_input_mode",
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=THRESHOLD_DEFAULT,
        step=0.05,
        key="fraud_threshold",
        help="A lower threshold flags more transactions; a higher threshold usually reduces alerts but can miss more fraud.",
    )
    st.caption(f"Current rule: flag as potential fraud when the model score is at least **{threshold:.0%}**.")

    if mode == "Historical transaction":
        picker_col, detail_col = st.columns([1, 1], gap="large")
        with picker_col:
            row_index = st.number_input(
                "Dataset row index", min_value=0, max_value=len(df) - 1,
                value=min(0, len(df) - 1), step=1, key="historical_row_index",
            )
            row = df.iloc[int(row_index)]
            st.metric("Transaction amount", money(float(row["Amount"])))
            st.metric("Dataset label", "Fraud" if int(row["Class"]) == 1 else "Legitimate")
        with detail_col:
            st.markdown("#### Transaction preview")
            preview_cols = [c for c in ["Time", "Amount", "V1", "V2", "V3", "Class"] if c in df.columns]
            preview = row[preview_cols].to_frame(name="Value")
            st.dataframe(preview, width="stretch", height=270)

        if st.button("Analyze transaction", type="primary", width="stretch", key="analyze_historical"):
            input_df = df.loc[[int(row_index)], features].copy()
            actual = int(row["Class"])
            prob = float(model.predict_proba(input_df)[0, 1])
            is_fraud = prob >= threshold
            _, metrics = threshold_metrics(np.array([actual]), np.array([prob]), threshold)
            st.markdown("---")
            st.markdown("### Analysis result")
            result_col, score_col, truth_col = st.columns(3, gap="medium")
            with result_col:
                if is_fraud:
                    st.error("🚨 Potential fraud detected")
                else:
                    st.success("✓ Likely legitimate")
                st.caption("Prediction based on the selected threshold.")
            with score_col:
                st.metric("Model fraud score", f"{prob:.2%}")
                st.progress(float(np.clip(prob, 0, 1)))
            with truth_col:
                st.metric("Actual dataset label", "Fraud" if actual else "Legitimate")
                st.metric("Historical prediction", "Correct" if int(is_fraud) == actual else "Incorrect",
                          delta="Matches label" if int(is_fraud) == actual else "Differs from label",
                          delta_color="normal" if int(is_fraud) == actual else "inverse")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%", "font": {"color": "#edf4ff", "size": 32}},
                title={"text": "Fraud score", "font": {"color": "#dce8ff", "size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#93a5c6"},
                    "bar": {"color": "#ff6f91" if is_fraud else "#54efbd"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(139,166,220,.25)",
                    "steps": [
                        {"range": [0, threshold * 100], "color": "rgba(84,239,189,.10)"},
                        {"range": [threshold * 100, 100], "color": "rgba(255,111,145,.12)"},
                    ],
                    "threshold": {"line": {"color": "#5ee7f5", "width": 4}, "thickness": .8, "value": threshold * 100},
                },
            ))
            st.plotly_chart(plotly_dark(gauge, 270), width="stretch", config={"displayModeBar": False})
            st.caption("The selected row's actual label is shown only because this is a labeled historical dataset.")
            render_local_explanation(model, input_df, features, df, f"historical_{int(row_index)}")

    else:
        st.warning("The V1–V28 columns are anonymized PCA features. For a meaningful test, use values from the dataset; arbitrary values may be unrealistic.")
        defaults = df[features].median(numeric_only=True)
        with st.form("manual_transaction_form", border=True):
            st.markdown("#### Transaction details")
            amount = st.number_input("Transaction amount", min_value=0.0,
                                     value=float(df["Amount"].median()), step=1.0, format="%.2f")
            st.markdown("#### Anonymized model features")
            vals = {}
            feature_cols = st.columns(3, gap="medium")
            for i, feat in enumerate(features):
                with feature_cols[i % 3]:
                    vals[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)),
                                                 format="%.5f", key=f"manual_feature_{feat}")
            submitted = st.form_submit_button("Run fraud analysis", type="primary", width="stretch")
        if submitted:
            vals["Amount"] = amount
            input_df = pd.DataFrame([[vals[c] for c in features]], columns=features)
            prob = float(model.predict_proba(input_df)[0, 1])
            is_fraud = prob >= threshold
            st.markdown("---")
            st.markdown("### Prediction result")
            result_col, score_col = st.columns([1, 1], gap="large")
            with result_col:
                if is_fraud:
                    st.error("🚨 Potential fraud detected")
                else:
                    st.success("✓ Likely legitimate")
                st.metric("Transaction amount", money(amount))
                st.caption("No actual label exists for manually entered transactions.")
            with score_col:
                st.metric("Model fraud score", f"{prob:.2%}")
                st.progress(float(np.clip(prob, 0, 1)))
                st.plotly_chart(plotly_dark(go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%", "font": {"color": "#edf4ff", "size": 28}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ff6f91" if is_fraud else "#54efbd"},
                           "steps": [{"range": [0, threshold * 100], "color": "rgba(84,239,189,.10)"},
                                     {"range": [threshold * 100, 100], "color": "rgba(255,111,145,.12)"}],
                           "threshold": {"line": {"color": "#5ee7f5", "width": 3}, "value": threshold * 100}},
                )), 250), width="stretch", config={"displayModeBar": False})
            render_local_explanation(model, input_df, features, df, "manual")

# ----------------------------- MODEL EVALUATION -----------------------------
elif page == "Model Evaluation":
    section_header("Model performance", "Evaluation laboratory", "Held-out metrics and threshold trade-offs for the current Extra Trees estimator.")
    threshold = st.slider("Decision threshold", .05, .95, THRESHOLD_DEFAULT, .05, key="eval_threshold")
    pred, m = threshold_metrics(bundle["y_test"], bundle["test_prob"], threshold)
    k = st.columns(4, gap="medium")
    with k[0]: glass_kpi("Precision", f"{m['precision']:.1%}", "Fraud alerts that are correct")
    with k[1]: glass_kpi("Recall", f"{m['recall']:.1%}", "Fraud cases detected")
    with k[2]: glass_kpi("F1 score", f"{m['f1']:.1%}", "Precision/recall balance")
    with k[3]: glass_kpi("Average precision", f"{bundle['test_ap']:.3f}", "Across score thresholds")

    left, right = st.columns(2, gap="large")
    with left:
        section_header("Confusion matrix", "Actual vs predicted")
        cm = confusion_matrix(bundle["y_test"], pred, labels=[0, 1])
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted legitimate", "Predicted fraud"],
                                   y=["Actual legitimate", "Actual fraud"], text=cm, texttemplate="%{text}",
                                   colorscale=[[0, "#101b34"], [.5, "#247ba0"], [1, "#a78bfa"]],
                                   showscale=False, hovertemplate="Count: %{z}<extra></extra>"))
        st.plotly_chart(plotly_dark(fig, 340), width="stretch", config={"displayModeBar": False})
        st.caption(f"TN {m['tn']:,} · FP {m['fp']:,} · FN {m['fn']:,} · TP {m['tp']:,}")
    with right:
        section_header("Precision–recall", "Ranking quality")
        precision, recall, _ = precision_recall_curve(bundle["y_test"], bundle["test_prob"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", fill="tozeroy",
                                 line=dict(color="#5ee7f5", width=3), fillcolor="rgba(94,231,245,.10)",
                                 name="Precision–recall"))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(plotly_dark(fig, 340), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Classification report")
    report = classification_report(bundle["y_test"], pred, labels=[0, 1],
                                   target_names=["Legitimate", "Fraud"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.4f}"), width="stretch")
    st.caption("The held-out test set is for final evaluation. Avoid repeatedly choosing thresholds based on test results; use validation data for tuning.")

# ----------------------------- MODEL COMPARISON -----------------------------
elif page == "Model Comparison":
    section_header(
        "Experiment tracker",
        "Model comparison",
        "Compare previously recorded validation Average Precision (AP) scores. Higher is better.",
    )
    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest (baseline)",
            "Random Forest (tuned)",
            "Extra Trees",
        ],
        "Validation AP": [0.6857, 0.7970, 0.8070, 0.8090],
    }).sort_values("Validation AP", ascending=True)

    best_score = float(comparison_df["Validation AP"].max())
    best_model = comparison_df.loc[comparison_df["Validation AP"].idxmax(), "Model"]
    metric_cols = st.columns(3, gap="medium")
    with metric_cols[0]:
        glass_kpi("Top recorded model", best_model, "By validation AP")
    with metric_cols[1]:
        glass_kpi("Best validation AP", f"{best_score:.3f}", "Recorded experiment")
    with metric_cols[2]:
        glass_kpi("Models compared", str(len(comparison_df)), "Historical results")

    fig = px.bar(
        comparison_df,
        x="Validation AP",
        y="Model",
        orientation="h",
        text="Validation AP",
        range_x=[0, 1],
        color="Validation AP",
        color_continuous_scale=["#43c6f4", "#a78bfa", "#54efbd"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Validation Average Precision", yaxis_title="")
    st.plotly_chart(plotly_dark(fig, max(360, len(comparison_df) * 75)), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Recorded validation results")
    st.dataframe(
        comparison_df.sort_values("Validation AP", ascending=False).reset_index(drop=True).style.format({"Validation AP": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "These are previously recorded validation scores, rounded for display. They are not recalculated by this page. "
        "For a rigorous comparison, rerun each experiment with the same stratified train/validation split and preprocessing. "
        "Keep the test set for the final evaluation rather than choosing a model based on test performance."
    )

# ----------------------------- DATASET EXPLORER -----------------------------
elif page == "Dataset Explorer":
    section_header("Explore the data", "Dataset explorer", "Filter transactions and inspect records from the original labeled dataset.")
    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    with c1:
        classes = st.multiselect("Transaction class", ["Legitimate", "Fraud"], default=["Legitimate", "Fraud"])
    with c2:
        min_amount = float(df["Amount"].min())
        max_amount = float(df["Amount"].max())
        amount_range = st.slider("Amount range", min_value=min_amount, max_value=max_amount,
                                 value=(min_amount, max_amount))
    with c3:
        sample_size = st.selectbox("Rows to display", [25, 50, 100, 250], index=1)
    class_values = [0 if x == "Legitimate" else 1 for x in classes]
    filtered = df[df["Class"].isin(class_values)]
    filtered = filtered[filtered["Amount"].between(amount_range[0], amount_range[1])]
    st.metric("Matching transactions", f"{len(filtered):,}")
    display_df = filtered.head(sample_size).copy()
    display_df["Class"] = display_df["Class"].map({0: "Legitimate", 1: "Fraud"})
    st.dataframe(display_df, width="stretch", height=360)
    st.download_button("Download filtered CSV", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="fraudguard_filtered_transactions.csv", mime="text/csv")
    st.markdown("#### Amount by transaction class")
    amount_plot = filtered.copy()
    amount_plot["Transaction type"] = amount_plot["Class"].map({0: "Legitimate", 1: "Fraud"})
    fig = px.box(amount_plot.sample(min(15000, len(amount_plot)), random_state=42),
                 x="Transaction type", y="Amount", color="Transaction type", points=False,
                 color_discrete_map={"Legitimate":"#43c6f4", "Fraud":"#ff6f91"})
    fig.update_yaxes(type="log", title="Amount (log scale)")
    st.plotly_chart(plotly_dark(fig, 360), width="stretch", config={"displayModeBar": False})

# ----------------------------- INSIGHTS -----------------------------
elif page == "Insights & Features":
    section_header("Explainability", "Insights & feature importance", "Inspect the model's built-in feature importance when supported.")
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False)
        n = st.slider("Features to display", 5, min(30, len(imp)), min(15, len(imp)))
        fig = px.bar(imp.head(n).sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", color="Importance", color_continuous_scale=["#43c6f4", "#a78bfa", "#ff75c8"])
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Relative importance", yaxis_title="")
        st.plotly_chart(plotly_dark(fig, max(360, n * 25)), width="stretch", config={"displayModeBar": False})
        st.dataframe(imp.head(n), width="stretch")
        st.caption("Feature importance describes how the fitted tree ensemble uses features; it does not prove causation or explain an individual prediction.")
    else:
        st.info("This estimator does not expose built-in feature importance.")

    st.markdown("#### Feature distributions")
    selected_feature = st.selectbox("Choose a feature", features, index=features.index("Amount") if "Amount" in features else 0)
    chart_df = df[[selected_feature, "Class"]].copy()
    chart_df["Class label"] = chart_df["Class"].map({0: "Legitimate", 1: "Fraud"})
    fig = px.histogram(chart_df.sample(min(25000, len(chart_df)), random_state=42),
                       x=selected_feature, color="Class label", nbins=60, barmode="overlay",
                       opacity=.7, color_discrete_map={"Legitimate":"#43c6f4", "Fraud":"#ff6f91"})
    st.plotly_chart(plotly_dark(fig, 350), width="stretch", config={"displayModeBar": False})

# ----------------------------- ABOUT -----------------------------
elif page == "About Project":
    section_header("Project notes", "About FraudGuard", "An educational machine-learning project for exploring imbalanced classification.")
    st.markdown(
        """
        <div class="glass">
          <h3>What this project demonstrates</h3>
          <p>FraudGuard combines exploratory analysis, model scoring, threshold selection, and evaluation visuals in a small interactive application.</p>
          <ul>
            <li><b>Dataset:</b> anonymized credit-card transactions with a binary fraud label.</li>
            <li><b>Estimator:</b> Extra Trees classifier with class weighting.</li>
            <li><b>Evaluation:</b> average precision, ROC AUC, precision, recall, F1, and a confusion matrix.</li>
            <li><b>Threshold:</b> configurable so users can see the trade-off between catching fraud and generating false alerts.</li>
          </ul>
          <h3>Important limitations</h3>
          <p>Historical benchmark performance does not guarantee performance on new transactions. Dataset features are anonymized, model scores may not be calibrated probabilities, and this app is not suitable for real financial decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Evaluation snapshot")
    st.write(f"- Held-out average precision: **{bundle['test_ap']:.4f}**")
    st.write(f"- Held-out ROC AUC: **{bundle['test_roc']:.4f}**")
    st.write(f"- Dataset size: **{len(df):,}** rows")
    st.write(f"- Fraud cases: **{fraud_count:,}**")
    if bundle["loaded_saved_model"]:
        st.success(f"Loaded saved model from `{bundle['model_path']}`.")
    else:
        st.info("No compatible saved model was found. The app trained an Extra Trees model on the training split and cached it for this session.")

st.markdown(
    """
    <hr>
    <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#7184a8;font-size:.75rem">
      <span>🛡️ FraudGuard Analytics · Educational project</span>
      <span>Built with Streamlit, scikit-learn, Plotly & pandas</span>
    </div>
    """,
    unsafe_allow_html=True,
)
