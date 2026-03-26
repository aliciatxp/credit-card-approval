import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTENC
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@300;400&display=swap');

  html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

  .stApp { background-color: #0e0e0e; color: #f0ece4; }

  .metric-card {
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7a7570;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f0ece4;
  }
  .approved-box {
    background: linear-gradient(135deg, rgba(6,214,160,0.15), rgba(6,214,160,0.05));
    border: 2px solid #06d6a0;
    border-radius: 6px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .rejected-box {
    background: linear-gradient(135deg, rgba(255,95,46,0.15), rgba(255,95,46,0.05));
    border: 2px solid #ff5f2e;
    border-radius: 6px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .verdict-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
  }
  .section-tag {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff5f2e;
    margin-bottom: 0.3rem;
  }
  .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9a9590 !important;
  }
  div[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
  }
  .stButton > button {
    background: #ff5f2e;
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 3px;
    padding: 0.7rem 2rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #e04e20;
    transform: translate(-2px, -2px);
    box-shadow: 4px 4px 0 #ffd166;
  }
  hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#111111",
    "axes.facecolor": "#181818",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#9a9590",
    "xtick.color": "#9a9590",
    "ytick.color": "#9a9590",
    "text.color": "#f0ece4",
    "grid.color": "#2a2a2a",
    "grid.linestyle": "--",
    "font.family": "monospace",
})

# ── Constants ──────────────────────────────────────────────────────────────────
EDUCATION_ORDER = [
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]
OHE_COLS = ["NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]
INCOME_TYPES    = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
FAMILY_STATUSES = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPES   = ["House / apartment", "With parents", "Municipal apartment",
                   "Rented apartment", "Office apartment", "Co-op apartment"]
OCCUPATION_TYPES = [
    "Laborers", "Core staff", "Accountants", "Managers", "Drivers",
    "Sales staff", "Cleaning staff", "Cooking staff", "Private service staff",
    "Medicine staff", "Security staff", "High skill tech staff", "Waiters/barmen staff",
    "Low-skill Laborers", "Realty agents", "Secretaries", "IT staff", "HR staff",
    "Unknown", "None",
]

# ── Data generation (synthetic, matching notebook distributions) ───────────────
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    n_bad = int(n * 0.015)
    n_good = n - n_bad

    def make_block(size, label):
        return pd.DataFrame({
            "CODE_GENDER":          rng.choice([0, 1], size),
            "FLAG_OWN_CAR":         rng.choice([0, 1], size),
            "FLAG_OWN_REALTY":      rng.choice([0, 1], size),
            "CNT_FAM_MEMBERS":      rng.integers(1, 6, size).clip(max=5),
            "AMT_INCOME_TOTAL":     rng.uniform(50000, 500000, size),
            "NAME_INCOME_TYPE":     rng.choice(INCOME_TYPES, size),
            "NAME_EDUCATION_TYPE":  rng.choice(EDUCATION_ORDER, size),
            "NAME_FAMILY_STATUS":   rng.choice(FAMILY_STATUSES, size),
            "NAME_HOUSING_TYPE":    rng.choice(HOUSING_TYPES, size),
            "OCCUPATION_TYPE":      rng.choice(OCCUPATION_TYPES, size),
            "AGE":                  rng.integers(20, 70, size),
            "EMPLOYMENT_DURATION":  rng.integers(-1, 20, size),
            "FLAG_WORK_PHONE":      rng.choice([0, 1], size),
            "FLAG_PHONE":           rng.choice([0, 1], size),
            "FLAG_EMAIL":           rng.choice([0, 1], size),
            "CNT_CHILDREN":         rng.integers(0, 4, size),
            "Classification":       label,
        })

    df = pd.concat([make_block(n_good, "1"), make_block(n_bad, "0")], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Pipeline ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_pipeline():
    df = generate_synthetic_data()
    X = df.drop("Classification", axis=1).drop("CNT_CHILDREN", axis=1)
    y = df["Classification"]

    X["CNT_FAM_MEMBERS"] = X["CNT_FAM_MEMBERS"].clip(upper=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    oe  = OrdinalEncoder(categories=[EDUCATION_ORDER])

    def encode(df_in, fit=False):
        df_in = df_in.copy()
        if fit:
            ohe_arr = ohe.fit_transform(df_in[OHE_COLS])
            df_in["NAME_EDUCATION_TYPE_ORDINAL"] = oe.fit_transform(df_in[["NAME_EDUCATION_TYPE"]])
        else:
            ohe_arr = ohe.transform(df_in[OHE_COLS])
            df_in["NAME_EDUCATION_TYPE_ORDINAL"] = oe.transform(df_in[["NAME_EDUCATION_TYPE"]])
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(OHE_COLS), index=df_in.index)
        df_in  = df_in.drop(columns=OHE_COLS + ["NAME_EDUCATION_TYPE"])
        return pd.concat([df_in, ohe_df], axis=1)

    X_train_enc = encode(X_train, fit=True)
    X_test_enc  = encode(X_test,  fit=False)

    cat_idx = [i for i, c in enumerate(X_train_enc.columns)
               if X_train_enc[c].nunique() <= 20]

    smote = SMOTENC(categorical_features=cat_idx, random_state=42)
    X_res, y_res = smote.fit_resample(X_train_enc, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train_enc.columns)

    sc  = StandardScaler()
    pca = PCA(n_components=30, random_state=42)

    X_res_sc   = sc.fit_transform(X_res)
    X_test_sc  = sc.transform(X_test_enc)
    X_train_pca = pca.fit_transform(X_res_sc)
    X_test_pca  = pca.transform(X_test_sc)

    models = {
        "Decision Tree":     DecisionTreeClassifier(criterion="entropy", max_depth=None,
                                                    min_samples_leaf=2, min_samples_split=10,
                                                    random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=100, max_features="log2",
                                                    max_depth=None, random_state=42),
        "Logistic Regression": LogisticRegression(C=1, penalty="l2", max_iter=100,
                                                  solver="liblinear", random_state=42),
    }
    trained, metrics = {}, {}
    for name, m in models.items():
        m.fit(X_train_pca, y_res)
        y_pred = m.predict(X_test_pca)
        trained[name] = m
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred, pos_label="1", zero_division=0), 3),
            "Recall":    round(recall_score(y_test, y_pred, pos_label="1", zero_division=0), 3),
            "F1":        round(f1_score(y_test, y_pred, pos_label="1", zero_division=0), 3),
            "ROC-AUC":   round(roc_auc_score(y_test, m.predict_proba(X_test_pca)[:, 1]), 3),
            "y_pred":    y_pred,
            "y_test":    y_test.values,
            "proba":     m.predict_proba(X_test_pca)[:, 1],
        }

    return trained, metrics, ohe, oe, sc, pca, X_train_enc.columns.tolist()


def encode_input(row_dict, ohe, oe, sc, pca, feature_cols):
    df_in = pd.DataFrame([row_dict])
    df_in["CNT_FAM_MEMBERS"] = df_in["CNT_FAM_MEMBERS"].clip(upper=5)
    ohe_arr = ohe.transform(df_in[OHE_COLS])
    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(OHE_COLS))
    df_in["NAME_EDUCATION_TYPE_ORDINAL"] = oe.transform(df_in[["NAME_EDUCATION_TYPE"]])
    df_in   = df_in.drop(columns=OHE_COLS + ["NAME_EDUCATION_TYPE"])
    df_in   = pd.concat([df_in.reset_index(drop=True), ohe_df], axis=1)
    df_in   = df_in.reindex(columns=feature_cols, fill_value=0)
    scaled  = sc.transform(df_in)
    return pca.transform(scaled)


# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Training models on synthetic data…"):
    trained_models, all_metrics, ohe, oe, sc, pca, feature_cols = train_pipeline()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — applicant form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💳 Applicant Details")
    st.markdown("---")

    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    age    = st.slider("Age", 18, 75, 32)
    st.markdown("---")

    income = st.number_input("Annual Income (SGD)", min_value=10000, max_value=1_000_000,
                              value=120000, step=5000)
    income_type = st.selectbox("Income Type", INCOME_TYPES)
    employment  = st.slider("Employment Duration (years)", -1, 30, 5,
                             help="-1 = retired / not employed")
    st.markdown("---")

    education    = st.selectbox("Education Level", EDUCATION_ORDER, index=3)
    family_status = st.selectbox("Family Status", FAMILY_STATUSES)
    fam_members  = st.slider("Family Members", 1, 5, 2)
    st.markdown("---")

    housing      = st.selectbox("Housing Type", HOUSING_TYPES)
    occupation   = st.selectbox("Occupation Type", OCCUPATION_TYPES)
    st.markdown("---")

    own_car      = st.checkbox("Owns a Car")
    own_realty   = st.checkbox("Owns Property")
    work_phone   = st.checkbox("Has Work Phone")
    phone        = st.checkbox("Has Home Phone")
    email        = st.checkbox("Has Email")
    st.markdown("---")

    model_choice = st.selectbox(
        "Model",
        ["Decision Tree", "Random Forest", "Logistic Regression"],
        index=0,
        help="Decision Tree has the highest macro recall — best for minimising missed bad applicants."
    )
    predict_btn = st.button("🔍 Predict Approval", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-tag">Portfolio Project · Alicia Tan</p>', unsafe_allow_html=True)
st.title("Credit Card Approval Predictor")
st.markdown(
    "An end-to-end ML pipeline that handles class imbalance via **SMOTE-NC**, "
    "reduces dimensionality with **PCA**, and compares three classifiers tuned with **GridSearchCV**. "
    "Fill in the applicant form on the left and hit **Predict**."
)
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "🔬 Methodology"])

# ─────────────────────────────  TAB 1: PREDICTION  ────────────────────────────
with tab1:
    if predict_btn:
        applicant = {
            "CODE_GENDER":         1 if gender == "Male" else 0,
            "FLAG_OWN_CAR":        int(own_car),
            "FLAG_OWN_REALTY":     int(own_realty),
            "CNT_FAM_MEMBERS":     fam_members,
            "AMT_INCOME_TOTAL":    income,
            "NAME_INCOME_TYPE":    income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS":  family_status,
            "NAME_HOUSING_TYPE":   housing,
            "OCCUPATION_TYPE":     occupation,
            "AGE":                 age,
            "EMPLOYMENT_DURATION": employment,
            "FLAG_WORK_PHONE":     int(work_phone),
            "FLAG_PHONE":          int(phone),
            "FLAG_EMAIL":          int(email),
        }

        X_input = encode_input(applicant, ohe, oe, sc, pca, feature_cols)
        model   = trained_models[model_choice]
        pred    = model.predict(X_input)[0]
        proba   = model.predict_proba(X_input)[0]
        conf    = max(proba) * 100

        col_res, col_gauge = st.columns([1, 1])
        with col_res:
            if pred == "1":
                st.markdown(f"""
                <div class="approved-box">
                  <div class="verdict-text" style="color:#06d6a0">✅ APPROVED</div>
                  <div style="font-size:0.85rem;color:#9a9590;margin-top:0.4rem">
                    Model confidence: <strong style="color:#f0ece4">{conf:.1f}%</strong>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected-box">
                  <div class="verdict-text" style="color:#ff5f2e">❌ REJECTED</div>
                  <div style="font-size:0.85rem;color:#9a9590;margin-top:0.4rem">
                    Model confidence: <strong style="color:#f0ece4">{conf:.1f}%</strong>
                  </div>
                </div>""", unsafe_allow_html=True)

        with col_gauge:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            p_good = proba[1] if len(proba) > 1 else proba[0]
            p_bad  = 1 - p_good
            bars   = ax.barh(["Rejected", "Approved"], [p_bad, p_good],
                              color=["#ff5f2e", "#06d6a0"], height=0.5)
            for bar, val in zip(bars, [p_bad, p_good]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{val*100:.1f}%", va="center", fontsize=11, color="#f0ece4")
            ax.set_xlim(0, 1.18)
            ax.set_xlabel("Probability", labelpad=8)
            ax.set_title(f"Output Probabilities — {model_choice}", pad=10, fontsize=11)
            ax.grid(axis="x")
            st.pyplot(fig)

        # Applicant summary
        st.markdown("#### Applicant Summary")
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            [c1, c2, c3, c4],
            ["Age", "Annual Income", "Employment", "Family Size"],
            [f"{age} yrs", f"${income:,.0f}", f"{employment} yrs", f"{fam_members} members"],
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="font-size:1.3rem">{val}</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.info("👈 Fill in the applicant details in the sidebar and click **Predict Approval**.")
        st.markdown("#### How it works")
        cols = st.columns(3)
        steps = [
            ("01", "Input", "Enter applicant demographics, financials, and lifestyle details."),
            ("02", "Transform", "Features are one-hot encoded, scaled, and projected via PCA."),
            ("03", "Predict", "The selected model outputs Approved / Rejected with a confidence score."),
        ]
        for col, (n, title, desc) in zip(cols, steps):
            col.markdown(f"""
            <div class="metric-card" style="text-align:left;padding:1.5rem">
              <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                          color:#ff5f2e;line-height:1">{n}</div>
              <div style="font-family:Syne,sans-serif;font-weight:700;margin:0.5rem 0 0.4rem">{title}</div>
              <div style="font-size:0.78rem;color:#7a7570;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────  TAB 2: PERFORMANCE  ──────────────────────────
with tab2:
    st.markdown("### Model Comparison")
    st.caption("Trained on synthetic data mirroring the original notebook's distributions.")

    # Metrics table
    metric_rows = []
    for name, m in all_metrics.items():
        metric_rows.append({
            "Model": name,
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall ↑": m["Recall"],
            "F1": m["F1"],
            "ROC-AUC": m["ROC-AUC"],
        })
    df_metrics = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(df_metrics.style.highlight_max(axis=0, color="#1e3a2f"), use_container_width=True)

    st.markdown("---")
    left, right = st.columns(2)

    # ROC curves
    with left:
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = {"Decision Tree": "#ff5f2e", "Random Forest": "#ffd166", "Logistic Regression": "#a78bfa"}
        for name, m in all_metrics.items():
            y_num = (m["y_test"] == "1").astype(int)
            fpr, tpr, _ = roc_curve(y_num, m["proba"])
            ax.plot(fpr, tpr, label=f"{name} ({m['ROC-AUC']})", color=colors[name], lw=2)
        ax.plot([0,1],[0,1], "--", color="#555", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Curves", pad=10)
        ax.legend(fontsize=8, framealpha=0.2)
        ax.grid(True)
        st.pyplot(fig)

    # Metric bar chart
    with right:
        st.markdown("#### Recall Comparison (primary metric)")
        fig, ax = plt.subplots(figsize=(5, 4))
        names   = list(all_metrics.keys())
        recalls = [all_metrics[n]["Recall"] for n in names]
        bar_colors = ["#ff5f2e", "#ffd166", "#a78bfa"]
        bars = ax.bar(names, recalls, color=bar_colors, width=0.5)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Macro Recall")
        ax.set_title("Recall — Primary Metric\n(minimises missed bad applicants)", pad=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y")
        st.pyplot(fig)

    # Confusion matrices
    st.markdown("#### Confusion Matrices")
    c1, c2, c3 = st.columns(3)
    for col, (name, m) in zip([c1, c2, c3], all_metrics.items()):
        with col:
            cm = confusion_matrix(m["y_test"], m["y_pred"], labels=["0","1"])
            fig, ax = plt.subplots(figsize=(3.5, 3))
            im = ax.imshow(cm, cmap="YlOrRd", vmin=0)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Rejected","Approved"], fontsize=8)
            ax.set_yticklabels(["Rejected","Approved"], fontsize=8, rotation=90, va="center")
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.set_title(name, fontsize=9, pad=6)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                            color="white" if cm[i,j] > cm.max()/2 else "#f0ece4", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)


# ─────────────────────────────  TAB 3: METHODOLOGY  ──────────────────────────
with tab3:
    st.markdown("### Pipeline Walkthrough")

    steps_info = [
        ("🧹 Data Cleaning", [
            "Removed duplicate IDs (94 occurrences assumed as data entry errors)",
            "Filled 30% missing OCCUPATION_TYPE values with 'Unknown'",
            "Dropped FLAG_MOBIL (only 1 unique value — no predictive power)",
            "Converted DAYS_BIRTH → AGE and DAYS_EMPLOYED → EMPLOYMENT_DURATION",
            "Treated EMPLOYMENT_DURATION of −1001 (retired) as −1",
        ]),
        ("🏷️ Labelling Strategy", [
            "Bad (0): STATUS = 2, 3, 4, 5 at most recent month",
            "Good (1): STATUS = 0, C, X at most recent month",
            "Intermediate (STATUS=1): classified bad if >1 overdue month within recent 2 months",
            "Final dataset: 35,901 good clients vs 556 bad clients",
        ]),
        ("⚙️ Feature Engineering", [
            "One-hot encoding: NAME_INCOME_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE",
            "Ordinal encoding: NAME_EDUCATION_TYPE (Lower secondary → Academic degree)",
            "Dropped CNT_CHILDREN (highly correlated with CNT_FAM_MEMBERS, r=0.89)",
            "Capped CNT_FAM_MEMBERS at 5 to reduce outlier influence",
        ]),
        ("⚖️ Class Imbalance — SMOTE-NC", [
            "Extreme imbalance: 35,901 good vs 556 bad (ratio ≈ 65:1)",
            "Applied SMOTE-NC (handles both continuous and categorical features)",
            "Applied to training set only — no leakage into test set",
        ]),
        ("📉 Dimensionality Reduction — PCA", [
            "StandardScaler applied before PCA to normalise variance",
            "PCA reduces ~47 features to 30 principal components",
            "Preserves the majority of explained variance while reducing noise",
        ]),
        ("🏆 Model Selection", [
            "Baseline: Decision Tree, Random Forest, Logistic Regression",
            "Hyperparameter tuning via GridSearchCV (5-fold CV, scoring = macro recall)",
            "Decision Tree final params: criterion=entropy, min_samples_leaf=2, min_samples_split=10",
            "Random Forest final params: n_estimators=100, max_features=log2",
            "Logistic Regression final params: C=1, penalty=l2, solver=liblinear",
            "Decision Tree selected as best model — highest macro recall (0.63)",
        ]),
    ]

    for title, bullets in steps_info:
        with st.expander(title, expanded=False):
            for b in bullets:
                st.markdown(f"- {b}")

    st.markdown("---")
    st.markdown("#### Why Recall?")
    st.markdown("""
    > Recall is prioritised over accuracy because the cost of a **False Positive** 
    (approving a bad applicant) far outweighs the cost of a **False Negative** 
    (rejecting a good applicant) in credit risk management.
    > According to MAS guidelines, financial institutions cannot grant further unsecured credit 
    to borrowers who are 60+ days past due — making early detection of bad clients critical.
    """)
