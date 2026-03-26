import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
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
    background: #181818; border: 1px solid #2a2a2a;
    border-radius: 4px; padding: 1.2rem 1.4rem; text-align: center;
  }
  .metric-label { font-size:0.65rem; letter-spacing:0.15em; text-transform:uppercase; color:#7a7570; margin-bottom:0.4rem; }
  .metric-value { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#f0ece4; }
  .approved-box {
    background: linear-gradient(135deg,rgba(6,214,160,.15),rgba(6,214,160,.05));
    border: 2px solid #06d6a0; border-radius:6px; padding:1.5rem 2rem; text-align:center;
  }
  .rejected-box {
    background: linear-gradient(135deg,rgba(255,95,46,.15),rgba(255,95,46,.05));
    border: 2px solid #ff5f2e; border-radius:6px; padding:1.5rem 2rem; text-align:center;
  }
  .verdict-text { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; margin-bottom:0.3rem; }
  .section-tag { font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:#ff5f2e; margin-bottom:0.3rem; }
  .stSelectbox label,.stSlider label,.stNumberInput label,.stRadio label {
    font-size:0.75rem !important; letter-spacing:0.08em; text-transform:uppercase; color:#9a9590 !important;
  }
  div[data-testid="stSidebar"] { background-color:#111111; border-right:1px solid #222; }
  .stButton > button {
    background:#ff5f2e; color:white; font-family:'Syne',sans-serif; font-weight:700;
    font-size:0.85rem; letter-spacing:0.1em; text-transform:uppercase;
    border:none; border-radius:3px; padding:0.7rem 2rem; transition:all 0.2s;
  }
  .stButton > button:hover { background:#e04e20; transform:translate(-2px,-2px); box-shadow:4px 4px 0 #ffd166; }
  hr { border-color:#222; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#111111", "axes.facecolor": "#181818",
    "axes.edgecolor": "#333", "axes.labelcolor": "#9a9590",
    "xtick.color": "#9a9590", "ytick.color": "#9a9590",
    "text.color": "#f0ece4", "grid.color": "#2a2a2a",
    "grid.linestyle": "--", "font.family": "monospace",
})

# ── Constants ──────────────────────────────────────────────────────────────────
EDUCATION_ORDER = [
    "Lower secondary", "Secondary / secondary special",
    "Incomplete higher", "Higher education", "Academic degree",
]
OHE_COLS = ["NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]
INCOME_TYPES     = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
FAMILY_STATUSES  = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPES    = ["House / apartment", "With parents", "Municipal apartment",
                    "Rented apartment", "Office apartment", "Co-op apartment"]
OCCUPATION_TYPES = [
    "Laborers", "Core staff", "Accountants", "Managers", "Drivers",
    "Sales staff", "Cleaning staff", "Cooking staff", "Private service staff",
    "Medicine staff", "Security staff", "High skill tech staff", "Waiters/barmen staff",
    "Low-skill Laborers", "Realty agents", "Secretaries", "IT staff", "HR staff",
    "Unknown", "None",
]

# ── Data path — works locally and on Streamlit Cloud ──────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

# ── Step 1: Load & label (exact notebook pipeline) ────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_label_data():
    credit_path = DATA_DIR / "credit_record.csv"
    app_path    = DATA_DIR / "application_record.csv"

    if not credit_path.exists() or not app_path.exists():
        return None, (
            "Data files not found in `data/`. "
            "Add `credit_record.csv` and `application_record.csv` from "
            "[Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) "
            "to your repo's `data/` folder."
        )

    credit_record      = pd.read_csv(credit_path)
    application_record = pd.read_csv(app_path)

    # Remove duplicate IDs
    dup_ids = application_record["ID"].value_counts()
    dup_ids = dup_ids[dup_ids > 1].index
    application_record = application_record[~application_record["ID"].isin(dup_ids)]

    # Drop FLAG_MOBIL
    if "FLAG_MOBIL" in application_record.columns:
        application_record = application_record.drop("FLAG_MOBIL", axis=1)

    # Binary encode
    application_record["CODE_GENDER"]     = application_record["CODE_GENDER"].map({"M": 1, "F": 0})
    application_record["FLAG_OWN_CAR"]    = application_record["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})
    application_record["FLAG_OWN_REALTY"] = application_record["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})

    # Days → years
    application_record["AGE"]                = application_record["DAYS_BIRTH"]    // -365
    application_record["EMPLOYMENT_DURATION"] = application_record["DAYS_EMPLOYED"] // -365
    application_record = application_record.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1)

    # Missing occupation
    application_record["OCCUPATION_TYPE"] = application_record["OCCUPATION_TYPE"].fillna("Unknown")

    # Retired / unemployed
    application_record["OCCUPATION_TYPE"] = np.where(
        application_record["EMPLOYMENT_DURATION"] < 0, "None",
        application_record["OCCUPATION_TYPE"],
    )
    application_record["EMPLOYMENT_DURATION"] = np.where(
        application_record["EMPLOYMENT_DURATION"] < 0, -1,
        application_record["EMPLOYMENT_DURATION"],
    )

    # Label using MAS 60-day rule
    credit_record = credit_record.sort_values(["ID", "MONTHS_BALANCE"], ascending=[True, False])

    def classify_client(client):
        most_recent = client[client["MONTHS_BALANCE"] == 0]
        if len(most_recent) > 0:
            s = most_recent["STATUS"].iloc[0]
            if s in ["2", "3", "4", "5"]:
                return "0"
            elif s in ["0", "C", "X"]:
                return "1"
        overdue = client[client["STATUS"].isin(["1", "2", "3", "4", "5"])]
        if len(overdue) > 1 and overdue["MONTHS_BALANCE"].iloc[0] >= -2:
            return "0"
        return "1"

    result = (
        credit_record.groupby("ID")
        .apply(classify_client)
        .reset_index()
        .rename(columns={0: "Classification"})
    )

    df = application_record.merge(result, on="ID", how="inner").drop("ID", axis=1)
    return df, None


# ── Step 2: Full ML pipeline ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_pipeline(df):
    X = df.drop("Classification", axis=1).drop("CNT_CHILDREN", axis=1)
    y = df["Classification"]
    X["CNT_FAM_MEMBERS"] = X["CNT_FAM_MEMBERS"].clip(upper=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    oe  = OrdinalEncoder(
        categories=[EDUCATION_ORDER],
        handle_unknown="use_encoded_value", unknown_value=-1,
    )

    def encode(df_in, fit=False):
        df_in = df_in.copy()
        ohe_arr = ohe.fit_transform(df_in[OHE_COLS]) if fit else ohe.transform(df_in[OHE_COLS])
        df_in["NAME_EDUCATION_TYPE_ORDINAL"] = (
            oe.fit_transform(df_in[["NAME_EDUCATION_TYPE"]]) if fit
            else oe.transform(df_in[["NAME_EDUCATION_TYPE"]])
        )
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(OHE_COLS), index=df_in.index)
        df_in  = df_in.drop(columns=OHE_COLS + ["NAME_EDUCATION_TYPE"])
        return pd.concat([df_in, ohe_df], axis=1)

    X_train_enc = encode(X_train, fit=True)
    X_test_enc  = encode(X_test)

    # SMOTE-NC
    cat_idx = [i for i, c in enumerate(X_train_enc.columns) if X_train_enc[c].nunique() <= 20]
    smote   = SMOTENC(categorical_features=cat_idx, random_state=42)
    X_res, y_res = smote.fit_resample(X_train_enc, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train_enc.columns)

    # Scale + PCA
    sc  = StandardScaler()
    pca = PCA(n_components=min(30, X_res.shape[1] - 1), random_state=42)
    X_train_pca = pca.fit_transform(sc.fit_transform(X_res))
    X_test_pca  = pca.transform(sc.transform(X_test_enc))

    # Models with best hyperparams from GridSearchCV in notebook
    models = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=None,
            min_samples_leaf=2, min_samples_split=10, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_features="log2", max_depth=None, random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            C=1, penalty="l2", max_iter=100, solver="liblinear", random_state=42,
        ),
    }

    trained, metrics = {}, {}
    for name, m in models.items():
        m.fit(X_train_pca, y_res)
        y_pred = m.predict(X_test_pca)
        trained[name] = m
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred, pos_label="1", zero_division=0), 3),
            "Recall":    round(recall_score(y_test, y_pred,    pos_label="1", zero_division=0), 3),
            "F1":        round(f1_score(y_test, y_pred,        pos_label="1", zero_division=0), 3),
            "ROC-AUC":   round(roc_auc_score(y_test, m.predict_proba(X_test_pca)[:, 1]), 3),
            "y_pred": y_pred, "y_test": y_test.values,
            "proba":  m.predict_proba(X_test_pca)[:, 1],
        }

    return trained, metrics, ohe, oe, sc, pca, X_train_enc.columns.tolist()


def encode_input(row_dict, ohe, oe, sc, pca, feature_cols):
    df_in = pd.DataFrame([row_dict])
    df_in["CNT_FAM_MEMBERS"] = df_in["CNT_FAM_MEMBERS"].clip(upper=5)
    ohe_arr = ohe.transform(df_in[OHE_COLS])
    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(OHE_COLS))
    df_in["NAME_EDUCATION_TYPE_ORDINAL"] = oe.transform(df_in[["NAME_EDUCATION_TYPE"]])
    df_in = df_in.drop(columns=OHE_COLS + ["NAME_EDUCATION_TYPE"])
    df_in = pd.concat([df_in.reset_index(drop=True), ohe_df], axis=1)
    df_in = df_in.reindex(columns=feature_cols, fill_value=0)
    return pca.transform(sc.transform(df_in))


# ── Load & train ───────────────────────────────────────────────────────────────
with st.spinner("Loading real applicant data…"):
    df, data_error = load_and_label_data()

if data_error:
    st.error(data_error)
    st.markdown("""
    **Repo structure needed:**
    ```
    data/
      credit_record.csv
      application_record.csv
    app.py
    requirements.txt
    ```
    """)
    st.stop()

n_good = (df["Classification"] == "1").sum()
n_bad  = (df["Classification"] == "0").sum()

with st.spinner(f"Training models on {len(df):,} real applicants…"):
    trained_models, all_metrics, ohe, oe, sc, pca, feature_cols = train_pipeline(df)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Applicant Details")
    st.markdown("---")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    age    = st.slider("Age", 18, 75, 32)
    st.markdown("---")
    income      = st.number_input("Annual Income (USD)", min_value=10_000, max_value=1_000_000, value=120_000, step=5_000)
    income_type = st.selectbox("Income Type", INCOME_TYPES)
    employment  = st.slider("Employment Duration (years)", -1, 30, 5, help="-1 = retired / unemployed")
    st.markdown("---")
    education     = st.selectbox("Education Level", EDUCATION_ORDER, index=3)
    family_status = st.selectbox("Family Status", FAMILY_STATUSES)
    fam_members   = st.slider("Family Members", 1, 5, 2)
    st.markdown("---")
    housing    = st.selectbox("Housing Type", HOUSING_TYPES)
    occupation = st.selectbox("Occupation Type", OCCUPATION_TYPES)
    st.markdown("---")
    own_car    = st.checkbox("Owns a Car")
    own_realty = st.checkbox("Owns Property")
    work_phone = st.checkbox("Has Work Phone")
    phone      = st.checkbox("Has Home Phone")
    email_flag = st.checkbox("Has Email")
    st.markdown("---")
    model_choice = st.selectbox(
        "Model", ["Decision Tree", "Random Forest", "Logistic Regression"],
        help="Decision Tree achieved the highest macro recall in the original study.",
    )
    predict_btn = st.button("🔍 Predict Approval", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-tag">Alicia Tan</p>', unsafe_allow_html=True)
st.title("Credit Card Approval Predictor")
st.markdown(
    f"Trained on **{len(df):,} real applicants** "
    f"({n_good:,} good clients · {n_bad:,} bad clients). "
    "Full pipeline: SMOTE-NC → PCA → tuned classifiers."
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "Methodology"])

# ─── TAB 1 ────────────────────────────────────────────────────────────────────
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
            "FLAG_EMAIL":          int(email_flag),
        }
        X_input = encode_input(applicant, ohe, oe, sc, pca, feature_cols)
        model   = trained_models[model_choice]
        pred    = model.predict(X_input)[0]
        proba   = model.predict_proba(X_input)[0]
        classes = list(model.classes_)
        p_good  = proba[classes.index("1")] if "1" in classes else proba[1]
        p_bad   = 1 - p_good
        conf    = max(proba) * 100

        col_res, col_gauge = st.columns(2)
        with col_res:
            if pred == "1":
                st.markdown(f"""<div class="approved-box">
                  <div class="verdict-text" style="color:#06d6a0">✅ APPROVED</div>
                  <div style="font-size:0.85rem;color:#9a9590;margin-top:0.4rem">
                    Model confidence: <strong style="color:#f0ece4">{conf:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="rejected-box">
                  <div class="verdict-text" style="color:#ff5f2e">❌ REJECTED</div>
                  <div style="font-size:0.85rem;color:#9a9590;margin-top:0.4rem">
                    Model confidence: <strong style="color:#f0ece4">{conf:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)

        with col_gauge:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.barh(["Rejected", "Approved"], [p_bad, p_good],
                    color=["#ff5f2e", "#06d6a0"], height=0.5)
            for val, y in [(p_bad, 0), (p_good, 1)]:
                ax.text(val + 0.01, y, f"{val*100:.1f}%", va="center", fontsize=11, color="#f0ece4")
            ax.set_xlim(0, 1.18)
            ax.set_xlabel("Probability", labelpad=8)
            ax.set_title(f"Output Probabilities — {model_choice}", pad=10, fontsize=11)
            ax.grid(axis="x")
            st.pyplot(fig)

        st.markdown("#### Applicant Summary")
        for col, label, val in zip(
            st.columns(4),
            ["Age", "Annual Income", "Employment", "Family Size"],
            [f"{age} yrs", f"${income:,.0f}", f"{employment} yrs", f"{fam_members} members"],
        ):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="font-size:1.3rem">{val}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Fill in the applicant details in the sidebar and click **Predict Approval**.")
        for col, (n, title, desc) in zip(st.columns(3), [
            ("01", "Input",     "Enter applicant demographics, financials, and lifestyle details."),
            ("02", "Transform", "Features are one-hot encoded, scaled, and projected via PCA."),
            ("03", "Predict",   "The selected model outputs Approved / Rejected with confidence."),
        ]):
            col.markdown(f"""<div class="metric-card" style="text-align:left;padding:1.5rem">
              <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#ff5f2e;line-height:1">{n}</div>
              <div style="font-family:Syne,sans-serif;font-weight:700;margin:0.5rem 0 0.4rem">{title}</div>
              <div style="font-size:0.78rem;color:#7a7570;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Comparison")
    st.caption(f"Results on held-out test set (30% of {len(df):,} real applicants).")

    rows = [{"Model": n, "Accuracy": m["Accuracy"], "Precision": m["Precision"],
             "Recall ↑": m["Recall"], "F1": m["F1"], "ROC-AUC": m["ROC-AUC"]}
            for n, m in all_metrics.items()]
    st.dataframe(
        pd.DataFrame(rows).set_index("Model").style.highlight_max(axis=0, color="#1e3a2f"),
        use_container_width=True,
    )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = {"Decision Tree": "#ff5f2e", "Random Forest": "#ffd166", "Logistic Regression": "#a78bfa"}
        for name, m in all_metrics.items():
            fpr, tpr, _ = roc_curve((m["y_test"] == "1").astype(int), m["proba"])
            ax.plot(fpr, tpr, label=f"{name} ({m['ROC-AUC']})", color=colors[name], lw=2)
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Curves", pad=10)
        ax.legend(fontsize=8, framealpha=0.2); ax.grid(True)
        st.pyplot(fig)

    with right:
        st.markdown("#### Recall Comparison (primary metric)")
        fig, ax = plt.subplots(figsize=(5, 4))
        names   = list(all_metrics.keys())
        recalls = [all_metrics[n]["Recall"] for n in names]
        bars    = ax.bar(names, recalls, color=["#ff5f2e", "#ffd166", "#a78bfa"], width=0.5)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=11)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Macro Recall")
        ax.set_title("Recall — Primary Metric\n(minimises missed bad applicants)", pad=10)
        ax.tick_params(axis="x", labelsize=9); ax.grid(axis="y")
        st.pyplot(fig)

    st.markdown("#### Confusion Matrices")
    for col, (name, m) in zip(st.columns(3), all_metrics.items()):
        with col:
            cm = confusion_matrix(m["y_test"], m["y_pred"], labels=["0", "1"])
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.imshow(cm, cmap="YlOrRd", vmin=0)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Rejected", "Approved"], fontsize=8)
            ax.set_yticklabels(["Rejected", "Approved"], fontsize=8, rotation=90, va="center")
            ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
            ax.set_title(name, fontsize=9, pad=6)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "#f0ece4", fontsize=14)
            plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Dataset Statistics")
    for col, label, val in zip(
        st.columns(4),
        ["Total Applicants", "Good Clients", "Bad Clients", "Class Ratio"],
        [f"{len(df):,}", f"{n_good:,}", f"{n_bad:,}", f"{n_good // max(n_bad, 1)}:1"],
    ):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="font-size:1.4rem">{val}</div>
        </div>""", unsafe_allow_html=True)

# ─── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Pipeline Walkthrough")
    for title, bullets in [
        (" Data Cleaning", [
            "Removed duplicate IDs (94 pairs — assumed data entry errors or joint applications)",
            "Filled 30% missing OCCUPATION_TYPE values with 'Unknown'",
            "Dropped FLAG_MOBIL (only 1 unique value — zero predictive power)",
            "Converted DAYS_BIRTH → AGE and DAYS_EMPLOYED → EMPLOYMENT_DURATION in years",
            "Retired/unemployed applicants (EMPLOYMENT_DURATION < 0) set to −1; occupation set to 'None'",
        ]),
        (" Labelling Strategy", [
            "Bad (0): STATUS = 2, 3, 4, or 5 at the most recent month (60+ days overdue per MAS)",
            "Good (1): STATUS = 0, C, or X at the most recent month",
            "Intermediate (STATUS=1): classified Bad if >1 overdue month within the last 2 months",
            f"Result: {n_good:,} good clients vs {n_bad:,} bad clients",
        ]),
        (" Feature Engineering", [
            "One-hot encoding: NAME_INCOME_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE",
            "Ordinal encoding: NAME_EDUCATION_TYPE (Lower secondary → Academic degree)",
            "Dropped CNT_CHILDREN — highly correlated with CNT_FAM_MEMBERS (r = 0.89)",
            "Capped CNT_FAM_MEMBERS at 5 to reduce extreme outlier influence",
        ]),
        (" Class Imbalance — SMOTE-NC", [
            f"Extreme imbalance: {n_good:,} good vs {n_bad:,} bad (≈ {n_good // max(n_bad, 1)}:1 ratio)",
            "SMOTE-NC applied — handles both continuous and nominal categorical features simultaneously",
            "Oversampling applied to training set only — test set untouched to avoid leakage",
        ]),
        (" Dimensionality Reduction — PCA", [
            "StandardScaler applied before PCA to normalise feature variance across ~47 columns",
            "PCA reduces to 30 principal components, preserving majority of explained variance",
            "Reduces noise and multicollinearity before passing to classifiers",
        ]),
        (" Model Selection", [
            "Three models evaluated: Decision Tree, Random Forest, Logistic Regression",
            "Hyperparameter tuning via GridSearchCV (5-fold CV, scoring = macro recall)",
            "Decision Tree best params: criterion=entropy, min_samples_leaf=2, min_samples_split=10",
            "Random Forest best params: n_estimators=100, max_features=log2",
            "Logistic Regression best params: C=1, penalty=l2, solver=liblinear",
            "Decision Tree selected — highest macro recall, best at catching bad applicants",
        ]),
    ]:
        with st.expander(title, expanded=False):
            for b in bullets:
                st.markdown(f"- {b}")

    st.markdown("---")
    st.info(
        "**Why Recall?** The cost of a False Positive (approving a bad applicant) far exceeds "
        "that of a False Negative (rejecting a good one) in credit risk management. "
        "Per MAS guidelines, banks cannot extend further unsecured credit to borrowers 60+ days past due — "
        "making early detection of bad clients critical."
    )