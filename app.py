from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📶", layout="wide")

DATA_PATH = Path("data/telco_churn_sample.csv")

@st.cache_data
def load_sample():
    return pd.read_csv(DATA_PATH)

def prepare_data(df):
    df = df.copy()
    # Accept the common IBM Telco dataset column name and target values.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" not in df.columns:
        raise ValueError("The CSV must contain a target column named 'Churn'.")

    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    )
    df = df.dropna(subset=["Churn"])
    if df["Churn"].nunique() < 2:
        raise ValueError("The Churn column must contain both churn and non-churn examples.")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

st.title("📶 Telecom Customer Churn Prediction")
st.write("A supervised machine-learning demo that estimates whether a telecom customer may leave.")

st.info(
    "This is an educational prototype. Predictions are estimates, not guarantees. "
    "Do not use them to make consequential customer decisions without validation, fairness checks, and human review."
)

with st.expander("What is churn? How does this app work?"):
    st.markdown("""
    **Churn** means a customer stops using a service. The model learns from historical customer records labelled `Yes` or `No`.

    - Numerical columns are imputed and scaled.
    - Categorical columns are imputed and one-hot encoded.
    - Logistic Regression learns to classify churn vs. non-churn.
    - A held-out test set is used for the evaluation metrics.

    Accuracy alone is not enough: precision and recall help explain different types of mistakes.
    """)

uploaded = st.file_uploader(
    "Optional: upload a CSV dataset (for example, IBM Telco Customer Churn)",
    type=["csv"],
    help="Expected target column: Churn, containing Yes/No or 1/0. Other columns are used as features."
)

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_sample()
    df = prepare_data(raw)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    if len(df) < 20:
        st.error("Please provide at least 20 labelled rows for a meaningful demonstration.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows used", f"{len(df):,}")
    c2.metric("Churn rate", f"{y.mean():.1%}")
    c3.metric("Input features", str(X.shape[1]))
    st.caption("The bundled dataset is synthetic and is only for demonstrating the workflow.")

    st.subheader("Held-out test results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    st.write("Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual: No churn", "Actual: Churn"],
            columns=["Predicted: No churn", "Predicted: Churn"]
        ),
        use_container_width=True
    )
    st.caption("Metrics depend on the selected dataset and split. Results on synthetic data do not demonstrate real-world performance.")

    st.subheader("Try a customer profile")
    st.write("Change the fields below. This demo creates a profile from a sample row and changes common fields when present.")
    sample = X.iloc[[0]].copy()

    if "tenure" in sample.columns:
        max_tenure = int(pd.to_numeric(X["tenure"], errors="coerce").max())
        tenure = st.slider("Tenure (months)", 0, max(1, min(max_tenure, 120)), min(12, max(1, max_tenure)))
        sample.loc[:, "tenure"] = tenure

    if "MonthlyCharges" in sample.columns:
        charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=float(pd.to_numeric(X["MonthlyCharges"], errors="coerce").median())
        )
        sample.loc[:, "MonthlyCharges"] = charges

    if "Contract" in sample.columns:
        options = sorted(X["Contract"].dropna().astype(str).unique().tolist())
        contract = st.selectbox("Contract type", options)
        sample.loc[:, "Contract"] = contract

    if st.button("Estimate churn probability", type="primary"):
        churn_probability = float(model.predict_proba(sample)[0][1])
        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
        st.caption("This is a model estimate for the supplied profile, not a guarantee or a validated business risk score.")

    with st.expander("Preview data"):
        st.dataframe(df.head(20), use_container_width=True)

except FileNotFoundError:
    st.error("Bundled sample dataset not found. Ensure data/telco_churn_sample.csv exists.")
except (ValueError, KeyError) as exc:
    st.error(f"Could not use this dataset: {exc}")
    st.write("Check that the CSV has a `Churn` column and contains both churn and non-churn examples.")
