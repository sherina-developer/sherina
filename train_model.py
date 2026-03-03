from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
MODEL_BUNDLE_PATH = Path("model_bundle.pkl")


FEATURE_ORDER = [
    "Age",
    "Gender",
    "BMI",
    "Cholesterol",
    "Sleep",
    "Activity",
    "Breakfast",
]


COLUMN_MAP = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "BMXBMI": "BMI",
    "LBXTC": "Cholesterol",
    "SLD010H": "Sleep",
    "PAQ605": "ActivityRaw",
    "DRABF": "BreakfastRaw",
}


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def safe_select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df[["SEQN", *present]].copy()


def build_dataset() -> pd.DataFrame:
    demographic = read_csv_safely(DATA_DIR / "demographic.csv")
    examination = read_csv_safely(DATA_DIR / "examination.csv")
    labs = read_csv_safely(DATA_DIR / "labs.csv")
    questionnaire = read_csv_safely(DATA_DIR / "questionnaire.csv")
    diet = read_csv_safely(DATA_DIR / "diet.csv")

    merged = (
        safe_select(demographic, ["RIDAGEYR", "RIAGENDR"])
        .merge(safe_select(examination, ["BMXBMI"]), on="SEQN", how="left")
        .merge(safe_select(labs, ["LBXTC"]), on="SEQN", how="left")
        .merge(safe_select(questionnaire, ["SLD010H", "PAQ605"]), on="SEQN", how="left")
        .merge(safe_select(diet, ["DRABF"]), on="SEQN", how="left")
        .rename(columns=COLUMN_MAP)
    )

    merged["Activity"] = merged["ActivityRaw"].map({1: 1, 2: 0})
    merged["Breakfast"] = merged["BreakfastRaw"].map({1: 1, 2: 0})

    merged.loc[(merged["Sleep"] < 2) | (merged["Sleep"] > 14), "Sleep"] = pd.NA

    for col in ["Age", "Gender", "BMI", "Cholesterol", "Sleep", "Activity", "Breakfast"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    risk_score = (
        (merged["BMI"] >= 30).astype(float)
        + (merged["Cholesterol"] >= 240).astype(float)
        + (merged["Sleep"] < 6).astype(float)
        + (merged["Activity"] == 0).astype(float)
        + (merged["Breakfast"] == 0).astype(float)
    )

    merged["Metabolic_Risk"] = (risk_score >= 2).astype(int)
    return merged


def train_and_save(df: pd.DataFrame) -> None:
    X = df[FEATURE_ORDER].copy()
    y = df["Metabolic_Risk"].copy()

    imputer_values = X.median(numeric_only=True).to_dict()
    X = X.fillna(imputer_values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    numeric_cols = ["Age", "BMI", "Cholesterol", "Sleep"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
    model.fit(X_train_scaled, y_train)

    test_pred = model.predict(X_test_scaled)
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("Model performance on holdout set")
    print(classification_report(y_test, test_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")

    bundle = {
        "model": model,
        "scaler": scaler,
        "imputer_values": imputer_values,
        "feature_order": FEATURE_ORDER,
        "numeric_cols": numeric_cols,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"Saved trained bundle -> {MODEL_BUNDLE_PATH.resolve()}")


def main() -> None:
    df = build_dataset()
    print(f"Prepared dataset shape: {df.shape}")
    print("Risk distribution:")
    print(df["Metabolic_Risk"].value_counts())
    train_and_save(df)


if __name__ == "__main__":
    main()
