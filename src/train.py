"""Train and evaluate a telecom churn classifier.

Run from the project root: python src/train.py

What this does, in order:
  1. Load and clean the data.
  2. Split into train/test (stratified, so the churn rate is preserved).
  3. Build a preprocessing + model pipeline (so the saved model can score
     raw customer rows later, with no separate preprocessing code).
  4. Compare a DummyClassifier baseline, Logistic Regression and
     RandomForest using 5-fold cross-validation ROC-AUC on the training set.
  5. Fit the winner on the full training set and report honest test metrics.
  6. Save the model, metadata, metrics and three plots.
"""
import datetime
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed; we only save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# A CV edge smaller than this is treated as noise. RandomForest beat
# LogisticRegression by 0.0008 here, far below this and the ~0.016
# fold-to-fold standard deviation, so we prefer the simpler model.
SELECTION_TOLERANCE = 0.005


def load_and_clean(path):
    """Load the CSV and apply the cleaning rules. Return X, y and the full df."""
    path = Path(path)
    if not path.exists():
        print("ERROR: dataset not found at", path)
        print()
        print("Run:")
        print("  python src/00_download.py")
        print()
        print("If that fails, use one of these fallbacks:")
        print("  1) Kaggle (needs a free Kaggle account):")
        print("     https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("     Download Telco-Customer-Churn.csv and save it to:")
        print("     " + str(path))
        print("  2) Fully offline (no internet):")
        print("     python src/00_generate_sample.py --out data/raw/telco_churn.csv")
        print("     (Use --force only if the file already exists.)")
        print("     The output is SYNTHETIC; label it as such.")
        sys.exit(1)
    print("Loading data from", path)
    df = pd.read_csv(path)
    original_rows = len(df)

    # customerID is an identifier, not a signal. Keeping it would let the
    # model memorise individual customers instead of learning patterns.
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as text and has 11 blanks. All 11 are customers
    # with tenure 0, so their lifetime charges are genuinely 0, not missing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blanks = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Make SeniorCitizen readable so the saved categories and the app inputs
    # use the same "Yes"/"No" vocabulary as every other binary column.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Defensive: drop exact duplicate rows if any exist.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates = before - len(df)

    y = (df["Churn"] == "Yes").astype(int)
    X = df[ALL_FEATURES].copy()

    print("  rows:", len(df), "(started with", original_rows, ")")
    print("  churn rate:", round(y.mean() * 100, 2), "%")
    print("  TotalCharges blanks filled with 0:", blanks)
    print("  duplicate rows removed:", duplicates)
    return X, y, df


def build_preprocessor():
    """Fresh preprocessor. One per model so they do not share fitted state."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def evaluate(name, model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def pretty_feature(name):
    """Turn 'Contract_Month-to-month' into 'Contract = Month-to-month'."""
    if "_" in name and name not in ("tenure", "MonthlyCharges", "TotalCharges"):
        left, right = name.split("_", 1)
        return f"{left} = {right}"
    return name


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, df = load_and_clean(DATA_PATH)

    # Stratified split keeps the 26.4% churn rate in both halves.
    print("\nSplitting 80/20 (stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("  train:", len(X_train), "  test:", len(X_test))

    # Baseline first: a model that always predicts the majority class.
    print("\nTraining baseline (DummyClassifier, strategy=prior)...")
    dummy = DummyClassifier(strategy="prior", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate("DummyClassifier (prior)", dummy, X_test, y_test)

    # Two real candidates.
    print("Training Logistic Regression...")
    lr_pipe = build_pipeline(LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ))
    print("Training Random Forest (300 trees)...")
    rf_pipe = build_pipeline(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ))

    # Select by cross-validation ROC-AUC on the TRAINING set only, so the
    # test set stays untouched until the final evaluation.
    print("\n5-fold cross-validation ROC-AUC on training set:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "LogisticRegression": lr_pipe,
        "RandomForest": rf_pipe,
    }
    cv_scores = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                scoring="roc_auc", n_jobs=1)
        cv_scores[name] = (float(scores.mean()), float(scores.std()))
        print(f"  {name:20s} {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Prefer the simpler model unless the forest wins by more than the
    # tolerance. A 0.0008 edge is noise, not evidence.
    lr_cv = cv_scores["LogisticRegression"][0]
    rf_cv = cv_scores["RandomForest"][0]
    margin = rf_cv - lr_cv
    if margin > SELECTION_TOLERANCE:
        winner_name = "RandomForest"
        selection_reason = (
            f"RF margin {margin:+.4f} > tolerance {SELECTION_TOLERANCE:.3f}"
        )
    else:
        winner_name = "LogisticRegression"
        selection_reason = (
            f"RF margin {margin:+.4f} <= tolerance {SELECTION_TOLERANCE:.3f}; "
            "prefer the simpler model"
        )
    print(f"  CV margin (RF - LR): {margin:+.4f} (tolerance {SELECTION_TOLERANCE:.3f})")
    print("  Selection:", selection_reason)

    print("\nSelected model:", winner_name)

    # Fit both on the full training set so we can report both test scores.
    print("Fitting both candidates on the full training set...")
    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    pipes = {"LogisticRegression": lr_pipe, "RandomForest": rf_pipe}
    selected = pipes[winner_name]

    lr_metrics = evaluate("LogisticRegression", lr_pipe, X_test, y_test)
    rf_metrics = evaluate("RandomForest", rf_pipe, X_test, y_test)
    selected_metrics = lr_metrics if winner_name == "LogisticRegression" else rf_metrics

    # Business metric: how many of the real churners do we catch if we only
    # have time to call the top 20% riskiest customers?
    proba_test = selected.predict_proba(X_test)[:, 1]
    order = np.argsort(proba_test)[::-1]
    top_k = int(0.2 * len(y_test))
    churners_caught = int(y_test.iloc[order[:top_k]].sum())
    total_churners = int(y_test.sum())
    capture_rate = churners_caught / total_churners

    pred_test = selected.predict(X_test)
    cm = confusion_matrix(y_test, pred_test)

    # EDA for the findings section.
    eda = {}
    for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
        rates = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean())
        eda[col] = {str(k): float(v) for k, v in rates.items()}

    # Marginal churn rates used by the app to sanity-check occlusion signs.
    # Occlusion swaps one feature at a time, so correlated features can flip
    # the apparent sign; the app cross-checks against these marginal rates.
    marginal = {
        "overall_churn_rate": float(y_train.mean()),
        "categorical": {},
        "numeric": {},
    }
    for col in CATEGORICAL_FEATURES:
        rates = y_train.groupby(X_train[col]).mean()
        marginal["categorical"][col] = {
            str(k): float(v) for k, v in rates.items()
        }
    for col in NUMERIC_FEATURES:
        med = float(X_train[col].median())
        low_mask = X_train[col] <= med
        high_mask = X_train[col] > med
        marginal["numeric"][col] = {
            "median": med,
            "low_churn_rate": float(y_train[low_mask].mean()),
            "high_churn_rate": float(y_train[high_mask].mean()),
        }

    # Feature importance of the selected model.
    preprocessor = selected.named_steps["preprocessor"]
    classifier = selected.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    if hasattr(classifier, "coef_"):
        raw_importance = classifier.coef_[0]
        importance_kind = "coefficient (log-odds, standardised features)"
    else:
        raw_importance = classifier.feature_importances_
        importance_kind = "impurity importance"
    importance = pd.Series(raw_importance, index=feature_names)
    importance_sorted = importance.reindex(
        importance.abs().sort_values(ascending=False).index
    )
    fmt_val = "{:+.4f}" if hasattr(classifier, "coef_") else "{:.4f}"
    top_features = [(pretty_feature(n), float(v))
                    for n, v in importance_sorted.head(15).items()]

    # ---- Save model + metadata ----
    print("\nSaving model to", MODELS_DIR / "churn_model.pkl")
    joblib.dump(selected, MODELS_DIR / "churn_model.pkl")

    categories = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    baselines = {}
    for col in NUMERIC_FEATURES:
        baselines[col] = float(X_train[col].median())
    for col in CATEGORICAL_FEATURES:
        baselines[col] = str(X_train[col].mode(dropna=True).iloc[0])

    metadata = {
        "schema_version": 1,
        "model_name": winner_name,
        "selection_metric": "5-fold CV ROC-AUC on the training set",
        "cv_roc_auc": cv_scores[winner_name][0],
        "cv_roc_auc_std": cv_scores[winner_name][1],
        "cv_scores": {k: {"mean": v[0], "std": v[1]} for k, v in cv_scores.items()},
        "selection_margin_rf_minus_lr": float(margin),
        "selection_tolerance": SELECTION_TOLERANCE,
        "selection_reason": selection_reason,
        "test_metrics": selected_metrics,
        "all_test_metrics": {
            "LogisticRegression": lr_metrics,
            "RandomForest": rf_metrics,
        },
        "dummy_test_metrics": dummy_metrics,
        "top_20pct_churn_capture": float(capture_rate),
        "top_20pct_churners_caught": churners_caught,
        "top_20pct_customers_scored": top_k,
        "total_test_churners": total_churners,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "feature_order": ALL_FEATURES,
        "categories": categories,
        "baselines": baselines,
        "importance_kind": importance_kind,
        "top_features": [{"feature": f, "value": v} for f, v in top_features],
        "risk_tiers": {"low": 0.35, "high": 0.65},
        "decision_threshold": 0.5,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "data_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "eda_churn_rate_by": eda,
        "marginal": marginal,
    }
    with open(MODELS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to", MODELS_DIR / "model_meta.json")

    # ---- Plots ----
    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(f"Confusion matrix - {winner_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2. ROC curve (both models + dummy)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe, color in [("LogisticRegression", lr_pipe, "#0F766E"),
                              ("RandomForest", rf_pipe, "#1D4ED8")]:
        p = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")
    # Dummy: constant score -> AUC 0.5 diagonal.
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Dummy (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve on the test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # 3. Feature importance (top 15)
    top15 = importance_sorted.head(15).iloc[::-1]  # ascending for barh
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = []
    for v in top15.values:
        if hasattr(classifier, "coef_"):
            colors.append("#B91C1C" if v > 0 else "#15803D")
        else:
            colors.append("#1D4ED8")
    ax.barh([pretty_feature(n) for n in top15.index], top15.values, color=colors)
    ax.set_xlabel(importance_kind)
    ax.set_title(f"Top 15 features - {winner_name}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("Saved 3 plots to", OUTPUT_DIR)

    # ---- metrics.txt ----
    def fmt(m):
        return (f"{m['model']:<28s} "
                f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
                f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  "
                f"auc={m['roc_auc']:.3f}")

    lines = []
    lines.append("TELCO CUSTOMER CHURN - TRAINING REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {metadata['trained_at']}")
    lines.append(f"Python: {sys.version.split()[0]}  scikit-learn: {sklearn.__version__}  "
                 f"pandas: {pd.__version__}")
    lines.append(f"Data: {DATA_PATH} ({len(df)} rows, churn rate {y.mean()*100:.2f}%)")
    lines.append(f"Split: 80/20 stratified (train {len(X_train)}, test {len(X_test)})")
    lines.append("")
    lines.append("MODEL COMPARISON (test set)")
    lines.append("-" * 60)
    lines.append(fmt(dummy_metrics))
    lines.append(fmt(lr_metrics))
    lines.append(fmt(rf_metrics))
    lines.append("")
    lines.append("SELECTION (5-fold CV ROC-AUC on training set)")
    lines.append("-" * 60)
    for name, (m, s) in cv_scores.items():
        lines.append(f"  {name:<20s} {m:.4f} (+/- {s:.4f})")
    lines.append(f"  CV margin (RF - LR): {margin:+.4f}")
    lines.append(f"  Tolerance: {SELECTION_TOLERANCE:.3f} (prefer simpler model unless RF wins by more)")
    lines.append(f"  Winner: {winner_name} ({selection_reason})")
    lines.append("")
    lines.append("SELECTED MODEL TEST METRICS")
    lines.append("-" * 60)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lines.append(f"  {k:<10s} {selected_metrics[k]:.4f}")
    lines.append("")
    lines.append("CONFUSION MATRIX (rows=true, cols=predicted)")
    lines.append("-" * 60)
    lines.append(f"  [[{cm[0, 0]:4d} {cm[0, 1]:4d}]   (stay)")
    lines.append(f"   [{cm[1, 0]:4d} {cm[1, 1]:4d}]]  (churn)")
    lines.append("")
    lines.append("CLASSIFICATION REPORT")
    lines.append("-" * 60)
    lines.append(classification_report(y_test, pred_test,
                                     target_names=["Stay", "Churn"],
                                     zero_division=0))
    lines.append("BUSINESS CHECK")
    lines.append("-" * 60)
    lines.append(f"  Scoring only the top 20% riskiest customers catches "
                 f"{churners_caught} of {total_churners} churners "
                 f"({capture_rate*100:.1f}%).")
    lines.append("")
    lines.append("TOP 15 FEATURES")
    lines.append("-" * 60)
    for feat, val in top_features:
        lines.append(f"  {feat:<34s} {fmt_val.format(val)}")
    lines.append("")
    lines.append("EDA: churn rate by category")
    lines.append("-" * 60)
    for col, rates in eda.items():
        lines.append(f"  {col}:")
        for k, v in sorted(rates.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:<28s} {v*100:5.1f}%")
    lines.append("")
    lines.append("NOTE: feature importance shows association, not causation.")

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "metrics.txt").write_text(report, encoding="utf-8")

    # Console summary.
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(fmt(dummy_metrics))
    print(fmt(lr_metrics))
    print(fmt(rf_metrics))
    print()
    print("Winner:", winner_name, "| CV AUC:", round(cv_scores[winner_name][0], 4))
    print("  CV margin (RF - LR):", f"{margin:+.4f}", "| tolerance:", SELECTION_TOLERANCE)
    print("  Selection:", selection_reason)
    print("Test ROC-AUC:", round(selected_metrics["roc_auc"], 4),
          "| Recall:", round(selected_metrics["recall"], 4))
    print("Top 20% capture:", f"{capture_rate*100:.1f}%")
    print("Confusion matrix:", cm.tolist())
    print("Top 5 features:")
    for feat, val in top_features[:5]:
        print(f"  {feat}: {fmt_val.format(val)}")
    print("\nSaved:")
    print("  model   ->", MODELS_DIR / "churn_model.pkl")
    print("  meta    ->", MODELS_DIR / "model_meta.json")
    print("  metrics ->", OUTPUT_DIR / "metrics.txt")
    print("  plots   ->", OUTPUT_DIR / "confusion_matrix.png, roc_curve.png, feature_importance.png")


if __name__ == "__main__":
    main()