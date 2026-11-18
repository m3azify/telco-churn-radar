"""Streamlit app: score a telecom customer's churn risk.

Run from the project root: streamlit run src/app.py
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
META_PATH = ROOT / "models" / "model_meta.json"
DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
TRAIN_SCRIPT = ROOT / "src" / "train.py"

st.set_page_config(
    page_title="Churn Radar",
    page_icon="📉",
    layout="wide",
)

FRIENDLY = {
    "tenure": "tenure",
    "MonthlyCharges": "monthly charges",
    "TotalCharges": "total charges",
    "Contract": "contract",
    "InternetService": "internet service",
    "PaymentMethod": "payment method",
    "TechSupport": "tech support",
    "OnlineSecurity": "online security",
    "OnlineBackup": "online backup",
    "DeviceProtection": "device protection",
    "StreamingTV": "streaming TV",
    "StreamingMovies": "streaming movies",
    "MultipleLines": "multiple lines",
    "PhoneService": "phone service",
    "PaperlessBilling": "paperless billing",
    "Partner": "partner",
    "Dependents": "dependents",
    "gender": "gender",
    "SeniorCitizen": "senior citizen",
}


@st.cache_resource(show_spinner="Loading model (first run may train it, ~30s)...")
def load_bundle():
    """Load the saved model + metadata. Retrain once if loading fails."""
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            if meta.get("schema_version") == 1 and meta.get("feature_order"):
                return model, meta, None
        except Exception as exc:  # fall through to retrain
            load_error = str(exc)
        else:
            load_error = "saved metadata is out of date"
    else:
        load_error = "model files are missing"

    if not DATA_PATH.exists():
        return None, None, (
            "Model is unavailable and the training data is missing. "
            "Run `python src/00_download.py` first, then `python src/train.py`."
        )

    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        return None, None, f"Retraining failed:\n{tail}"

    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, f"Model still could not be loaded after retraining: {exc}"
    return model, meta, None


def risk_band(probability, meta):
    low = meta["risk_tiers"]["low"]
    high = meta["risk_tiers"]["high"]
    if probability < low:
        return "Low", "success"
    if probability < high:
        return "Medium", "warning"
    return "High", "error"


def factor_text(column, value, baselines):
    label = FRIENDLY.get(column, column)
    if column in ("tenure", "MonthlyCharges", "TotalCharges"):
        try:
            v = float(value)
            b = float(baselines[column])
            if v > b:
                return f"high {label} ({v:g})"
            if v < b:
                return f"low {label} ({v:g})"
            return f"{label} ({v:g})"
        except (TypeError, ValueError):
            pass
    return f"{label} = {value}"


def marginal_signal(column, value, meta):
    """Compare the model's occlusion sign to marginal churn.

    Returns +1 if the feature value is associated with higher churn than
    the overall rate, -1 if lower, 0 if unknown or neutral. Used to stop
    the app from calling a protective feature a "risk factor".
    """
    marginal = meta.get("marginal") or {}
    overall = marginal.get("overall_churn_rate")
    if overall is None:
        return 0
    categorical = marginal.get("categorical") or {}
    numeric = marginal.get("numeric") or {}
    tolerance = 0.005

    if column in categorical:
        rate = categorical[column].get(str(value))
        if rate is None:
            return 0
        if rate > overall + tolerance:
            return 1
        if rate < overall - tolerance:
            return -1
        return 0

    info = numeric.get(column)
    if not info:
        return 0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    med = float(info["median"])
    if v < med:
        rate = info["low_churn_rate"]
    elif v > med:
        rate = info["high_churn_rate"]
    else:
        return 0
    if rate > overall + tolerance:
        return 1
    if rate < overall - tolerance:
        return -1
    return 0


def marginal_label(column, value, meta):
    signal = marginal_signal(column, value, meta)
    if signal > 0:
        return "higher churn"
    if signal < 0:
        return "lower churn"
    return "similar"


def explain_one(model, meta, row):
    """Return (one_line_explanation, list_of_(column, delta, value))."""
    feature_order = meta["feature_order"]
    baselines = meta["baselines"]
    X = pd.DataFrame([row], columns=feature_order)
    base_p = float(model.predict_proba(X)[0, 1])

    deltas = []
    for col in feature_order:
        perturbed = X.copy()
        perturbed[col] = baselines[col]
        p2 = float(model.predict_proba(perturbed)[0, 1])
        deltas.append((col, base_p - p2, row[col]))

    positives = sorted([d for d in deltas if d[1] > 0], key=lambda d: -d[1])
    consistent = [d for d in positives if marginal_signal(d[0], d[2], meta) >= 0]
    conflicting = [d for d in positives if marginal_signal(d[0], d[2], meta) < 0]

    if consistent:
        chosen = consistent[:2]
        if len(chosen) == 2 and chosen[1][1] < 0.005:
            chosen = chosen[:1]
        phrase = "Main risk factors: " + "; ".join(
            factor_text(c, v, baselines) for c, _, v in chosen
        )
    elif conflicting:
        chosen = conflicting[:1]
        phrase = (
            "Conditional signal only: " + factor_text(chosen[0][0], chosen[0][2], baselines)
            + " (the model flags it, but this group's marginal churn is lower; "
            "correlated features can flip the sign)."
        )
    else:
        chosen = [min(deltas, key=lambda d: d[1])]
        phrase = "No strong risk factors; strongest protective factor: " + factor_text(
            chosen[0][0], chosen[0][2], baselines
        )
    return phrase, deltas


def batch_explain(model, meta, X):
    feature_order = meta["feature_order"]
    baselines = meta["baselines"]
    base = model.predict_proba(X)[:, 1]
    deltas = np.zeros((len(X), len(feature_order)))
    for i, col in enumerate(feature_order):
        perturbed = X.copy()
        perturbed[col] = baselines[col]
        p2 = model.predict_proba(perturbed)[:, 1]
        deltas[:, i] = base - p2

    signals = [
        [marginal_signal(col, X.iloc[r][col], meta) for col in feature_order]
        for r in range(len(X))
    ]

    factors = []
    for r in range(len(X)):
        positive_idx = [i for i in range(len(feature_order)) if deltas[r, i] > 0]
        consistent = [i for i in positive_idx if signals[r][i] >= 0]
        if consistent:
            i = max(consistent, key=lambda j: deltas[r, j])
            factors.append("risk: " + factor_text(feature_order[i], X.iloc[r][feature_order[i]], baselines))
        elif positive_idx:
            i = max(positive_idx, key=lambda j: deltas[r, j])
            factors.append("conditional: " + factor_text(feature_order[i], X.iloc[r][feature_order[i]], baselines))
        else:
            i = int(np.argmin(deltas[r]))
            factors.append("protective: " + factor_text(feature_order[i], X.iloc[r][feature_order[i]], baselines))
    return factors


def clean_uploaded(df, meta):
    feature_order = meta["feature_order"]
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    out = df.copy()
    # SeniorCitizen may arrive as 0/1 from the original dataset.
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].replace(
            {0: "No", 1: "Yes", "0": "No", "1": "Yes"}
        )

    for col in ("tenure", "MonthlyCharges"):
        out[col] = pd.to_numeric(out[col], errors="raise")

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    mask = out["TotalCharges"].isna()
    out.loc[mask, "TotalCharges"] = out.loc[mask, "tenure"] * out.loc[mask, "MonthlyCharges"]
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    missing = []
    for col in meta["features"]["categorical"]:
        if col in out.columns:
            n_missing = int(out[col].isna().sum())
            if n_missing:
                missing.append(f"{col}: {n_missing} missing value(s)")
    if missing:
        raise ValueError(
            "Missing category value(s) in the upload. Fix the CSV or use the "
            "downloaded template.\n" + "\n".join(missing)
        )

    for col in meta["features"]["categorical"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    unknown = []
    for col in meta["features"]["categorical"]:
        if col not in out.columns:
            continue
        allowed = set(meta["categories"][col])
        bad = sorted(set(out[col].dropna().astype(str).unique()) - allowed)
        if bad:
            unknown.append(f"{col}: {', '.join(bad)} (allowed: {', '.join(sorted(allowed))})")
    if unknown:
        raise ValueError(
            "Unknown category value(s) in the upload. Fix the CSV or use the "
            "downloaded template.\n" + "\n".join(unknown)
        )
    return out


def main():
    model, meta, err = load_bundle()
    if err:
        st.error(err)
        st.stop()

    feature_order = meta["feature_order"]
    categories = meta["categories"]
    baselines = meta["baselines"]

    # ---------- sidebar: customer inputs ----------
    with st.sidebar:
        st.header("Customer profile")
        tenure = st.slider("Tenure (months)", 0, 72, 8, 1)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 85.0, 0.5)
        st.caption(f"Estimated lifetime charges: ${tenure * monthly:,.2f}")

        contract = st.selectbox("Contract", categories["Contract"])
        internet = st.selectbox("Internet service", categories["InternetService"])
        payment = st.selectbox("Payment method", categories["PaymentMethod"])
        paperless = st.selectbox("Paperless billing", categories["PaperlessBilling"])

        st.subheader("Demographics")
        gender = st.selectbox("Gender", categories["gender"])
        senior = st.selectbox("Senior citizen", categories["SeniorCitizen"])
        partner = st.selectbox("Partner", categories["Partner"])
        dependents = st.selectbox("Dependents", categories["Dependents"])

        st.subheader("Services")
        phone = st.selectbox("Phone service", categories["PhoneService"])
        multiple = st.selectbox("Multiple lines", categories["MultipleLines"])
        online_sec = st.selectbox("Online security", categories["OnlineSecurity"])
        online_bak = st.selectbox("Online backup", categories["OnlineBackup"])
        device = st.selectbox("Device protection", categories["DeviceProtection"])
        tech = st.selectbox("Tech support", categories["TechSupport"])
        stream_tv = st.selectbox("Streaming TV", categories["StreamingTV"])
        stream_mv = st.selectbox("Streaming movies", categories["StreamingMovies"])

        st.divider()
        st.subheader("About this demo")
        st.caption(f"Model: {meta['model_name']}")
        st.caption(f"Test ROC-AUC: {meta['test_metrics']['roc_auc']:.3f}")
        st.caption(f"Recall: {meta['test_metrics']['recall']:.3f}  "
                  f"Precision: {meta['test_metrics']['precision']:.3f}")
        st.caption(f"Trained on {meta['data_rows']} customers "
                  f"({meta['churn_rate'] * 100:.1f}% churn).")
        st.caption("Data: IBM Telco Customer Churn (public).")

    # ---------- main ----------
    st.title("Churn Radar")
    st.write(
        "Score a telecom customer's risk of leaving. Adjust the profile in the "
        "sidebar, or upload a CSV for batch scoring."
    )

    tab_single, tab_batch = st.tabs(["Single customer", "Batch scoring"])

    with tab_single:
        if st.button("Score customer", type="primary"):
            row = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": online_sec,
                "OnlineBackup": online_bak,
                "DeviceProtection": device,
                "TechSupport": tech,
                "StreamingTV": stream_tv,
                "StreamingMovies": stream_mv,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": tenure * monthly,
            }
            X = pd.DataFrame([row], columns=feature_order)
            prob = float(model.predict_proba(X)[0, 1])
            phrase, deltas = explain_one(model, meta, row)
            st.session_state["last_result"] = (prob, phrase, deltas)

        if "last_result" in st.session_state:
            prob, phrase, deltas = st.session_state["last_result"]
            band, kind = risk_band(prob, meta)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Churn probability", f"{prob:.1%}")
                getattr(st, kind)(f"{band} risk")
                st.progress(min(max(prob, 0.0), 1.0))
                st.caption(f"Decision threshold: {meta['decision_threshold']:.0%}")
            with col2:
                st.info(phrase)
                st.caption(
                    "Caveat: occlusion is conditional. It swaps one feature at a "
                    "time, so correlated features (tenure / total charges) can "
                    "flip the apparent sign. The headline is cross-checked "
                    "against marginal churn, but treat it as a prompt to "
                    "investigate, not proof of cause."
                )
                with st.expander("Why this score?"):
                    contrib = sorted(deltas, key=lambda d: -d[1])
                    contrib_df = pd.DataFrame(
                        [
                            {
                                "Factor": factor_text(c, v, baselines),
                                "Effect on risk": round(d, 4),
                                "Marginal churn": marginal_label(c, v, meta),
                            }
                            for c, d, v in contrib
                        ]
                    )
                    st.dataframe(contrib_df, width="stretch", hide_index=True)
                    st.caption(
                        "Effect = change in churn probability if that feature were "
                        "set to its typical value. Positive pushes risk up. "
                        "The headline ignores effects that conflict with marginal "
                        "churn (marked 'lower churn')."
                    )
        else:
            st.caption("Click 'Score customer' to see a prediction.")

    with tab_batch:
        st.subheader("Score a CSV of customers")
        template_rows = [
            {
                "gender": "Female", "SeniorCitizen": "No", "Partner": "No",
                "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5,
                "TotalCharges": 179.0,
            },
            {
                "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes",
                "Dependents": "Yes", "tenure": 48, "PhoneService": "Yes",
                "MultipleLines": "Yes", "InternetService": "DSL",
                "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
                "DeviceProtection": "Yes", "TechSupport": "Yes",
                "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "One year", "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 65.0, "TotalCharges": 3120.0,
            },
            {
                "gender": "Male", "SeniorCitizen": "Yes", "Partner": "Yes",
                "Dependents": "No", "tenure": 60, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "Two year", "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 25.0, "TotalCharges": 1500.0,
            },
        ]
        template_df = pd.DataFrame(template_rows, columns=feature_order)
        st.download_button(
            "Download CSV template",
            template_df.to_csv(index=False).encode("utf-8"),
            "churn_template.csv",
            "text/csv",
        )

        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                st.stop()

            try:
                cleaned = clean_uploaded(raw, meta)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            if len(cleaned) > 500:
                st.warning("More than 500 rows; scoring the first 500.")
                cleaned = cleaned.head(500).copy()
                raw = raw.head(500).copy()

            try:
                X = cleaned[feature_order].copy()
                for col in meta["features"]["numeric"]:
                    X[col] = pd.to_numeric(X[col], errors="raise")
                proba = model.predict_proba(X)[:, 1]
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")
                st.stop()

            factors = batch_explain(model, meta, X)
            result = raw.copy()
            result["Churn probability"] = np.round(proba, 4)
            result["Risk"] = [risk_band(p, meta)[0] for p in proba]
            result["Main factor"] = factors
            result = result.sort_values("Churn probability", ascending=False)

            st.dataframe(result, width="stretch")
            st.caption(
                "Caveat: occlusion is conditional. It swaps one feature at a "
                "time, so correlated features (tenure / total charges) can flip "
                "the apparent sign. The main factor is cross-checked against "
                "marginal churn; a conflicting effect is marked 'conditional' "
                "rather than 'risk'. Treat it as a prompt to investigate, not "
                "proof of cause."
            )
            st.download_button(
                "Download scored CSV",
                result.to_csv(index=False).encode("utf-8"),
                "churn_scored.csv",
                "text/csv",
            )
            st.caption(f"Scored {len(result)} customers.")


if __name__ == "__main__":
    main()