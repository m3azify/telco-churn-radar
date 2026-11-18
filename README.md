# Churn Radar - telco customer churn prediction

A small, end-to-end machine learning project: predict which telecom
customers are likely to leave, explain why, and ship it as a clickable
web app. Built as a portfolio piece for Moaz (final-year CS & Data
Science, Alexandria) to show clients that he can take a problem from
raw CSV to a deployed AI product.

Live demo: not deployed yet. Follow `docs/DEPLOY.md` (about 10 minutes,
free tier). A screenshot of the running app is in
`output/app_screenshot.png` once you run it locally.

## Results

Trained on 7,021 customers (26.5% churn), 80/20 stratified split,
random seed 42. Model selected by 5-fold cross-validation ROC-AUC on the
training set, then evaluated once on the untouched test set (1,405
customers).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (always "stays") | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.742 | 0.508 | 0.780 | 0.615 | 0.840 |
| **Random Forest (selected)** | **0.762** | **0.537** | **0.737** | **0.621** | **0.838** |

- Selected model: RandomForest (CV ROC-AUC 0.8467 +/- 0.0161).
- Confusion matrix: 797 true stays, 236 false alarms, 98 missed churners,
  274 caught churners.
- Business check: scoring only the top 20% riskiest customers catches
  49% of all churners in the test set.

## Findings and recommendations

Churn is concentrated and explainable. From the EDA and the model's
feature importance:

1. Contract type is the strongest single lever. Month-to-month customers
   churn far more than one- or two-year customers.
2. Tenure matters most early. Most churn happens in the first ~12 months.
3. Fiber-optic customers churn more than DSL or no-internet customers.
4. Electronic-check payers churn more than automatic-payment customers.
5. Customers without TechSupport or OnlineSecurity churn more.

Three concrete actions a telecom could take:

1. **Convert month-to-month customers to annual contracts.** Offer a
   discount or one free month for signing a 12-month contract. Target the
   highest-risk month-to-month customers first using the model's ranking.
2. **Run a first-year onboarding programme.** Proactive check-ins and a
   small loyalty incentive before month 12, where most churn happens.
   This is the cheapest intervention per churner prevented.
3. **Bundle TechSupport and OnlineSecurity into the fiber plan and
   incentivise automatic payment.** Both are associated with materially
   lower churn, and automatic payment also reduces billing friction.

Note: these are associations, not proven causes. A/B test each action
before rolling it out company-wide.

## How it works

1. `src/00_download.py` fetches the IBM CSV (or prints exact fallback
   instructions if offline).
2. `src/train.py` cleans the data, builds a preprocessing + model
   pipeline, compares a dummy baseline against Logistic Regression and
   Random Forest, selects by cross-validation ROC-AUC, and saves the
   model, metadata, metrics and three plots.
3. `src/app.py` is a Streamlit app: sidebar customer inputs or CSV upload
   -> churn probability, risk band, a one-line explanation, and a
   contribution table. Explanations use occlusion (replace each feature
   with its training baseline and measure the probability drop), so they
   are derived from the model rather than hard-coded.
4. `docs/DEPLOY.md` has exact free deploy steps.

## Quickstart (Windows)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/00_download.py
    python src/train.py
    streamlit run src/app.py

Then open http://localhost:8501.

If you have no internet, generate a synthetic dataset instead (labelled
synthetic in the README and in the script output):

    python src/00_generate_sample.py --out data/raw/telco_churn.csv
    python src/train.py

## Project structure

    proj3/
      README.md
      LEARNING.md
      requirements.txt
      .gitignore
      data/raw/telco_churn.csv
      src/00_download.py
      src/00_generate_sample.py
      src/train.py
      src/app.py
      models/churn_model.pkl
      models/model_meta.json
      output/metrics.txt
      output/confusion_matrix.png
      output/roc_curve.png
      output/feature_importance.png
      docs/DEPLOY.md
      docs/HOW_TO_REBUILD.md

## Dataset

Primary: IBM Telco Customer Churn (public, no login).
Fallback: Kaggle `blastchar/telco-customer-churn` (free account) or the
synthetic generator in `src/00_generate_sample.py` (seed 42). Any run on
synthetic data is labelled as such in the README and script output.

## Limitations and honest caveats

- The model is trained on a 2018 public sample. Real deployments need
  recent data and retraining.
- Feature importance shows association, not causation.
- Occlusion explanations share credit across correlated features
  (tenure and TotalCharges).
- Class imbalance is handled with `class_weight="balanced"`, but the
  decision threshold is left at 0.5. A production system should tune it
  against the cost of a missed churner vs. a wasted call.
- No personal data is used; the dataset is public and anonymised.

## What I would do next

- Calibrate probabilities and tune the decision threshold.
- Temporal validation on a later month of data.
- Compare a gradient-boosted model.
- Add monitoring of the score distribution and churn rate.

## Credits

Dataset: IBM Telco Customer Churn sample, public. This project is not
affiliated with IBM. Built with scikit-learn, pandas, matplotlib and
Streamlit. Free tools only, no API keys.