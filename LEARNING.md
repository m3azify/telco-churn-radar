# Learning guide: how this project works

Written for Moaz (and for future-you in an interview). Every claim below
maps to a real file in this repository. Read the code and this document
together.

## 1. The business problem

A telecom loses money every time a customer leaves. Retention is cheaper
than acquisition, but a retention team cannot call every customer. So we
build a **ranking tool**: score every customer by churn probability, then
call the top of the list. Accuracy alone is a trap here (see section 6).

## 2. The data

`data/raw/telco_churn.csv` is the IBM Telco Customer Churn sample:
7,043 customers, 21 columns, 26.5% churn. Target column is `Churn`
(Yes/No). The rest are demographics, services and billing.

Key facts discovered in `src/train.py`:
- `customerID` is an identifier, not a signal. Dropping it prevents the
  model from memorising individuals.
- `TotalCharges` is stored as text and has 11 blanks. All 11 are
  tenure-0 customers, so their lifetime charges are genuinely 0.
- 22 exact duplicate rows were removed.
- Final training set: 7,021 rows, 5,616 train / 1,405 test.

## 3. Cleaning rules (and why)

- Drop `customerID`: identifiers do not generalise.
- `TotalCharges` blanks -> 0: a new customer has paid nothing yet.
  Imputing the median would invent money that was never charged.
- `SeniorCitizen` 0/1 -> "No"/"Yes": one vocabulary for every binary
  column, so the app and the model agree.
- `Churn` -> 1/0: scikit-learn expects a numeric target.

## 4. Train/test split

    train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

- 80/20 is the standard starting point.
- `stratify=y` keeps the 26.5% churn rate in both halves. Without it, a
  random split could give the test set 22% or 31% churn and make the
  metrics meaningless.
- `random_state=42` makes the split reproducible: rerunning gives the
  same numbers, which is essential for a portfolio project.

## 5. Preprocessing

A `ColumnTransformer` applies different steps to numeric and categorical
columns, all inside one `Pipeline`:

- Numeric: `SimpleImputer(median)` then `StandardScaler`.
- Categorical: `SimpleImputer(most_frequent)` then
  `OneHotEncoder(handle_unknown="ignore")`.

Why inside a Pipeline: the fitted preprocessor is saved with the model.
The app can hand the model a raw customer row and get a probability back,
with no separate preprocessing code to keep in sync.

`handle_unknown="ignore"` means a category seen only in production (for
example a new payment method) does not crash the app; it contributes
zero to every one-hot column.

## 6. Why accuracy is a trap

The dummy baseline predicts "no churn" for everyone and scores 73.5%
accuracy, because 73.5% of customers stay. A model that never flags a
single churner looks great on accuracy and is useless to the business.
That is why we report precision, recall, F1 and ROC-AUC, and why the
dummy is included in `output/metrics.txt` as proof of value.

## 7. The two candidate models

- **Logistic Regression**: linear, fast, coefficients are directly
  interpretable ("each extra month of tenure lowers log-odds by X").
- **Random Forest**: 300 trees, captures non-linear interactions
  (tenure x contract type), harder to explain but usually strong on tabular
  data.

Both use `class_weight="balanced"` because churn is the minority class.
This tells the model to pay more attention to churners, trading some
precision for recall, which is usually the right business trade-off.

## 8. Metrics (what each one means)

- **Accuracy**: fraction correct. Misleading under imbalance.
- **Precision**: of the customers we flagged, how many actually churned.
- **Recall**: of the customers who churned, how many we flagged.
- **F1**: harmonic mean of precision and recall.
- **ROC-AUC**: probability that a random churner scores higher than a
  random stayer. 0.5 = useless, 1.0 = perfect. Robust to imbalance.

Actual results (test set, 1,405 customers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (prior) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.742 | 0.508 | 0.780 | 0.615 | 0.840 |
| Random Forest (selected) | 0.762 | 0.537 | 0.737 | 0.621 | 0.838 |

The winner is chosen by **5-fold cross-validation ROC-AUC on the training
set** (Random Forest 0.8467 vs Logistic Regression 0.8459), then
evaluated once on the untouched test set. Selecting on the test set would
leak information and flatter the result.

## 9. Cross-validation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` splits the
training data into 5 folds, trains on 4, validates on 1, and repeats.
The mean ROC-AUC is a more honest estimate of how the model will do on
new data than a single split. We keep the test set completely out of this.

## 10. Feature importance

For the selected Random Forest, `feature_importances_` (impurity-based)
ranks the features. Top 5:

1. tenure (0.128)
2. Contract = Month-to-month (0.121)
3. TotalCharges (0.106)
4. MonthlyCharges (0.078)
5. OnlineSecurity = No (0.059)

Caveat: impurity importance measures how much a feature is used in splits,
not causation. A feature can rank high because it is a proxy for another.
Always pair it with business reasoning (section 12).

## 11. The one-line explanation (occlusion)

The app must explain *why* a specific customer is risky. Instead of
adding a heavy dependency (SHAP), we use **occlusion**: for each raw
feature, replace it with its training-set baseline (median for numbers,
mode for categories), re-score, and measure the drop in probability.
A large positive drop means that feature is pushing this customer toward
churn. We report the top one or two. This is model-agnostic, free, and
genuinely derived from the model.

Limitation: correlated features (tenure and TotalCharges) share credit, so
the explanation is a ranking aid, not a causal claim.

## 12. Findings and recommendations

From the EDA in `output/metrics.txt`:
- Month-to-month customers churn far more than one- or two-year customers.
- Fiber-optic customers churn more than DSL or no-internet customers.
- Electronic-check payers churn more than automatic-payment customers.
- Customers without TechSupport or OnlineSecurity churn more.
- Churn is concentrated in the first ~12 months of tenure.

Three concrete actions:
1. **Convert month-to-month to annual contracts** with a discount or a
   free month; contract type is the strongest single lever.
2. **First-year onboarding programme**: proactive contact and a loyalty
   incentive before month 12, where most churn happens.
3. **Bundle TechSupport and OnlineSecurity** into the base fiber plan and
   incentivise automatic payment; both are associated with materially
   lower churn.

Business check: scoring only the top 20% riskiest customers catches 49%
of all churners in the test set, so the retention team's time is well
spent.

## 13. Saving the model

`joblib.dump(pipeline, "models/churn_model.pkl")` stores the entire
preprocessor + model as one object. `model_meta.json` stores the schema,
categories, baselines, metrics and versions the app needs. Keeping them
separate means the app can display provenance and retrain safely if the
pickle is incompatible with the host Python.

## 14. The Streamlit app

`src/app.py` runs top-to-bottom on every interaction, which is why we use
`@st.cache_resource` for the model (load once) and `st.session_state` for
the last result. Structure:
- Sidebar: customer inputs + model card.
- Main tabs: single customer (score + explanation + contribution table)
  and batch scoring (CSV upload, validation, scoring, download).
- If the pickle cannot load, the app retrains once from the bundled CSV
  and then serves predictions. No silent failures.

## 15. Deployment

See `docs/DEPLOY.md`. Two free options: Streamlit Community Cloud and
Hugging Face Spaces. Both install `requirements.txt` and run
`streamlit run src/app.py`. No secrets, no API keys.

## 16. Interview Q&A

**Q: Why not just use accuracy?**
A: The dummy baseline gets 73.5% accuracy by predicting "no churn" for
everyone. We report ROC-AUC and recall to prove the model adds value.

**Q: Why stratify the split?**
A: Churn is 26.5%. A random split could unbalance the test set and make
metrics noisy. Stratification preserves the rate.

**Q: How did you pick the model?**
A: 5-fold CV ROC-AUC on the training set only, then one final evaluation
on the untouched test set. Random Forest won by 0.0008.

**Q: How does the explanation work?**
A: Occlusion: replace each feature with its training baseline, measure
the probability drop, report the biggest. Model-agnostic, no SHAP.

**Q: What would you improve next?**
A: Calibrate probabilities (Platt/ isotonic), tune the decision threshold
for the business cost ratio, add temporal validation on a later month, and
retrain monthly. Also test a gradient-boosted model.

**Q: What is the biggest risk in this system?**
A: Data drift: if the telecom's plans or pricing change, the model goes
stale. Mitigation: monitor the score distribution and churn rate monthly,
and retrain on fresh data.

## 17. Glossary

- **Churn**: customer leaves the service.
- **Stratified split**: train/test split that preserves class proportions.
- **One-hot encoding**: turning a category into 0/1 columns.
- **ROC-AUC**: ranking quality of a classifier, 0.5 to 1.0.
- **Recall**: fraction of real churners caught.
- **Precision**: fraction of flagged customers who actually churn.
- **Cross-validation**: rotating validation folds to estimate generalisation.
- **Occlusion**: measuring a feature's effect by replacing it with a baseline.
- **Pipeline**: chained preprocessing + model that can be saved and reused.