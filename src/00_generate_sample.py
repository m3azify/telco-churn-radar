"""Generate a fully-offline synthetic telecom churn dataset.

This is a FALLBACK for machines without internet. The real IBM dataset
is used by default; this script is only for offline demos. Any dataset
produced here is synthetic and labelled as such in the README.

Run:
  python src/00_generate_sample.py
  python src/00_generate_sample.py --out data/raw/telco_churn.csv
"""
import argparse
import os
import numpy as np
import pandas as pd


def generate(n, seed=42):
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n)
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, size=n)

    phone = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple = np.where(phone == "No", "No phone service",
                        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]))

    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])

    def addon():
        out = np.where(internet == "No", "No internet service",
                       rng.choice(["Yes", "No"], size=n, p=[0.38, 0.62]))
        return out

    online_sec = addon()
    online_bak = addon()
    device_prot = addon()
    tech_sup = addon()
    stream_tv = addon()
    stream_mv = addon()

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n,
                          p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges correlated with services.
    monthly = np.full(n, 20.0)
    monthly += np.where(phone == "Yes", 10.0, 0.0)
    monthly += np.where(internet == "DSL", 25.0, 0.0)
    monthly += np.where(internet == "Fiber optic", 45.0, 0.0)
    for a in (online_sec, online_bak, device_prot, tech_sup, stream_tv, stream_mv):
        monthly += np.where(a == "Yes", 5.0, 0.0)
    monthly += rng.normal(0, 4, size=n)
    monthly = np.round(np.clip(monthly, 18.0, 120.0), 2)

    total = np.where(tenure == 0, 0.0,
                     np.round(tenure * monthly + rng.normal(0, 30, size=n), 2))
    total = np.clip(total, 0.0, None)

    # Churn score: higher = more likely to leave.
    score = -1.2
    score = score + np.where(contract == "Month-to-month", 1.3, 0.0)
    score = score + np.where(contract == "Two year", -0.9, 0.0)
    score = score - 0.04 * tenure
    score = score + 0.015 * monthly
    score = score + np.where(internet == "Fiber optic", 0.6, 0.0)
    score = score + np.where(tech_sup == "No", 0.5, 0.0)
    score = score + np.where(online_sec == "No", 0.3, 0.0)
    score = score + np.where(payment == "Electronic check", 0.5, 0.0)
    score = score + np.where(senior == 1, 0.3, 0.0)
    score = score - np.where(partner == "Yes", 0.3, 0.0)
    score = score - np.where(dependents == "Yes", 0.3, 0.0)
    score = score + rng.normal(0, 0.6, size=n)

    prob = 1.0 / (1.0 + np.exp(-score))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"C-{i:05d}" for i in range(1, n + 1)],
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
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mv,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic churn data.")
    parser.add_argument("--out", default="data/raw/telco_churn_synthetic.csv",
                        help="output CSV path")
    parser.add_argument("--n", type=int, default=2000, help="number of rows")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Generated", len(df), "synthetic rows ->", out_path)
    print("Churn rate:", round((df["Churn"] == "Yes").mean() * 100, 2), "%")
    print("NOTE: this dataset is SYNTHETIC (seed=%d). Label it as such." % args.seed)


if __name__ == "__main__":
    main()