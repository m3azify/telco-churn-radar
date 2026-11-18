"""Download the telecom churn dataset.

Checks for data/raw/telco_churn.csv first. If it is missing, tries to
download it from the IBM GitHub mirror (free, no login). If the network
is unavailable, prints exact fallback instructions instead of failing
silently.

Run from the project root: python src/00_download.py
"""
import os
import sys
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "data", "raw", "telco_churn.csv")
PRIMARY_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def validate(path):
    """Return True if the file exists and has the expected schema."""
    if not os.path.exists(path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return list(df.columns) == EXPECTED_COLUMNS and len(df) > 1000
    except Exception:
        return False


def main():
    print("Checking for", TARGET)
    if validate(TARGET):
        import pandas as pd
        df = pd.read_csv(TARGET)
        print("OK: dataset present")
        print("  rows:", len(df))
        print("  churn rate:", round((df["Churn"] == "Yes").mean() * 100, 2), "%")
        return

    print("Dataset not found or invalid. Trying primary source...")
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    try:
        urllib.request.urlretrieve(PRIMARY_URL, TARGET)
        if validate(TARGET):
            print("Downloaded successfully.")
            import pandas as pd
            df = pd.read_csv(TARGET)
            print("  rows:", len(df))
            return
        raise RuntimeError("downloaded file failed validation")
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        # Remove a half-written file so the next run starts clean.
        if os.path.exists(TARGET):
            os.remove(TARGET)
        print("ERROR: could not download the dataset.")
        print("Reason:", exc)
        print()
        print("Exact fallback instructions:")
        print("  1) Kaggle (needs a free Kaggle account):")
        print("     https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("     Download Telco-Customer-Churn.csv and save it to:")
        print("     " + TARGET)
        print("  2) Fully offline (no internet):")
        print("     python src/00_generate_sample.py")
        print("     python src/00_generate_sample.py --out data/raw/telco_churn.csv")
        print("     The second command labels the data as synthetic in output.")
        sys.exit(1)


if __name__ == "__main__":
    main()