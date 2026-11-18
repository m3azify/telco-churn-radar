# How to rebuild this project without AI

This guide assumes Windows 11 and Python 3.11 or newer. Every command is
run from the project root unless noted.

## 1. Create the project folder

    mkdir freelance-portfolio\proj3
    cd freelance-portfolio\proj3

## 2. Create and activate a virtual environment

    python -m venv .venv
    .venv\Scripts\Activate.ps1

If PowerShell blocks the script, run this once in the same window:

    Set-ExecutionPolicy -Scope Process Bypass

Verify:

    python --version

## 3. Install pinned dependencies

    pip install -r requirements.txt

Verify:

    python -c "import sklearn, pandas, streamlit; print(sklearn.__version__, pandas.__version__, streamlit.__version__)"

## 4. Get the data

    python src/00_download.py

Expected: "OK: dataset present" with about 7000 rows.

If you are offline, generate a synthetic dataset instead (labelled
synthetic in the README):

    python src/00_generate_sample.py --out data/raw/telco_churn.csv

## 5. Train the model

    python src/train.py

Expected: a metrics table, a selected model, 3 PNGs in output/, and
models/churn_model.pkl. Training takes about 15-30 seconds.

Check:

    dir output
    dir models

## 6. Inspect the results

Open output/metrics.txt. Confirm:
- The dummy baseline has ROC-AUC 0.500 and recall 0.000.
- Both real models beat it on ROC-AUC.
- The confusion matrix and top features are written.

## 7. Run the web app

    streamlit run src/app.py

Open http://localhost:8501. Then:
- Move the sliders and click "Score customer". A probability, a risk band
  and a one-line explanation should appear.
- Download the CSV template, upload it back, and check the scored table
  and the download button.

Stop the app with Ctrl+C.

## 8. Verify the model loads independently

    python -c "import joblib; m=joblib.load('models/churn_model.pkl'); print(type(m).__name__)"

## Common problems

- "python is not recognized": install Python from python.org and tick
  "Add Python to PATH", then reopen the terminal.
- "No module named streamlit": the virtual environment is not active;
  run the activation command from step 2.
- Port 8501 already in use: stop the other process or run
  `streamlit run src/app.py --server.port 8502`.
- Blank TotalCharges errors: the cleaning code converts blanks to 0 for
  tenure-0 customers. If you edit the cleaning, keep that rule.
- Model file missing after training: check that models/ is not in
  .gitignore and that the script ran to the end (look for "Saved:").