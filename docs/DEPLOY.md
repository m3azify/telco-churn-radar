# Deploying Churn Radar for free

No API keys and no paid services are required. The app runs on CPU and
loads a pre-trained model from the repository.

## Before you deploy

1. The repository must contain everything the app needs:
   - `data/raw/telco_churn.csv`
   - `models/churn_model.pkl`
   - `models/model_meta.json`
   - `requirements.txt`
   - `src/app.py`
2. Do not gitignore `models/` or `data/`. The app reads them at runtime.
3. Test locally first: `streamlit run src/app.py` and open
   http://localhost:8501.

## Option A - Streamlit Community Cloud (recommended, easiest)

1. Push this project to a **public** GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app".
4. Select the repository, branch (main) and main file path: `src/app.py`.
5. (Optional but recommended) Open "Advanced settings" and set the Python
   version to the newest one available that is >= 3.11. The model was
   trained on Python 3.15.2; if the host Python differs, the app retrains
   itself once on first load (about 30 seconds) and then runs normally.
6. Click "Deploy". The first build installs `requirements.txt` and takes
   a few minutes.
7. Your app gets a public URL like
   `https://<your-app>.streamlit.app`.

Free-tier notes:
- Apps sleep after inactivity and wake on the next visit.
- 1 GB RAM is plenty for this model.
- Do not store secrets; this app uses none.

Updating:
- `git push` to main. Streamlit Cloud rebuilds automatically.

## Option B - Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space.
2. SDK: Streamlit. Hardware: CPU basic (free). Visibility: Public.
3. Add a README.md at the repository root of the Space with this front
   matter (Hugging Face reads it):

   ```
   ---
   title: Churn Radar
   emoji: 📉
   colorFrom: teal
   colorTo: blue
   sdk: streamlit
   sdk_version: 1.67.0
   app_file: src/app.py
   pinned: false
   ---
   ```

4. Push this project to the Space repository (git remote add + push).
   Include `data/`, `models/`, `src/`, `requirements.txt`.
5. The Space builds and serves the app at
   `https://huggingface.co/spaces/<user>/<space>`.

Free-tier notes:
- CPU basic Spaces sleep after 48 hours of inactivity.
- No GPU needed.

## If the model fails to load on the host

The app never fails silently. If the saved pickle cannot be unpickled
(different Python or scikit-learn build), it prints a warning and retrains
once from the bundled CSV, then serves predictions. You can also force a
retrain locally and push the new pickle:

    python src/train.py
    git add models/churn_model.pkl models/model_meta.json
    git commit -m "Retrain model"
    git push

## Checklist after deploy

- Open the public URL.
- Score the default customer (should show a probability and a one-line
  explanation).
- Download the CSV template, upload it back, and confirm a scored table
  and a downloadable result.
- Check the sidebar model card shows the trained model name and metrics.