# California Water System Enforcement Risk Predictor

Machine learning dashboard predicting which California drinking water systems are at risk of a formal Safe Drinking Water Act (SDWA) enforcement action, with per-facility explainability.

**Key result:** ~0.815 cross-validated ROC-AUC on real historical enforcement outcomes.

## What it does

- Loads EPA ECHO facility data, filtered to active Safe Drinking Water Act (SDWIS-flagged) facilities in California
- Trains a Random Forest classifier (`scikit-learn` `Pipeline` with median imputation, class-balanced) to predict whether a facility will receive a formal enforcement action, using six features: population density, percent minority population, inspection count, days since last inspection, informal enforcement count, and quarters with non-compliance
- Validates with stratified train/test splitting to avoid data leakage
- Explains individual predictions with SHAP (per-facility waterfall plots), not just global feature importance
- Serves the result through an interactive Streamlit dashboard: risk tier filtering, a probability distribution histogram, a Folium map of facilities colored by risk tier, a top-20 highest-risk table, and per-facility SHAP explanations

## Tech stack

Python, scikit-learn (RandomForestClassifier), SHAP, Streamlit, Plotly, Folium

## Data source

EPA ECHO facility data — the full `ECHO_EXPORTER.csv` bulk download, available from [EPA ECHO Data Downloads](https://echo.epa.gov/tools/data-downloads). Not included in this repo (too large for GitHub) — see `.gitignore`.

## Running locally

```bash
pip install -r requirements.txt
streamlit run risk_app.py
```

You'll need `ECHO_EXPORTER.csv` in the project root (see Data source above).

## Live demo

[Add live Streamlit Cloud link here once deployed]

## Author

Prathyusha Paresh — [github.com/prathyusha0618](https://github.com/prathyusha0618)
