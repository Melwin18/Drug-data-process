# Early Guard — Understand Risk. Act Early.

An explainable AI-powered early drug-abuse risk assessment and early warning platform. It uses multidimensional psychological, behavioural, social, family, lifestyle, academic/work, and substance-related factors to classify assessed risk into **Low**, **Moderate**, or **High** categories, with SHAP-based explainability for transparent, model-associated factor contributions.

> Early Guard is a research-oriented decision-support and awareness system. It is **not** a medical diagnosis and does **not** guarantee prediction of future addiction.

## 1. Project Overview
This project is an academic prototype for an AI-Based Early Drug Abuse Risk Assessment and Early Warning System. It estimates substance-use risk using 35 carefully selected behavioural, psychological, social, family, lifestyle, academic/work, digital, and attitudinal factors.

## 2. Features
- **Risk Assessment**: Interactive form capturing 35 structured inputs.
- **Prediction**: Real-time risk estimation (Low / Moderate / High) via a pre-trained Logistic Regression model.
- **SHAP Explanation**: Dynamic feature influence explanations to understand the model's decision-making process.
- **Model Transparency**: Performance metrics, cross-validation details, and dataset information are strictly documented in the app.

## 3. ML Methodology
The underlying model uses a completely leakage-free, Scikit-learn Pipeline (SimpleImputer + StandardScaler/OneHotEncoder + Logistic Regression). Feature selection (Kruskal-Wallis, Chi-Squared, RF/XGB/Permutation Importance) and hyperparameter tuning were performed exclusively on a 4,000-row training set, keeping a 1,000-row test set completely isolated.

## 4. Model Information
- **Best Model**: Logistic Regression
- **Predictors**: 35
- **Target**: `Drug_Abuse_Risk`

## 5. Installation
Ensure you have Python 3.8+ installed.

## 6. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## 7. Running Locally

### Web client (Early Guard — recommended)
The primary interface is a dependency-light web client served by a standard-library HTTP server. It serves the static client and a small JSON API over the trained model and audit artefacts.

```bash
python webapp/server.py                 # http://127.0.0.1:8000
python webapp/server.py --port 9000     # custom port
```

Then open:
- `http://127.0.0.1:8000/` — landing page
- `http://127.0.0.1:8000/app` — desktop assessment app
- `http://127.0.0.1:8000/mobile` — mobile screens

### Streamlit client (alternative)
```bash
streamlit run app/app.py
```

## 8. Project Structure
```
├── app/
│   ├── app.py                     # Main application entry point
│   ├── pages/                     # Application pages
│   │   ├── 1_Home.py
│   │   ├── 2_Risk_Assessment.py
│   │   ├── 3_Model_Performance.py
│   │   ├── 4_Feature_Importance.py
│   │   └── 5_Model_Information.py
│   └── utils/
│       └── utils.py               # Shared logic (loading models, SHAP, configs)
├── models/                        # Pre-trained models and metadata
├── results/                       # Performance metrics and plots
├── data/                          # Locked datasets
└── requirements.txt
```

## 9. Deployment

### Render (via GitHub)
The web client is deployment-ready for [Render](https://render.com). The server binds `0.0.0.0` and reads the `PORT` environment variable that Render injects, so no code changes are needed.

**Option A — Blueprint (uses `render.yaml`):**
1. Push this repository to GitHub.
2. In Render: **New + → Blueprint**, connect the repo, and apply.

**Option B — Manual Web Service:**
| Setting | Value |
|---------|-------|
| Environment | Python 3 |
| Build Command | `pip install -r webapp/requirements.txt` |
| Start Command | `python webapp/server.py` |
| Health Check Path | `/api/health` |
| Python Version | `3.12.7` (env var `PYTHON_VERSION`) |

> **Python ≥ 3.11 is required.** `models/best_model_pipeline.pkl` was produced with scikit-learn 1.9.0, which only ships wheels for Python 3.11+. The pinned `webapp/requirements.txt` installs the matching scikit-learn so the model loads without version warnings.

The Streamlit client can alternatively be deployed on Streamlit Community Cloud. No retraining occurs during deployment.

## 10. Limitations
- **Data Constraints**: The underlying dataset relies on cross-sectional self-reported data, making it susceptible to response bias.
- **Target Constraints**: The target `Drug_Abuse_Risk` is an operational research label inferred from usage indicators, not a clinically established diagnostic boundary.

## 11. Research Disclaimer
**IMPORTANT DISCLAIMER**: This application is intended for academic and research demonstration only. It provides an estimated risk classification based on statistical patterns learned from the training dataset. It is NOT a medical diagnostic system and cannot determine whether a person is currently using drugs or will develop addiction. SHAP values explain the model's mathematical behaviour and do not establish biological or psychosocial causality. Always consult qualified healthcare professionals for medical advice or assessment.
