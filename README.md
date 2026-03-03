# MetaHealth End-to-End Project

This project trains a metabolic risk classifier from CSV files in `data/` and serves predictions with:
- modern predictor UI
- user authentication (register/login/logout)
- saved per-user prediction history
- dashboard charts (risk trend, risk distribution, feature importance)

## Local run

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open: `http://127.0.0.1:5000`

## One-command Docker deployment

Build and run:

```bash
docker compose up --build
```

Open: `http://127.0.0.1:5000`

Notes:
- `model_bundle.pkl` must exist before starting container. Generate it with `python train_model.py`.
- User data and prediction history are stored in `metahealth.db`.

## Project flow

1. Load and merge demographic, examination, labs, questionnaire, and diet data by `SEQN`
2. Build engineered target label `Metabolic_Risk`
3. Train/test split, scale numeric features, train logistic regression
4. Predict via Flask app and persist results per authenticated user
5. Visualize trends and feature effects on dashboard
