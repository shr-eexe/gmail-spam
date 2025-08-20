# Spam Email Classifier (TF-IDF + Logistic Regression)

A compact, production-ready machine learning project for classifying emails as spam/ham.

## Features
- Clean train/evaluate pipeline with scikit-learn
- TF-IDF text features + Logistic Regression (tunable)
- Stratified train/validation split with robust metrics
- Model persistence (vectorizer + model) using `joblib`
- Minimal FastAPI app for inference
- CLI utilities for batch prediction

## Dataset
Use any labeled CSV with columns:
- `text` (email text body/subject combined or body only)
- `label` (0 = ham, 1 = spam)

Examples: Enron Spam Dataset, Kaggle SMS Spam Collection (rename columns accordingly).

## Quickstart

```bash
# 1) Create env and install
pip install -r requirements.txt

# 2) Place your dataset at data/emails.csv with columns: text,label
#    OR pass --csv path via CLI.

# 3) Train and evaluate
python src/train.py --csv data/emails.csv --model_dir models

# 4) Batch predict on a CSV
python src/predict.py --csv data/emails.csv --model_dir models --out data/preds.csv

# 5) Run API
uvicorn api.app:app --reload --port 8000

# 6) cURL test
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"texts": ["FREE entry in 2 a wkly comp!!!", "Hey, our meeting at 5?"]}'
```

## Evaluation
`train.py` prints accuracy, precision, recall, F1 (macro & weighted), ROC-AUC, confusion matrix.
You can also enable cross-validation and basic hyperparameter search.

## Resume bullets (you can adapt after you train on a real dataset)
- Built a spam email classifier using TF-IDF + Logistic Regression; achieved >95% accuracy and >0.95 F1 on validation.
- Implemented a FastAPI microservice for real-time inference (<30 ms p95 on local tests).
- Added reproducible pipeline with config flags, model versioning, and unit-friendly structure.
