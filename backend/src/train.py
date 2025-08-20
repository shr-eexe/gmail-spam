import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns.")
    df = df.dropna(subset=['text', 'label'])
    # Coerce label to int (0/1)
    df['label'] = df['label'].astype(int)
    return df

def build_pipeline(max_features: int = 30000, ngram_max: int = 2, C: float = 2.0):
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max), lowercase=True)),
        ('clf', LogisticRegression(max_iter=2000, C=C, n_jobs=None, class_weight='balanced'))
    ])
    return pipe

def evaluate(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)
    print("Accuracy:", round(acc, 4))
    print("Precision (binary):", round(p, 4), "Recall:", round(r, 4), "F1:", round(f1, 4))
    print("Precision (weighted):", round(p_w, 4), "Recall:", round(r_w, 4), "F1:", round(f1_w, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/emails.csv')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--max_features', type=int, default=30000)
    parser.add_argument('--ngram_max', type=int, default=2)
    parser.add_argument('--C', type=float, default=2.0)
    parser.add_argument('--grid_search', action='store_true', help='Run a small grid search for C and ngrams')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading data from:", args.csv)
    df = load_data(args.csv)

    X = df['text'].astype(str).values
    y = df['label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    if args.grid_search:
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True)),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
        ])
        param_grid = {
            'tfidf__max_features': [20000, 30000, 40000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__C': [0.5, 1.0, 2.0, 3.0]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
        gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=1, scoring='f1')
        gs.fit(X_train, y_train)
        print("Best params:", gs.best_params_)
        model = gs.best_estimator_
    else:
        model = build_pipeline(args.max_features, args.ngram_max, args.C)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    try:
        y_prob = model.predict_proba(X_val)[:,1]
    except Exception:
        # Some classifiers may not support predict_proba
        y_prob = None

    print("Evaluation on validation set:")
    evaluate(y_val, y_pred, y_prob)

    # Persist
    model_path = os.path.join(args.model_dir, 'spam_model.joblib')
    joblib.dump(model, model_path)
    print("Saved model to:", model_path)

if __name__ == '__main__':
    main()
