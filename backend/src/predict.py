import argparse
import os
import pandas as pd
import joblib

def load_model(model_dir: str):
    path = os.path.join(model_dir, 'spam_model.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No model found at {path}. Run training first.')
    return joblib.load(path)

def predict_texts(texts, model):
    preds = model.predict(texts)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)[:,1]
    return preds, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='CSV with a column named text')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--out', type=str, default='preds.csv')
    args = parser.parse_args()

    model = load_model(args.model_dir)
    df = pd.read_csv(args.csv)
    if 'text' not in df.columns:
        raise ValueError("CSV must have a 'text' column.")
    preds, probs = predict_texts(df['text'].astype(str).values, model)
    df_out = df.copy()
    df_out['pred'] = preds
    if probs is not None:
        df_out['prob_spam'] = probs
    df_out.to_csv(args.out, index=False)
    print(f"Wrote predictions to {args.out}")

if __name__ == '__main__':
    main()
