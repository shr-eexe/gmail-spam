from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Email Classifier API")

model = joblib.load("models/spam_model.joblib")

class PredictRequest(BaseModel):
    texts: list[str]

class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float] | None = None

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    preds = model.predict(req.texts).tolist()
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(req.texts)[:, 1].tolist()
    return PredictResponse(predictions=preds, probabilities=probs)
