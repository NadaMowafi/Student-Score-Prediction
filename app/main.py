from fastapi import FastAPI, HTTPException
from app.model.load import get_model, get_feature_names
from app.schemas import PredictRequest, PredictResponse
import pandas as pd

app = FastAPI(title="Student Score Predictor", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    pipeline = get_model()
    feature_names = get_feature_names()
    if pipeline is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Model or schema missing in artifacts/")

    try:
        df = pd.DataFrame(payload.instances)
        df = df.reindex(columns=feature_names)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Input schema mismatch: {e}")

    try:
        preds = pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictResponse(predictions=[float(x) for x in preds])
