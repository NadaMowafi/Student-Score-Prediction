
# Student Score Predictor — Deployment

Production-ready scaffold to deploy your **scikit-learn Pipeline** for predicting student exam scores.

## Quick Start (local)

1) Create a virtualenv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Export your trained Pipeline:
```python
# In your training notebook/script
from joblib import dump
# pipeline is a single sklearn Pipeline that includes preprocessing + estimator
dump(pipeline, "artifacts/model.joblib")

# also save the exact training feature names/order
import json
with open("artifacts/columns.json","w") as f:
    json.dump(feature_names, f)  # e.g., list of strings in order used for training
```

3) Run the API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Swagger docs at: `http://localhost:8000/docs`

5) Try a request:
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d @sample_request.json
```

## Streamlit UI (optional)
```bash
streamlit run streamlit_app.py
```

## Docker
```bash
docker build -t student-score:latest .
docker run -p 8000:8000 student-score:latest
```

## Deployment targets
- **Render**: push repo, create a Web Service (for FastAPI) or a Static/Service for Streamlit.
- **Railway/Fly.io**: similar Docker-based deploy.
- **Hugging Face Spaces**: use the Streamlit app (`streamlit_app.py`) and `requirements.txt`.

## Notes
- Keep the model wrapped as a single `Pipeline` that includes all preprocessing (encoders/scalers) to avoid schema drift.
- `columns.json` ensures you pass features in the same order used at training time.
- Validate inputs with Pydantic; unknown fields are ignored and missing required ones trigger 422 errors.

