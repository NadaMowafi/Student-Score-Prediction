from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]
    model_config = ConfigDict(extra="ignore")

class PredictResponse(BaseModel):
    predictions: List[float]
