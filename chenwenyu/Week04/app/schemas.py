from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResult(BaseModel):
    text: str
    predicted_label: str
    predicted_class: int
    confidence: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class TrainRequest(BaseModel):
    epochs: Optional[int] = 4
    batch_size: Optional[int] = 16