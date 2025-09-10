# app/schemas.py
from pydantic import BaseModel, Field
from typing import List

class InlineFeature(BaseModel):
    cut_torque: float
    cut_lag_error: float
    cut_position: float
    cut_speed: float
    film_position: float
    film_speed: float
    film_lag_error: float

class InferenceRequestCSV(BaseModel):
    run_id: str = Field(..., description="실행 ID")
    csv_path: str
    window_size: int = 128
    stride: int = 1   # ✅

class InferenceRequestInline(BaseModel):
    run_id: str
    features: List[InlineFeature]
    window_size: int = 128
    stride: int = 1   # ✅

class InferenceSummary(BaseModel):
    run_id: str
    rows_inserted: int
    message: str
