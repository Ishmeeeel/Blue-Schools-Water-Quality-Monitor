from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from bayesian_model import BoreholeWaterQualityModel

app = FastAPI(
    title="Water Quality Bayesian API",
    version="1.0"
)

# Load model ONCE (important)
model = BoreholeWaterQualityModel()


class EvidenceRequest(BaseModel):
    evidence: Dict[str, int]


@app.get("/")
def health_check():
    return {"status": "API running"}


@app.post("/predict/contamination")
def predict_contamination(req: EvidenceRequest):
    return model.predict_contamination_risk(req.evidence)


@app.get("/predict/pump/{pump_age}")
def predict_pump(pump_age: int):
    return model.predict_pump_status(pump_age)
