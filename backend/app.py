"""
SmartWater Agriculture — FastAPI Backend
=========================================
REST API serving the SmartWater Bayesian Decision Engine.

Endpoints
─────────
GET  /              → API info
GET  /health        → Health check (used by Streamlit frontend)
GET  /structure     → Bayesian network structure
POST /predict-irrigation  → Irrigation advisory
POST /predict-drilling    → Borehole / drilling risk
POST /predict-warning     → Early warning system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
import uvicorn

from bayesian_model import SmartWaterModel


# ══════════════════════════════════════════════════════════════════════════════
# APP INIT
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SmartWater Agriculture API",
    description=(
        "Bayesian Decision Support System for groundwater & irrigation advisory. "
        "Igabi & Zaria LGAs · Kaduna State, Nigeria. "
        "ABU Zaria & IAR Zaria · M4D Open Innovation Challenge 2025/26."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Streamlit URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the Bayesian model once at startup (singleton)
model = SmartWaterModel()


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class IrrigationRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "Rainfall": 0,
        "SeasonalForecast": 1,
        "SoilMoisture": 0,
        "CropWaterNeed": 2,
        "GroundwaterStress": 1,
    }})

    # Directly observed (preferred — overrides network priors)
    SoilMoisture: Optional[int] = Field(
        None, ge=0, le=2,
        description="0=Dry, 1=Adequate, 2=Saturated"
    )
    CropWaterNeed: Optional[int] = Field(
        None, ge=0, le=2,
        description="0=Low, 1=Medium, 2=High"
    )
    GroundwaterStress: Optional[int] = Field(
        None, ge=0, le=2,
        description="0=Low, 1=Moderate, 2=High"
    )
    # Optional upstream evidence (improves inference accuracy)
    Rainfall: Optional[int] = Field(
        None, ge=0, le=2,
        description="0=Low, 1=Moderate, 2=High"
    )
    SeasonalForecast: Optional[int] = Field(
        None, ge=0, le=2,
        description="0=WetSeason, 1=DrySpell, 2=DroughtWarning"
    )


class IrrigationResponse(BaseModel):
    p_safe:     float
    p_careful:  float
    p_delay:    float
    verdict:    str
    confidence: str
    timestamp:  str


class DrillingRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "BoreholeDepth": 2,
        "GeologyFavorability": 2,
    }})

    BoreholeDepth: int = Field(
        ..., ge=0, le=2,
        description="0=Shallow(<20m), 1=Medium(20-60m), 2=Deep(>60m)"
    )
    GeologyFavorability: int = Field(
        ..., ge=0, le=2,
        description="0=Unfavorable, 1=Moderate, 2=Favorable"
    )


class DrillingResponse(BaseModel):
    p_success:   float
    p_uncertain: float
    p_risk:      float
    verdict:     str
    confidence:  str
    timestamp:   str


class WarningRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "PumpAge": 2,
        "SeasonalForecast": 2,
        "GroundwaterStress": 2,
    }})

    PumpAge: int = Field(
        ..., ge=0, le=2,
        description="0=New(<2yrs), 1=Mid(2-5yrs), 2=Old(>5yrs)"
    )
    SeasonalForecast: int = Field(
        ..., ge=0, le=2,
        description="0=WetSeason, 1=DrySpell, 2=DroughtWarning"
    )
    GroundwaterStress: int = Field(
        ..., ge=0, le=2,
        description="0=Low, 1=Moderate, 2=High"
    )


class WarningResponse(BaseModel):
    p_normal:   float
    p_watch:    float
    p_critical: float
    verdict:    str
    confidence: str
    timestamp:  str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    timestamp:    str
    version:      str


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
async def root():
    return {
        "api":         "SmartWater Agriculture",
        "version":     "1.0.0",
        "description": "Bayesian Decision Engine for groundwater & irrigation advisory",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints": {
            "irrigation": "POST /predict-irrigation",
            "drilling":   "POST /predict-drilling",
            "warning":    "POST /predict-warning",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check — called by Streamlit frontend every 30 seconds."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
    )


@app.get("/structure", tags=["Info"])
async def get_structure():
    """Return Bayesian network nodes and edges for documentation."""
    return model.get_structure()


@app.post("/predict-irrigation", response_model=IrrigationResponse, tags=["Advisory"])
async def predict_irrigation(req: IrrigationRequest):
    """
    Irrigation Advisory — should the farmer irrigate today?

    Provide at minimum SoilMoisture + GroundwaterStress.
    Adding Rainfall and SeasonalForecast improves accuracy.
    """
    evidence = {k: v for k, v in req.model_dump().items() if v is not None}

    if not evidence:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one observation (SoilMoisture, GroundwaterStress, etc.)"
        )

    try:
        result = model.predict_irrigation(evidence)
        return IrrigationResponse(
            **result,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-drilling", response_model=DrillingResponse, tags=["Advisory"])
async def predict_drilling(req: DrillingRequest):
    """
    Borehole Drilling Risk — is this a good site to drill?

    Requires BoreholeDepth and GeologyFavorability.
    """
    evidence = req.model_dump()

    try:
        result = model.predict_drilling(evidence)
        return DrillingResponse(
            **result,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-warning", response_model=WarningResponse, tags=["Advisory"])
async def predict_warning(req: WarningRequest):
    """
    Early Warning System — anticipate water stress before it hits.

    Requires PumpAge, SeasonalForecast, and GroundwaterStress.
    """
    evidence = req.model_dump()

    try:
        result = model.predict_warning(evidence)
        return WarningResponse(
            **result,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("🌊 SMARTWATER AGRICULTURE API")
    print("=" * 55)
    print("Docs  : http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("=" * 55)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")