"""
SmartWater Agriculture — FastAPI Backend v2.0
=============================================
Endpoints
─────────
GET  /                        API info
GET  /health                  Health check
GET  /stats                   System statistics
GET  /model-info              Bayesian network metadata

POST /farmers/register        Register a new farmer
GET  /farmers/{farmer_id}     Get farmer profile
GET  /farmers/{farmer_id}/history   Full assessment history

POST /surveys/submit          Submit a field survey observation
GET  /surveys/{farmer_id}     Get farmer's past surveys

POST /predict-irrigation      Irrigation advisory
POST /predict-drilling        Borehole / drilling risk
POST /predict-warning         Early water stress warning

POST /feedback                Log farmer feedback (thumbs up/down)
POST /import-csv              Bulk import historical CSV data
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime, date
import json
import csv
import io

from bayesian_model import SmartWaterModel
import database as db


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
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton Bayesian model ─────────────────────────────────────────────────
model = SmartWaterModel()


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS — FARMERS
# ══════════════════════════════════════════════════════════════════════════════

class FarmerRegisterRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "name": "Musa Ibrahim", "village": "Rigachikun", "lga": "Igabi",
        "phone": "08012345678", "farm_size_ha": 2.5,
        "crops_grown": ["Maize", "Tomato"],
        "borehole_depth": 1, "pump_age": 2,
        "geology_obs": 1, "language_pref": "ha"
    }})
    name:           str
    village:        Optional[str] = None
    lga:            Optional[str] = None
    phone:          Optional[str] = None
    farm_size_ha:   Optional[float] = None
    crops_grown:    Optional[List[str]] = None
    borehole_depth: Optional[int] = Field(None, ge=0, le=2)
    pump_age:       Optional[int] = Field(None, ge=0, le=2)
    geology_obs:    Optional[int] = Field(None, ge=0, le=2)
    language_pref:  Optional[str] = "en"


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS — SURVEYS
# ══════════════════════════════════════════════════════════════════════════════

class SurveySubmitRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "farmer_id": "uuid-here",
        "survey_date": "2026-05-02",
        "rainfall_obs": 0, "soil_moisture": 0,
        "groundwater_level": 1, "seasonal_forecast": 1,
        "community_obs": 0, "crop_type": "Maize",
        "crop_stage": 2, "crop_water_need": 2,
        "pump_working": True, "pump_age_obs": 2,
        "actual_outcome": "irrigated",
        "outcome_successful": True,
        "notes": "Soil very dry, irrigated in the morning",
        "recorded_by": "Musa"
    }})
    farmer_id:          Optional[str]  = None
    survey_date:        Optional[str]  = None   # ISO date string YYYY-MM-DD
    rainfall_obs:       Optional[int]  = Field(None, ge=0, le=2)
    soil_moisture:      Optional[int]  = Field(None, ge=0, le=2)
    groundwater_level:  Optional[int]  = Field(None, ge=0, le=2)
    seasonal_forecast:  Optional[int]  = Field(None, ge=0, le=2)
    community_obs:      Optional[int]  = Field(None, ge=0, le=2)
    crop_type:          Optional[str]  = None
    crop_stage:         Optional[int]  = Field(None, ge=0, le=2)
    crop_water_need:    Optional[int]  = Field(None, ge=0, le=2)
    pump_working:       Optional[bool] = None
    pump_age_obs:       Optional[int]  = Field(None, ge=0, le=2)
    actual_outcome:     Optional[str]  = None
    outcome_successful: Optional[bool] = None
    notes:              Optional[str]  = None
    recorded_by:        Optional[str]  = None


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

class IrrigationRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "Rainfall": 0, "SeasonalForecast": 1,
        "SoilMoisture": 0, "CropWaterNeed": 2,
        "GroundwaterStress": 1,
        "farmer_id": "uuid-here",
    }})
    SoilMoisture:       Optional[int] = Field(None, ge=0, le=2)
    CropWaterNeed:      Optional[int] = Field(None, ge=0, le=2)
    GroundwaterStress:  Optional[int] = Field(None, ge=0, le=2)
    Rainfall:           Optional[int] = Field(None, ge=0, le=2)
    SeasonalForecast:   Optional[int] = Field(None, ge=0, le=2)
    farmer_id:          Optional[str] = None
    survey_id:          Optional[str] = None


class DrillingRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "BoreholeDepth": 2, "GeologyFavorability": 2,
        "farmer_id": "uuid-here",
    }})
    BoreholeDepth:        int = Field(..., ge=0, le=2)
    GeologyFavorability:  int = Field(..., ge=0, le=2)
    farmer_id:            Optional[str] = None
    survey_id:            Optional[str] = None


class WarningRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "PumpAge": 2, "SeasonalForecast": 2,
        "GroundwaterStress": 2, "farmer_id": "uuid-here",
    }})
    PumpAge:            int = Field(..., ge=0, le=2)
    SeasonalForecast:   int = Field(..., ge=0, le=2)
    GroundwaterStress:  int = Field(..., ge=0, le=2)
    farmer_id:          Optional[str] = None
    survey_id:          Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    assessment_id:  Optional[str]  = None
    farmer_id:      Optional[str]  = None
    module:         str
    verdict:        str
    helpful:        bool
    comment:        Optional[str]  = None


# ══════════════════════════════════════════════════════════════════════════════
# GENERAL ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
async def root():
    return {
        "api":      "SmartWater Agriculture",
        "version":  "2.0.0",
        "docs":     "/docs",
        "health":   "/health",
        "stats":    "/stats",
    }


@app.get("/health", tags=["General"])
async def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "db_status":    "connected" if db.DB_AVAILABLE else "offline",
        "timestamp":    datetime.utcnow().isoformat(),
        "version":      "2.0.0",
    }


@app.get("/stats", tags=["General"])
async def stats():
    """System-wide statistics — useful for researchers and dashboard."""
    return db.get_system_stats()


@app.get("/model-info", tags=["General"])
async def model_info():
    return model.info()


# ══════════════════════════════════════════════════════════════════════════════
# FARMER ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/farmers/register", tags=["Farmers"])
async def register_farmer(req: FarmerRegisterRequest):
    """
    Register a new farmer. Returns farmer_id which should be stored
    on the frontend for all future requests.
    """
    # Check if phone already exists
    if req.phone:
        existing = db.get_farmer_by_phone(req.phone)
        if existing:
            return {
                "message":   "Farmer already registered",
                "farmer_id": existing["farmer_id"],
                "farmer":    existing,
            }

    farmer_data = req.model_dump(exclude_none=True)
    saved = db.register_farmer(farmer_data)

    if not saved:
        # DB offline — return a temporary session ID
        return {
            "message":   "Registered (session only — DB offline)",
            "farmer_id": None,
            "farmer":    farmer_data,
        }

    return {
        "message":   "Farmer registered successfully",
        "farmer_id": saved["farmer_id"],
        "farmer":    saved,
    }


@app.get("/farmers/{farmer_id}", tags=["Farmers"])
async def get_farmer(farmer_id: str):
    """Get farmer profile by UUID."""
    farmer = db.get_farmer_by_id(farmer_id)
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    return farmer


@app.get("/farmers/{farmer_id}/history", tags=["Farmers"])
async def get_farmer_history(farmer_id: str, limit: int = 50):
    """
    Full assessment history for a farmer.
    Powers the personalised history tab in the frontend.
    """
    history = db.get_farmer_history(farmer_id, limit=limit)
    surveys = db.get_farmer_surveys(farmer_id, limit=limit)
    return {
        "farmer_id":   farmer_id,
        "assessments": history,
        "surveys":     surveys,
        "total":       len(history),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SURVEY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/surveys/submit", tags=["Surveys"])
async def submit_survey(req: SurveySubmitRequest):
    """
    Submit a field observation from an extension agent or farmer.
    This data feeds the CPT parameter learning pipeline.
    """
    survey_data = req.model_dump(exclude_none=True)
    saved = db.submit_survey(survey_data)

    if not saved:
        return {
            "message":  "Survey received (DB offline — not persisted)",
            "survey_id": None,
        }

    return {
        "message":   "Survey saved successfully",
        "survey_id": saved["survey_id"],
        "survey":    saved,
    }


@app.get("/surveys/{farmer_id}", tags=["Surveys"])
async def get_surveys(farmer_id: str, limit: int = 20):
    """Get recent field surveys for a farmer."""
    surveys = db.get_farmer_surveys(farmer_id, limit=limit)
    return {"farmer_id": farmer_id, "surveys": surveys, "total": len(surveys)}


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _log_prediction(module: str, farmer_id, survey_id,
                    evidence: dict, result: dict):
    """Background task — log prediction to assessments table."""
    db.log_assessment({
        "farmer_id":    farmer_id,
        "survey_id":    survey_id,
        "module":       module,
        "prediction":   result.get("prediction", ""),
        "p_values":     {k: v for k, v in result.items()
                         if k.startswith("p_")},
        "confidence":   result.get("confidence", ""),
        "model_used":   result.get("model_used", ""),
        "evidence_used": evidence,
    })


@app.post("/predict-irrigation", tags=["Advisory"])
async def predict_irrigation(req: IrrigationRequest,
                              background_tasks: BackgroundTasks):
    evidence = {
        k: v for k, v in req.model_dump().items()
        if v is not None and k not in ["farmer_id", "survey_id"]
    }
    if not evidence:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one observation."
        )
    try:
        result = model.predict_irrigation(
            soil_moisture      = evidence.get("SoilMoisture", 1),
            crop_water_need    = evidence.get("CropWaterNeed", 1),
            groundwater_stress = evidence.get("GroundwaterStress", 1),
            rainfall           = evidence.get("Rainfall", 1),
            seasonal_forecast  = evidence.get("SeasonalForecast", 1),
        )
        if db.DB_AVAILABLE and req.farmer_id:
            background_tasks.add_task(
                _log_prediction, "irrigation",
                req.farmer_id, req.survey_id, evidence, result
            )
        return {**result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-drilling", tags=["Advisory"])
async def predict_drilling(req: DrillingRequest,
                            background_tasks: BackgroundTasks):
    try:
        result = model.predict_drilling(
            borehole_depth      = req.BoreholeDepth,
            geology_favorability = req.GeologyFavorability,
        )
        evidence = {"BoreholeDepth": req.BoreholeDepth,
                    "GeologyFavorability": req.GeologyFavorability}
        if db.DB_AVAILABLE and req.farmer_id:
            background_tasks.add_task(
                _log_prediction, "drilling",
                req.farmer_id, req.survey_id, evidence, result
            )
        return {**result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-warning", tags=["Advisory"])
async def predict_warning(req: WarningRequest,
                           background_tasks: BackgroundTasks):
    try:
        result = model.predict_warning(
            pump_age           = req.PumpAge,
            groundwater_stress = req.GroundwaterStress,
            seasonal_forecast  = req.SeasonalForecast,
        )
        evidence = {"PumpAge": req.PumpAge,
                    "GroundwaterStress": req.GroundwaterStress,
                    "SeasonalForecast": req.SeasonalForecast}
        if db.DB_AVAILABLE and req.farmer_id:
            background_tasks.add_task(
                _log_prediction, "warning",
                req.farmer_id, req.survey_id, evidence, result
            )
        return {**result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(req: FeedbackRequest):
    """Save farmer thumbs up / down feedback."""
    saved = db.log_feedback(req.model_dump(exclude_none=True))
    if req.assessment_id and req.helpful is not None:
        db.update_assessment_feedback(req.assessment_id, req.helpful)
    return {
        "message": "Feedback saved" if saved else "Feedback received (DB offline)",
        "saved":   saved is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CSV BULK IMPORT
# ══════════════════════════════════════════════════════════════════════════════

# Mapping from plain English CSV values → CPT integer codes
RAINFALL_MAP    = {"low":0,"little":0,"none":0,"moderate":1,"some":1,"high":2,"heavy":2}
SOIL_MAP        = {"dry":0,"very dry":0,"moist":1,"adequate":1,"wet":2,"saturated":2,"muddy":2}
GW_MAP          = {"normal":0,"stable":0,"low":1,"slightly low":1,"very low":2,"much lower":2}
SEASON_MAP      = {"wet":0,"rains":0,"dry spell":1,"dry":1,"drought":2}
COMMUNITY_MAP   = {"fine":0,"ok":0,"few issues":1,"some":1,"many":2,"struggling":2}
CROP_STAGE_MAP  = {"early":0,"seedling":0,"mid":1,"vegetative":1,"flowering":2,"fruiting":2}
DEPTH_MAP       = {"shallow":0,"<20m":0,"medium":1,"20-60m":1,"deep":2,">60m":2}
PUMP_MAP        = {"new":0,"<2yrs":0,"mid":1,"2-5yrs":1,"old":2,">5yrs":2}
GEO_MAP         = {"unfavorable":0,"poor":0,"moderate":1,"mixed":1,"favorable":2,"good":2}


def _map(value: str, mapping: dict, default: int = 1) -> int:
    if value is None: return default
    return mapping.get(str(value).strip().lower(), default)


@app.post("/import-csv", tags=["Data Import"])
async def import_csv(file: UploadFile = File(...)):
    """
    Bulk import historical farmer survey data from CSV.

    Expected CSV columns (plain English values — see /docs for mapping):
    farmer_name, village, lga, phone, farm_size_ha,
    borehole_depth, pump_age, geology,
    date, rainfall, soil_moisture, groundwater_level,
    seasonal_forecast, crop_type, crop_stage, crop_water_need,
    community_obs, pump_working, actual_outcome,
    outcome_successful, notes, recorded_by
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    contents = await file.read()
    reader   = csv.DictReader(io.StringIO(contents.decode("utf-8-sig")))

    imported  = 0
    skipped   = 0
    errors    = []
    farmer_cache: dict = {}   # phone → farmer_id

    for i, row in enumerate(reader, start=2):  # row 1 = header
        try:
            # ── Register / look up farmer ────────────────────────────
            phone = row.get("phone", "").strip() or None
            farmer_id = None

            if phone and phone in farmer_cache:
                farmer_id = farmer_cache[phone]
            else:
                farmer_data = {
                    "name":           row.get("farmer_name","").strip() or "Unknown",
                    "village":        row.get("village","").strip() or None,
                    "lga":            row.get("lga","").strip() or None,
                    "phone":          phone,
                    "borehole_depth": _map(row.get("borehole_depth"), DEPTH_MAP),
                    "pump_age":       _map(row.get("pump_age"), PUMP_MAP),
                    "geology_obs":    _map(row.get("geology"), GEO_MAP),
                }
                try:
                    farm_size = float(row.get("farm_size_ha","0") or 0)
                    farmer_data["farm_size_ha"] = farm_size if farm_size > 0 else None
                except ValueError:
                    pass

                # Check if already exists
                existing = db.get_farmer_by_phone(phone) if phone else None
                if existing:
                    farmer_id = existing["farmer_id"]
                else:
                    saved = db.register_farmer(
                        {k: v for k, v in farmer_data.items() if v is not None}
                    )
                    farmer_id = saved["farmer_id"] if saved else None

                if phone and farmer_id:
                    farmer_cache[phone] = farmer_id

            # ── Build survey row ──────────────────────────────────────
            pump_raw = str(row.get("pump_working","")).strip().lower()
            pump_bool = True if pump_raw in ["yes","true","1"] else (
                        False if pump_raw in ["no","false","0"] else None)

            outcome_raw = str(row.get("outcome_successful","")).strip().lower()
            outcome_bool = True if outcome_raw in ["yes","true","1"] else (
                           False if outcome_raw in ["no","false","0"] else None)

            survey_data = {
                "farmer_id":         farmer_id,
                "survey_date":       row.get("date","").strip() or None,
                "rainfall_obs":      _map(row.get("rainfall"), RAINFALL_MAP),
                "soil_moisture":     _map(row.get("soil_moisture"), SOIL_MAP),
                "groundwater_level": _map(row.get("groundwater_level"), GW_MAP),
                "seasonal_forecast": _map(row.get("seasonal_forecast"), SEASON_MAP),
                "community_obs":     _map(row.get("community_obs"), COMMUNITY_MAP),
                "crop_type":         row.get("crop_type","").strip() or None,
                "crop_stage":        _map(row.get("crop_stage"), CROP_STAGE_MAP),
                "crop_water_need":   _map(row.get("crop_water_need"), CROP_STAGE_MAP),
                "pump_working":      pump_bool,
                "pump_age_obs":      _map(row.get("pump_age"), PUMP_MAP),
                "actual_outcome":    row.get("actual_outcome","").strip() or None,
                "outcome_successful":outcome_bool,
                "notes":             row.get("notes","").strip() or None,
                "recorded_by":       row.get("recorded_by","").strip() or None,
            }
            survey_data = {k: v for k, v in survey_data.items() if v is not None}

            saved = db.submit_survey(survey_data)
            if saved:
                imported += 1
            else:
                skipped += 1

        except Exception as e:
            errors.append({"row": i, "error": str(e)})
            skipped += 1

    return {
        "message":        f"Import complete — {imported} surveys saved, {skipped} skipped.",
        "imported":       imported,
        "skipped":        skipped,
        "errors":         errors[:20],   # show first 20 errors max
        "farmers_cached": len(farmer_cache),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("🌊 SMARTWATER AGRICULTURE API v2.0")
    print("=" * 55)
    print("Docs  : http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("=" * 55)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)