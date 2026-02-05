"""
FastAPI Backend Server for Blue Schools Bayesian Water Quality System
======================================================================

This API provides endpoints for:
1. Real-time contamination risk prediction
2. Pump failure assessment
3. Model information and variable details
4. Health checks

Author: Blue Schools Project
Date: 2026
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime

from bayesian_model import BoreholeWaterQualityModel

# Initialize FastAPI app
app = FastAPI(
    title="Blue Schools Water Quality API",
    description="Bayesian Decision Support System for monitoring water quality in Nigerian school boreholes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Bayesian model (singleton)
bayesian_model = BoreholeWaterQualityModel()

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class ObservationInput(BaseModel):
    """
    Input model for water quality observations.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rainfall": 2,
                "turbidity": 1,
                "latrine_distance": 0,
                "school_name": "Government Primary School Kano",
                "location": "Kano City",
                "reporter_name": "Musa Ahmed"
            }
        }
    )
    
    rainfall: Optional[int] = Field(
        None,
        ge=0,
        le=2,
        description="Rainfall level: 0=Low, 1=Medium, 2=High"
    )
    turbidity: Optional[int] = Field(
        None,
        ge=0,
        le=2,
        description="Water clarity: 0=Clear, 1=Slightly Cloudy, 2=Very Cloudy"
    )
    surface_runoff: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Surface runoff: 0=No, 1=Yes"
    )
    latrine_distance: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Latrine proximity: 0=Safe (>30m), 1=Risky (<30m)"
    )
    pump_age: Optional[int] = Field(
        None,
        ge=0,
        le=2,
        description="Pump age: 0=New (<2yr), 1=Medium (2-5yr), 2=Old (>5yr)"
    )
    
    school_name: Optional[str] = Field(None, description="Name of the school")
    location: Optional[str] = Field(None, description="School location")
    reporter_name: Optional[str] = Field(None, description="Name of person reporting")


class ContaminationRiskResponse(BaseModel):
    """
    Response model for contamination risk assessment.
    """
    safe_probability: float = Field(..., description="Probability water is safe (0-1)")
    contamination_probability: float = Field(..., description="Probability of contamination (0-1)")
    risk_level: str = Field(..., description="Risk category: LOW, MODERATE, HIGH, CRITICAL")
    recommendation: str = Field(..., description="Action recommendation")
    evidence_used: Dict[str, int] = Field(..., description="Variables used in calculation")
    timestamp: str = Field(..., description="Time of assessment")
    confidence: str = Field(..., description="Confidence level of prediction")


class PumpStatusResponse(BaseModel):
    """
    Response model for pump status assessment.
    """
    working_probability: float
    failure_probability: float
    maintenance_needed: bool
    recommendation: str
    pump_age_category: str


class HealthCheckResponse(BaseModel):
    """
    Response model for API health check.
    """
    status: str
    timestamp: str
    model_loaded: bool
    api_version: str


class VariableInfoResponse(BaseModel):
    """
    Response model for variable information.
    """
    variables: Dict
    total_nodes: int
    total_edges: int


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Blue Schools Water Quality Monitoring API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "github": "https://github.com/blue-schools/bayesian-ai"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
async def health_check():
    """
    Check if API is running and model is loaded.
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=bayesian_model is not None,
        api_version="1.0.0"
    )


@app.post("/predict", response_model=ContaminationRiskResponse, tags=["Prediction"])
async def predict_contamination(observation: ObservationInput):
    """
    Predict water contamination risk based on observations.
    
    This endpoint accepts field observations and returns a contamination
    risk assessment with actionable recommendations.
    
    **Example Request:**
    ```json
    {
        "rainfall": 2,
        "turbidity": 1,
        "latrine_distance": 0,
        "school_name": "GPS Kano"
    }
    ```
    """
    try:
        # Build evidence dictionary (only include non-None values)
        evidence = {}
        
        if observation.rainfall is not None:
            evidence['Rainfall'] = observation.rainfall
        if observation.turbidity is not None:
            evidence['Turbidity'] = observation.turbidity
        if observation.surface_runoff is not None:
            evidence['Surface_Runoff'] = observation.surface_runoff
        if observation.latrine_distance is not None:
            evidence['Latrine_Dist'] = observation.latrine_distance
        
        # Check if we have at least one observation
        if not evidence:
            raise HTTPException(
                status_code=400,
                detail="At least one observation (rainfall, turbidity, surface_runoff, or latrine_distance) must be provided"
            )
        
        # Get prediction from model
        result = bayesian_model.predict_contamination_risk(evidence)
        
        # Generate recommendation based on risk level
        recommendation = _generate_recommendation(
            result['risk_level'],
            result['contamination_probability']
        )
        
        # Determine confidence based on number of observations
        confidence = _calculate_confidence(len(evidence))
        
        return ContaminationRiskResponse(
            safe_probability=result['safe_probability'],
            contamination_probability=result['contamination_probability'],
            risk_level=result['risk_level'],
            recommendation=recommendation,
            evidence_used=evidence,
            timestamp=datetime.now().isoformat(),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-pump", response_model=PumpStatusResponse, tags=["Prediction"])
async def predict_pump_status(
    pump_age: int = Query(..., ge=0, le=2, description="Pump age: 0=New (<2yr), 1=Medium (2-5yr), 2=Old (>5yr)")
):
    """
    Predict pump failure probability based on age.
    
    **Pump Age Categories:**
    - 0: New (<2 years)
    - 1: Medium (2-5 years)
    - 2: Old (>5 years)
    """
    try:
        result = bayesian_model.predict_pump_status(pump_age)
        
        age_categories = ["New (<2 years)", "Medium (2-5 years)", "Old (>5 years)"]
        
        # Generate maintenance recommendation
        if result['failure_probability'] > 0.2:
            recommendation = "⚠️ URGENT: Schedule immediate inspection and maintenance"
        elif result['failure_probability'] > 0.1:
            recommendation = "⚡ IMPORTANT: Plan maintenance within next 2 weeks"
        else:
            recommendation = "✅ GOOD: Continue routine monitoring"
        
        return PumpStatusResponse(
            working_probability=result['working_probability'],
            failure_probability=result['failure_probability'],
            maintenance_needed=result['maintenance_needed'],
            recommendation=recommendation,
            pump_age_category=age_categories[pump_age]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pump prediction error: {str(e)}")


@app.get("/variables", response_model=VariableInfoResponse, tags=["Information"])
async def get_variable_info():
    """
    Get information about all variables in the Bayesian model.
    
    Returns variable names, possible states, and descriptions.
    """
    try:
        variables = bayesian_model.get_variable_info()
        
        return VariableInfoResponse(
            variables=variables,
            total_nodes=len(bayesian_model.model.nodes()),
            total_edges=len(bayesian_model.model.edges())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching variables: {str(e)}")


@app.get("/model-structure", tags=["Information"])
async def get_model_structure():
    """
    Get the structure of the Bayesian network.
    
    Returns nodes and edges for visualization.
    """
    try:
        nodes = list(bayesian_model.model.nodes())
        edges = list(bayesian_model.model.edges())
        
        return {
            "nodes": nodes,
            "edges": [{"from": edge[0], "to": edge[1]} for edge in edges],
            "description": "Causal relationships in the water quality model"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching structure: {str(e)}")


@app.post("/sensitivity-analysis", tags=["Analysis"])
async def analyze_sensitivity(observation: ObservationInput):
    """
    Perform sensitivity analysis to identify most impactful variables.
    
    Shows which observations have the biggest effect on contamination risk.
    """
    try:
        # Build evidence dictionary
        evidence = {}
        if observation.rainfall is not None:
            evidence['Rainfall'] = observation.rainfall
        if observation.turbidity is not None:
            evidence['Turbidity'] = observation.turbidity
        if observation.surface_runoff is not None:
            evidence['Surface_Runoff'] = observation.surface_runoff
        if observation.latrine_distance is not None:
            evidence['Latrine_Dist'] = observation.latrine_distance
        
        if not evidence:
            raise HTTPException(status_code=400, detail="At least one observation required")
        
        sensitivity = bayesian_model.sensitivity_analysis('Contamination', evidence)
        
        # Rank by impact
        ranked = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "sensitivity_scores": sensitivity,
            "most_impactful": ranked[0][0] if ranked else None,
            "ranked_variables": [{"variable": var, "impact": score} for var, score in ranked]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis error: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_recommendation(risk_level: str, probability: float) -> str:
    """
    Generate actionable recommendation based on risk level.
    """
    recommendations = {
        "LOW": "✅ Water appears safe. Continue routine monitoring.",
        "MODERATE": "⚠️ CAUTION: Consider boiling water or using water purification tablets. Monitor closely.",
        "HIGH": "🚨 WARNING: Do NOT drink without treatment. Boil water for at least 3 minutes or use chlorine treatment.",
        "CRITICAL": "🔴 DANGER: Water likely contaminated. DO NOT USE for drinking or cooking. Contact health authorities immediately."
    }
    
    return recommendations.get(risk_level, "Monitor water quality closely")


def _calculate_confidence(num_observations: int) -> str:
    """
    Calculate confidence level based on number of observations.
    """
    if num_observations >= 4:
        return "HIGH"
    elif num_observations >= 2:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌊 BLUE SCHOOLS WATER QUALITY API")
    print("=" * 60)
    print("Starting server...")
    print("Documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
