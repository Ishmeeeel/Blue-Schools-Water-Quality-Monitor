import numpy as np
from typing import Dict

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


def _confidence(probs: np.ndarray) -> str:
    m = float(np.max(probs))
    if m >= 0.75: return "HIGH"
    if m >= 0.50: return "MEDIUM"
    return "LOW"


def build_model():
    model = DiscreteBayesianNetwork([
        ("Rainfall",            "SoilMoisture"),
        ("SeasonalForecast",    "SoilMoisture"),
        ("SeasonalForecast",    "GroundwaterStress"),
        ("Rainfall",            "GroundwaterStress"),
        ("SoilMoisture",        "IrrigationRisk"),
        ("CropWaterNeed",       "IrrigationRisk"),
        ("GroundwaterStress",   "IrrigationRisk"),
        ("BoreholeDepth",       "DrillingSuccess"),
        ("GeologyFavorability", "DrillingSuccess"),
        ("PumpAge",             "EarlyWarning"),
        ("GroundwaterStress",   "EarlyWarning"),
        ("SeasonalForecast",    "EarlyWarning"),
    ])

    cpd_rainfall = TabularCPD("Rainfall", 3, [[0.50],[0.35],[0.15]])
    cpd_forecast = TabularCPD("SeasonalForecast", 3, [[0.40],[0.40],[0.20]])
    cpd_crop     = TabularCPD("CropWaterNeed", 3, [[0.30],[0.50],[0.20]])
    cpd_depth    = TabularCPD("BoreholeDepth", 3, [[0.40],[0.40],[0.20]])
    cpd_geology  = TabularCPD("GeologyFavorability", 3, [[0.30],[0.40],[0.30]])
    cpd_pump     = TabularCPD("PumpAge", 3, [[0.30],[0.40],[0.30]])

    cpd_soil = TabularCPD(
        "SoilMoisture", 3,
        values=[
            [0.10,0.60,0.85, 0.05,0.30,0.60, 0.02,0.10,0.25],
            [0.60,0.35,0.13, 0.60,0.55,0.35, 0.28,0.55,0.55],
            [0.30,0.05,0.02, 0.35,0.15,0.05, 0.70,0.35,0.20],
        ],
        evidence=["Rainfall","SeasonalForecast"],
        evidence_card=[3,3],
    )

    cpd_gw = TabularCPD(
        "GroundwaterStress", 3,
        values=[
            [0.20,0.55,0.80, 0.15,0.40,0.65, 0.10,0.20,0.40],
            [0.50,0.35,0.15, 0.55,0.45,0.28, 0.40,0.50,0.45],
            [0.30,0.10,0.05, 0.30,0.15,0.07, 0.50,0.30,0.15],
        ],
        evidence=["Rainfall","SeasonalForecast"],
        evidence_card=[3,3],
    )

    safe_r, care_r, delay_r = [], [], []
    for s in range(3):
        for c in range(3):
            for g in range(3):
                if   s==0 and c==2 and g==0: safe_r.append(0.85); care_r.append(0.12); delay_r.append(0.03)
                elif s==2 and g==2:          safe_r.append(0.02); care_r.append(0.08); delay_r.append(0.90)
                elif s==1 and g==1:          safe_r.append(0.30); care_r.append(0.55); delay_r.append(0.15)
                elif s==0 and g==2:          safe_r.append(0.10); care_r.append(0.40); delay_r.append(0.50)
                elif s==2 and g==0:          safe_r.append(0.05); care_r.append(0.30); delay_r.append(0.65)
                elif s==0:                   safe_r.append(0.65); care_r.append(0.25); delay_r.append(0.10)
                elif s==2:                   safe_r.append(0.05); care_r.append(0.20); delay_r.append(0.75)
                else:                        safe_r.append(0.35); care_r.append(0.45); delay_r.append(0.20)

    cpd_irr = TabularCPD(
        "IrrigationRisk", 3,
        values=[safe_r, care_r, delay_r],
        evidence=["SoilMoisture","CropWaterNeed","GroundwaterStress"],
        evidence_card=[3,3,3],
    )

    cpd_drill = TabularCPD(
        "DrillingSuccess", 3,
        values=[
            [0.10,0.30,0.60, 0.20,0.50,0.80, 0.35,0.65,0.90],
            [0.25,0.45,0.30, 0.35,0.38,0.17, 0.40,0.28,0.09],
            [0.65,0.25,0.10, 0.45,0.12,0.03, 0.25,0.07,0.01],
        ],
        evidence=["BoreholeDepth","GeologyFavorability"],
        evidence_card=[3,3],
    )

    wn, ww, wc = [], [], []
    for p in range(3):
        for g in range(3):
            for s in range(3):
                if   p==2 and g==2 and s==2: wn.append(0.02); ww.append(0.08); wc.append(0.90)
                elif p==2 and g>=1 and s>=1: wn.append(0.05); ww.append(0.35); wc.append(0.60)
                elif g==2 and s>=1:          wn.append(0.05); ww.append(0.30); wc.append(0.65)
                elif g==1 and s==1:          wn.append(0.20); ww.append(0.60); wc.append(0.20)
                elif g==0 and s==0:          wn.append(0.88); ww.append(0.10); wc.append(0.02)
                elif p==2:                   wn.append(0.30); ww.append(0.50); wc.append(0.20)
                else:                        wn.append(0.55); ww.append(0.35); wc.append(0.10)

    cpd_warn = TabularCPD(
        "EarlyWarning", 3,
        values=[wn, ww, wc],
        evidence=["PumpAge","GroundwaterStress","SeasonalForecast"],
        evidence_card=[3,3,3],
    )

    model.add_cpds(
        cpd_rainfall, cpd_forecast, cpd_crop,
        cpd_depth, cpd_geology, cpd_pump,
        cpd_soil, cpd_gw, cpd_irr, cpd_drill, cpd_warn,
    )
    assert model.check_model(), "CPT validation failed"
    return model, VariableElimination(model)


class SmartWaterModel:

    def __init__(self):
        self.pgmpy_available = PGMPY_AVAILABLE
        self.model = None
        self.infer = None
        if PGMPY_AVAILABLE:
            try:
                self.model, self.infer = build_model()
                print(f"✅ SmartWater BN loaded — {len(self.model.nodes())} nodes")
            except Exception as e:
                print(f"⚠️  Build failed: {e}. Using heuristic fallback.")
                self.pgmpy_available = False
        else:
            print("⚠️  pgmpy not installed. Using heuristic fallback.")

    def _query(self, var: str, evidence: Dict) -> np.ndarray:
        return self.infer.query([var], evidence=evidence, show_progress=False).values

    @staticmethod
    def _fb_irr(s, c, g):
        if   s==0 and c==2 and g==0: return np.array([0.85,0.12,0.03])
        elif s==2 and g==2:          return np.array([0.02,0.08,0.90])
        elif s==1 and g==1:          return np.array([0.30,0.55,0.15])
        elif s==0 and g==2:          return np.array([0.10,0.40,0.50])
        elif s==2 and g==0:          return np.array([0.05,0.30,0.65])
        elif s==0:                   return np.array([0.65,0.25,0.10])
        elif s==2:                   return np.array([0.05,0.20,0.75])
        else:                        return np.array([0.35,0.45,0.20])

    @staticmethod
    def _fb_drill(d, g):
        t = {
            (0,0):[0.10,0.25,0.65],(0,1):[0.30,0.45,0.25],(0,2):[0.60,0.30,0.10],
            (1,0):[0.20,0.35,0.45],(1,1):[0.50,0.38,0.12],(1,2):[0.80,0.17,0.03],
            (2,0):[0.35,0.40,0.25],(2,1):[0.65,0.28,0.07],(2,2):[0.90,0.09,0.01],
        }
        return np.array(t.get((d,g),[0.40,0.35,0.25]))

    @staticmethod
    def _fb_warn(p, f, g):
        if   p==2 and g==2 and f==2: return np.array([0.02,0.08,0.90])
        elif p==2 and g>=1 and f>=1: return np.array([0.05,0.35,0.60])
        elif g==2 and f>=1:          return np.array([0.05,0.30,0.65])
        elif g==1 and f==1:          return np.array([0.20,0.60,0.20])
        elif g==0 and f==0:          return np.array([0.88,0.10,0.02])
        elif p==2:                   return np.array([0.30,0.50,0.20])
        else:                        return np.array([0.55,0.35,0.10])

    def predict_irrigation(self, soil_moisture, crop_water_need,
                           groundwater_stress, rainfall, seasonal_forecast) -> Dict:
        if self.pgmpy_available:
            probs = self._query("IrrigationRisk", {
                "Rainfall": rainfall, "SeasonalForecast": seasonal_forecast,
                "SoilMoisture": soil_moisture, "CropWaterNeed": crop_water_need,
                "GroundwaterStress": groundwater_stress,
            })
        else:
            probs = self._fb_irr(soil_moisture, crop_water_need, groundwater_stress)
        idx = int(np.argmax(probs))
        return {
            "prediction": ["SafeToIrrigate","IrrigateCarefully","DelayIrrigation"][idx],
            "p_safe": float(probs[0]), "p_careful": float(probs[1]), "p_delay": float(probs[2]),
            "confidence": _confidence(probs),
            "model_used": "pgmpy" if self.pgmpy_available else "heuristic",
        }

    def predict_drilling(self, borehole_depth, geology_favorability) -> Dict:
        if self.pgmpy_available:
            probs = self._query("DrillingSuccess", {
                "BoreholeDepth": borehole_depth,
                "GeologyFavorability": geology_favorability,
            })
        else:
            probs = self._fb_drill(borehole_depth, geology_favorability)
        idx = int(np.argmax(probs))
        return {
            "prediction": ["HighSuccess","Uncertain","HighRisk"][idx],
            "p_success": float(probs[0]), "p_uncertain": float(probs[1]), "p_risk": float(probs[2]),
            "confidence": _confidence(probs),
            "model_used": "pgmpy" if self.pgmpy_available else "heuristic",
        }

    def predict_warning(self, pump_age, groundwater_stress, seasonal_forecast) -> Dict:
        if self.pgmpy_available:
            probs = self._query("EarlyWarning", {
                "PumpAge": pump_age,
                "GroundwaterStress": groundwater_stress,
                "SeasonalForecast": seasonal_forecast,
            })
        else:
            probs = self._fb_warn(pump_age, seasonal_forecast, groundwater_stress)
        idx = int(np.argmax(probs))
        return {
            "prediction": ["Normal","WatchAlert","CriticalAlert"][idx],
            "p_normal": float(probs[0]), "p_watch": float(probs[1]), "p_critical": float(probs[2]),
            "confidence": _confidence(probs),
            "model_used": "pgmpy" if self.pgmpy_available else "heuristic",
        }

    def info(self) -> Dict:
        if self.pgmpy_available and self.model:
            return {
                "status": "pgmpy",
                "nodes": list(self.model.nodes()),
                "edges": [{"from": e[0], "to": e[1]} for e in self.model.edges()],
            }
        return {"status": "heuristic_fallback", "nodes": [], "edges": []}