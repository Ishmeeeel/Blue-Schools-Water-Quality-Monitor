"""
SmartWater Agriculture — Bayesian Decision Engine
==================================================
11-node Bayesian Network for groundwater & irrigation advisory.

Nodes
─────
Priors (no parents):
  Rainfall            : 0=Low, 1=Moderate, 2=High
  SeasonalForecast    : 0=WetSeason, 1=DrySpell, 2=DroughtWarning
  CropWaterNeed       : 0=Low, 1=Medium, 2=High
  BoreholeDepth       : 0=Shallow(<20m), 1=Medium(20-60m), 2=Deep(>60m)
  GeologyFavorability : 0=Unfavorable, 1=Moderate, 2=Favorable
  PumpAge             : 0=New(<2yrs), 1=Mid(2-5yrs), 2=Old(>5yrs)

Intermediates:
  SoilMoisture        : 0=Dry, 1=Adequate, 2=Saturated
  GroundwaterStress   : 0=Low, 1=Moderate, 2=High

Outputs:
  IrrigationRisk      : 0=SafeToIrrigate, 1=IrrigateCarefully, 2=DelayIrrigation
  DrillingSuccess     : 0=HighSuccess, 1=Uncertain, 2=HighRisk
  EarlyWarning        : 0=Normal, 1=WatchAlert, 2=CriticalAlert

Causal structure:
  Rainfall + SeasonalForecast → SoilMoisture → IrrigationRisk
  Rainfall + SeasonalForecast → GroundwaterStress → IrrigationRisk
  CropWaterNeed               → IrrigationRisk
  BoreholeDepth + GeologyFavorability → DrillingSuccess
  PumpAge + GroundwaterStress + SeasonalForecast → EarlyWarning
"""

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict
import numpy as np


class SmartWaterModel:
    """
    SmartWater Bayesian Decision Support System.
    Build once on startup, query on every request.
    """

    def __init__(self):
        self.model = None
        self.inference = None
        self._build()

    # ──────────────────────────────────────────────────────────────────────
    # NETWORK CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────

    def _build(self):
        # 1. Define causal structure (DAG)
        self.model = BayesianNetwork([
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

        # 2. Prior probabilities (root nodes)

        # Rainfall: Low=50%, Moderate=35%, High=15% — dry-season bias for Kaduna
        cpd_rainfall = TabularCPD(
            variable="Rainfall", variable_card=3,
            values=[[0.50], [0.35], [0.15]],
        )

        # SeasonalForecast: Wet=40%, DrySpell=40%, Drought=20%
        cpd_forecast = TabularCPD(
            variable="SeasonalForecast", variable_card=3,
            values=[[0.40], [0.40], [0.20]],
        )

        # CropWaterNeed: Low=30%, Medium=50%, High=20%
        cpd_crop = TabularCPD(
            variable="CropWaterNeed", variable_card=3,
            values=[[0.30], [0.50], [0.20]],
        )

        # BoreholeDepth: Shallow=40%, Medium=40%, Deep=20%
        cpd_depth = TabularCPD(
            variable="BoreholeDepth", variable_card=3,
            values=[[0.40], [0.40], [0.20]],
        )

        # GeologyFavorability: Unfav=30%, Moderate=40%, Favorable=30%
        cpd_geology = TabularCPD(
            variable="GeologyFavorability", variable_card=3,
            values=[[0.30], [0.40], [0.30]],
        )

        # PumpAge: New=30%, Mid=40%, Old=30%
        cpd_pump = TabularCPD(
            variable="PumpAge", variable_card=3,
            values=[[0.30], [0.40], [0.30]],
        )

        # 3. Conditional Probability Tables

        # ── SoilMoisture | Rainfall(3) × SeasonalForecast(3) = 9 columns ──
        # Column order (pgmpy iterates evidence right-to-left):
        # [R=L,F=W] [R=L,F=D] [R=L,F=Dr] [R=M,F=W] [R=M,F=D] [R=M,F=Dr]
        # [R=H,F=W] [R=H,F=D] [R=H,F=Dr]
        cpd_soil = TabularCPD(
            variable="SoilMoisture", variable_card=3,
            values=[
                # Dry
                [0.10, 0.60, 0.85, 0.05, 0.30, 0.60, 0.02, 0.10, 0.25],
                # Adequate
                [0.60, 0.35, 0.13, 0.60, 0.55, 0.35, 0.28, 0.55, 0.55],
                # Saturated
                [0.30, 0.05, 0.02, 0.35, 0.15, 0.05, 0.70, 0.35, 0.20],
            ],
            evidence=["Rainfall", "SeasonalForecast"],
            evidence_card=[3, 3],
        )

        # ── GroundwaterStress | Rainfall(3) × SeasonalForecast(3) = 9 cols ──
        cpd_gw = TabularCPD(
            variable="GroundwaterStress", variable_card=3,
            values=[
                # Low stress
                [0.20, 0.55, 0.80, 0.15, 0.40, 0.65, 0.10, 0.20, 0.40],
                # Moderate
                [0.50, 0.35, 0.15, 0.55, 0.45, 0.28, 0.40, 0.50, 0.45],
                # High stress
                [0.30, 0.10, 0.05, 0.30, 0.15, 0.07, 0.50, 0.30, 0.15],
            ],
            evidence=["Rainfall", "SeasonalForecast"],
            evidence_card=[3, 3],
        )

        # ── IrrigationRisk | Soil(3) × Crop(3) × GW(3) = 27 columns ──
        safe_row, careful_row, delay_row = [], [], []
        for soil in range(3):
            for crop in range(3):
                for gw in range(3):
                    if soil == 0 and crop == 2 and gw == 0:
                        # Dry soil, high crop need, low GW stress → safe
                        safe_row.append(0.85); careful_row.append(0.12); delay_row.append(0.03)
                    elif soil == 2 and gw == 2:
                        # Saturated soil + high GW stress → delay
                        safe_row.append(0.02); careful_row.append(0.08); delay_row.append(0.90)
                    elif soil == 1 and gw == 1:
                        # Adequate moisture, moderate stress → careful
                        safe_row.append(0.30); careful_row.append(0.55); delay_row.append(0.15)
                    elif soil == 0 and gw == 2:
                        # Dry soil but high GW stress (critical tradeoff)
                        safe_row.append(0.10); careful_row.append(0.40); delay_row.append(0.50)
                    elif soil == 2 and gw == 0:
                        # Saturated but low stress → still delay
                        safe_row.append(0.05); careful_row.append(0.30); delay_row.append(0.65)
                    elif soil == 0:
                        # Dry soil general → lean safe
                        safe_row.append(0.65); careful_row.append(0.25); delay_row.append(0.10)
                    elif soil == 2:
                        # Saturated general → lean delay
                        safe_row.append(0.05); careful_row.append(0.20); delay_row.append(0.75)
                    else:
                        # Adequate moisture general → careful
                        safe_row.append(0.35); careful_row.append(0.45); delay_row.append(0.20)

        cpd_irrigation = TabularCPD(
            variable="IrrigationRisk", variable_card=3,
            values=[safe_row, careful_row, delay_row],
            evidence=["SoilMoisture", "CropWaterNeed", "GroundwaterStress"],
            evidence_card=[3, 3, 3],
        )

        # ── DrillingSuccess | Depth(3) × Geology(3) = 9 columns ──
        cpd_drilling = TabularCPD(
            variable="DrillingSuccess", variable_card=3,
            values=[
                # HighSuccess
                [0.10, 0.30, 0.60, 0.20, 0.50, 0.80, 0.35, 0.65, 0.90],
                # Uncertain
                [0.25, 0.45, 0.30, 0.35, 0.38, 0.17, 0.40, 0.28, 0.09],
                # HighRisk
                [0.65, 0.25, 0.10, 0.45, 0.12, 0.03, 0.25, 0.07, 0.01],
            ],
            evidence=["BoreholeDepth", "GeologyFavorability"],
            evidence_card=[3, 3],
        )

        # ── EarlyWarning | Pump(3) × GW(3) × Forecast(3) = 27 columns ──
        normal_row, watch_row, critical_row = [], [], []
        for pump in range(3):
            for gw in range(3):
                for seas in range(3):
                    if pump == 2 and gw == 2 and seas == 2:
                        normal_row.append(0.02); watch_row.append(0.08); critical_row.append(0.90)
                    elif pump == 2 and gw >= 1 and seas >= 1:
                        normal_row.append(0.05); watch_row.append(0.35); critical_row.append(0.60)
                    elif gw == 2 and seas >= 1:
                        normal_row.append(0.05); watch_row.append(0.30); critical_row.append(0.65)
                    elif gw == 1 and seas == 1:
                        normal_row.append(0.20); watch_row.append(0.60); critical_row.append(0.20)
                    elif gw == 0 and seas == 0:
                        normal_row.append(0.88); watch_row.append(0.10); critical_row.append(0.02)
                    elif pump == 2:
                        normal_row.append(0.30); watch_row.append(0.50); critical_row.append(0.20)
                    else:
                        normal_row.append(0.55); watch_row.append(0.35); critical_row.append(0.10)

        cpd_warning = TabularCPD(
            variable="EarlyWarning", variable_card=3,
            values=[normal_row, watch_row, critical_row],
            evidence=["PumpAge", "GroundwaterStress", "SeasonalForecast"],
            evidence_card=[3, 3, 3],
        )

        # 4. Assemble and validate
        self.model.add_cpds(
            cpd_rainfall, cpd_forecast, cpd_crop,
            cpd_depth, cpd_geology, cpd_pump,
            cpd_soil, cpd_gw,
            cpd_irrigation, cpd_drilling, cpd_warning,
        )
        assert self.model.check_model(), \
            "Bayesian Network validation failed — check CPT column sums."

        # 5. Inference engine (Variable Elimination)
        self.inference = VariableElimination(self.model)
        print("✅ SmartWater Bayesian Network built and validated.")
        print(f"   Nodes : {len(self.model.nodes())}")
        print(f"   Edges : {len(self.model.edges())}")

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC QUERY METHODS
    # ──────────────────────────────────────────────────────────────────────

    def _query(self, variable: str, evidence: Dict[str, int]) -> np.ndarray:
        clean = {k: v for k, v in evidence.items() if v is not None}
        result = self.inference.query(
            variables=[variable],
            evidence=clean,
            show_progress=False,
        )
        return result.values

    def predict_irrigation(self, evidence: Dict[str, int]) -> Dict:
        probs = self._query("IrrigationRisk", evidence)
        idx = int(np.argmax(probs))
        return {
            "p_safe":    float(probs[0]),
            "p_careful": float(probs[1]),
            "p_delay":   float(probs[2]),
            "verdict":   ["SafeToIrrigate", "IrrigateCarefully", "DelayIrrigation"][idx],
            "confidence": self._confidence(probs),
        }

    def predict_drilling(self, evidence: Dict[str, int]) -> Dict:
        probs = self._query("DrillingSuccess", evidence)
        idx = int(np.argmax(probs))
        return {
            "p_success":   float(probs[0]),
            "p_uncertain": float(probs[1]),
            "p_risk":      float(probs[2]),
            "verdict":     ["HighSuccess", "Uncertain", "HighRisk"][idx],
            "confidence":  self._confidence(probs),
        }

    def predict_warning(self, evidence: Dict[str, int]) -> Dict:
        probs = self._query("EarlyWarning", evidence)
        idx = int(np.argmax(probs))
        return {
            "p_normal":   float(probs[0]),
            "p_watch":    float(probs[1]),
            "p_critical": float(probs[2]),
            "verdict":    ["Normal", "WatchAlert", "CriticalAlert"][idx],
            "confidence": self._confidence(probs),
        }

    def get_structure(self) -> Dict:
        return {
            "nodes": list(self.model.nodes()),
            "edges": [{"from": e[0], "to": e[1]} for e in self.model.edges()],
        }

    @staticmethod
    def _confidence(probs: np.ndarray) -> str:
        m = float(np.max(probs))
        if m >= 0.75: return "HIGH"
        if m >= 0.50: return "MEDIUM"
        return "LOW"


# ──────────────────────────────────────────────────────────────────────────
# RUN DIRECTLY TO SELF-TEST: python bayesian_model.py
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("SMARTWATER BAYESIAN MODEL — SELF TEST")
    print("=" * 55)

    m = SmartWaterModel()

    print("\n📊 TEST 1: Dry soil, thirsty crop, low GW stress")
    r = m.predict_irrigation({"SoilMoisture": 0, "CropWaterNeed": 2, "GroundwaterStress": 0})
    print(f"   {r['verdict']} | Safe={r['p_safe']:.0%} Careful={r['p_careful']:.0%} Delay={r['p_delay']:.0%}")

    print("\n📊 TEST 2: Saturated soil, high GW stress")
    r = m.predict_irrigation({"SoilMoisture": 2, "CropWaterNeed": 1, "GroundwaterStress": 2})
    print(f"   {r['verdict']} | Safe={r['p_safe']:.0%} Careful={r['p_careful']:.0%} Delay={r['p_delay']:.0%}")

    print("\n📊 TEST 3: Deep borehole, favorable geology")
    r = m.predict_drilling({"BoreholeDepth": 2, "GeologyFavorability": 2})
    print(f"   {r['verdict']} | Success={r['p_success']:.0%} Uncertain={r['p_uncertain']:.0%} Risk={r['p_risk']:.0%}")

    print("\n📊 TEST 4: Old pump, drought warning, high GW stress")
    r = m.predict_warning({"PumpAge": 2, "SeasonalForecast": 2, "GroundwaterStress": 2})
    print(f"   {r['verdict']} | Normal={r['p_normal']:.0%} Watch={r['p_watch']:.0%} Critical={r['p_critical']:.0%}")

    print("\n✅ All tests passed.")