from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, List, Tuple
import numpy as np


class BoreholeWaterQualityModel:
    """
    Bayesian Decision Support System for water quality assessment.
    
    This model considers:
    - Environmental factors (Rainfall)
    - Infrastructure conditions (Latrine Distance, Pump Age)
    - Observable indicators (Turbidity, Water Color)
    - Target outcome (Contamination Risk)
    """
    
    def __init__(self):
        """Initialize the Bayesian network structure and probabilities."""
        self.model = None
        self.inference_engine = None
        self._build_network()
        
    def _build_network(self):
        """
        Construct the Bayesian network structure.
        
        Network Structure:
        - Rainfall → Turbidity
        - Rainfall → Surface_Runoff
        - Surface_Runoff → Contamination
        - Turbidity → Contamination
        - Latrine_Dist → Contamination
        - Pump_Age → Pump_Failure
        """
        
        # 1. DEFINE NETWORK STRUCTURE (Directed Acyclic Graph)
        self.model = DiscreteBayesianNetwork([
            ('Rainfall', 'Turbidity'),
            ('Rainfall', 'Surface_Runoff'),
            ('Turbidity', 'Contamination'),
            ('Surface_Runoff', 'Contamination'),
            ('Latrine_Dist', 'Contamination'),
            ('Pump_Age', 'Pump_Failure')
        ])
        
        # 2. DEFINE PRIOR PROBABILITIES (Variables with no parents)
        
        # Rainfall: [Low=0, Medium=1, High=2]
        # Based on Nigerian seasonal patterns
        cpd_rainfall = TabularCPD(
            variable='Rainfall',
            variable_card=3,
            values=[[0.5],   # Low (50% - dry season)
                    [0.3],   # Medium (30%)
                    [0.2]]   # High (20% - rainy season)
        )
        
        # Latrine Distance: [Safe=0 (>30m), Risky=1 (<30m)]
        # Based on WHO guidelines
        cpd_latrine = TabularCPD(
            variable='Latrine_Dist',
            variable_card=2,
            values=[[0.7],   # Safe distance (70% of cases)
                    [0.3]]   # Too close (30% of cases)
        )
        
        # Pump Age: [New=0 (<2yrs), Medium=1 (2-5yrs), Old=2 (>5yrs)]
        cpd_pump_age = TabularCPD(
            variable='Pump_Age',
            variable_card=3,
            values=[[0.3],   # New (30%)
                    [0.5],   # Medium (50%)
                    [0.2]]   # Old (20%)
        )
        
        # 3. DEFINE CONDITIONAL PROBABILITY TABLES
        
        # Turbidity given Rainfall
        # Columns: [Low Rain, Medium Rain, High Rain]
        # Rows: [Clear=0, Slightly Cloudy=1, Very Cloudy=2]
        cpd_turbidity = TabularCPD(
            variable='Turbidity',
            variable_card=3,
            values=[
                [0.90, 0.50, 0.10],  # Clear water
                [0.08, 0.35, 0.30],  # Slightly cloudy
                [0.02, 0.15, 0.60]   # Very cloudy
            ],
            evidence=['Rainfall'],
            evidence_card=[3]
        )
        
        # Surface Runoff given Rainfall
        # Columns: [Low Rain, Medium Rain, High Rain]
        # Rows: [No Runoff=0, Runoff=1]
        cpd_runoff = TabularCPD(
            variable='Surface_Runoff',
            variable_card=2,
            values=[
                [0.95, 0.40, 0.10],  # No runoff
                [0.05, 0.60, 0.90]   # Runoff present
            ],
            evidence=['Rainfall'],
            evidence_card=[3]
        )
        
        # Pump Failure given Pump Age
        # Columns: [New, Medium, Old]
        # Rows: [Working=0, Failed=1]
        cpd_pump_failure = TabularCPD(
            variable='Pump_Failure',
            variable_card=2,
            values=[
                [0.98, 0.90, 0.70],  # Working
                [0.02, 0.10, 0.30]   # Failed
            ],
            evidence=['Pump_Age'],
            evidence_card=[3]
        )
        
        # CONTAMINATION RISK (The critical output!)
        # This is the most complex CPT with 3 parent variables
        # Parents: Turbidity (3 states), Surface_Runoff (2 states), Latrine_Dist (2 states)
        # Total combinations: 3 × 2 × 2 = 12 scenarios
        
        # Scenario ordering (left to right):
        # Turb=Clear, Runoff=No,  Latrine=Safe  → Very low risk
        # Turb=Clear, Runoff=No,  Latrine=Risky → Low risk
        # Turb=Clear, Runoff=Yes, Latrine=Safe  → Medium-low risk
        # Turb=Clear, Runoff=Yes, Latrine=Risky → Medium risk
        # ... and so on for Slightly Cloudy and Very Cloudy
        
        cpd_contamination = TabularCPD(
            variable='Contamination',
            variable_card=2,
            values=[
                # Safe (row 0)
                [0.99, 0.95, 0.85, 0.70,  # Clear water scenarios
                 0.80, 0.60, 0.50, 0.30,  # Slightly cloudy scenarios
                 0.40, 0.20, 0.15, 0.05], # Very cloudy scenarios
                
                # Contaminated (row 1)
                [0.01, 0.05, 0.15, 0.30,  # Clear water scenarios
                 0.20, 0.40, 0.50, 0.70,  # Slightly cloudy scenarios
                 0.60, 0.80, 0.85, 0.95]  # Very cloudy scenarios
            ],
            evidence=['Turbidity', 'Surface_Runoff', 'Latrine_Dist'],
            evidence_card=[3, 2, 2]
        )
        
        # 4. ADD ALL CPDs TO THE MODEL
        self.model.add_cpds(
            cpd_rainfall,
            cpd_latrine,
            cpd_pump_age,
            cpd_turbidity,
            cpd_runoff,
            cpd_pump_failure,
            cpd_contamination
        )
        
        # 5. VALIDATE THE MODEL
        # This checks if the network is valid (DAG, probability sums, etc.)
        assert self.model.check_model(), "Model validation failed!"
        
        # 6. INITIALIZE INFERENCE ENGINE
        self.inference_engine = VariableElimination(self.model)
        
        print("✅ Bayesian Network successfully built and validated!")
        print(f"   Nodes: {len(self.model.nodes())}")
        print(f"   Edges: {len(self.model.edges())}")
    
    def predict_contamination_risk(
        self,
        evidence: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Calculate contamination probability given observations.
        
        Args:
            evidence: Dictionary of observed variables
                     e.g., {'Turbidity': 2, 'Rainfall': 1}
        
        Returns:
            Dictionary with probabilities for Safe and Contaminated states
        """
        
        # Run inference
        result = self.inference_engine.query(
            variables=['Contamination'],
            evidence=evidence
        )
        
        # Extract probabilities
        probs = result.values
        
        return {
            'safe_probability': float(probs[0]),
            'contamination_probability': float(probs[1]),
            'risk_level': self._categorize_risk(float(probs[1])),
            'evidence_used': evidence
        }
    
    def predict_pump_status(self, pump_age: int) -> Dict[str, float]:
        """
        Predict pump failure probability based on age.
        
        Args:
            pump_age: 0=New, 1=Medium, 2=Old
        
        Returns:
            Dictionary with pump status probabilities
        """
        
        result = self.inference_engine.query(
            variables=['Pump_Failure'],
            evidence={'Pump_Age': pump_age}
        )
        
        probs = result.values
        
        return {
            'working_probability': float(probs[0]),
            'failure_probability': float(probs[1]),
            'maintenance_needed': probs[1] > 0.15
        }
    
    def get_most_likely_scenario(self, evidence: Dict[str, int]) -> Dict:
        """
        Get the most probable explanation for observed evidence.
        
        Args:
            evidence: Observed variables
        
        Returns:
            Most probable values for unobserved variables
        """
        
        # Get all variables
        all_vars = set(self.model.nodes())
        unobserved = all_vars - set(evidence.keys())
        
        results = {}
        for var in unobserved:
            query_result = self.inference_engine.query(
                variables=[var],
                evidence=evidence
            )
            most_probable_state = int(np.argmax(query_result.values))
            results[var] = most_probable_state
        
        return results
    
    def _categorize_risk(self, prob: float) -> str:
        """
        Convert probability to human-readable risk level.
        
        Args:
            prob: Contamination probability (0-1)
        
        Returns:
            Risk category string
        """
        if prob < 0.2:
            return "LOW"
        elif prob < 0.5:
            return "MODERATE"
        elif prob < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def get_variable_info(self) -> Dict:
        """
        Get information about all variables in the model.
        
        Returns:
            Dictionary with variable names, states, and descriptions
        """
        return {
            'Rainfall': {
                'states': ['Low', 'Medium', 'High'],
                'description': 'Recent rainfall intensity'
            },
            'Turbidity': {
                'states': ['Clear', 'Slightly Cloudy', 'Very Cloudy'],
                'description': 'Water clarity/cloudiness'
            },
            'Surface_Runoff': {
                'states': ['No', 'Yes'],
                'description': 'Presence of surface water runoff'
            },
            'Latrine_Dist': {
                'states': ['Safe (>30m)', 'Risky (<30m)'],
                'description': 'Distance of latrine from borehole'
            },
            'Pump_Age': {
                'states': ['New (<2yr)', 'Medium (2-5yr)', 'Old (>5yr)'],
                'description': 'Age of water pump'
            },
            'Pump_Failure': {
                'states': ['Working', 'Failed'],
                'description': 'Pump operational status'
            },
            'Contamination': {
                'states': ['Safe', 'Contaminated'],
                'description': 'Water contamination risk'
            }
        }
    
    def sensitivity_analysis(self, target: str, evidence: Dict[str, int]) -> Dict:
        """
        Analyze which variables have the most impact on the target.
        
        Args:
            target: Variable to analyze (usually 'Contamination')
            evidence: Base evidence
        
        Returns:
            Sensitivity scores for each variable
        """
        # This is a simplified sensitivity analysis
        # For production, use more sophisticated methods
        
        base_result = self.inference_engine.query(
            variables=[target],
            evidence=evidence
        )
        base_prob = base_result.values[1]  # Contaminated probability
        
        sensitivity = {}
        
        # Test impact of each evidence variable
        for var in evidence.keys():
            temp_evidence = evidence.copy()
            
            # Try flipping to different state
            original_state = temp_evidence[var]
            variable_info = self.get_variable_info()
            num_states = len(variable_info[var]['states'])
            
            impacts = []
            for state in range(num_states):
                if state != original_state:
                    temp_evidence[var] = state
                    new_result = self.inference_engine.query(
                        variables=[target],
                        evidence=temp_evidence
                    )
                    new_prob = new_result.values[1]
                    impact = abs(new_prob - base_prob)
                    impacts.append(impact)
            
            sensitivity[var] = max(impacts) if impacts else 0.0
        
        return sensitivity


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("BAYESIAN WATER QUALITY MODEL - TESTING")
    print("=" * 60)
    
    # Initialize model
    model = BoreholeWaterQualityModel()
    
    # Test Case 1: Good conditions
    print("\n📊 TEST CASE 1: Ideal Conditions")
    print("-" * 40)
    evidence1 = {
        'Rainfall': 0,      # Low
        'Turbidity': 0,     # Clear
        'Latrine_Dist': 0   # Safe distance
    }
    result1 = model.predict_contamination_risk(evidence1)
    print(f"Evidence: {evidence1}")
    print(f"Contamination Risk: {result1['contamination_probability']:.2%}")
    print(f"Risk Level: {result1['risk_level']}")
    
    # Test Case 2: Risky conditions
    print("\n📊 TEST CASE 2: High Risk Conditions")
    print("-" * 40)
    evidence2 = {
        'Rainfall': 2,      # High
        'Turbidity': 2,     # Very cloudy
        'Latrine_Dist': 1   # Too close
    }
    result2 = model.predict_contamination_risk(evidence2)
    print(f"Evidence: {evidence2}")
    print(f"Contamination Risk: {result2['contamination_probability']:.2%}")
    print(f"Risk Level: {result2['risk_level']}")
    
    # Test Case 3: Pump age
    print("\n📊 TEST CASE 3: Pump Status (Old Pump)")
    print("-" * 40)
    pump_result = model.predict_pump_status(pump_age=2)
    print(f"Failure Risk: {pump_result['failure_probability']:.2%}")
    print(f"Maintenance Needed: {pump_result['maintenance_needed']}")
    
    # Test Case 4: Real-world scenario (caretaker report)
    print("\n📊 TEST CASE 4: Real-World Scenario")
    print("-" * 40)
    print("Scenario: After heavy rain, caretaker reports cloudy water")
    evidence4 = {
        'Rainfall': 2,      # Heavy rain reported
        'Turbidity': 1      # Slightly cloudy observed
    }
    result4 = model.predict_contamination_risk(evidence4)
    print(f"Contamination Risk: {result4['contamination_probability']:.2%}")
    print(f"⚠️  Recommendation: {result4['risk_level']} risk - Consider boiling water")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
