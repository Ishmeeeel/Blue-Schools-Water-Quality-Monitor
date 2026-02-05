"""
Streamlit Dashboard for Blue Schools Water Quality Monitoring
==============================================================

Interactive web interface for field workers and school caretakers
to assess water quality risk in real-time.

Author: Blue Schools Project
Date: 2026
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Blue Schools Water Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .risk-critical {
        background-color: #f5c6cb;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #721c24;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

# For local testing
API_BASE_URL = "http://localhost:8000"

# For production, use environment variable or deployed URL
# API_BASE_URL = os.getenv("API_URL", "https://your-api.herokuapp.com")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if backend API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_prediction(observations):
    """Get contamination prediction from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=observations,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend API. Please ensure the server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_pump_prediction(pump_age):
    """Get pump status prediction from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-pump",
            params={"pump_age": pump_age},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None


def create_risk_gauge(probability):
    """Create a gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Contamination Risk (%)", 'font': {'size': 24}},
        delta={'reference': 20, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d4edda'},
                {'range': [20, 50], 'color': '#fff3cd'},
                {'range': [50, 80], 'color': '#f8d7da'},
                {'range': [80, 100], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def display_risk_result(result):
    """Display risk assessment result with appropriate styling."""
    risk_level = result['risk_level']
    prob = result['contamination_probability']
    
    # Determine CSS class
    css_class_map = {
        'LOW': 'risk-low',
        'MODERATE': 'risk-moderate',
        'HIGH': 'risk-high',
        'CRITICAL': 'risk-critical'
    }
    css_class = css_class_map.get(risk_level, 'risk-moderate')
    
    # Emoji map
    emoji_map = {
        'LOW': '✅',
        'MODERATE': '⚠️',
        'HIGH': '🚨',
        'CRITICAL': '🔴'
    }
    emoji = emoji_map.get(risk_level, '⚠️')
    
    st.markdown(f"""
        <div class="{css_class}">
            <h2>{emoji} Risk Level: {risk_level}</h2>
            <p style="font-size: 18px;">Contamination Probability: {prob:.1%}</p>
            <p style="font-size: 16px;"><b>Recommendation:</b> {result['recommendation']}</p>
            <p style="font-size: 14px; color: #666;">Confidence: {result['confidence']}</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.title("💧 Blue Schools Water Quality Monitor")
st.markdown("### Real-time Water Contamination Risk Assessment")
st.markdown("---")

# Check API status
api_healthy = check_api_health()

if api_healthy:
    st.success("✅ Connected to backend API")
else:
    st.error("❌ Backend API not available. Please start the server with: `cd backend && uvicorn app:app --reload`")
    st.stop()

# Sidebar for school information
with st.sidebar:
    st.header("📍 School Information")
    
    school_name = st.text_input("School Name", placeholder="e.g., GPS Kano")
    location = st.text_input("Location", placeholder="e.g., Kano City")
    reporter_name = st.text_input("Your Name", placeholder="e.g., Musa Ahmed")
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.info("""
    This tool uses Bayesian AI to assess water contamination risk based on field observations.
    
    **How to use:**
    1. Select observations below
    2. Click 'Assess Risk'
    3. Follow recommendations
    
    **Need help?**
    Contact: support@blueschools.org
    """)

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Risk Assessment", "🔧 Pump Status", "📊 History", "📚 Guide"])

# ============================================================================
# TAB 1: RISK ASSESSMENT
# ============================================================================

with tab1:
    st.header("Water Quality Observations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environmental Factors")
        
        rainfall = st.select_slider(
            "Recent Rainfall",
            options=[0, 1, 2],
            format_func=lambda x: ["☀️ Low/None", "🌧️ Medium", "⛈️ Heavy"][x],
            help="Select the rainfall intensity in the past 24 hours"
        )
        
        surface_runoff = st.radio(
            "Surface Water Runoff",
            options=[0, 1],
            format_func=lambda x: ["❌ No runoff observed", "✅ Runoff present"][x],
            help="Is there visible water flowing on the ground surface?"
        )
        
        st.subheader("Infrastructure")
        
        latrine_distance = st.radio(
            "Latrine Distance from Borehole",
            options=[0, 1],
            format_func=lambda x: ["✅ Safe (>30 meters)", "⚠️ Too close (<30 meters)"][x],
            help="Distance between latrine and water source"
        )
    
    with col2:
        st.subheader("Water Quality Indicators")
        
        turbidity = st.select_slider(
            "Water Clarity (Turbidity)",
            options=[0, 1, 2],
            format_func=lambda x: ["💎 Clear", "☁️ Slightly Cloudy", "🌫️ Very Cloudy"][x],
            help="Hold a glass of water up to light. How cloudy is it?"
        )
        
        st.image("https://via.placeholder.com/300x150.png?text=Turbidity+Reference+Chart", 
                 caption="Reference: Compare your water to this chart",
                 use_container_width=True)
    
    st.markdown("---")
    
    # Assessment button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        assess_button = st.button("🔍 ASSESS CONTAMINATION RISK", type="primary", use_container_width=True)
    
    if assess_button:
        with st.spinner("Analyzing water quality..."):
            # Prepare observation data
            observations = {
                "rainfall": rainfall,
                "turbidity": turbidity,
                "surface_runoff": surface_runoff,
                "latrine_distance": latrine_distance,
                "school_name": school_name,
                "location": location,
                "reporter_name": reporter_name
            }
            
            # Get prediction
            result = get_prediction(observations)
            
            if result:
                st.success("✅ Analysis complete!")
                
                # Display gauge
                st.plotly_chart(
                    create_risk_gauge(result['contamination_probability']),
                    use_container_width=True
                )
                
                # Display result
                display_risk_result(result)
                
                # Additional details in expander
                with st.expander("📋 Technical Details"):
                    col_detail1, col_detail2 = st.columns(2)
                    
                    with col_detail1:
                        st.metric("Safe Probability", f"{result['safe_probability']:.1%}")
                        st.metric("Contamination Probability", f"{result['contamination_probability']:.1%}")
                    
                    with col_detail2:
                        st.metric("Confidence Level", result['confidence'])
                        st.text(f"Timestamp: {result['timestamp']}")
                    
                    st.json(result['evidence_used'])
                
                # Save to history
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'school': school_name or "Unknown",
                    'risk_level': result['risk_level'],
                    'probability': result['contamination_probability'],
                    'observations': observations
                }
                st.session_state.history.append(history_entry)

# ============================================================================
# TAB 2: PUMP STATUS
# ============================================================================

with tab2:
    st.header("🔧 Pump Failure Assessment")
    
    st.info("Assess the likelihood of pump failure based on age and condition.")
    
    col_pump1, col_pump2 = st.columns(2)
    
    with col_pump1:
        pump_age = st.radio(
            "Pump Age",
            options=[0, 1, 2],
            format_func=lambda x: ["🆕 New (<2 years)", "⚙️ Medium (2-5 years)", "🔧 Old (>5 years)"][x],
            help="Approximate age of the water pump"
        )
        
        check_pump_button = st.button("Check Pump Status", type="primary")
    
    with col_pump2:
        if check_pump_button:
            pump_result = get_pump_prediction(pump_age)
            
            if pump_result:
                st.metric("Failure Risk", f"{pump_result['failure_probability']:.1%}")
                
                if pump_result['maintenance_needed']:
                    st.error(pump_result['recommendation'])
                else:
                    st.success(pump_result['recommendation'])
                
                # Pump status gauge
                fig_pump = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pump_result['failure_probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Failure Risk (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 15], 'color': "lightgreen"},
                            {'range': [15, 30], 'color': "orange"},
                            {'range': [30, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                st.plotly_chart(fig_pump, use_container_width=True)

# ============================================================================
# TAB 3: HISTORY
# ============================================================================

with tab3:
    st.header("📊 Assessment History")
    
    if st.session_state.history:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.history)
        
        # Summary statistics
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Total Assessments", len(df))
        
        with col_stat2:
            high_risk_count = len(df[df['risk_level'].isin(['HIGH', 'CRITICAL'])])
            st.metric("High Risk Cases", high_risk_count)
        
        with col_stat3:
            avg_prob = df['probability'].mean()
            st.metric("Average Risk", f"{avg_prob:.1%}")
        
        st.markdown("---")
        
        # Display history table
        display_df = df[['timestamp', 'school', 'risk_level', 'probability']].copy()
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export button
        if st.button("📥 Export History as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"water_quality_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No assessments yet. Complete a risk assessment to see history here.")

# ============================================================================
# TAB 4: USER GUIDE
# ============================================================================

with tab4:
    st.header("📚 User Guide")
    
    st.markdown("""
    ### How to Use This Tool
    
    #### 1️⃣ Gather Observations
    Before using the tool, observe the following at your school's borehole:
    
    - **Rainfall**: Has it rained recently? How heavy was it?
    - **Water Clarity**: Fill a clear glass with water and look through it
    - **Surface Runoff**: Is there water flowing on the ground?
    - **Latrine Distance**: How far is the nearest latrine? (Should be >30m)
    
    #### 2️⃣ Input Information
    - Enter school details in the sidebar
    - Select your observations using the sliders and buttons
    - Click "Assess Risk"
    
    #### 3️⃣ Understand Results
    
    **Risk Levels:**
    - ✅ **LOW** (0-20%): Water appears safe
    - ⚠️ **MODERATE** (20-50%): Use caution, consider treatment
    - 🚨 **HIGH** (50-80%): Treatment required
    - 🔴 **CRITICAL** (80-100%): Do not use
    
    #### 4️⃣ Take Action
    Follow the recommendations provided. When in doubt, always treat water before use.
    
    ---
    
    ### Water Treatment Methods
    
    1. **Boiling**: Boil water for at least 3 minutes
    2. **Chlorine**: Add 2 drops of bleach per liter, wait 30 minutes
    3. **Filtration**: Use ceramic or sand filters
    
    ---
    
    ### Emergency Contacts
    
    - **Health Emergency**: 123 (Nigeria Emergency Number)
    - **Water Quality Issues**: [Local Health Department]
    - **Technical Support**: support@blueschools.org
    
    ---
    
    ### About the Technology
    
    This tool uses **Bayesian Networks** - a type of AI that reasons with uncertainty,
    similar to how a doctor diagnoses patients. It combines multiple observations to
    calculate the probability of water contamination.
    
    **Developed by**: Blue Schools Project  
    **Version**: 1.0.0  
    **Last Updated**: February 2026
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💧 Blue Schools Water Quality Monitoring System v1.0</p>
        <p>Powered by Bayesian AI | Built with ❤️ for Nigerian Schools</p>
        <p><small>For technical support: support@blueschools.org</small></p>
    </div>
""", unsafe_allow_html=True)