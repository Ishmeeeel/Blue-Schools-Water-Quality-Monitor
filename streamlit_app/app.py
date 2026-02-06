"""
Streamlit Dashboard for Blue Schools Water Quality Monitoring
==============================================================

Interactive web interface for field workers and school caretakers
to assess water quality risk in real-time.

Author: Blue Schools Project
Date: 2026
Version: 2.0 - Enhanced with Percentage Sliders
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
    .percentage-display {
        font-size: 24px;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

# For local testing
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def percentage_to_category(percentage, categories):
    """
    Convert percentage (0-100) to discrete category (0, 1, or 2).
    
    Args:
        percentage: Value from 0-100
        categories: Number of categories (2 or 3)
    
    Returns:
        Category index (0, 1, or 2)
    """
    if categories == 3:
        # For 3 categories: 0-33% = 0, 34-66% = 1, 67-100% = 2
        if percentage <= 33:
            return 0
        elif percentage <= 66:
            return 1
        else:
            return 2
    else:  # categories == 2
        # For 2 categories: 0-50% = 0, 51-100% = 1
        if percentage <= 50:
            return 0
        else:
            return 1


def get_category_label(percentage, labels):
    """
    Get the category label based on percentage.
    
    Args:
        percentage: Value from 0-100
        labels: List of category labels
    
    Returns:
        Current category label
    """
    categories = len(labels)
    category_index = percentage_to_category(percentage, categories)
    return labels[category_index]


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
    1. Adjust percentage sliders
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
        
        # Rainfall Slider (0-100%)
        st.markdown("**Recent Rainfall**")
        rainfall_pct = st.slider(
            "Rainfall Intensity",
            min_value=0,
            max_value=100,
            value=50,
            help="Slide to indicate rainfall intensity in the past 24 hours",
            label_visibility="collapsed"
        )
        
        # Show current category
        rainfall_labels = ["☀️ Low/None (0-33%)", "🌧️ Medium (34-66%)", "⛈️ Heavy (67-100%)"]
        current_rainfall = get_category_label(rainfall_pct, rainfall_labels)
        st.markdown(f"""
            <div class="percentage-display">
                {rainfall_pct}% - {current_rainfall}
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Surface Runoff Slider (0-100%)
        st.markdown("**Surface Water Runoff**")
        runoff_pct = st.slider(
            "Surface Runoff Presence",
            min_value=0,
            max_value=100,
            value=0,
            help="Slide to indicate amount of surface water runoff observed",
            label_visibility="collapsed"
        )
        
        runoff_labels = ["❌ No runoff (0-50%)", "✅ Runoff present (51-100%)"]
        current_runoff = get_category_label(runoff_pct, runoff_labels)
        st.markdown(f"""
            <div class="percentage-display">
                {runoff_pct}% - {current_runoff}
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("Infrastructure")
        
        # Latrine Distance Slider (0-100%)
        st.markdown("**Latrine Distance from Borehole**")
        latrine_pct = st.slider(
            "Latrine Safety",
            min_value=0,
            max_value=100,
            value=100,
            help="0% = Too close (<30m), 100% = Safe distance (>30m)",
            label_visibility="collapsed"
        )
        
        latrine_labels = ["⚠️ Too close (<30m)", "✅ Safe (>30m)"]
        current_latrine = get_category_label(latrine_pct, latrine_labels)
        st.markdown(f"""
            <div class="percentage-display">
                {latrine_pct}% - {current_latrine}
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Water Quality Indicators")
        
        # Turbidity Slider (0-100%)
        st.markdown("**Water Clarity (Turbidity)**")
        turbidity_pct = st.slider(
            "Turbidity Level",
            min_value=0,
            max_value=100,
            value=0,
            help="Hold a glass of water up to light. Slide to indicate how cloudy it is",
            label_visibility="collapsed"
        )
        
        turbidity_labels = ["💎 Clear (0-33%)", "☁️ Slightly Cloudy (34-66%)", "🌫️ Very Cloudy (67-100%)"]
        current_turbidity = get_category_label(turbidity_pct, turbidity_labels)
        st.markdown(f"""
            <div class="percentage-display">
                {turbidity_pct}% - {current_turbidity}
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visual guide
        st.info("""
        **💡 Turbidity Guide:**
        - **0-33%**: Crystal clear water
        - **34-66%**: Slightly hazy/cloudy
        - **67-100%**: Very cloudy, cannot see through
        """)
    
    st.markdown("---")
    
    # Assessment button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        assess_button = st.button("🔍 ASSESS CONTAMINATION RISK", type="primary", use_container_width=True)
    
    if assess_button:
        with st.spinner("Analyzing water quality..."):
            # Convert percentages to categories
            rainfall = percentage_to_category(rainfall_pct, 3)
            turbidity = percentage_to_category(turbidity_pct, 3)
            surface_runoff = percentage_to_category(runoff_pct, 2)
            latrine_distance = percentage_to_category(latrine_pct, 2)
            
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
                
                # Display school information if provided
                if school_name or location or reporter_name:
                    st.info(f"""
                    **📍 Assessment Details:**
                    - School: {school_name or 'Not provided'}
                    - Location: {location or 'Not provided'}
                    - Assessed by: {reporter_name or 'Not provided'}
                    - Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    
                    **📊 Input Percentages:**
                    - Rainfall: {rainfall_pct}% ({current_rainfall})
                    - Turbidity: {turbidity_pct}% ({current_turbidity})
                    - Surface Runoff: {runoff_pct}% ({current_runoff})
                    - Latrine Distance: {latrine_pct}% ({current_latrine})
                    """)
                
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
                
                # Save to history with percentages
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'school': school_name or "Unknown",
                    'location': location or "Unknown",
                    'reporter': reporter_name or "Unknown",
                    'rainfall_pct': rainfall_pct,
                    'turbidity_pct': turbidity_pct,
                    'runoff_pct': runoff_pct,
                    'latrine_pct': latrine_pct,
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
        st.markdown("**Pump Age**")
        pump_age_pct = st.slider(
            "Pump Age Percentage",
            min_value=0,
            max_value=100,
            value=0,
            help="0% = New (<2yr), 50% = Medium (2-5yr), 100% = Old (>5yr)",
            label_visibility="collapsed"
        )
        
        pump_labels = ["🆕 New (<2 years)", "⚙️ Medium (2-5 years)", "🔧 Old (>5 years)"]
        current_pump = get_category_label(pump_age_pct, pump_labels)
        st.markdown(f"""
            <div class="percentage-display">
                {pump_age_pct}% - {current_pump}
            </div>
        """, unsafe_allow_html=True)
        
        check_pump_button = st.button("Check Pump Status", type="primary")
    
    with col_pump2:
        if check_pump_button:
            pump_age = percentage_to_category(pump_age_pct, 3)
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
        
        # Display history table with percentages
        display_df = df[['timestamp', 'school', 'location', 'reporter', 'rainfall_pct', 'turbidity_pct', 'risk_level', 'probability']].copy()
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
        display_df.columns = ['Time', 'School', 'Location', 'Reporter', 'Rainfall %', 'Turbidity %', 'Risk', 'Contamination %']
        
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
    ### How to Use This Tool (Version 2.0 - Enhanced)
    
    #### 1️⃣ Gather Observations
    Before using the tool, observe the following at your school's borehole:
    
    - **Rainfall**: Has it rained recently? How heavy was it?
    - **Water Clarity**: Fill a clear glass with water and look through it
    - **Surface Runoff**: Is there water flowing on the ground?
    - **Latrine Distance**: How far is the nearest latrine? (Should be >30m)
    
    #### 2️⃣ Adjust Percentage Sliders
    - **NEW!** Use continuous sliders (0-100%) for precise input
    - See real-time category labels as you slide
    - Current percentage is displayed prominently
    
    **Slider Guide:**
    - **Rainfall**: 0% = No rain, 100% = Heavy rain
    - **Turbidity**: 0% = Crystal clear, 100% = Very cloudy
    - **Surface Runoff**: 0% = None, 100% = Significant runoff
    - **Latrine Distance**: 0% = Too close, 100% = Safe distance
    
    #### 3️⃣ Understand Results
    
    **Risk Levels:**
    - ✅ **LOW** (0-20%): Water appears safe
    - ⚠️ **MODERATE** (20-50%): Use caution, consider treatment
    - 🚨 **HIGH** (50-80%): Treatment required
    - 🔴 **CRITICAL** (80-100%): Do not use
    
    #### 4️⃣ Take Action
    Follow the recommendations provided. When in doubt, always treat water before use.
    
    ---
    
    ### What's in this Version 1.0
    
    ✅ **Continuous Percentage Sliders**: More precise control (0-100%)  
    ✅ **Real-time Category Display**: See category as you move slider  
    ✅ **Percentage History**: Track exact values over time  
    ✅ **Enhanced Visual Feedback**: Clear percentage indicators  
    
    ---
    
    ### Water Treatment Methods
    
    1. **Boiling**: Boil water for at least 3 minutes
    2. **Chlorine**: Add 2 drops of bleach per liter, wait 30 minutes
    3. **Filtration**: Use ceramic or sand filters
    
    ---
    
    ### Emergency Contacts
    
    - **Health Emergency**: +2348012345678
    - **Water Quality Issues**: [Minstry of Health]
    - **Technical Support**: support@blueschools.org
    
    ---
    
    ### About the Technology
    
    This tool uses **Bayesian Networks** - a type of AI that reasons with uncertainty.
    Your percentage inputs are converted to categories for the AI model while preserving
    the exact values for your records.
    
    **Developed by**: Ismail Usman and Blue Team Project  
    **Version**: 1.0 
    **Last Updated**: February 2026
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💧 Blue Schools Water Quality Monitoring System v2.0</p>
        <p>Powered by Bayesian AI | Built with ❤️ for Nigerian Schools</p>
        <p><small>For technical support: support@blueschools.org</small></p>
    </div>
""", unsafe_allow_html=True)