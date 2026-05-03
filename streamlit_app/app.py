"""
SmartWater Agriculture — Streamlit Frontend
============================================
AI-Powered Groundwater & Irrigation Advisory
Igabi & Zaria LGAs · Kaduna State, Nigeria

Powered by a Bayesian Decision Engine
ABU Zaria & IAR Zaria · M4D Open Innovation Challenge 2025/26
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import urllib.parse
import os


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SmartWater Agriculture",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Light, professional, field-grade design
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    /* ── Background — ice blue ── */
    .stApp,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #f0f6fb !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
        background-color: #f0f6fb !important;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: 'Sora', sans-serif !important;
        color: #0f172a !important;
    }

    /* ── App Header ── */
    .sw-header {
        background: linear-gradient(135deg, #1a6fa8 0%, #0e4d7a 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .sw-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .sw-header::after {
        content: '';
        position: absolute;
        bottom: -40px; left: -40px;
        width: 150px; height: 150px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .sw-header h1 {
        font-family: 'Sora', sans-serif !important;
        color: white !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.3rem 0 !important;
        letter-spacing: -0.3px;
    }
    .sw-header p {
        color: rgba(255,255,255,0.82) !important;
        font-size: 0.95rem;
        margin: 0;
    }
    .sw-header .badge-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.8rem;
        flex-wrap: wrap;
    }
    .sw-badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.3px;
    }

    /* ── Status Banner ── */
    .status-connected {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #15803d;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .status-demo {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #d97706;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #92400e;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"],
    [data-baseweb="tab-list"] {
        background: white !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 2px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
        border: 1px solid #e2e8f0 !important;
    }
    [data-testid="stTabs"] [role="tab"],
    [data-baseweb="tab"] {
        font-family: 'Sora', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.8rem !important;
        color: #1a6fa8 !important;
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"] {
        background: #1a6fa8 !important;
        color: white !important;
    }
    [data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]),
    [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: #e4eef7 !important;
        color: #0e4d7a !important;
    }
    /* Tab content background */
    [data-testid="stTabsContent"],
    [data-baseweb="tab-panel"] {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-top: 0.5rem !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stTabsContent"] {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 0.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }

    /* ── Question Card ── */
    .q-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .q-label {
        font-family: 'Sora', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }

    /* ── Result Cards ── */
    .result-card {
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        border-radius: 14px 14px 0 0;
    }
    .result-safe {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .result-safe::before { background: #16a34a; }
    .result-caution {
        background: #fffbeb;
        border: 1px solid #fde68a;
    }
    .result-caution::before { background: #d97706; }
    .result-danger {
        background: #fff1f2;
        border: 1px solid #fecdd3;
    }
    .result-danger::before { background: #dc2626; }

    .result-verdict {
        font-family: 'Sora', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .result-safe .result-verdict  { color: #15803d; }
    .result-caution .result-verdict { color: #92400e; }
    .result-danger .result-verdict  { color: #991b1b; }

    .result-advice {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #374151;
        margin: 0;
    }

    /* ── Probability Bar ── */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.4rem 0;
    }
    .prob-label {
        font-size: 0.78rem;
        font-weight: 600;
        width: 140px;
        color: #475569;
        flex-shrink: 0;
    }
    .prob-track {
        flex: 1;
        background: #e2e8f0;
        border-radius: 99px;
        height: 8px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s ease;
    }
    .prob-fill-safe    { background: #16a34a; }
    .prob-fill-caution { background: #d97706; }
    .prob-fill-danger  { background: #dc2626; }
    .prob-pct {
        font-size: 0.8rem;
        font-weight: 700;
        width: 36px;
        text-align: right;
        color: #0f172a;
    }
    .prob-section-title {
        font-family: 'Sora', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin: 1rem 0 0.5rem 0;
    }

    /* ── Action Box ── */
    .action-box {
        background: #f0f6fb;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
    }
    .action-box-title {
        font-family: 'Sora', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: #1a6fa8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .action-box ul {
        margin: 0;
        padding-left: 1.2rem;
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.7;
    }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a6fa8, #0e4d7a) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.6rem 1.5rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 2px 8px rgba(26,111,168,0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 14px rgba(26,111,168,0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* ── WhatsApp Button ── */
    .wa-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #25D366;
        color: white !important;
        text-decoration: none !important;
        border-radius: 10px;
        padding: 0.55rem 1.2rem;
        font-family: 'Sora', sans-serif;
        font-size: 0.84rem;
        font-weight: 600;
        margin-top: 0.8rem;
        transition: background 0.2s ease;
        border: none;
        cursor: pointer;
    }
    .wa-btn:hover { background: #1ebe5d; }

    /* ── Feedback ── */
    .feedback-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    .feedback-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
    }

    /* ── Section Divider ── */
    .sw-divider {
        height: 1px;
        background: linear-gradient(to right, #e2e8f0, transparent);
        margin: 1.2rem 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }
    /* Hide the collapse arrow */
    [data-testid="collapsedControl"],
    button[kind="header"] {
        display: none !important;
    }

    /* ── Farmer Profile Card ── */
    .profile-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }
    .profile-card-title {
        font-family: 'Sora', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: #1a6fa8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.8rem;
    }
    .profile-saved {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.84rem;
        color: #15803d;
        font-weight: 500;
        margin-top: 0.5rem;
        display: inline-block;
    }
    .sidebar-logo {
        text-align: center;
        padding: 0.8rem 0 1.2rem 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 1.2rem;
    }
    .sidebar-logo h2 {
        font-family: 'Sora', sans-serif !important;
        font-size: 1.05rem !important;
        color: #1a6fa8 !important;
        margin: 0.4rem 0 0.1rem 0 !important;
    }
    .sidebar-logo p {
        font-size: 0.72rem;
        color: #94a3b8;
        margin: 0;
    }
    .sidebar-section {
        font-family: 'Sora', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 1rem 0 0.5rem 0;
    }

    /* ── Radio button labels always visible ── */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span,
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #0f172a !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.8rem !important;
    }

    /* ── History Table ── */
    .history-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .history-meta {
        font-size: 0.8rem;
        color: #64748b;
    }
    .history-verdict {
        font-family: 'Sora', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .hv-safe    { background: #dcfce7; color: #15803d; }
    .hv-caution { background: #fef9c3; color: #92400e; }
    .hv-danger  { background: #fee2e2; color: #991b1b; }

    /* ── Footer ── */
    .sw-footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 0.75rem;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
        line-height: 1.8;
    }
    .sw-footer strong { color: #64748b; }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border-color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }

    /* ── Metric ── */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE SYSTEM — English + Hausa
# ══════════════════════════════════════════════════════════════════════════════

EN = {
    # App header
    "app_title":        "SmartWater Agriculture",
    "app_subtitle":     "AI-Powered Groundwater & Irrigation Advisory",
    "badge_location":   "Igabi & Zaria LGAs · Kaduna State",
    "badge_engine":     "Bayesian Decision Engine",
    "badge_partners":   "ABU Zaria · IAR Zaria",

    # Status
    "connected":        "✅  Connected to SmartWater API — live predictions enabled",
    "demo_mode":        "⚡  Demo Mode — showing example outputs. Backend will be connected soon.",

    # Sidebar
    "sidebar_profile":  "Farmer Profile",
    "your_name":        "Your Name / Village",
    "name_placeholder": "e.g. Musa, Rigachikun",
    "main_crop":        "Main Crop",
    "phone_type":       "Phone Type",
    "phone_basic":      "Basic phone (calls/SMS only)",
    "phone_feature":    "Feature phone (WhatsApp)",
    "phone_smart":      "Smartphone",
    "language":         "Display Language",
    "sidebar_about":    "About SmartWater",
    "about_text":       "SmartWater uses Bayesian AI to help farmers make smarter decisions about irrigation, borehole drilling, and early water stress warnings.",
    "version":          "SmartWater Agriculture v1.0",
    "m4d_credit":       "M4D Digital Extension Track · © ABU Zaria 2026",

    # Tab names
    "tab_irrigation":   "💧  Irrigation Advisory",
    "tab_drilling":     "🪨  Borehole / Drilling Risk",
    "tab_warning":      "⚠️  Early Warning",
    "tab_history":      "📋  History",

    # ── Tab 1: Irrigation ──
    "irr_heading":      "Should I Irrigate Today?",
    "irr_caption":      "Answer the questions below about your farm and water source to get a personalised recommendation.",
    "q_rainfall":       "How much rain fell in the last 3 days?",
    "opt_rain_low":     "Little or none",
    "opt_rain_mid":     "Some rain",
    "opt_rain_high":    "Heavy rain",
    "q_soil":           "How does your soil feel right now?",
    "opt_soil_dry":     "Very dry — cracks visible",
    "opt_soil_moist":   "Moist but not wet",
    "opt_soil_wet":     "Wet / muddy",
    "q_season":         "What is the current season / forecast?",
    "opt_seas_wet":     "Rains expected soon",
    "opt_seas_dry":     "Dry spell forecast",
    "opt_seas_drought": "Drought warning issued",
    "q_crop_need":      "How thirsty is your crop right now?",
    "opt_crop_low":     "Not very thirsty (early stage)",
    "opt_crop_mid":     "Moderate water need",
    "opt_crop_high":    "Very thirsty (flowering / fruiting)",
    "q_gw":             "Has your well or borehole water level dropped recently?",
    "opt_gw_normal":    "No — level looks normal",
    "opt_gw_low":       "Slightly lower than usual",
    "opt_gw_very_low":  "Much lower — very concerned",
    "btn_irrigate":     "Get Irrigation Advice",

    # Irrigation results
    "irr_safe_label":   "✅  Safe to Irrigate",
    "irr_safe_advice":  "Soil moisture, groundwater stress, and crop needs are all balanced. You can irrigate normally today.",
    "irr_careful_label":"⚡  Irrigate Carefully",
    "irr_careful_advice":"Some stress detected. Use 20–30% less water than usual. Prioritise morning irrigation to reduce evaporation.",
    "irr_delay_label":  "🛑  Delay Irrigation",
    "irr_delay_advice": "High groundwater stress or saturated soil detected. Rest your source for 24–48 hours. Check your pump and water level before proceeding.",
    "irr_actions_careful": ["Contact your extension agent this week",
                             "Irrigate in early morning only",
                             "Check and clear any blocked drip lines",
                             "Monitor borehole level daily"],
    "irr_actions_delay":   ["Do NOT irrigate in the next 24–48 hours",
                             "Check pump and water level immediately",
                             "Prepare water-saving measures",
                             "Contact Kaduna State Water Agency if level keeps dropping"],

    # ── Tab 2: Drilling ──
    "drill_heading":    "Is This a Good Place to Drill a Borehole?",
    "drill_caption":    "Use this before investing in a new borehole or well to assess your probability of success.",
    "q_terrain":        "What is the terrain like at the proposed drill site?",
    "opt_terrain_valley":"Flat / valley bottom (low ground)",
    "opt_terrain_slope": "Gentle slope",
    "opt_terrain_ridge": "Hilltop / ridge (high ground)",
    "q_nearby":         "Are there successful boreholes nearby?",
    "opt_nearby_yes":   "Yes — productive boreholes within 1 km",
    "opt_nearby_mixed": "Some boreholes but mixed results",
    "opt_nearby_no":    "No boreholes, or most have failed",
    "q_geology":        "What do you know about the soil / rock here?",
    "opt_geo_sandy":    "Sandy or alluvial soil (good signs)",
    "opt_geo_mixed":    "Mixed clay and sandy soil",
    "opt_geo_rock":     "Hard rock / granite near the surface",
    "q_depth":          "How deep are you planning to drill?",
    "opt_depth_shallow":"Shallow well (less than 20 metres)",
    "opt_depth_mid":    "Medium borehole (20–60 metres)",
    "opt_depth_deep":   "Deep borehole (more than 60 metres)",
    "btn_drilling":     "Assess Drilling Risk",

    # Drilling results
    "drill_good_label":     "✅  High Success Probability",
    "drill_good_advice":    "Geological and depth conditions are favourable. This is a good zone to invest in a new borehole.",
    "drill_unsure_label":   "🔶  Uncertain — Seek Expert Advice",
    "drill_unsure_advice":  "Mixed signals from geology and depth data. Commission a Vertical Electrical Sounding (VES) survey before drilling.",
    "drill_risk_label":     "🛑  High Drilling Risk",
    "drill_risk_advice":    "Unfavourable geology or insufficient depth. Avoid investing here without a full hydrogeological assessment.",
    "drill_note":           "Note: This is a probabilistic estimate. A VES survey will always improve accuracy before drilling.",

    # ── Tab 3: Early Warning ──
    "warn_heading":     "Water Stress Early Warning",
    "warn_caption":     "Anticipate water problems before they affect your crops or water supply.",
    "q_pump_age":       "How old is your water pump or borehole?",
    "opt_pump_new":     "New — less than 2 years old",
    "opt_pump_mid":     "2 to 5 years old",
    "opt_pump_old":     "More than 5 years old",
    "q_forecast":       "What is the seasonal outlook for the next month?",
    "opt_fc_normal":    "Normal rains expected",
    "opt_fc_dry":       "Dry spell forecast",
    "opt_fc_drought":   "Drought warning issued",
    "q_gw_trend":       "How has your water level been trending lately?",
    "opt_trend_stable": "Stable — no change",
    "opt_trend_slight": "Slightly declining",
    "opt_trend_fast":   "Declining fast — very concerned",
    "q_community":      "Are other farmers in your community reporting water problems?",
    "opt_comm_no":      "No — everyone seems fine",
    "opt_comm_few":     "A few are reporting issues",
    "opt_comm_many":    "Many farmers are struggling",
    "btn_warning":      "Check Early Warning Status",

    # Warning results
    "warn_normal_label":   "✅  Normal Conditions",
    "warn_normal_advice":  "No unusual water stress expected. Continue regular farming activities and monitor weekly.",
    "warn_watch_label":    "⚡  Watch Alert",
    "warn_watch_advice":   "Moderate stress signals detected. Reduce water use where possible, check pump condition, and prepare water storage.",
    "warn_critical_label": "🔴  Critical Alert",
    "warn_critical_advice":"High risk of water shortage or pump failure in coming weeks. Activate water-saving measures immediately. Contact your extension agent.",
    "warn_actions_watch":    ["Contact your extension agent this week",
                               "Begin water-saving irrigation practices",
                               "Service your pump if it is more than 3 years old",
                               "Store water in tanks or jerry cans as a buffer"],
    "warn_actions_critical": ["Contact Kaduna State Water Agency immediately",
                               "Switch to deficit irrigation — water only at critical crop stages",
                               "Coordinate with cooperative on shared water access",
                               "Postpone any new drilling until conditions improve",
                               "Monitor NIMET and Kaduna Met Agency bulletins daily"],

    # ── Shared UI ──
    "prob_breakdown":       "Probability Breakdown",
    "whatsapp_share":       "📲  Share via WhatsApp",
    "feedback_prompt":      "Was this advice helpful?",
    "feedback_yes":         "👍  Yes",
    "feedback_no":          "👎  No",
    "feedback_thanks":      "Thank you for your feedback!",
    "assessed_for":         "Advice generated for",
    "crop_label":           "Crop",
    "actions_title":        "Recommended Actions",

    # ── History tab ──
    "history_heading":      "Assessment History",
    "history_empty":        "No assessments yet. Complete a risk assessment to see history here.",
    "history_total":        "Total Assessments",
    "history_high_risk":    "Caution / Delay Cases",
    "history_last":         "Last Assessment",
    "history_clear":        "🗑  Clear History",
    "history_export":       "📥  Export as CSV",
    "col_time":             "Time",
    "col_farmer":           "Farmer",
    "col_type":             "Type",
    "col_verdict":          "Verdict",
    "col_confidence":       "Confidence",
}


HA = {
    # App header
    "app_title":        "SmartWater Noma",
    "app_subtitle":     "Shawarar Ruwan Ƙasa Mai Amfani da AI",
    "badge_location":   "Igabi & Zaria LGAs · Jihar Kaduna",
    "badge_engine":     "Injin Yanke Shawara na Bayesian",
    "badge_partners":   "ABU Zaria · IAR Zaria",

    # Status
    "connected":        "✅  An haɗa da API na SmartWater — ana yin annabci kai tsaye",
    "demo_mode":        "⚡  Yanayin Demo — ana nuna misalai. Za a haɗa backend nan ba da jimawa ba.",

    # Sidebar
    "sidebar_profile":  "Bayanin Manomi",
    "your_name":        "Sunanka / Ƙauye",
    "name_placeholder": "misali: Musa, Rigachikun",
    "main_crop":        "Babban Amfanin Gona",
    "phone_type":       "Nau'in Waya",
    "phone_basic":      "Wayar gargajiya (kira/SMS kawai)",
    "phone_feature":    "Wayar fasali (WhatsApp)",
    "phone_smart":      "Wayar Smart",
    "language":         "Harshen Nuni",
    "sidebar_about":    "Game da SmartWater",
    "about_text":       "SmartWater yana amfani da AI na Bayesian don taimaka wa manoma su yanke shawara mafi kyau game da ban ruwa, hako rijiya, da gargaɗi na farkon wahalar ruwa.",
    "version":          "SmartWater Noma v1.0",
    "m4d_credit":       "M4D Digital Extension Track · © ABU Zaria 2026",

    # Tab names
    "tab_irrigation":   "💧  Shawarar Ban Ruwa",
    "tab_drilling":     "🪨  Haɗarin Hako Rijiya",
    "tab_warning":      "⚠️  Gargaɗin Farko",
    "tab_history":      "📋  Tarihi",

    # ── Tab 1: Irrigation ──
    "irr_heading":      "Shin Ya Kamata In Yi Ban Ruwa Yau?",
    "irr_caption":      "Amsa tambayoyin da ke ƙasa game da gonar ka da tushen ruwa don samun shawarar da ta dace da ka.",
    "q_rainfall":       "Nawa ne ruwan sama da ya fadi a cikin kwanaki 3 da suka gabata?",
    "opt_rain_low":     "Kaɗan ko babu",
    "opt_rain_mid":     "Ruwan sama kaɗan",
    "opt_rain_high":    "Ruwan sama mai yawa",
    "q_soil":           "Yaya ƙasar gonar ka take a yanzu?",
    "opt_soil_dry":     "Busasshe sosai — ana ganin tsagewa",
    "opt_soil_moist":   "Mai laima amma ba mai jike ba",
    "opt_soil_wet":     "Mai jike / laushi",
    "q_season":         "Mene ne yanayin kakar noma a yanzu?",
    "opt_seas_wet":     "Ana sa ran ruwan sama nan ba da jimawa ba",
    "opt_seas_dry":     "An annabta lokacin rani",
    "opt_seas_drought": "An sanar da gargaɗin fari",
    "q_crop_need":      "Nawa ne bukatar ruwa ta amfanin gona ka a yanzu?",
    "opt_crop_low":     "Babu ƙishirwa (farkon matakin)",
    "opt_crop_mid":     "Bukatar ruwa ta matsakaici",
    "opt_crop_high":    "Ƙishirwa sosai (lokacin fure / ɗanye)",
    "q_gw":             "Shin matakin ruwa a cikin rijiyar ka ya sauka kwanan nan?",
    "opt_gw_normal":    "A'a — matakin yana da kyau",
    "opt_gw_low":       "Ya sauka kaɗan fiye da al'ada",
    "opt_gw_very_low":  "Ya sauka sosai — na damu ƙwarai",
    "btn_irrigate":     "Sami Shawarar Ban Ruwa",

    # Irrigation results
    "irr_safe_label":   "✅  Ana iya Ban Ruwa",
    "irr_safe_advice":  "Laimar ƙasa, damuwar ruwan ƙasa, da buƙatar amfanin gona sun dace. Zaka iya yin ban ruwa al'ada yau.",
    "irr_careful_label":"⚡  Yi Ban Ruwa da Hankali",
    "irr_careful_advice":"An gano wani damuwa. Yi amfani da ruwa 20–30% ƙasa da al'ada. Fifita ban ruwa da safe don rage ƙwararrawa.",
    "irr_delay_label":  "🛑  Jinkirta Ban Ruwa",
    "irr_delay_advice": "An gano damuwar ruwan ƙasa mai yawa ko ƙasar da ta cika ruwa. Huta tushen ruwa na sa'o'i 24–48. Duba famfo da matakin ruwa kafin ci gaba.",
    "irr_actions_careful": ["Tuntuɓi wakili na faɗaɗawa wannan mako",
                             "Yi ban ruwa da safe kawai",
                             "Duba da share duk bututu da suka toshe",
                             "Kula da matakin rijiya kowace rana"],
    "irr_actions_delay":   ["KADA ku yi ban ruwa cikin sa'o'i 24–48",
                             "Duba famfo da matakin ruwa nan da nan",
                             "Shirya matakan adana ruwa",
                             "Tuntuɓi Hukumar Ruwa ta Jihar Kaduna idan matakin ya ci gaba da sauka"],

    # ── Tab 2: Drilling ──
    "drill_heading":    "Shin Wannan Wuri ne Mai Kyau don Hako Rijiya?",
    "drill_caption":    "Yi amfani da wannan kafin ka zuba jari a rijiya ko maɓuɓɓugar ruwa don kimanta damar nasararku.",
    "q_terrain":        "Yaya yanayin ƙasar da aka tsara don hako rijiya?",
    "opt_terrain_valley":"Fili / ƙasan kwari (ƙasa mai ƙasƙanci)",
    "opt_terrain_slope": "Gangaren ƙasa mai taushi",
    "opt_terrain_ridge": "Kan tudun ƙasa / gefen dutse (ƙasa mai tsayi)",
    "q_nearby":         "Shin akwai rijiyoyin da suke aiki a kusa?",
    "opt_nearby_yes":   "Eh — rijiyoyin da suke aiki a cikin km 1",
    "opt_nearby_mixed": "Wasu rijiyoyi amma sakamako ya bambanta",
    "opt_nearby_no":    "Babu rijiyoyi, ko mafi yawa sun kasa",
    "q_geology":        "Me kuka sani game da ƙasan / dutse a nan?",
    "opt_geo_sandy":    "Ƙasar yashi ko alluvial (alamomi masu kyau)",
    "opt_geo_mixed":    "Cakuɗen yumɓu da yashi",
    "opt_geo_rock":     "Dutse mai ƙarfi / granite kusa da saman ƙasa",
    "q_depth":          "Nawa ne zurfin da kuke shirin hako?",
    "opt_depth_shallow":"Rijiya mai zurfi kaɗan (ƙasa da mita 20)",
    "opt_depth_mid":    "Rijiya ta matsakaici (mita 20–60)",
    "opt_depth_deep":   "Rijiya mai zurfi (fiye da mita 60)",
    "btn_drilling":     "Kimanta Haɗarin Hako",

    # Drilling results
    "drill_good_label":     "✅  Yiwuwar Nasara Mai Yawa",
    "drill_good_advice":    "Yanayin ƙasa da zurfin sun dace. Wannan yanki ne mai kyau don zuba jari a sabuwar rijiya.",
    "drill_unsure_label":   "🔶  Ba a Tabbata — Nemi Shawarar Ƙwararru",
    "drill_unsure_advice":  "Alamomin sun bambanta daga ƙasa da zurfin. Yi binciken VES kafin ka fara hako.",
    "drill_risk_label":     "🛑  Haɗarin Hako Mai Yawa",
    "drill_risk_advice":    "Yanayin ƙasa mara kyau ko bayanai ba su isa ba. Guji zuba jari a nan ba tare da cikakken kimantawa ba.",
    "drill_note":           "Lura: Wannan ƙididdiga ce ta yuwuwa. Binciken VES zai koyaushe inganta daidaito kafin hako.",

    # ── Tab 3: Early Warning ──
    "warn_heading":     "Gargaɗin Farkon Wahalar Ruwa",
    "warn_caption":     "Annabci matsalolin ruwa kafin su shafi amfanin gona ko wadatar ruwa.",
    "q_pump_age":       "Yaya tsufa famfon ruwa ko rijiyar ka?",
    "opt_pump_new":     "Sabo — ƙasa da shekaru 2",
    "opt_pump_mid":     "Shekaru 2 zuwa 5",
    "opt_pump_old":     "Fiye da shekaru 5",
    "q_forecast":       "Yaya hasashen kakar noma ga wata mai zuwa?",
    "opt_fc_normal":    "Ana sa ran ruwan sama na al'ada",
    "opt_fc_dry":       "An annabta lokacin rani",
    "opt_fc_drought":   "An sanar da gargaɗin fari",
    "q_gw_trend":       "Yaya matakin ruwan ka ya kasance kwanan nan?",
    "opt_trend_stable": "Tsayayye — babu sauyi",
    "opt_trend_slight": "Yana sauka kaɗan kaɗan",
    "opt_trend_fast":   "Yana sauka da sauri — na damu sosai",
    "q_community":      "Shin wasu manoma a ƙaunuwarku suna ba da rahoton matsalolin ruwa?",
    "opt_comm_no":      "A'a — kowa yana da kyau",
    "opt_comm_few":     "Wasu kaɗan suna ba da rahoton matsaloli",
    "opt_comm_many":    "Yawancin manoma suna fama da matsala",
    "btn_warning":      "Duba Matsayin Gargaɗi",

    # Warning results
    "warn_normal_label":   "✅  Yanayi na Al'ada",
    "warn_normal_advice":  "Ba a sa ran wata wahalar ruwa ta musamman. Ci gaba da ayyukan noma na yau da kullum kuma kula kowace mako.",
    "warn_watch_label":    "⚡  Gargaɗin Kallo",
    "warn_watch_advice":   "An gano alamomin damuwa na matsakaici. Rage amfani da ruwa inda zai yiwu, duba yanayin famfo, kuma shirya adana ruwa.",
    "warn_critical_label": "🔴  Gargaɗi Mai Muhimmanci",
    "warn_critical_advice":"Akwai haɗarin rashin ruwa ko karyewar famfo a makonni masu zuwa. Fara matakan adana ruwa nan da nan. Tuntuɓi wakilin faɗaɗawa.",
    "warn_actions_watch":    ["Tuntuɓi wakilin faɗaɗawa wannan mako",
                               "Fara ayyukan ban ruwa na adana ruwa",
                               "Yi gyaran famfo idan ya haura shekaru 3",
                               "Adana ruwa a tankuna ko kwalaben ruwa"],
    "warn_actions_critical": ["Tuntuɓi Hukumar Ruwa ta Jihar Kaduna nan da nan",
                               "Canja zuwa ban ruwa na ƙarancin ruwa — ba da ruwa a matakai masu mahimmanci kawai",
                               "Haɗin gwiwa da ƙungiyar manoma don raba ruwa",
                               "Jinkirta duk wani hako rijiya sai yanayi ya inganta",
                               "Kula da sanarwar NIMET da Hukumar Kula da Yanayi ta Kaduna kowace rana"],

    # ── Shared UI ──
    "prob_breakdown":       "Rabon Yuwuwa",
    "whatsapp_share":       "📲  Raba ta WhatsApp",
    "feedback_prompt":      "Shin shawarar ta taimaka?",
    "feedback_yes":         "👍  Eh",
    "feedback_no":          "👎  A'a",
    "feedback_thanks":      "Na gode da ra'ayinka!",
    "assessed_for":         "An yi shawarar don",
    "crop_label":           "Amfanin Gona",
    "actions_title":        "Ayyuka da Ake Ba da Shawarar",

    # ── History tab ──
    "history_heading":      "Tarihin Kimantawa",
    "history_empty":        "Babu kimantawa tukuna. Kammala kimantawar haɗari don ganin tarihin a nan.",
    "history_total":        "Jimlar Kimantawa",
    "history_high_risk":    "Lokuta na Hankali / Jinkiri",
    "history_last":         "Ƙarshen Kimantawa",
    "history_clear":        "🗑  Share Tarihi",
    "history_export":       "📥  Fitar a matsayin CSV",
    "col_time":             "Lokaci",
    "col_farmer":           "Manomi",
    "col_type":             "Nau'in",
    "col_verdict":          "Shawara",
    "col_confidence":       "Tabbaci",
}


def L(key):
    """Return string in the currently selected language."""
    lang = st.session_state.get("lang", "en")
    d = HA if lang == "ha" else EN
    return d.get(key, EN.get(key, key))


CROP_OPTIONS = {
    "en": ["Maize", "Tomato", "Onion", "Sorghum", "Cowpea", "Pepper", "Other"],
    "ha": ["Masara", "Tomato", "Albasa", "Dawa", "Wake", "Tattasai", "Sauran"],
}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "lang":         "en",
        "history":      [],
        "api_ok":       None,
        "farmer_name":  "",
        "crop":         "Maize",
        "phone":        "Basic phone (calls/SMS only)",
        # feedback tracking
        "fb_irr":       None,
        "fb_drill":     None,
        "fb_warn":      None,
        "farmer_id":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Restore farmer_id from URL on refresh ────────────────────────────────
    try:
        params = st.query_params
        if "fid" in params and not st.session_state.get("farmer_id"):
            st.session_state["farmer_id"] = params["fid"]
        if "fname" in params and not st.session_state.get("farmer_name"):
            st.session_state["farmer_name"] = params["fname"]
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# DEMO INFERENCE — mirrors PDF CPT logic without pgmpy
# ══════════════════════════════════════════════════════════════════════════════

def demo_irrigation(soil: int, crop_need: int, gw_stress: int) -> np.ndarray:
    """Returns [p_safe, p_careful, p_delay] from heuristic CPT."""
    if soil == 0 and crop_need == 2 and gw_stress == 0:
        return np.array([0.85, 0.12, 0.03])
    elif soil == 2 and gw_stress == 2:
        return np.array([0.02, 0.08, 0.90])
    elif soil == 1 and gw_stress == 1:
        return np.array([0.30, 0.55, 0.15])
    elif soil == 0 and gw_stress == 2:
        return np.array([0.10, 0.40, 0.50])
    elif soil == 2 and gw_stress == 0:
        return np.array([0.05, 0.30, 0.65])
    elif soil == 0:
        return np.array([0.65, 0.25, 0.10])
    elif soil == 2:
        return np.array([0.05, 0.20, 0.75])
    else:
        return np.array([0.35, 0.45, 0.20])


def demo_drilling(depth: int, geology: int) -> np.ndarray:
    """Returns [p_success, p_uncertain, p_risk] from CPT."""
    table = {
        (0, 0): [0.10, 0.25, 0.65],
        (0, 1): [0.30, 0.45, 0.25],
        (0, 2): [0.60, 0.30, 0.10],
        (1, 0): [0.20, 0.35, 0.45],
        (1, 1): [0.50, 0.38, 0.12],
        (1, 2): [0.80, 0.17, 0.03],
        (2, 0): [0.35, 0.40, 0.25],
        (2, 1): [0.65, 0.28, 0.07],
        (2, 2): [0.90, 0.09, 0.01],
    }
    return np.array(table.get((depth, geology), [0.40, 0.35, 0.25]))


def demo_warning(pump: int, forecast: int, gw_stress: int) -> np.ndarray:
    """Returns [p_normal, p_watch, p_critical] from heuristic CPT."""
    if pump == 2 and gw_stress == 2 and forecast == 2:
        return np.array([0.02, 0.08, 0.90])
    elif pump == 2 and gw_stress >= 1 and forecast >= 1:
        return np.array([0.05, 0.35, 0.60])
    elif gw_stress == 2 and forecast >= 1:
        return np.array([0.05, 0.30, 0.65])
    elif gw_stress == 1 and forecast == 1:
        return np.array([0.20, 0.60, 0.20])
    elif gw_stress == 0 and forecast == 0:
        return np.array([0.88, 0.10, 0.02])
    elif pump == 2:
        return np.array([0.30, 0.50, 0.20])
    else:
        return np.array([0.55, 0.35, 0.10])


# ══════════════════════════════════════════════════════════════════════════════
# API LAYER — calls Render backend, falls back to demo
# ══════════════════════════════════════════════════════════════════════════════

API_URL = os.getenv(
    "API_BASE_URL",
    "https://smartwater-api.onrender.com"   # ← replace when backend is deployed
)


@st.cache_data(ttl=30)
def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def call_irrigation(evidence: dict) -> np.ndarray:
    farmer_id = st.session_state.get("farmer_id")
    payload   = {**evidence}
    if farmer_id:
        payload["farmer_id"] = farmer_id
    try:
        r = requests.post(f"{API_URL}/predict-irrigation", json=payload, timeout=8)
        if r.status_code == 200:
            d = r.json()
            return np.array([d["p_safe"], d["p_careful"], d["p_delay"]])
    except Exception:
        pass
    return demo_irrigation(
        evidence["SoilMoisture"],
        evidence["CropWaterNeed"],
        evidence["GroundwaterStress"],
    )


def call_drilling(evidence: dict) -> np.ndarray:
    farmer_id = st.session_state.get("farmer_id")
    payload   = {**evidence}
    if farmer_id:
        payload["farmer_id"] = farmer_id
    try:
        r = requests.post(f"{API_URL}/predict-drilling", json=payload, timeout=8)
        if r.status_code == 200:
            d = r.json()
            return np.array([d["p_success"], d["p_uncertain"], d["p_risk"]])
    except Exception:
        pass
    return demo_drilling(evidence["BoreholeDepth"], evidence["GeologyFavorability"])


def call_warning(evidence: dict) -> np.ndarray:
    farmer_id = st.session_state.get("farmer_id")
    payload   = {**evidence}
    if farmer_id:
        payload["farmer_id"] = farmer_id
    try:
        r = requests.post(f"{API_URL}/predict-warning", json=payload, timeout=8)
        if r.status_code == 200:
            d = r.json()
            return np.array([d["p_normal"], d["p_watch"], d["p_critical"]])
    except Exception:
        pass
    return demo_warning(
        evidence["PumpAge"],
        evidence["SeasonalForecast"],
        evidence["GroundwaterStress"],
    )


def register_farmer_api(name: str, village: str = None,
                        crop: str = None, phone: str = None) -> str:
    """Register farmer on backend, return farmer_id."""
    try:
        payload = {"name": name}
        if village: payload["village"] = village
        if crop:    payload["crops_grown"] = [crop]
        if phone:   payload["phone"] = phone
        r = requests.post(f"{API_URL}/farmers/register", json=payload, timeout=8)
        if r.status_code == 200:
            return r.json().get("farmer_id")
    except Exception:
        pass
    return None


def fetch_farmer_history(farmer_id: str) -> list:
    """Pull farmer assessment history from Supabase via backend."""
    try:
        r = requests.get(f"{API_URL}/farmers/{farmer_id}/history", timeout=8)
        if r.status_code == 200:
            return r.json().get("assessments", [])
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

CARD_CLASSES  = ["result-safe",    "result-caution",  "result-danger"]
FILL_CLASSES  = ["prob-fill-safe", "prob-fill-caution","prob-fill-danger"]
VERDICT_KEYS  = {
    "irr":   ["irr_safe_label",   "irr_careful_label",   "irr_delay_label"],
    "drill": ["drill_good_label", "drill_unsure_label",   "drill_risk_label"],
    "warn":  ["warn_normal_label","warn_watch_label",     "warn_critical_label"],
}
ADVICE_KEYS   = {
    "irr":   ["irr_safe_advice",   "irr_careful_advice",   "irr_delay_advice"],
    "drill": ["drill_good_advice", "drill_unsure_advice",   "drill_risk_advice"],
    "warn":  ["warn_normal_advice","warn_watch_advice",     "warn_critical_advice"],
}
PROB_LABELS   = {
    "irr":   ["irr_safe_label",   "irr_careful_label",   "irr_delay_label"],
    "drill": ["drill_good_label", "drill_unsure_label",   "drill_risk_label"],
    "warn":  ["warn_normal_label","warn_watch_label",     "warn_critical_label"],
}


def render_result(module: str, probs: np.ndarray):
    """Render a full result card — verdict, advice, probability bars."""
    idx     = int(np.argmax(probs))
    card_cls = CARD_CLASSES[idx]
    verdict  = L(VERDICT_KEYS[module][idx])
    advice   = L(ADVICE_KEYS[module][idx])

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <p class="result-verdict">{verdict}</p>
        <p class="result-advice">{advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown(f'<p class="prob-section-title">{L("prob_breakdown")}</p>', unsafe_allow_html=True)
    for i, (p, fill) in enumerate(zip(probs, FILL_CLASSES)):
        label = L(PROB_LABELS[module][i])
        pct   = int(round(p * 100))
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-label">{label}</span>
            <div class="prob-track">
                <div class="prob-fill {fill}" style="width:{pct}%"></div>
            </div>
            <span class="prob-pct">{pct}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_actions(actions: list):
    """Render an action recommendation box."""
    items_html = "".join(f"<li>{a}</li>" for a in actions)
    st.markdown(f"""
    <div class="action-box">
        <p class="action-box-title">⚡ {L("actions_title")}</p>
        <ul>{items_html}</ul>
    </div>
    """, unsafe_allow_html=True)


def whatsapp_share(verdict: str, advice: str):
    """WhatsApp share button with pre-filled message."""
    farmer  = st.session_state.get("farmer_name", "") or "Farmer"
    crop    = st.session_state.get("crop", "")
    text    = (
        f"🌊 SmartWater Agriculture\n"
        f"Farmer: {farmer} | Crop: {crop}\n\n"
        f"{verdict}\n{advice}\n\n"
        f"Powered by Bayesian AI — ABU Zaria & IAR Zaria"
    )
    url = "https://wa.me/?text=" + urllib.parse.quote(text)
    st.markdown(
        f'<a class="wa-btn" href="{url}" target="_blank">{L("whatsapp_share")}</a>',
        unsafe_allow_html=True,
    )


def feedback_widget(module_key: str, verdict: str, probs: np.ndarray):
    """Thumbs up / down feedback — Supabase logging placeholder."""
    st.markdown(f"""
    <div class="sw-divider"></div>
    <div class="feedback-row">
        <span class="feedback-label">{L("feedback_prompt")}</span>
    </div>
    """, unsafe_allow_html=True)

    col_y, col_n, _ = st.columns([1, 1, 6])
    with col_y:
        if st.button(L("feedback_yes"), key=f"fb_yes_{module_key}"):
            _save_feedback(module_key, verdict, probs, helpful=True)
    with col_n:
        if st.button(L("feedback_no"), key=f"fb_no_{module_key}"):
            _save_feedback(module_key, verdict, probs, helpful=False)

    if st.session_state.get(f"fb_{module_key}") is not None:
        st.caption(f"✓ {L('feedback_thanks')}")


def _save_feedback(module: str, verdict: str, probs: np.ndarray, helpful: bool):
    """
    ── Supabase placeholder ──────────────────────────────────────────────────
    Replace the pass below with your Supabase insert when backend is ready.

    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    supabase.table("feedback").insert({
        "module":   module,
        "verdict":  verdict,
        "probs":    probs.tolist(),
        "helpful":  helpful,
        "farmer":   st.session_state.get("farmer_name",""),
        "crop":     st.session_state.get("crop",""),
        "ts":       datetime.utcnow().isoformat(),
    }).execute()
    ─────────────────────────────────────────────────────────────────────────
    """
    st.session_state[f"fb_{module}"] = helpful


def save_history(module: str, verdict: str, probs: np.ndarray, confidence: str):
    entry = {
        "ts":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        "farmer":     st.session_state.get("farmer_name", "—"),
        "type":       module.upper(),
        "verdict":    verdict,
        "confidence": confidence,
        "probs":      probs.tolist(),
    }
    st.session_state["history"].append(entry)


def confidence_from_probs(probs: np.ndarray) -> str:
    """Map max probability to a confidence label."""
    m = float(np.max(probs))
    if m >= 0.75: return "HIGH"
    if m >= 0.50: return "MEDIUM"
    return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # Logo / branding
        st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size:2rem">💧</div>
            <h2>SmartWater</h2>
            <p>Agriculture Advisory</p>
        </div>
        """, unsafe_allow_html=True)

        # Language toggle — top of sidebar so it affects everything below
        st.markdown(f'<p class="sidebar-section">{L("language")}</p>', unsafe_allow_html=True)
        lang_choice = st.radio(
            label="Display Language",
            options=["English", "Hausa"],
            index=0 if st.session_state.lang == "en" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.lang = "en" if lang_choice == "English" else "ha"

        st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

        # About — dark text so it's visible on light sidebar
        st.markdown(f'<p class="sidebar-section">{L("sidebar_about")}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.82rem; color:#334155; line-height:1.6; margin:0">'
            f'{L("about_text")}</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.75rem; color:#64748b; margin:0.2rem 0">{L("version")}</p>'
            f'<p style="font-size:0.72rem; color:#94a3b8; margin:0">{L("m4d_credit")}</p>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IRRIGATION ADVISORY
# ══════════════════════════════════════════════════════════════════════════════

def tab_irrigation():
    st.markdown(f"### {L('irr_heading')}")
    st.caption(L("irr_caption"))
    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container():
            rainfall_opt = st.selectbox(
                L("q_rainfall"),
                options=[L("opt_rain_low"), L("opt_rain_mid"), L("opt_rain_high")],
                key="irr_rainfall",
            )
            soil_opt = st.selectbox(
                L("q_soil"),
                options=[L("opt_soil_dry"), L("opt_soil_moist"), L("opt_soil_wet")],
                key="irr_soil",
            )
            season_opt = st.selectbox(
                L("q_season"),
                options=[L("opt_seas_wet"), L("opt_seas_dry"), L("opt_seas_drought")],
                key="irr_season",
            )

    with col2:
        crop_need_opt = st.selectbox(
            L("q_crop_need"),
            options=[L("opt_crop_low"), L("opt_crop_mid"), L("opt_crop_high")],
            key="irr_crop",
        )
        gw_opt = st.selectbox(
            L("q_gw"),
            options=[L("opt_gw_normal"), L("opt_gw_low"), L("opt_gw_very_low")],
            key="irr_gw",
        )

    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button(
            f"💧  {L('btn_irrigate')}",
            type="primary",
            use_container_width=True,
            key="btn_irr",
        )

    if run:
        # Map options → CPT indices
        rain_map  = {L("opt_rain_low"): 0,    L("opt_rain_mid"): 1,    L("opt_rain_high"): 2}
        soil_map  = {L("opt_soil_dry"): 0,    L("opt_soil_moist"): 1,  L("opt_soil_wet"): 2}
        seas_map  = {L("opt_seas_wet"): 0,    L("opt_seas_dry"): 1,    L("opt_seas_drought"): 2}
        crop_map  = {L("opt_crop_low"): 0,    L("opt_crop_mid"): 1,    L("opt_crop_high"): 2}
        gw_map    = {L("opt_gw_normal"): 0,   L("opt_gw_low"): 1,      L("opt_gw_very_low"): 2}

        evidence = {
            "Rainfall":          rain_map[rainfall_opt],
            "SeasonalForecast":  seas_map[season_opt],
            "SoilMoisture":      soil_map[soil_opt],
            "CropWaterNeed":     crop_map[crop_need_opt],
            "GroundwaterStress": gw_map[gw_opt],
        }

        with st.spinner(""):
            probs = call_irrigation(evidence)

        idx        = int(np.argmax(probs))
        verdict    = L(VERDICT_KEYS["irr"][idx])
        advice     = L(ADVICE_KEYS["irr"][idx])
        confidence = confidence_from_probs(probs)

        # Footer attribution
        farmer = st.session_state.get("farmer_name", "") or "Farmer"
        crop   = st.session_state.get("crop", "")
        st.caption(f"{L('assessed_for')}: **{farmer}** · {L('crop_label')}: {crop} · {datetime.now().strftime('%d %b %Y, %H:%M')}")

        render_result("irr", probs)

        # Action boxes for caution / delay
        if idx == 1:
            render_actions(L("irr_actions_careful"))
        elif idx == 2:
            render_actions(L("irr_actions_delay"))

        whatsapp_share(verdict, advice)
        feedback_widget("irr", verdict, probs)
        save_history("irr", verdict, probs, confidence)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BOREHOLE / DRILLING RISK
# ══════════════════════════════════════════════════════════════════════════════

def tab_drilling():
    st.markdown(f"### {L('drill_heading')}")
    st.caption(L("drill_caption"))
    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        terrain_opt = st.selectbox(
            L("q_terrain"),
            options=[L("opt_terrain_valley"), L("opt_terrain_slope"), L("opt_terrain_ridge")],
            key="drill_terrain",
        )
        nearby_opt = st.selectbox(
            L("q_nearby"),
            options=[L("opt_nearby_yes"), L("opt_nearby_mixed"), L("opt_nearby_no")],
            key="drill_nearby",
        )

    with col2:
        geology_opt = st.selectbox(
            L("q_geology"),
            options=[L("opt_geo_sandy"), L("opt_geo_mixed"), L("opt_geo_rock")],
            key="drill_geology",
        )
        depth_opt = st.selectbox(
            L("q_depth"),
            options=[L("opt_depth_shallow"), L("opt_depth_mid"), L("opt_depth_deep")],
            key="drill_depth",
        )

    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button(
            f"🪨  {L('btn_drilling')}",
            type="primary",
            use_container_width=True,
            key="btn_drill",
        )

    if run:
        terrain_map = {L("opt_terrain_valley"): 2, L("opt_terrain_slope"): 1, L("opt_terrain_ridge"): 0}
        nearby_map  = {L("opt_nearby_yes"): 2,     L("opt_nearby_mixed"): 1,  L("opt_nearby_no"): 0}
        geology_map = {L("opt_geo_sandy"): 2,      L("opt_geo_mixed"): 1,     L("opt_geo_rock"): 0}
        depth_map   = {L("opt_depth_shallow"): 0,  L("opt_depth_mid"): 1,     L("opt_depth_deep"): 2}

        geo_score = round(
            (terrain_map[terrain_opt] + nearby_map[nearby_opt] + geology_map[geology_opt]) / 3
        )
        geo_score = max(0, min(2, geo_score))

        evidence = {
            "BoreholeDepth":        depth_map[depth_opt],
            "GeologyFavorability":  geo_score,
        }

        with st.spinner(""):
            probs = call_drilling(evidence)

        idx        = int(np.argmax(probs))
        verdict    = L(VERDICT_KEYS["drill"][idx])
        advice     = L(ADVICE_KEYS["drill"][idx])
        confidence = confidence_from_probs(probs)

        farmer = st.session_state.get("farmer_name", "") or "Farmer"
        st.caption(f"{L('assessed_for')}: **{farmer}** · {datetime.now().strftime('%d %b %Y, %H:%M')}")

        render_result("drill", probs)
        st.caption(L("drill_note"))

        whatsapp_share(verdict, advice)
        feedback_widget("drill", verdict, probs)
        save_history("drill", verdict, probs, confidence)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EARLY WARNING
# ══════════════════════════════════════════════════════════════════════════════

def tab_warning():
    st.markdown(f"### {L('warn_heading')}")
    st.caption(L("warn_caption"))
    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        pump_opt = st.selectbox(
            L("q_pump_age"),
            options=[L("opt_pump_new"), L("opt_pump_mid"), L("opt_pump_old")],
            key="warn_pump",
        )
        forecast_opt = st.selectbox(
            L("q_forecast"),
            options=[L("opt_fc_normal"), L("opt_fc_dry"), L("opt_fc_drought")],
            key="warn_forecast",
        )

    with col2:
        trend_opt = st.selectbox(
            L("q_gw_trend"),
            options=[L("opt_trend_stable"), L("opt_trend_slight"), L("opt_trend_fast")],
            key="warn_trend",
        )
        community_opt = st.selectbox(
            L("q_community"),
            options=[L("opt_comm_no"), L("opt_comm_few"), L("opt_comm_many")],
            key="warn_community",
        )

    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button(
            f"⚠️  {L('btn_warning')}",
            type="primary",
            use_container_width=True,
            key="btn_warn",
        )

    if run:
        pump_map     = {L("opt_pump_new"): 0,     L("opt_pump_mid"): 1,      L("opt_pump_old"): 2}
        forecast_map = {L("opt_fc_normal"): 0,    L("opt_fc_dry"): 1,        L("opt_fc_drought"): 2}
        trend_map    = {L("opt_trend_stable"): 0, L("opt_trend_slight"): 1,  L("opt_trend_fast"): 2}
        comm_map     = {L("opt_comm_no"): 0,      L("opt_comm_few"): 1,      L("opt_comm_many"): 2}

        raw_gw  = trend_map[trend_opt]
        comm_b  = comm_map[community_opt]
        combined_gw = min(2, max(raw_gw, round((raw_gw + comm_b) / 2 + 0.4)))

        evidence = {
            "PumpAge":           pump_map[pump_opt],
            "SeasonalForecast":  forecast_map[forecast_opt],
            "GroundwaterStress": combined_gw,
        }

        with st.spinner(""):
            probs = call_warning(evidence)

        idx        = int(np.argmax(probs))
        verdict    = L(VERDICT_KEYS["warn"][idx])
        advice     = L(ADVICE_KEYS["warn"][idx])
        confidence = confidence_from_probs(probs)

        farmer = st.session_state.get("farmer_name", "") or "Farmer"
        st.caption(f"{L('assessed_for')}: **{farmer}** · {datetime.now().strftime('%d %b %Y, %H:%M')}")

        render_result("warn", probs)

        if idx == 1:
            render_actions(L("warn_actions_watch"))
        elif idx == 2:
            render_actions(L("warn_actions_critical"))

        whatsapp_share(verdict, advice)
        feedback_widget("warn", verdict, probs)
        save_history("warn", verdict, probs, confidence)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def tab_history():
    st.markdown(f"### {L('history_heading')}")

    farmer_id = st.session_state.get("farmer_id")

    # ── Try loading from Supabase first ──────────────────────────────────────
    if farmer_id:
        remote = fetch_farmer_history(farmer_id)
        if remote:
            # Normalise remote records to match local history format
            for rec in remote:
                ts      = rec.get("created_at","")[:16].replace("T"," ")
                pvals   = rec.get("p_values", {})
                module  = rec.get("module","").upper()[:4]
                verdict = rec.get("prediction","")
                probs   = list(pvals.values()) if pvals else [0.33,0.33,0.34]
                # Avoid duplicates
                existing_ts = [h["ts"] for h in st.session_state["history"]]
                if ts not in existing_ts:
                    st.session_state["history"].append({
                        "ts":         ts,
                        "farmer":     st.session_state.get("farmer_name","—"),
                        "type":       module if module else rec.get("module","IRR").upper(),
                        "verdict":    verdict,
                        "confidence": rec.get("confidence","—"),
                        "probs":      probs,
                    })

    history = st.session_state.get("history", [])

    if not history:
        st.info(L("history_empty"))
        return

    # Summary metrics
    total      = len(history)
    high_risk  = sum(1 for h in history if "Carefully" in h["verdict"] or
                     "Delay" in h["verdict"] or "Watch" in h["verdict"] or
                     "Critical" in h["verdict"] or "Uncertain" in h["verdict"] or
                     "Risk" in h["verdict"] or "Hankali" in h["verdict"] or
                     "Jinkirta" in h["verdict"] or "Gargaɗi" in h["verdict"])
    last_ts    = history[-1]["ts"] if history else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric(L("history_total"),     total)
    c2.metric(L("history_high_risk"), high_risk)
    c3.metric(L("history_last"),      last_ts)

    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    # History cards
    verdict_class_map = {
        "safe": "hv-safe", "Safe": "hv-safe", "Ana iya": "hv-safe",
        "Carefully": "hv-caution", "Watch": "hv-caution", "Uncertain": "hv-caution",
        "Hankali": "hv-caution", "Kallo": "hv-caution",
        "Delay": "hv-danger", "Critical": "hv-danger", "Risk": "hv-danger",
        "Jinkirta": "hv-danger", "Muhimmanci": "hv-danger", "Haɗari": "hv-danger",
    }

    for h in reversed(history):
        # Determine badge colour
        badge_cls = "hv-safe"
        for keyword, cls in verdict_class_map.items():
            if keyword in h["verdict"]:
                badge_cls = cls
                break

        module_icon = {"IRR": "💧", "DRILL": "🪨", "WARN": "⚠️"}.get(h["type"], "📋")

        st.markdown(f"""
        <div class="history-card">
            <div>
                <div class="history-meta">
                    {module_icon} <strong>{h['type']}</strong> · {h['farmer']} · {h['ts']}
                </div>
            </div>
            <span class="history-verdict {badge_cls}">{h['verdict']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)

    col_exp, col_clr, _ = st.columns([2, 2, 4])
    with col_exp:
        df = pd.DataFrame([
            {
                L("col_time"):       h["ts"],
                L("col_farmer"):     h["farmer"],
                L("col_type"):       h["type"],
                L("col_verdict"):    h["verdict"],
                L("col_confidence"): h["confidence"],
            }
            for h in history
        ])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=L("history_export"),
            data=csv,
            file_name=f"smartwater_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    with col_clr:
        if st.button(L("history_clear"), type="secondary"):
            st.session_state["history"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# FARMER PROFILE CARD
# ══════════════════════════════════════════════════════════════════════════════

def render_profile_card():
    """Farmer profile card — sits between header and tabs in main content area."""
    with st.container():
        st.markdown(f'<div class="profile-card-title">👤 {L("sidebar_profile")}</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([3, 2, 2, 1], gap="medium")

        with col1:
            name_val = st.text_input(
                L("your_name"),
                value=st.session_state.get("farmer_name", ""),
                placeholder=L("name_placeholder"),
                key="profile_name_input",
                label_visibility="visible",
            )
        with col2:
            crops = CROP_OPTIONS[st.session_state.lang]
            crop_val = st.selectbox(
                L("main_crop"),
                options=crops,
                key="profile_crop_input",
            )
        with col3:
            phone_opts = [L("phone_basic"), L("phone_feature"), L("phone_smart")]
            phone_val = st.selectbox(
                L("phone_type"),
                options=phone_opts,
                key="profile_phone_input",
            )
        with col4:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            save_btn = st.button(
                "💾  Save",
                type="primary",
                use_container_width=True,
                key="profile_save_btn",
            )

        if save_btn:
            st.session_state["farmer_name"] = name_val
            st.session_state["crop"]        = crop_val
            st.session_state["phone"]       = phone_val
            st.session_state["profile_saved"] = True
            # Register on backend → get persistent farmer_id
            fid = register_farmer_api(
                name=name_val, crop=crop_val, phone=phone_val or None
            )
            if fid:
                st.session_state["farmer_id"] = fid
                # Save to URL so it survives page refresh
                try:
                    st.query_params["fid"]   = fid
                    st.query_params["fname"] = name_val
                except Exception:
                    pass

        # Restore saved values into session on load (without button press)
        if not save_btn:
            if st.session_state.get("farmer_name", "") == "" and name_val:
                pass  # Only save on button press
            # Keep existing saved values in place
            if "farmer_name" not in st.session_state:
                st.session_state["farmer_name"] = ""

        if st.session_state.get("profile_saved"):
            farmer = st.session_state.get("farmer_name", "")
            st.markdown(
                f'<span class="profile-saved">✅ Profile saved'
                f'{" — " + farmer if farmer else ""}!</span>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sw-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    inject_css()
    render_sidebar()

    # ── App Header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sw-header">
        <h1>💧 {L('app_title')}</h1>
        <p>{L('app_subtitle')}</p>
        <div class="badge-row">
            <span class="sw-badge">📍 {L('badge_location')}</span>
            <span class="sw-badge">🤖 {L('badge_engine')}</span>
            <span class="sw-badge">🏛 {L('badge_partners')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Status Banner ───────────────────────────────────────────────────
    api_ok = check_api()
    if api_ok:
        st.markdown(f'<div class="status-connected">{L("connected")}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-demo">{L("demo_mode")}</div>', unsafe_allow_html=True)

    # ── Farmer Profile Card ────────────────────────────────────────────────
    render_profile_card()

    # ── Tabs ────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        L("tab_irrigation"),
        L("tab_drilling"),
        L("tab_warning"),
        L("tab_history"),
    ])

    with t1: tab_irrigation()
    with t2: tab_drilling()
    with t3: tab_warning()
    with t4: tab_history()

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sw-footer">
        <strong>SmartWater Agriculture</strong> · Ahmadu Bello University, Zaria ·
        Institute for Agricultural Research, Zaria<br>
        Achesae Farmers NGO · Kaduna State, Nigeria<br>
        <em>Built for the M4D Open Innovation Challenge 2025/26 — Digital Extension Track</em><br>
        Advisory outputs are probabilistic estimates. Always consult a local extension agent for major decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()