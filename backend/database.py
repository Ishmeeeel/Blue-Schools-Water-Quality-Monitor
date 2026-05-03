"""
SmartWater Agriculture — Supabase Database Layer
=================================================
All database operations live here.
app.py imports from this module — clean separation.
"""

import os
from typing import Optional, Dict, List
from datetime import datetime

# ── Supabase client ──────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        DB_AVAILABLE = True
        print("✅ Supabase connected.")
    else:
        supabase = None
        DB_AVAILABLE = False
        print("⚠️  SUPABASE_URL / SUPABASE_KEY not set. DB logging disabled.")
except ImportError:
    supabase = None
    DB_AVAILABLE = False
    print("⚠️  supabase-py not installed. DB logging disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# FARMERS
# ══════════════════════════════════════════════════════════════════════════════

def register_farmer(data: Dict) -> Optional[Dict]:
    """Insert or upsert a farmer record. Returns the farmer row."""
    if not DB_AVAILABLE:
        return None
    try:
        result = supabase.table("farmers").insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"DB error register_farmer: {e}")
        return None


def get_farmer_by_phone(phone: str) -> Optional[Dict]:
    """Look up a farmer by phone number."""
    if not DB_AVAILABLE:
        return None
    try:
        result = (
            supabase.table("farmers")
            .select("*")
            .eq("phone", phone)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"DB error get_farmer_by_phone: {e}")
        return None


def get_farmer_by_id(farmer_id: str) -> Optional[Dict]:
    """Look up a farmer by UUID."""
    if not DB_AVAILABLE:
        return None
    try:
        result = (
            supabase.table("farmers")
            .select("*")
            .eq("farmer_id", farmer_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"DB error get_farmer_by_id: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FIELD SURVEYS
# ══════════════════════════════════════════════════════════════════════════════

def submit_survey(data: Dict) -> Optional[Dict]:
    """Save a field survey observation. Returns the saved row."""
    if not DB_AVAILABLE:
        return None
    try:
        result = supabase.table("field_surveys").insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"DB error submit_survey: {e}")
        return None


def get_farmer_surveys(farmer_id: str, limit: int = 20) -> List[Dict]:
    """Retrieve recent surveys for a farmer."""
    if not DB_AVAILABLE:
        return []
    try:
        result = (
            supabase.table("field_surveys")
            .select("*")
            .eq("farmer_id", farmer_id)
            .order("survey_date", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"DB error get_farmer_surveys: {e}")
        return []


def get_all_surveys_for_training(limit: int = 10000) -> List[Dict]:
    """
    Pull all field surveys for CPT parameter learning.
    Called by the CPT updater script.
    """
    if not DB_AVAILABLE:
        return []
    try:
        result = (
            supabase.table("field_surveys")
            .select("*")
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"DB error get_all_surveys_for_training: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# ASSESSMENTS
# ══════════════════════════════════════════════════════════════════════════════

def log_assessment(data: Dict) -> Optional[Dict]:
    """Log every prediction the AI makes."""
    if not DB_AVAILABLE:
        return None
    try:
        result = supabase.table("assessments").insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"DB error log_assessment: {e}")
        return None


def get_farmer_history(farmer_id: str, limit: int = 50) -> List[Dict]:
    """Retrieve full assessment history for a farmer."""
    if not DB_AVAILABLE:
        return []
    try:
        result = (
            supabase.table("assessments")
            .select("*")
            .eq("farmer_id", farmer_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"DB error get_farmer_history: {e}")
        return []


def update_assessment_feedback(assessment_id: str, helpful: bool) -> bool:
    """Update was_helpful flag after farmer gives feedback."""
    if not DB_AVAILABLE:
        return False
    try:
        supabase.table("assessments").update(
            {"was_helpful": helpful}
        ).eq("assessment_id", assessment_id).execute()
        return True
    except Exception as e:
        print(f"DB error update_assessment_feedback: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

def log_feedback(data: Dict) -> Optional[Dict]:
    """Save a farmer thumbs up/down feedback."""
    if not DB_AVAILABLE:
        return None
    try:
        result = supabase.table("feedback").insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"DB error log_feedback: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STATS (for dashboard / research)
# ══════════════════════════════════════════════════════════════════════════════

def get_system_stats() -> Dict:
    """High-level system statistics."""
    if not DB_AVAILABLE:
        return {"db": "unavailable"}
    try:
        farmers_count    = supabase.table("farmers").select("farmer_id", count="exact").execute()
        surveys_count    = supabase.table("field_surveys").select("survey_id", count="exact").execute()
        assessments_count = supabase.table("assessments").select("assessment_id", count="exact").execute()
        return {
            "total_farmers":     farmers_count.count,
            "total_surveys":     surveys_count.count,
            "total_assessments": assessments_count.count,
            "db":                "connected",
        }
    except Exception as e:
        print(f"DB error get_system_stats: {e}")
        return {"db": "error", "detail": str(e)}