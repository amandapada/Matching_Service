"""
Supabase client wrapper for roommate matching agent.
Follows the same env pattern as src/lib/supabase.ts
"""
import os
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Reuse existing env var names with fallbacks (same pattern as TS)
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://sboglbzwiaqcovmlxsga.supabase.co')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')

if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")


class SupabaseClient:
    """Database client for fetching user profiles and candidates"""
    
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch complete user profile with all related data.
        
        Joins:
        - roommate_preferences
        - roommate_profiles  
        - spotify_profiles
        
        Returns None if user not found.
        """
        response = self.client.table("users").select(
            """
            id,
            first_name,
            last_name,
            mbti_type,
            mbti_vector,
            mbti_scores,
            roommate_preferences(*),
            roommate_profiles(*),
            spotify_profiles(*)
            """
        ).eq("id", user_id).maybe_single().execute()
        
        return response.data
    
    def get_candidate_users(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch eligible roommate candidates with same structure as get_user_profile.
        
        Applies filters:
        - budget_max
        - location
        - mbti_types
        - min_date, max_date
        
        Excludes the requesting user and non-tenant roles.
        """
        query = self.client.table("users").select(
            """
            id,
            first_name,
            last_name,
            mbti_type,
            mbti_vector,
            mbti_scores,
            roommate_preferences(*),
            roommate_profiles(*),
            spotify_profiles(*)
            """
        ).neq("id", user_id).eq("role", "tenant")
        
        # Apply filters (same logic as Edge Function)
        if filters:
            if filters.get("budget_max"):
                query = query.lte("roommate_preferences.budget", filters["budget_max"])
            if filters.get("location"):
                query = query.eq("roommate_preferences.location", filters["location"])
            if filters.get("mbti_types"):
                query = query.in_("mbti_type", filters["mbti_types"])
            if filters.get("min_date"):
                query = query.gte("roommate_preferences.move_in_date", filters["min_date"])
            if filters.get("max_date"):
                query = query.lte("roommate_preferences.move_in_date", filters["max_date"])
        
        response = query.execute()
        return response.data if response.data else []
    
    def upsert_match_records(self, match_records: List[Dict[str, Any]]):
        """
        Upsert calculated matches into roommate_matches table for history.
        Optional, can be disabled if not needed.
        """
        if not match_records:
            return
        
        self.client.table("roommate_matches").upsert(
            match_records,
            on_conflict="user_id,target_user_id",
            ignore_duplicates=False
        ).execute()


# Singleton instance
_supabase_client = None

def get_supabase_client() -> SupabaseClient:
    """Get or create Supabase client singleton"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client
