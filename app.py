"""
FastAPI application for roommate matching.

HYBRID APPROACH FOR COST SAFETY:
- No query → Direct Python scoring (NO LLM CALL, FREE, FAST)
- Query provided → LlamaIndex agent (LLM CALL, COSTS $0.01-0.05)

This keeps 90% of requests free while enabling AI features for power users.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Temporarily disabled due to compatibility issue with FastAPI 0.115.0
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded

from dotenv import load_dotenv
from supabase_client import get_supabase_client
from compatibility import calculate_compatibility_score

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cost control settings
USE_AGENT_BY_DEFAULT = os.getenv('USE_AGENT_BY_DEFAULT', 'true').lower() == 'true'  # Agent is default
ENABLE_RESULT_CACHING = os.getenv('ENABLE_RESULT_CACHING', 'true').lower() == 'true'
CACHE_TTL_MINUTES = int(os.getenv('CACHE_TTL_MINUTES', '5'))
FALLBACK_TO_DIRECT_ON_ERROR = os.getenv('FALLBACK_TO_DIRECT_ON_ERROR', 'true').lower() == 'true'

# CORS origins (same pattern as TS)
CORS_ORIGINS_STR = os.getenv('CORS_ORIGINS', '["http://localhost:5173"]')
import json
CORS_ORIGINS = json.loads(CORS_ORIGINS_STR)

# Initialize FastAPI app
app = FastAPI(
    title="BAABA Roommate Matching Agent",
    version="2.0.0-hybrid",
    description="Cost-safe hybrid roommate matching: direct scoring + optional AI agent"
)

# Rate limiting - temporarily disabled
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class MatchFilters(BaseModel):
    budget_max: Optional[int] = None
    location: Optional[str] = None
    mbti_types: Optional[List[str]] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    
    model_config = {"arbitrary_types_allowed": True}


class MatchRequest(BaseModel):
    user_id: str = Field(..., description="UUID of requesting user")
    limit: int = Field(default=20, ge=1, le=100)
    min_score: int = Field(default=60, ge=0, le=110)
    filters: Optional[MatchFilters] = None
    query: Optional[str] = Field(None, description="Natural language query (triggers agent)")
    
    model_config = {"arbitrary_types_allowed": True}


class RoommatePreferences(BaseModel):
    budget: Optional[int] = None
    location: Optional[str] = None
    move_in_date: Optional[str] = None
    cleanliness: Optional[int] = None
    noise: Optional[int] = None
    visitors: Optional[int] = None
    smoking_tolerance: Optional[bool] = None
    pets_tolerance: Optional[bool] = None
    
    model_config = {"arbitrary_types_allowed": True}


class RoommateProfile(BaseModel):
    bio: Optional[str] = None
    school: Optional[str] = None
    year_of_study: Optional[str] = None
    major: Optional[str] = None
    interests: Optional[List[str]] = None
    instagram_handle: Optional[str] = None
    
    model_config = {"arbitrary_types_allowed": True}


class TargetUser(BaseModel):
    id: str
    first_name: str
    last_name: str
    mbti_type: str
    roommate_preferences: Optional[RoommatePreferences] = None
    roommate_profiles: Optional[RoommateProfile] = None
    
    model_config = {"arbitrary_types_allowed": True}


class CompatibilityBreakdown(BaseModel):
    mbti_compatibility: float
    lifestyle_match: float
    budget_alignment: float
    location_match: float
    date_compatibility: float
    tolerance_match: float
    interests_overlap: float
    music_taste: float
    
    model_config = {"arbitrary_types_allowed": True}


class Match(BaseModel):
    match_id: str
    target_user: TargetUser
    compatibility_score: float
    compatibility_breakdown: CompatibilityBreakdown
    created_at: str
    
    model_config = {"arbitrary_types_allowed": True}


class MatchResponse(BaseModel):
    matches: List[Match]
    total_eligible_users: int
    algorithm_version: str
    
    model_config = {"arbitrary_types_allowed": True}


# Simple in-memory cache (can upgrade to Redis later)
_match_cache: Dict[str, Any] = {}


def get_cache_key(user_id: str, request: MatchRequest) -> str:
    """Generate cache key based on request parameters"""
    # Cache for 5-minute windows
    time_window = datetime.now().strftime('%Y%m%d%H%M')[:12]  # Round to 5-min blocks
    filters_str = str(request.filters.dict() if request.filters else {})
    return f"{user_id}:{request.limit}:{request.min_score}:{filters_str}:{time_window}"


def direct_matching(request: MatchRequest) -> MatchResponse:
    """
    Direct Python scoring without LLM.
    This is the DEFAULT path - fast, free, and predictable.
    Uses pure Python math functions from compatibility.py.
    """
    db = get_supabase_client()
    
    # Check cache first
    if ENABLE_RESULT_CACHING:
        cache_key = get_cache_key(request.user_id, request)
        if cache_key in _match_cache:
            logger.info(f"Cache hit for user {request.user_id}")
            return _match_cache[cache_key]
    
    # Fetch user profile
    user_profile = db.get_user_profile(request.user_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    # Check onboarding completion
    if not user_profile.get('mbti_type') or not user_profile.get('roommate_preferences'):
        raise HTTPException(
            status_code=400,
            detail="User has not completed onboarding"
        )
    
    # Fetch candidates
    filters_dict = request.filters.dict() if request.filters else None
    candidates = db.get_candidate_users(request.user_id, filters_dict)
    logger.info(f"Direct matching: {len(candidates)} candidates for user {request.user_id}")
    
    # Calculate scores using pure Python functions
    matches = []
    for candidate in candidates:
        # Skip if missing critical data
        if not candidate.get('mbti_type') or not candidate.get('roommate_preferences'):
            continue
        
        # Calculate compatibility (pure Python, no LLM)
        score_data = calculate_compatibility_score(user_profile, candidate)
        
        if score_data['total_score'] >= request.min_score:
            # Extract preferences (handle array vs object)
            prefs = candidate.get('roommate_preferences')
            if isinstance(prefs, list) and len(prefs) > 0:
                prefs = prefs[0]
            
            # Extract profile (handle array vs object)
            profile = candidate.get('roommate_profiles')
            if isinstance(profile, list) and len(profile) > 0:
                profile = profile[0]
            
            match = Match(
                match_id=f"{request.user_id}_{candidate['id']}",
                target_user=TargetUser(
                    id=candidate['id'],
                    first_name=candidate['first_name'],
                    last_name=candidate['last_name'],
                    mbti_type=candidate['mbti_type'],
                    roommate_preferences=RoommatePreferences(**prefs) if prefs else None,
                    roommate_profiles=RoommateProfile(**profile) if profile else None
                ),
                compatibility_score=score_data['total_score'],
                compatibility_breakdown=CompatibilityBreakdown(**score_data['breakdown']),
                created_at=datetime.utcnow().isoformat()
            )
            matches.append(match)
    
    # Sort by compatibility score
    matches.sort(key=lambda x: x.compatibility_score, reverse=True)
    matches = matches[:request.limit]
    
    response = MatchResponse(
        matches=matches,
        total_eligible_users=len(candidates),
        algorithm_version="v2.0.0-direct"
    )
    
    # Cache result
    if ENABLE_RESULT_CACHING:
        cache_key = get_cache_key(request.user_id, request)
        _match_cache[cache_key] = response
    
    return response


def agent_matching(request: MatchRequest) -> MatchResponse:
    """
    LlamaIndex agent-based matching with LLM-driven decisions.
    This is the OPT-IN path - costs money but provides AI-powered matching.
    Only called when request.query is provided OR USE_AGENT_BY_DEFAULT=true.
    
    The agent evaluates each candidate using:
    - Raw profile data (MBTI, preferences, Spotify, lifestyle)
    - Precomputed feature scores (from deterministic functions)
    
    The LLM makes the final compatibility decision, not a sum of scores.
    """
    from agent import get_agent
    
    logger.info(f"🤖 Agent matching for user {request.user_id}")
    
    try:
        # Get agent instance
        agent = get_agent()
        
        # Call agent to find matches
        # Note: query is not yet used, but could be used for semantic filtering
        filters_dict = request.filters.dict() if request.filters else None
        matches_data = agent.find_matches(
            user_id=request.user_id,
            limit=request.limit,
            min_score=request.min_score,
            filters=filters_dict
        )
        
        # Format to API response
        matches = []
        for match in matches_data:
            # Extract preferences/profiles (handle array vs object)
            prefs = match["target_user"].get("roommate_preferences")
            if isinstance(prefs, list) and len(prefs) > 0:
                prefs = prefs[0]
            
            profile = match["target_user"].get("roommate_profiles")
            if isinstance(profile, list) and len(profile) > 0:
                profile = profile[0]
            
            matches.append(Match(
                match_id=f"{request.user_id}_{match['target_user']['id']}",
                target_user=TargetUser(
                    id=match["target_user"]["id"],
                    first_name=match["target_user"]["first_name"],
                    last_name=match["target_user"]["last_name"],
                    mbti_type=match["target_user"]["mbti_type"],
                    roommate_preferences=RoommatePreferences(**prefs) if prefs else None,
                    roommate_profiles=RoommateProfile(**profile) if profile else None
                ),
                compatibility_score=match["compatibility_score"],
                compatibility_breakdown=CompatibilityBreakdown(**match["compatibility_breakdown"]),
                created_at=datetime.utcnow().isoformat()
            ))
        
        db = get_supabase_client()
        all_candidates = db.get_candidate_users(request.user_id, filters_dict)
        
        response = MatchResponse(
            matches=matches,
            total_eligible_users=len(all_candidates),
            algorithm_version="v2.0.0-agent-llm"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Agent matching error: {e}", exc_info=True)
        if FALLBACK_TO_DIRECT_ON_ERROR:
            logger.warning("Falling back to direct matching due to agent error")
            return direct_matching(request)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Agent matching failed: {str(e)}"
            )


@app.post("/roommate-matching", response_model=MatchResponse)
# @limiter.limit(os.getenv('RATE_LIMIT_PER_MINUTE', '10') + "/minute")  # Temporarily disabled
async def find_matches(request: MatchRequest, http_request: Request):
    """
    Primary endpoint for roommate matching.
    
    HYBRID BEHAVIOR:
    - If no 'query' provided → Direct Python scoring (FREE, FAST)
    - If 'query' provided → LlamaIndex agent (COSTS MONEY, SLOW)
    
    Returns same response structure as old Edge Function for backward compatibility.
    """
    try:
        # Decision tree: agent or direct?
        use_agent = (
            (request.query and request.query.strip()) or
            USE_AGENT_BY_DEFAULT
        )
        
        if use_agent:
            logger.info(f"Agent matching for user {request.user_id}: {request.query}")
            response = agent_matching(request)
        else:
            logger.info(f"Direct matching for user {request.user_id}")
            response = direct_matching(request)
        
        # Optional: Upsert matches to database for history
        # Disabled by default to reduce DB writes
        # db = get_supabase_client()
        # match_records = [...]
        # db.upsert_match_records(match_records)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in find_matches: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "roommate-matching-agent",
        "version": "2.0.0-hybrid",
        "agent_enabled": USE_AGENT_BY_DEFAULT,
        "cache_enabled": ENABLE_RESULT_CACHING
    }


@app.get("/admin/stats")
async def get_stats():
    """Admin endpoint for monitoring cache and usage"""
    return {
        "cache_size": len(_match_cache),
        "cache_enabled": ENABLE_RESULT_CACHING,
        "rate_limit": os.getenv('RATE_LIMIT_PER_MINUTE', '10'),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', '8001'))
    uvicorn.run(app, host="0.0.0.0", port=port)
