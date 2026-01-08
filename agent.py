"""
LlamaIndex-powered roommate matching agent.

The agent uses deterministic scoring functions as FEATURES (not as final scores)
and makes its own compatibility judgments using LLM reasoning.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

from supabase_client import get_supabase_client
from compatibility import compute_features

logger = logging.getLogger(__name__)

# Get settings from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
AGENT_TEMPERATURE = float(os.getenv('AGENT_TEMPERATURE', '0.8'))


# Response schema for structured output
class MatchDecision(BaseModel):
    """Structured decision output from the LLM"""
    decision: str = Field(description="One of: 'strong_match', 'possible_match', 'avoid'")
    score_0_100: float = Field(description="Overall compatibility score from 0-100", ge=0, le=100)
    breakdown: Dict[str, float] = Field(description="Score breakdown by dimension")
    reason: str = Field(description="Short explanation of why this is or isn't a good match")


# System prompt for the agent
MATCHING_SYSTEM_PROMPT = """You are a roommate matching expert with deep knowledge of human compatibility patterns.

For each candidate, you receive:
1) Raw profile data including:
   - MBTI personality type, vector, and detailed scores
   - Living preferences (budget, location, move-in date, cleanliness, noise tolerance, visitor preferences, smoking/pets tolerance)
   - Lifestyle information (interests, bio, school, major)
   - Spotify music profile (favorite genres, top artists, audio preferences)

2) Precomputed numeric feature scores:
   - mbti_compatibility_score (0-25): How well MBTI types align
   - lifestyle_match_score (0-25): Alignment on cleanliness, noise, visitors
   - budget_alignment_score (0-15): How close budgets are
   - location_match_score (0-10): Same preferred location
   - date_compatibility_score (0-10): Move-in date proximity
   - tolerance_match_score (0-10): Smoking/pets agreement
   - interests_overlap_score (0-5): Shared hobbies/interests
   - music_taste_score (0-10): Spotify genre similarity

IMPORTANT: The feature scores are HINTS, not a rigid formula. Use your own judgment based on patterns you know about human compatibility.

Your task:
1. Consider both the raw profile data AND the feature scores
2. Look for holistic patterns (e.g., "both introverts who value quiet study time")
3. Identify potential conflicts (e.g., "one is a night owl, other is early riser")
4. Weigh factors contextually (e.g., music taste matters more for music lovers)
5. Decide: strong_match, possible_match, or avoid
6. Provide a final score from 0-100
7. Break down your score by the 8 dimensions above
8. Explain your reasoning concisely

Remember: You are the final decision-maker. The features inform you, but YOUR judgment determines compatibility."""


def get_user_profile_tool(user_id: str) -> Dict[str, Any]:
    """
    Fetch complete user profile including MBTI, preferences, and Spotify data.
    
    Returns dict with:
    - MBTI data (type, vector, scores)
    - Living preferences (roommate_preferences)
    - Lifestyle/interest info (roommate_profiles) 
    - Spotify data (favorite_genres, top_artists, audio_preferences)
    """
    db = get_supabase_client()
    profile = db.get_user_profile(user_id)
    
    if not profile:
        raise ValueError(f"User profile not found for {user_id}")
    
    return profile


def get_candidates_tool(user_id: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Fetch eligible roommate candidates with same data structure as get_user_profile_tool.
    
    Applies filters (budget_max, location, mbti_types, min_date, max_date).
    Returns list of candidate profiles.
    """
    db = get_supabase_client()
    candidates = db.get_candidate_users(user_id, filters)
    return candidates


def compute_features_tool(user: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute compatibility features between user and candidate.
    
    Returns dict of 8 feature scores for LLM to consider:
    - mbti_compatibility_score
    - lifestyle_match_score
    - budget_alignment_score
    - location_match_score
    - date_compatibility_score
    - tolerance_match_score
    - interests_overlap_score
    - music_taste_score
    """
    return compute_features(user, candidate)


class RoommateMatchingAgent:
    """
    LlamaIndex agent for AI-driven roommate matching.
    
    The agent uses LLM reasoning to make compatibility decisions, informed by:
    - Raw profile data (MBTI, preferences, Spotify, lifestyle)
    - Precomputed feature scores (from deterministic functions)
    
    The LLM is the final decision-maker, not a sum of scores.
    """
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for agent")
        
        self.llm = OpenAI(
            model=OPENAI_MODEL,
            temperature=AGENT_TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
        
        # Create tools for the agent
        self.tools = [
            FunctionTool.from_defaults(fn=get_user_profile_tool),
            FunctionTool.from_defaults(fn=get_candidates_tool),
            FunctionTool.from_defaults(fn=compute_features_tool),
        ]
        
        # For structured output, we'll use LLMTextCompletionProgram
        self.decision_program = LLMTextCompletionProgram.from_defaults(
            output_cls=MatchDecision,
            llm=self.llm,
            prompt_template_str=MATCHING_SYSTEM_PROMPT + "\n\nUser Profile:\n{user_profile}\n\nCandidate Profile:\n{candidate_profile}\n\nFeature Scores:\n{features}\n\nProvide your matching decision as JSON."
        )
    
    def evaluate_candidate(
        self,
        user_profile: Dict[str, Any],
        candidate_profile: Dict[str, Any],
        features: Dict[str, float]
    ) -> MatchDecision:
        """
        Use LLM to evaluate a single candidate.
        
        Args:
            user_profile: Source user's complete profile
            candidate_profile: Candidate's complete profile
            features: Precomputed compatibility feature scores
            
        Returns:
            Structured MatchDecision with score, breakdown, and reasoning
        """
        # Extract relevant data for prompt (reduce token usage)
        user_summary = {
            "mbti_type": user_profile.get("mbti_type"),
            "mbti_vector": user_profile.get("mbti_vector"),
            "preferences": user_profile.get("roommate_preferences"),
            "profile": user_profile.get("roommate_profiles"),
            "spotify": {
                "favorite_genres": user_profile.get("spotify_profiles", {}).get("favorite_genres", [])[:10] if (isinstance(user_profile.get("spotify_profiles"), dict) and user_profile.get("spotify_profiles", {}).get("spotify_user_id")) else [],
                "top_artists": user_profile.get("spotify_profiles", {}).get("top_artists", [])[:5] if (isinstance(user_profile.get("spotify_profiles"), dict) and user_profile.get("spotify_profiles", {}).get("spotify_user_id")) else []
            }
        }
        
        candidate_summary = {
            "mbti_type": candidate_profile.get("mbti_type"),
            "mbti_vector": candidate_profile.get("mbti_vector"),
            "preferences": candidate_profile.get("roommate_preferences"),
            "profile": candidate_profile.get("roommate_profiles"),
            "spotify": {
                "favorite_genres": candidate_profile.get("spotify_profiles", {}).get("favorite_genres", [])[:10] if (isinstance(candidate_profile.get("spotify_profiles"), dict) and candidate_profile.get("spotify_profiles", {}).get("spotify_user_id")) else [],
                "top_artists": candidate_profile.get("spotify_profiles", {}).get("top_artists", [])[:5] if (isinstance(candidate_profile.get("spotify_profiles"), dict) and candidate_profile.get("spotify_profiles", {}).get("spotify_user_id")) else []
            }
        }
        
        try:
            # Call LLM with structured output
            decision = self.decision_program(
                user_profile=json.dumps(user_summary, indent=2),
                candidate_profile=json.dumps(candidate_summary, indent=2),
                features=json.dumps(features, indent=2)
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating candidate: {e}", exc_info=True)
            # Fallback: use feature scores
            total_features = sum(features.values())
            return MatchDecision(
                decision="possible_match",
                score_0_100=min(100, total_features),
                breakdown={
                    "mbti_compatibility": features["mbti_compatibility_score"],
                    "lifestyle_match": features["lifestyle_match_score"],
                    "budget_alignment": features["budget_alignment_score"],
                    "location_match": features["location_match_score"],
                    "date_compatibility": features["date_compatibility_score"],
                    "tolerance_match": features["tolerance_match_score"],
                    "interests_overlap": features["interests_overlap_score"],
                    "music_taste": features["music_taste_score"]
                },
                reason=f"Fallback decision due to error: {str(e)}"
            )
    
    def find_matches(
        self,
        user_id: str,
        limit: int = 20,
        min_score: int = 60,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find and rank compatible roommates using LLM-driven decisions.
        
        Args:
            user_id: UUID of requesting user
            limit: Maximum number of matches to return
            min_score: Minimum compatibility score (0-100)
            filters: Optional filters (budget, location, MBTI, dates)
            
        Returns:
            List of match dicts ready for API response
        """
        logger.info(f"Agent finding matches for user {user_id} (limit={limit}, min_score={min_score})")
        
        # Fetch user profile
        user_profile = get_user_profile_tool(user_id)
        
        # Fetch candidates
        candidates = get_candidates_tool(user_id, filters)
        logger.info(f"Evaluating {len(candidates)} candidates")
        
        # Evaluate each candidate with LLM
        evaluated_matches = []
        for candidate in candidates:
            try:
                # Compute features
                features = compute_features_tool(user_profile, candidate)
                
                # Get LLM decision
                decision = self.evaluate_candidate(user_profile, candidate, features)
                
                # Only include if meets minimum score
                if decision.score_0_100 >= min_score:
                    evaluated_matches.append({
                        "candidate": candidate,
                        "decision": decision
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluating candidate {candidate.get('id')}: {e}")
                continue
        
        # Sort by score descending
        evaluated_matches.sort(key=lambda x: x["decision"].score_0_100, reverse=True)
        
        # Take top N
        top_matches = evaluated_matches[:limit]
        
        # Format for API response
        matches = []
        for match_data in top_matches:
            candidate = match_data["candidate"]
            decision = match_data["decision"]
            
            matches.append({
                "target_user": {
                    "id": candidate["id"],
                    "first_name": candidate["first_name"],
                    "last_name": candidate["last_name"],
                    "mbti_type": candidate["mbti_type"],
                    "roommate_preferences": candidate.get("roommate_preferences"),
                    "roommate_profiles": candidate.get("roommate_profiles")
                },
                "compatibility_score": decision.score_0_100,
                "compatibility_breakdown": decision.breakdown,
                "decision": decision.decision,
                "reason": decision.reason
            })
        
        logger.info(f"Returning {len(matches)} matches")
        return matches


# Singleton instance
_agent = None

def get_agent() -> RoommateMatchingAgent:
    """Get or create agent singleton"""
    global _agent
    if _agent is None:
        _agent = RoommateMatchingAgent()
    return _agent
