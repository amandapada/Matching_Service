"""
Pure, deterministic compatibility scoring functions.
Ported from supabase/functions/find-roommate-matches/index.ts

These functions are intentionally kept as simple Python math functions
rather than being recalculated by the LLM. The LlamaIndex agent uses
these as tools for orchestration, not for re-implementing the logic.
"""
import math
from typing import Dict, Any, List, Set
from datetime import datetime


def calculate_mbti_distance(vector1: Dict[str, Any], vector2: Dict[str, Any]) -> float:
    """
    Calculate Euclidean distance between two MBTI vectors.
    Port of TypeScript function from lines 48-66.
    
    Returns: Distance value (lower = more similar)
    """
    if not vector1 or not vector2:
        return 1.0
    
    traits = ['E_I', 'S_N', 'T_F', 'J_P']
    sum_squared_diffs = 0
    
    for trait in traits:
        if vector1.get(trait) and vector2.get(trait):
            diff1 = (vector1[trait].get('E', 0) - vector2[trait].get('E', 0))
            diff2 = (vector1[trait].get('S', 0) - vector2[trait].get('S', 0))
            diff3 = (vector1[trait].get('T', 0) - vector2[trait].get('T', 0))
            diff4 = (vector1[trait].get('J', 0) - vector2[trait].get('J', 0))
            
            sum_squared_diffs += (diff1 ** 2) + (diff2 ** 2) + (diff3 ** 2) + (diff4 ** 2)
    
    return math.sqrt(sum_squared_diffs)


def calculate_mbti_compatibility(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate MBTI compatibility score (0-25 points).
    Port of TypeScript function from lines 69-102.
    """
    distance = calculate_mbti_distance(user1.get('mbti_vector'), user2.get('mbti_vector'))
    score = max(0, 25 - (distance * 10))
    
    type1 = user1.get('mbti_type')
    type2 = user2.get('mbti_type')
    
    if not type1 or not type2:
        return score
    
    # Same types bonus
    if type1 == type2:
        score = min(25, score + 5)
    
    # Compatible opposites (E-I balance)
    e1 = type1[0] == 'E'
    e2 = type2[0] == 'E'
    if e1 != e2:
        score = min(25, score + 3)
    
    # N-N pairing bonus
    if type1[1] == 'N' and type2[1] == 'N':
        score = min(25, score + 2)
    
    # S-S pairing bonus
    if type1[1] == 'S' and type2[1] == 'S':
        score = min(25, score + 2)
    
    return min(25, max(0, score))


def calculate_lifestyle_match(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate lifestyle alignment score (0-25 points).
    Port of TypeScript function from lines 105-116.
    
    Based on cleanliness, noise, and visitors preferences (1-5 scales).
    """
    prefs1 = user1.get('roommate_preferences')
    prefs2 = user2.get('roommate_preferences')
    
    # Handle array vs object format
    if isinstance(prefs1, list) and len(prefs1) > 0:
        prefs1 = prefs1[0]
    if isinstance(prefs2, list) and len(prefs2) > 0:
        prefs2 = prefs2[0]
    
    if not prefs1 or not prefs2:
        return 0.0
    
    cleanliness_score = 25 - (5 * abs((prefs1.get('cleanliness', 3) or 3) - (prefs2.get('cleanliness', 3) or 3)))
    noise_score = 25 - (5 * abs((prefs1.get('noise', 3) or 3) - (prefs2.get('noise', 3) or 3)))
    visitors_score = 25 - (5 * abs((prefs1.get('visitors', 3) or 3) - (prefs2.get('visitors', 3) or 3)))
    
    return min(25, (cleanliness_score + noise_score + visitors_score) / 3)


def calculate_budget_alignment(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate budget compatibility score (0-15 points).
    Port of TypeScript function from lines 119-134.
    """
    prefs1 = user1.get('roommate_preferences')
    prefs2 = user2.get('roommate_preferences')
    
    if isinstance(prefs1, list) and len(prefs1) > 0:
        prefs1 = prefs1[0]
    if isinstance(prefs2, list) and len(prefs2) > 0:
        prefs2 = prefs2[0]
    
    budget1 = prefs1.get('budget', 0) if prefs1 else 0
    budget2 = prefs2.get('budget', 0) if prefs2 else 0
    
    if budget1 == 0 or budget2 == 0:
        return 0.0
    
    diff = abs(budget1 - budget2)
    avg_budget = (budget1 + budget2) / 2
    percentage_diff = (diff / avg_budget) * 100
    
    if percentage_diff <= 10:
        return 15.0
    elif percentage_diff <= 20:
        return 12.0
    elif percentage_diff <= 30:
        return 8.0
    elif percentage_diff <= 50:
        return 4.0
    else:
        return 0.0


def calculate_location_match(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate location match score (0-10 points).
    Port of TypeScript function from lines 137-147.
    """
    prefs1 = user1.get('roommate_preferences')
    prefs2 = user2.get('roommate_preferences')
    
    if isinstance(prefs1, list) and len(prefs1) > 0:
        prefs1 = prefs1[0]
    if isinstance(prefs2, list) and len(prefs2) > 0:
        prefs2 = prefs2[0]
    
    location1 = prefs1.get('location') if prefs1 else None
    location2 = prefs2.get('location') if prefs2 else None
    
    if not location1 or not location2:
        return 0.0
    
    if location1.lower() == location2.lower():
        return 10.0
    
    # TODO: Add adjacent areas mapping in future
    return 0.0


def calculate_date_compatibility(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate move-in date compatibility score (0-10 points).
    Port of TypeScript function from lines 150-165.
    """
    prefs1 = user1.get('roommate_preferences')
    prefs2 = user2.get('roommate_preferences')
    
    if isinstance(prefs1, list) and len(prefs1) > 0:
        prefs1 = prefs1[0]
    if isinstance(prefs2, list) and len(prefs2) > 0:
        prefs2 = prefs2[0]
    
    date1 = prefs1.get('move_in_date') if prefs1 else None
    date2 = prefs2.get('move_in_date') if prefs2 else None
    
    if not date1 or not date2:
        return 0.0
    
    try:
        d1 = datetime.fromisoformat(str(date1).replace('Z', '+00:00'))
        d2 = datetime.fromisoformat(str(date2).replace('Z', '+00:00'))
        diff_days = abs((d1 - d2).days)
        
        if diff_days <= 14:
            return 10.0
        elif diff_days <= 30:
            return 8.0
        elif diff_days <= 60:
            return 5.0
        elif diff_days <= 90:
            return 3.0
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def calculate_tolerance_match(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate tolerance alignment score (0-10 points).
    Port of TypeScript function from lines 168-185.
    
    Based on smoking and pets tolerance.
    """
    prefs1 = user1.get('roommate_preferences')
    prefs2 = user2.get('roommate_preferences')
    
    if isinstance(prefs1, list) and len(prefs1) > 0:
        prefs1 = prefs1[0]
    if isinstance(prefs2, list) and len(prefs2) > 0:
        prefs2 = prefs2[0]
    
    if not prefs1 or not prefs2:
        return 0.0
    
    score = 0.0
    
    # Smoking tolerance
    if prefs1.get('smoking_tolerance') == prefs2.get('smoking_tolerance'):
        score += 5
    
    # Pets tolerance
    if prefs1.get('pets_tolerance') == prefs2.get('pets_tolerance'):
        score += 5
    
    return score


def calculate_interests_overlap(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate interests overlap score (0-5 points).
    Port of TypeScript function from lines 188-200.
    
    Uses Jaccard similarity on interests arrays.
    """
    profile1 = user1.get('roommate_profiles')
    profile2 = user2.get('roommate_profiles')
    
    if isinstance(profile1, list) and len(profile1) > 0:
        profile1 = profile1[0]
    if isinstance(profile2, list) and len(profile2) > 0:
        profile2 = profile2[0]
    
    interests1 = profile1.get('interests', []) if profile1 else []
    interests2 = profile2.get('interests', []) if profile2 else []
    
    if not interests1 or not interests2:
        return 0.0
    
    shared = len([i for i in interests1 if i in interests2])
    max_interests = max(len(interests1), len(interests2))
    
    return (shared / max_interests) * 5 if max_interests > 0 else 0.0


def calculate_music_taste(user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
    """
    Calculate music taste compatibility score (0-10 points).
    Port of TypeScript function from lines 203-238.
    
    Uses Jaccard similarity on Spotify favorite_genres.
    This is the PRIMARY source of music data - manual selections are ignored.
    
    Returns 0.0 if:
    - Either user has no spotify_profiles
    - spotify_user_id is NULL (incomplete OAuth - not whitelisted)
    - favorite_genres is missing or empty
    """
    spotify1 = user1.get('spotify_profiles')
    spotify2 = user2.get('spotify_profiles')
    
    # Handle array vs object format
    if isinstance(spotify1, list) and len(spotify1) > 0:
        spotify1 = spotify1[0]
    if isinstance(spotify2, list) and len(spotify2) > 0:
        spotify2 = spotify2[0]
    
    # Return 0 if no Spotify data
    if not spotify1 or not spotify2:
        return 0.0
    
    # Check for incomplete OAuth (NULL spotify_user_id)
    # This happens when users aren't whitelisted in Spotify Dashboard
    if not spotify1.get('spotify_user_id') or not spotify2.get('spotify_user_id'):
        return 0.0
    
    genres1 = spotify1.get('favorite_genres', [])
    genres2 = spotify2.get('favorite_genres', [])
    
    if not genres1 or not genres2:
        return 0.0
    
    # Jaccard similarity: intersection / union
    set1 = set(genres1)
    set2 = set(genres2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    similarity = intersection / union
    return min(10.0, similarity * 10.0)


def calculate_compatibility_score(user1: Dict[str, Any], user2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate total compatibility score and breakdown.
    Port of TypeScript function from lines 253-271.
    
    NOTE: This is now primarily used for backward compatibility with direct matching.
    For agent-driven matching, use compute_features() instead.
    
    Returns:
        {
            'total_score': float (0-110),
            'breakdown': {
                'mbti_compatibility': float (0-25),
                'lifestyle_match': float (0-25),
                'budget_alignment': float (0-15),
                'location_match': float (0-10),
                'date_compatibility': float (0-10),
                'tolerance_match': float (0-10),
                'interests_overlap': float (0-5),
                'music_taste': float (0-10)
            }
        }
    """
    breakdown = {
        'mbti_compatibility': calculate_mbti_compatibility(user1, user2),
        'lifestyle_match': calculate_lifestyle_match(user1, user2),
        'budget_alignment': calculate_budget_alignment(user1, user2),
        'location_match': calculate_location_match(user1, user2),
        'date_compatibility': calculate_date_compatibility(user1, user2),
        'tolerance_match': calculate_tolerance_match(user1, user2),
        'interests_overlap': calculate_interests_overlap(user1, user2),
        'music_taste': calculate_music_taste(user1, user2)
    }
    
    total_score = sum(breakdown.values())
    
    return {
        'total_score': round(total_score, 2),
        'breakdown': breakdown
    }


def compute_features(user1: Dict[str, Any], user2: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute compatibility features for LLM-driven matching.
    
    This function treats all scoring functions as FEATURES, not as a final score.
    The LlamaIndex agent uses these features + raw profile data to make its own
    compatibility judgment.
    
    Args:
        user1: Source user profile dict
        user2: Candidate user profile dict
        
    Returns:
        Dictionary of feature scores for the LLM to consider:
        {
            "mbti_compatibility_score": float (0-25),
            "lifestyle_match_score": float (0-25),
            "budget_alignment_score": float (0-15),
            "location_match_score": float (0-10),
            "date_compatibility_score": float (0-10),
            "tolerance_match_score": float (0-10),
            "interests_overlap_score": float (0-5),
            "music_taste_score": float (0-10)
        }
    """
    return {
        "mbti_compatibility_score": calculate_mbti_compatibility(user1, user2),
        "lifestyle_match_score": calculate_lifestyle_match(user1, user2),
        "budget_alignment_score": calculate_budget_alignment(user1, user2),
        "location_match_score": calculate_location_match(user1, user2),
        "date_compatibility_score": calculate_date_compatibility(user1, user2),
        "tolerance_match_score": calculate_tolerance_match(user1, user2),
        "interests_overlap_score": calculate_interests_overlap(user1, user2),
        "music_taste_score": calculate_music_taste(user1, user2),
    }

