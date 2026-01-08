"""
Diagnostic script to check Spotify data retrieval for all users.
Run this to identify which users have/don't have Spotify data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from supabase_client import get_supabase_client

def check_spotify_data():
    client = get_supabase_client()
    
    # Get all tenant users
    response = client.client.table("users").select(
        "id, first_name, last_name, spotify_profiles(*)"
    ).eq("role", "tenant").execute()
    
    users = response.data if response.data else []
    
    print(f"\n=== Spotify Data Check for {len(users)} Users ===\n")
    
    has_spotify = []
    missing_spotify = []
    
    for user in users:
        user_id = user.get('id')
        name = f"{user.get('first_name', 'Unknown')} {user.get('last_name', '')}"
        spotify = user.get('spotify_profiles')
        
        if spotify:
            # Check if it's a list or dict
            if isinstance(spotify, list):
                if len(spotify) > 0:
                    spotify_data = spotify[0]
                    genres = spotify_data.get('favorite_genres', [])
                    has_spotify.append((user_id, name, len(genres)))
                else:
                    missing_spotify.append((user_id, name, "Empty list"))
            elif isinstance(spotify, dict):
                genres = spotify.get('favorite_genres', [])
                has_spotify.append((user_id, name, len(genres)))
            else:
                missing_spotify.append((user_id, name, f"Unknown type: {type(spotify)}"))
        else:
            missing_spotify.append((user_id, name, "No Spotify data"))
    
    print(f"✅ Users WITH Spotify data ({len(has_spotify)}):")
    for user_id, name, genre_count in has_spotify:
        print(f"  - {name} ({user_id[:8]}...): {genre_count} genres")
    
    print(f"\n❌ Users WITHOUT Spotify data ({len(missing_spotify)}):")
    for user_id, name, reason in missing_spotify:
        print(f"  - {name} ({user_id[:8]}...): {reason}")
    
    print(f"\n=== Summary ===")
    print(f"Total: {len(users)}")
    print(f"With Spotify: {len(has_spotify)}")
    print(f"Without Spotify: {len(missing_spotify)}")
    print(f"Success Rate: {len(has_spotify)/len(users)*100:.1f}%")
    
    return has_spotify, missing_spotify

if __name__ == "__main__":
    check_spotify_data()
