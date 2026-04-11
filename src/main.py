"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    # Define at least six distinct profiles to test the algorithm's boundaries
    # Phase 4 Evaluation Profiles
    test_profiles = [
        {"name": "High-Energy Pop", "genre": "pop", "mood": "happy", "energy": 0.9},
        {"name": "Chill Lofi", "genre": "lofi", "mood": "chill", "energy": 0.2},
        {"name": "Deep Intense Rock", "genre": "rock", "mood": "intense", "energy": 0.85},
        {"name": "Conflicting Happy but Low Energy", "genre": "pop", "mood": "happy", "energy": 0.2},
        {"name": "Sad but High Energy", "genre": "lofi", "mood": "sad", "energy": 0.9},
        {"name": "Noisy Mismatch", "genre": "jazz", "mood": "intense", "energy": 0.1}
    ]

    for profile in test_profiles:
        print(f"{'='*30}")
        print(f"RUNNING TEST FOR: {profile['name']}")
        print(f"Preferences: {profile['genre']}, {profile['mood']}, Energy: {profile['energy']}")
        print(f"{'='*30}")

        # Note: Your recommend_songs takes the whole profile dict
        recommendations = recommend_songs(profile, songs, k=5)

        for rec in recommendations:
            song, score, reasons = rec
            # 1. Header with Emoji and Title
            print(f"⭐ {song['title'].upper()} ({song['artist']})")
            print(f"   Score: {score:.2f}")

            # 2. Handle the reasons (string or list)
            if isinstance(reasons, str):
                reasons_list = [r.strip() for r in reasons.split(",") if r.strip()]
            else:
                reasons_list = reasons

            # 3. Print reasons with a stylistic arrow
            for reason in reasons_list:
                print(f"   ↳ Because: {reason}")

            # 4. Divider and spacing for readability
            print(f"   {'-' * 20}\n")

if __name__ == "__main__":
    main()