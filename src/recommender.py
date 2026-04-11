from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """OOP implementation of the recommendation logic."""
    
    def __init__(self, songs: List[Song]):
        # FIXED: Assign attribute so it is accessible in self.recommend
        self.songs = songs

    @staticmethod
    def score_song(user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """
        The Single Source of Truth for scoring math.
        Weights: Genre +1, Mood +1, Energy Proximity * 3.
        """
        score = 0.0
        reasons: List[str] = []

        if song.genre == user.favorite_genre:
            score += 1.0
            reasons.append("genre match (+1.0)")

        if song.mood == user.favorite_mood:
            score += 1.0
            reasons.append("mood match (+1.0)")

        # Logic for Energy Experiment
        energy_proximity = max(0.0, 1.0 - abs(song.energy - user.target_energy))
        energy_score = energy_proximity * 3.0
        score += energy_score
        reasons.append(f"energy proximity (+{energy_score:.2f})")

        return score, reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Ranks songs based on the centralized score_song logic."""
        scored: List[Tuple[Song, float]] = []

        for song in self.songs:
            score, _ = self.score_song(user, song)
            scored.append((song, score))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return [song for song, _ in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Provides a string explanation for CLI output."""
        _, reasons = self.score_song(user, song)
        return ", ".join(reasons)

def load_songs(file_path: str) -> List[Dict]:
    """Loads songs from a CSV file into a list of dictionaries."""
    songs: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": int(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Bridges the dictionary-based main.py to the OOP Recommender.
    No math is performed here; it is delegated to the Recommender class.
    """
    # 1. Convert dicts to Song objects
    song_objects = [Song(**s) for s in songs]
    
    # 2. Create UserProfile object
    user = UserProfile(
        favorite_genre=user_prefs["genre"],
        favorite_mood=user_prefs["mood"],
        target_energy=user_prefs["energy"],
        likes_acoustic=user_prefs.get("likes_acoustic", False),
    )
    
    # 3. Use Recommender to find top songs
    engine = Recommender(song_objects)
    top_songs = engine.recommend(user, k)
    
    # 4. Pack results into the format expected by main.py
    result = []
    for song_obj in top_songs:
        score, _ = engine.score_song(user, song_obj)
        explanation = engine.explain_recommendation(user, song_obj)
        # Convert dataclass back to dict for the CLI runner
        result.append((song_obj.__dict__, score, explanation))
    
    return result