from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
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
    popularity: int = 50
    release_year: int = 2015
    mood_tag: str = "neutral"
    instrumentalness: float = 0.5
    speechiness: float = 0.1
    liveness: float = 0.2

@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    preferred_decade: Optional[int] = None
    preferred_mood_tag: str = ""
    target_instrumentalness: float = 0.5
    target_speechiness: float = 0.1
    target_liveness: float = 0.2


class ScoringStrategy(ABC):
    """Strategy interface for song scoring implementations."""

    @abstractmethod
    def calculate_score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Return a score and human-readable reasons for that score."""


class ConservativeStrategy(ScoringStrategy):
    """Balances taste matching with popularity and both trend/discovery bonuses."""

    def __init__(self, popularity_weight: float = 1.0):
        self.popularity_weight = popularity_weight

    def calculate_score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        if song.genre == user.favorite_genre:
            score += 1.0
            reasons.append("genre match (+1.0)")

        if song.mood == user.favorite_mood:
            score += 1.0
            reasons.append("mood match (+1.0)")

        energy_proximity = max(0.0, 1.0 - abs(song.energy - user.target_energy))
        energy_score = energy_proximity * 3.0
        score += energy_score
        reasons.append(f"energy proximity (+{energy_score:.2f})")

        popularity_score = (song.popularity / 100.0) * self.popularity_weight
        score += popularity_score
        reasons.append(f"popularity weighted (+{popularity_score:.2f})")

        if song.popularity > 80:
            score += 0.5
            reasons.append("Mainstream bonus (+0.5)")
        elif song.popularity < 30:
            score += 0.5
            reasons.append("Hidden Gem bonus (+0.5)")

        if user.preferred_decade is not None:
            song_decade = (song.release_year // 10) * 10
            decade_proximity = max(0.0, 1.0 - (abs(song_decade - user.preferred_decade) / 40.0))
            decade_score = decade_proximity * 0.7
            score += decade_score
            reasons.append(f"decade proximity (+{decade_score:.2f})")

        if user.preferred_mood_tag and song.mood_tag == user.preferred_mood_tag:
            score += 0.8
            reasons.append("detailed mood-tag match (+0.8)")

        instrumentalness_score = max(0.0, 1.0 - abs(song.instrumentalness - user.target_instrumentalness)) * 0.5
        score += instrumentalness_score
        reasons.append(f"instrumentalness proximity (+{instrumentalness_score:.2f})")

        speechiness_score = max(0.0, 1.0 - abs(song.speechiness - user.target_speechiness)) * 0.4
        score += speechiness_score
        reasons.append(f"speechiness proximity (+{speechiness_score:.2f})")

        liveness_score = max(0.0, 1.0 - abs(song.liveness - user.target_liveness)) * 0.3
        score += liveness_score
        reasons.append(f"liveness proximity (+{liveness_score:.2f})")

        return score, reasons


class DiscoveryStrategy(ScoringStrategy):
    """Prioritizes discovery by rewarding low-popularity songs more strongly."""

    def __init__(self, popularity_weight: float = 1.0):
        self.popularity_weight = popularity_weight

    def calculate_score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        if song.genre == user.favorite_genre:
            score += 1.0
            reasons.append("genre match (+1.0)")

        if song.mood == user.favorite_mood:
            score += 1.0
            reasons.append("mood match (+1.0)")

        energy_proximity = max(0.0, 1.0 - abs(song.energy - user.target_energy))
        energy_score = energy_proximity * 3.0
        score += energy_score
        reasons.append(f"energy proximity (+{energy_score:.2f})")

        popularity_score = (song.popularity / 100.0) * self.popularity_weight
        score += popularity_score
        reasons.append(f"popularity weighted (+{popularity_score:.2f})")

        if song.popularity < 30:
            score += 1.0
            reasons.append("Hidden Gem bonus (+1.0)")
        elif song.popularity > 80:
            score += 0.1
            reasons.append("Mainstream bonus (+0.1)")

        if user.preferred_decade is not None:
            song_decade = (song.release_year // 10) * 10
            decade_proximity = max(0.0, 1.0 - (abs(song_decade - user.preferred_decade) / 40.0))
            decade_score = decade_proximity * 0.4
            score += decade_score
            reasons.append(f"decade proximity (+{decade_score:.2f})")

        if user.preferred_mood_tag and song.mood_tag == user.preferred_mood_tag:
            score += 1.1
            reasons.append("detailed mood-tag match (+1.1)")

        instrumentalness_score = max(0.0, 1.0 - abs(song.instrumentalness - user.target_instrumentalness)) * 0.7
        score += instrumentalness_score
        reasons.append(f"instrumentalness proximity (+{instrumentalness_score:.2f})")

        speechiness_score = max(0.0, 1.0 - abs(song.speechiness - user.target_speechiness)) * 0.2
        score += speechiness_score
        reasons.append(f"speechiness proximity (+{speechiness_score:.2f})")

        liveness_score = max(0.0, 1.0 - abs(song.liveness - user.target_liveness)) * 0.5
        score += liveness_score
        reasons.append(f"liveness proximity (+{liveness_score:.2f})")

        return score, reasons


class HybridStrategy(ScoringStrategy):
    """Blends conservative and discovery scoring using an alpha weight."""

    def __init__(self, popularity_weight: float = 1.0, alpha: float = 0.5):
        # alpha=1.0 => conservative, alpha=0.0 => discovery
        self.alpha = max(0.0, min(1.0, alpha))
        self.conservative = ConservativeStrategy(popularity_weight)
        self.discovery = DiscoveryStrategy(popularity_weight)

    def calculate_score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        conservative_score, conservative_reasons = self.conservative.calculate_score(user, song)
        discovery_score, discovery_reasons = self.discovery.calculate_score(user, song)

        score = (self.alpha * conservative_score) + ((1.0 - self.alpha) * discovery_score)

        # Prefer a taste-facing reason first so dashboard views remain human-readable.
        primary_reason = "balanced fit"
        for reason in conservative_reasons + discovery_reasons:
            if "component" in reason.lower() or "hybrid blend" in reason.lower():
                continue
            primary_reason = reason
            break

        reasons = [
            primary_reason,
            f"hybrid blend alpha={self.alpha:.2f}",
            f"conservative component (+{(self.alpha * conservative_score):.2f})",
            f"discovery component (+{((1.0 - self.alpha) * discovery_score):.2f})",
        ]
        return score, reasons

class Recommender:
    """OOP implementation of the recommendation logic."""
    
    def __init__(
        self,
        songs: List[Song],
        popularity_weight: float = 1.0,
        strategy: Optional[ScoringStrategy] = None,
    ):
        # FIXED: Assign attribute so it is accessible in self.recommend
        self.songs = songs
        self.strategy = strategy or ConservativeStrategy(popularity_weight)

    @staticmethod
    def score_song(user: UserProfile, song: Song, popularity_weight: float = 1.0) -> Tuple[float, List[str]]:
        """Backward-compatible static helper that uses conservative scoring."""
        return ConservativeStrategy(popularity_weight).calculate_score(user, song)

    def set_strategy(self, strategy: ScoringStrategy) -> None:
        """Switch scoring behavior at runtime."""
        self.strategy = strategy

    def recommend_with_details(self, user: UserProfile, k: int = 5) -> List[Tuple[Song, float, List[str]]]:
        """Ranks songs and includes adjusted scores/reasons with diversity penalties applied."""
        base_details: Dict[int, Tuple[Song, float, List[str]]] = {}
        for song in self.songs:
            score, reasons = self.strategy.calculate_score(user, song)
            base_details[song.id] = (song, score, reasons)

        remaining_ids = list(base_details.keys())
        selected: List[Tuple[Song, float, List[str]]] = []

        while remaining_ids and len(selected) < k:
            seen_artists = {song.artist for song, _, _ in selected}
            best_song_id: Optional[int] = None
            best_adjusted_score = float("-inf")
            best_reasons: List[str] = []

            for song_id in remaining_ids:
                song, base_score, base_reasons = base_details[song_id]
                adjusted_score = base_score
                adjusted_reasons = list(base_reasons)
                if song.artist in seen_artists:
                    adjusted_score -= 0.5
                    adjusted_reasons.append("Diversity Penalty (-0.5)")

                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_song_id = song_id
                    best_reasons = adjusted_reasons

            if best_song_id is None:
                break

            chosen_song, _, _ = base_details[best_song_id]
            selected.append((chosen_song, best_adjusted_score, best_reasons))
            remaining_ids = [song_id for song_id in remaining_ids if song_id != best_song_id]

        return selected

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns only song objects from the diversity-aware ranking."""
        ranked = self.recommend_with_details(user, k)
        return [song for song, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Provides a string explanation for CLI output."""
        _, reasons = self.strategy.calculate_score(user, song)
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
                "popularity": int(row.get("popularity", 50)),
                "release_year": int(row.get("release_year", 2015)),
                "mood_tag": row.get("mood_tag", "neutral"),
                "instrumentalness": float(row.get("instrumentalness", 0.5)),
                "speechiness": float(row.get("speechiness", 0.1)),
                "liveness": float(row.get("liveness", 0.2)),
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
        preferred_decade=user_prefs.get("preferred_decade"),
        preferred_mood_tag=user_prefs.get("preferred_mood_tag", ""),
        target_instrumentalness=user_prefs.get("target_instrumentalness", 0.5),
        target_speechiness=user_prefs.get("target_speechiness", 0.1),
        target_liveness=user_prefs.get("target_liveness", 0.2),
    )
    
    # 3. Use Recommender with a selected strategy
    popularity_weight = user_prefs.get("popularity_weight", 1.0)
    strategy_name = str(user_prefs.get("strategy", user_prefs.get("mode", "conservative"))).lower()
    alpha = float(user_prefs.get("alpha", 0.5))
    if strategy_name == "discovery":
        strategy: ScoringStrategy = DiscoveryStrategy(popularity_weight)
    elif strategy_name == "hybrid":
        strategy = HybridStrategy(popularity_weight, alpha)
    else:
        strategy = ConservativeStrategy(popularity_weight)

    engine = Recommender(song_objects, popularity_weight=popularity_weight, strategy=strategy)
    ranked_songs = engine.recommend_with_details(user, k)
    
    # 4. Pack results into the format expected by main.py
    result = []
    for song_obj, score, reasons in ranked_songs:
        explanation = ", ".join(reasons)
        # Convert dataclass back to dict for the CLI runner
        result.append((song_obj.__dict__, score, explanation))
    
    return result