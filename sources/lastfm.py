"""
Last.fm API client for Music Recommender: Music Theory.

Retrieves songs matching the user's preferred tags, maps them into SongFeature
objects with estimated audio dimensions, and filters out any song whose tags
contain prohibited content before returning results to the graph.
"""

import logging
import os
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from models import SongFeature

logger = logging.getLogger(__name__)

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

# Tags that indicate content the system must not surface.
# Last.fm tags are user-generated and unmoderated — this blocklist is a
# first-pass filter, not a guarantee of complete content safety.
TAG_BLOCKLIST = {
    "hate",
    "racist",
    "racism",
    "nazi",
    "white power",
    "white supremacy",
    "antisemitic",
    "antisemitism",
    "homophobic",
    "homophobia",
    "transphobic",
    "slur",
    "gore",
    "snuff",
    "pedo",
    "pedophilia",
    "child abuse",
    "rape",
    "torture",
    "genocide",
    "terrorism",
    "jihad",
    "extremist",
    "propaganda",
}

# 0.5-second delay between API calls respects Last.fm's rate limit guidance.
_RATE_DELAY = 0.5


def _tag_is_blocked(tag: str) -> bool:
    """Return True if the tag matches any entry in the blocklist."""
    normalized = tag.strip().lower()
    return any(blocked in normalized for blocked in TAG_BLOCKLIST)


def _song_passes_filter(tags: list[str]) -> bool:
    """Return True if none of the song's tags are on the blocklist."""
    return not any(_tag_is_blocked(t) for t in tags)


def _estimate_features(tags: list[str]) -> dict[str, float]:
    """
    Derive audio feature estimates from Last.fm tags using keyword heuristics.

    Last.fm does not expose Spotify-style audio features. These estimates are
    derived from tag semantics — a song tagged 'ambient' scores high on
    acousticness, 'dance' scores high on danceability, etc. The estimates are
    intentionally coarse; they exist to populate the vector space for cosine
    similarity, not to replicate audio analysis.
    """
    tag_set = {t.lower() for t in tags}

    energy = 0.5
    valence = 0.5
    danceability = 0.5
    acousticness = 0.5

    # Energy signals
    if tag_set & {"metal", "hardcore", "punk", "drum and bass", "industrial", "noise"}:
        energy = 0.9
    elif tag_set & {"rock", "electronic", "edm", "hip-hop", "hip hop", "rap", "dance"}:
        energy = 0.75
    elif tag_set & {"ambient", "classical", "acoustic", "folk", "sleep", "meditation"}:
        energy = 0.2

    # Valence signals
    if tag_set & {"happy", "feel good", "upbeat", "party", "summer", "joy", "fun"}:
        valence = 0.85
    elif tag_set & {"sad", "melancholy", "depressing", "dark", "doom", "grief", "loss"}:
        valence = 0.2

    # Danceability signals
    if tag_set & {"dance", "club", "edm", "afrobeats", "reggaeton", "disco", "funk"}:
        danceability = 0.88
    elif tag_set & {"classical", "ambient", "drone", "experimental", "post-rock"}:
        danceability = 0.2

    # Acousticness signals
    if tag_set & {"acoustic", "folk", "singer-songwriter", "unplugged", "classical"}:
        acousticness = 0.85
    elif tag_set & {"electronic", "edm", "synth", "industrial", "techno", "house"}:
        acousticness = 0.1

    return {
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _get_tag_top_tracks(tag: str, api_key: str, limit: int = 10) -> list[dict]:
    """Fetch top tracks for a single Last.fm tag. Retries up to 3 times."""
    params = {
        "method": "tag.gettoptracks",
        "tag": tag,
        "api_key": api_key,
        "format": "json",
        "limit": limit,
    }
    response = requests.get(LASTFM_BASE, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("tracks", {}).get("track", [])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _get_track_tags(artist: str, title: str, api_key: str) -> list[str]:
    """Fetch top tags for a specific track. Retries up to 3 times."""
    params = {
        "method": "track.gettoptags",
        "artist": artist,
        "track": title,
        "api_key": api_key,
        "format": "json",
    }
    response = requests.get(LASTFM_BASE, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    raw_tags = data.get("toptags", {}).get("tag", [])
    return [t["name"] for t in raw_tags if isinstance(t, dict)]


def fetch_songs(preferred_tags: list[str], limit_per_tag: int = 8) -> list[SongFeature]:
    """
    Retrieve songs from Last.fm matching the user's preferred tags.

    For each tag, the top tracks are fetched, then each track's own tags are
    retrieved to populate the SongFeature. Songs with blocked tags are removed
    before the list is returned. A 0.5-second delay between calls respects
    Last.fm rate limits.

    Returns an empty list (with a logged warning) if the API is unavailable,
    allowing the graph to continue with Radio Browser data only.
    """
    api_key = os.getenv("LASTFM_API_KEY", "")
    if not api_key:
        logger.error("LASTFM_API_KEY not set — skipping Last.fm retrieval")
        return []

    songs: list[SongFeature] = []
    seen: set[str] = set()  # deduplicate by "artist|title"

    for tag in preferred_tags:
        try:
            tracks = _get_tag_top_tracks(tag, api_key, limit=limit_per_tag)
            logger.info("lastfm · tag '%s' · %d tracks returned", tag, len(tracks))
        except Exception as exc:
            logger.warning("lastfm · tag '%s' · fetch failed: %s", tag, exc)
            continue

        for track in tracks:
            try:
                artist = track.get("artist", {}).get("name", "Unknown Artist")
                title = track.get("name", "Unknown Title")
                url = track.get("url")
                key = f"{artist.lower()}|{title.lower()}"

                if key in seen:
                    continue
                seen.add(key)

                time.sleep(_RATE_DELAY)
                track_tags = _get_track_tags(artist, title, api_key)

                if not _song_passes_filter(track_tags):
                    logger.info(
                        "lastfm · filtered '%s' by '%s' — blocked tag detected",
                        title,
                        artist,
                    )
                    continue

                features = _estimate_features(track_tags or [tag])

                songs.append(
                    SongFeature(
                        title=title,
                        artist=artist,
                        source="lastfm",
                        energy=features["energy"],
                        valence=features["valence"],
                        danceability=features["danceability"],
                        acousticness=features["acousticness"],
                        tags=track_tags if track_tags else [tag],
                        url=url,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "lastfm · skipping track '%s' — %s",
                    track.get("name", "unknown"),
                    exc,
                )
                continue

    logger.info("lastfm · %d songs passed filter and returned", len(songs))
    return songs
