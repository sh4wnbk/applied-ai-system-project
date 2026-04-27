"""
Radio Browser API client for Music Recommender: Music Theory.

Radio Browser is a community-run, no-auth directory of live radio stations.
Stations are represented as SongFeature objects so they flow through the same
scoring and explanation pipeline as Last.fm tracks. The 'title' field holds the
station name; 'artist' holds the country or primary genre as a stand-in.

Because Radio Browser has no moderation infrastructure, this client applies
the same TAG_BLOCKLIST used by Last.fm plus an explicit exclusion list for
adult-content station tags.
"""

import logging
import random
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from models import SongFeature
from sources.lastfm import TAG_BLOCKLIST, _estimate_features, _song_passes_filter

logger = logging.getLogger(__name__)

# Radio Browser operates a pool of community mirrors. Selecting one at random
# distributes load and avoids hammering a single host.
_RADIO_MIRRORS = [
    "https://de1.api.radio-browser.info",
    "https://nl1.api.radio-browser.info",
    "https://at1.api.radio-browser.info",
]

# Minimum community votes required to include a station.
# Stations below this threshold are low-signal — few listens, sparse metadata.
# Documented as a design decision in model_card.md.
MIN_VOTES = 10

# Tags that explicitly mark adult, explicit, or age-restricted content.
# Applied in addition to the shared TAG_BLOCKLIST.
ADULT_TAGS = {"adult", "xxx", "explicit", "18+", "erotic", "sex", "nsfw"}


def _station_passes_filter(tags: list[str]) -> bool:
    """
    Return True if the station clears both content filters.

    Checks the shared blocklist (hate speech, extremism) and the adult-content
    exclusion list independently so each filter remains auditable on its own.
    """
    normalized = [t.strip().lower() for t in tags]
    if any(t in ADULT_TAGS for t in normalized):
        return False
    return _song_passes_filter(tags)


def _pick_mirror() -> str:
    """Return a random Radio Browser mirror URL."""
    return random.choice(_RADIO_MIRRORS)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _search_stations(tag: str, limit: int = 20) -> list[dict]:
    """
    Query Radio Browser for stations matching a tag. Retries up to 3 times.

    The API requires a User-Agent header that identifies the application —
    anonymous requests are rejected by some mirrors.
    """
    base = _pick_mirror()
    url = f"{base}/json/stations/search"
    headers = {"User-Agent": "MusicTheoryRecommender/1.0"}
    params = {
        "tag": tag,
        "limit": limit,
        "order": "votes",
        "reverse": "true",
        "hidebroken": "true",
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_stations(preferred_tags: list[str], limit_per_tag: int = 10) -> list[SongFeature]:
    """
    Retrieve radio stations from Radio Browser matching the user's preferred tags.

    Each station is mapped to a SongFeature so it flows through the same
    scoring and explanation pipeline as Last.fm tracks. Audio features are
    estimated from the station's reported tags using the same heuristics as
    Last.fm. Stations below MIN_VOTES or carrying blocked/adult tags are
    removed before results are returned.

    Returns an empty list (with a logged warning) if the API is unavailable,
    allowing the graph to continue with Last.fm data only.
    """
    stations: list[SongFeature] = []
    seen: set[str] = set()  # deduplicate by station name

    for tag in preferred_tags:
        try:
            results = _search_stations(tag, limit=limit_per_tag)
            logger.info("radiobrowser · tag '%s' · %d stations returned", tag, len(results))
        except Exception as exc:
            logger.warning("radiobrowser · tag '%s' · fetch failed: %s", tag, exc)
            continue

        for station in results:
            try:
                name = station.get("name", "").strip()
                if not name:
                    continue

                key = name.lower()
                if key in seen:
                    continue

                votes = int(station.get("votes", 0))
                if votes < MIN_VOTES:
                    continue

                raw_tags_str = station.get("tags", "")
                station_tags = (
                    [t.strip() for t in raw_tags_str.split(",") if t.strip()]
                    if raw_tags_str
                    else [tag]
                )

                if not _station_passes_filter(station_tags):
                    logger.info(
                        "radiobrowser · filtered station '%s' — blocked or adult tag",
                        name,
                    )
                    continue

                seen.add(key)

                country = station.get("country", "") or station.get("countrycode", "")
                # Use country as artist stand-in; gives Prestige something to reference
                # in cultural context explanations.
                artist_field = country if country else tag.title()

                features = _estimate_features(station_tags)
                url = station.get("url_resolved") or station.get("url")

                stations.append(
                    SongFeature(
                        title=name,
                        artist=artist_field,
                        source="radio",
                        energy=features["energy"],
                        valence=features["valence"],
                        danceability=features["danceability"],
                        acousticness=features["acousticness"],
                        tags=station_tags,
                        url=url,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "radiobrowser · skipping station '%s' — %s",
                    station.get("name", "unknown"),
                    exc,
                )
                continue

    logger.info("radiobrowser · %d stations passed filter and returned", len(stations))
    return stations
