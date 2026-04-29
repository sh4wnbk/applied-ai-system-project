"""
MeloData BPM enrichment client for Music Recommender: Music Theory.

Populates SongFeature.bpm using the MeloData audio analysis API. The
enrichment step runs after Misty's dual-source retrieval and only when
--mastermix is active, so free-tier quota is not spent on sessions that
do not use BPM matching.

Only Last.fm tracks are enriched — Radio Browser stations have no ISRC
and cannot be looked up in the MeloData catalog. Stations always receive
bpm=None and are treated as neutral by the MasterMix filter.

API base: https://melodata.voltenworks.com/api/v1
Auth: Bearer token via MELODATA_API_KEY environment variable
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_BASE = "https://melodata.voltenworks.com/api/v1"
_TIMEOUT = 10
_BATCH_TIMEOUT = 15
_SEARCH_WORKERS = 5  # free tier rate limit: 5 req/sec


def _key() -> Optional[str]:
    return os.getenv("MELODATA_API_KEY")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_key()}"}


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=False,
)
def _search_isrc(title: str, artist: str) -> Optional[str]:
    """Return the ISRC for a track, or None if not found or API unavailable."""
    try:
        resp = requests.get(
            f"{_BASE}/tracks/search",
            params={"q": f"{title} {artist}", "limit": 1},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            tracks = resp.json().get("data", [])
            if tracks:
                return tracks[0].get("isrc")
    except Exception as exc:
        logger.debug("melodata · search failed for '%s %s': %s", title, artist, exc)
    return None


def _batch_features(isrcs: list[str]) -> dict[str, Optional[float]]:
    """
    Return {isrc: bpm} for a list of ISRCs via the batch features endpoint.

    Tracks with status 202 (analysis queued) or unavailable return None and
    are skipped — the song remains eligible as a neutral candidate.
    Processes at most 50 ISRCs per call per API limit.
    """
    result: dict[str, Optional[float]] = {}
    try:
        resp = requests.post(
            f"{_BASE}/tracks/batch/features",
            json={"isrcs": isrcs},
            headers=_headers(),
            timeout=_BATCH_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            for isrc, features in data.items():
                if features and isinstance(features, dict):
                    bpm = features.get("bpm")
                    if bpm is not None:
                        result[isrc] = float(bpm)
    except Exception as exc:
        logger.warning("melodata · batch features failed: %s", exc)
    return result


def enrich_catalog_bpm(songs: list) -> tuple[list, int]:
    """
    Populate SongFeature.bpm for all Last.fm tracks in the catalog.

    Flow:
      1. Search for each Last.fm track's ISRC in parallel (capped at
         _SEARCH_WORKERS concurrent requests to respect rate limits).
      2. Batch-fetch audio features for all resolved ISRCs (chunks of 50).
      3. Return the enriched catalog and a hit count.

    Radio Browser stations are skipped — they have no ISRC.
    Returns the original catalog unchanged if the API key is absent.
    """
    if not _key():
        logger.info("melodata · MELODATA_API_KEY not set — skipping BPM enrichment")
        return songs, 0

    lastfm_indices = [i for i, s in enumerate(songs) if s.source == "lastfm"]
    if not lastfm_indices:
        logger.info("melodata · no Last.fm tracks in catalog — skipping")
        return songs, 0

    # Step 1: resolve ISRCs in parallel
    isrc_map: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_search_isrc, songs[i].title, songs[i].artist): i
            for i in lastfm_indices
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                isrc = future.result()
                if isrc:
                    isrc_map[idx] = isrc
                    logger.debug(
                        "melodata · resolved '%s' → %s",
                        songs[idx].title,
                        isrc,
                    )
            except Exception as exc:
                logger.debug("melodata · search future error at index %d: %s", idx, exc)

    if not isrc_map:
        logger.info("melodata · no ISRCs resolved — BPM enrichment produced 0 hits")
        return songs, 0

    # Step 2: batch-fetch features in chunks of 50
    all_isrcs = list(set(isrc_map.values()))
    bpm_by_isrc: dict[str, Optional[float]] = {}
    for chunk_start in range(0, len(all_isrcs), 50):
        chunk = all_isrcs[chunk_start : chunk_start + 50]
        bpm_by_isrc.update(_batch_features(chunk))

    # Step 3: apply BPM values to catalog
    enriched = list(songs)
    hits = 0
    for idx, isrc in isrc_map.items():
        bpm = bpm_by_isrc.get(isrc)
        if bpm is not None:
            enriched[idx] = enriched[idx].model_copy(update={"bpm": bpm})
            hits += 1
            logger.debug(
                "melodata · '%s' BPM = %.1f", songs[idx].title, bpm
            )

    logger.info(
        "melodata · enriched %d / %d Last.fm tracks with BPM",
        hits,
        len(lastfm_indices),
    )
    return enriched, hits
