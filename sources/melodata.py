"""
MeloData BPM enrichment and catalog discovery for Music Recommender: Music Theory.

Two complementary jobs in MasterMix mode:

1. Enrich existing Last.fm tracks — search each by title/artist to resolve an ISRC,
   then batch-fetch BPM via /v1/tracks/batch/features.

2. Discover MeloData catalog tracks — if any ISRCs were resolved, use them as seeds
   for /v1/recommendations with the TasteProfile's target features (BPM, energy,
   danceability, valence). Those recommendations come back with full audio features
   already attached; they are converted to SongFeature objects (source="melodata")
   and added to the catalog so the scoring and trajectory pipeline treats them as
   first-class candidates.

Only Last.fm tracks are enriched in step 1 — Radio Browser stations have no ISRC.
MeloData recommendation tracks carry bpm and full audio features natively.

API base: https://melodata.voltenworks.com/api/v1
Auth: Bearer token via MELODATA_API_KEY environment variable
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_BASE = "https://melodata.voltenworks.com/api/v1"
_TIMEOUT = 10
_BATCH_TIMEOUT = 15
_SEARCH_WORKERS = 5  # free tier rate limit: 5 req/sec
_REC_LIMIT = 20      # max MeloData recommendation tracks to add per session


def _key() -> Optional[str]:
    return os.getenv("MELODATA_API_KEY")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_key()}"}


def _clean_title(title: str) -> str:
    """Strip featured-artist credits so MeloData search matches the main title."""
    return re.sub(
        r"\s*[\(\[]\s*(?:feat|ft|with|f\/)[^\)\]]*[\)\]]",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=False,
)
def _search_isrc(title: str, artist: str) -> Optional[str]:
    """Return the ISRC for a track, or None if not found or API unavailable."""
    try:
        clean = _clean_title(title)
        resp = requests.get(
            f"{_BASE}/tracks/search",
            params={"q": f"{clean} {artist}", "limit": 1},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            tracks = resp.json().get("data", {}).get("results", [])
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


def _batch_full_features(isrcs: list[str]) -> dict[str, dict]:
    """
    Return {isrc: features_dict} for a list of ISRCs.

    Used to populate complete audio feature vectors (including acousticness)
    for MeloData recommendation tracks, which the recommendations endpoint
    does not return in full.
    """
    result: dict[str, dict] = {}
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
                    result[isrc] = features
    except Exception as exc:
        logger.warning("melodata · batch full features failed: %s", exc)
    return result


def _fetch_recommendations(seed_isrcs: list[str], profile) -> list[dict]:
    """
    Call /v1/recommendations using resolved ISRCs as seeds.

    Target features from the TasteProfile shift the recommendation space
    toward the user's desired BPM, energy, danceability, and valence.
    Returns a list of raw track dicts from the API, or [] on failure.
    """
    try:
        params: list[tuple] = [("seed", isrc) for isrc in seed_isrcs[:5]]
        params.append(("limit", _REC_LIMIT))
        if profile.target_bpm is not None:
            params.append(("target_bpm", profile.target_bpm))
        params.extend([
            ("target_energy", profile.energy),
            ("target_danceability", profile.danceability),
            ("target_valence", profile.valence),
        ])
        resp = requests.get(
            f"{_BASE}/recommendations",
            params=params,
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return data.get("results", []) if isinstance(data, dict) else data
        logger.warning("melodata · recommendations status %d", resp.status_code)
    except Exception as exc:
        logger.warning("melodata · recommendations failed: %s", exc)
    return []


def _recs_to_songfeatures(recs: list[dict], full_features: dict[str, dict]) -> list:
    """
    Convert MeloData recommendation dicts to SongFeature objects.

    Imports SongFeature locally to avoid a circular import at module load time.
    Tracks missing BPM or title/artist are skipped.
    """
    from models import SongFeature  # local import — avoids circular dependency

    songs = []
    for rec in recs:
        isrc = rec.get("isrc")
        title = rec.get("title", "").strip()
        artist = rec.get("artist", "").strip()
        if not title or not artist:
            continue

        feats = full_features.get(isrc) or rec.get("features") or {}
        bpm = feats.get("bpm")
        if bpm is None:
            continue

        songs.append(
            SongFeature(
                title=title,
                artist=artist,
                source="melodata",
                energy=float(feats.get("energy") or 0.5),
                valence=float(feats.get("valence") or 0.5),
                danceability=float(feats.get("danceability") or 0.5),
                acousticness=float(feats.get("acousticness") or 0.0),
                tags=[],
                bpm=float(bpm),
            )
        )
    return songs


def enrich_catalog_bpm(songs: list, profile=None) -> tuple[list, int]:
    """
    Populate SongFeature.bpm for Last.fm tracks and optionally discover new
    MeloData catalog tracks when a TasteProfile is provided.

    Flow:
      1. Search each Last.fm track's ISRC in parallel (capped at
         _SEARCH_WORKERS concurrent requests to respect rate limits).
      2. Batch-fetch BPM for all resolved ISRCs (chunks of 50) and apply
         to matching catalog entries.
      3. If profile is provided and at least one ISRC was resolved, call
         /v1/recommendations using those ISRCs as seeds with the profile's
         target features.  The returned tracks are fetched for full audio
         features and added to the catalog as source="melodata" entries
         with bpm populated.

    Radio Browser stations are always skipped — they have no ISRC.
    Returns (enriched_catalog, bpm_hits) where bpm_hits counts Last.fm
    tracks that received a BPM value.  MeloData recommendation tracks
    are appended to the catalog and counted separately in the log.
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

    enriched = list(songs)
    bpm_hits = 0

    # Step 2: enrich existing Last.fm tracks with BPM
    if isrc_map:
        all_isrcs = list(set(isrc_map.values()))
        bpm_by_isrc: dict[str, Optional[float]] = {}
        for chunk_start in range(0, len(all_isrcs), 50):
            chunk = all_isrcs[chunk_start : chunk_start + 50]
            bpm_by_isrc.update(_batch_features(chunk))

        for idx, isrc in isrc_map.items():
            bpm = bpm_by_isrc.get(isrc)
            if bpm is not None:
                enriched[idx] = enriched[idx].model_copy(update={"bpm": bpm})
                bpm_hits += 1
                logger.debug("melodata · '%s' BPM = %.1f", songs[idx].title, bpm)

    logger.info(
        "melodata · enriched %d / %d Last.fm tracks with BPM",
        bpm_hits,
        len(lastfm_indices),
    )

    # Step 3: discover MeloData catalog tracks via recommendations
    melodata_count = 0
    if profile is not None:
        seed_isrcs = list(set(isrc_map.values()))[:5]

        # Fallback: if no ISRCs found from catalog, search MeloData by artist name
        # alone for each Last.fm artist. An artist-only query is more likely to
        # match than a full title+artist query for tracks not in MeloData's index.
        if not seed_isrcs:
            seen_artists: set[str] = set()
            for idx in lastfm_indices:
                artist = songs[idx].artist.strip()
                if artist in seen_artists:
                    continue
                seen_artists.add(artist)
                try:
                    resp = requests.get(
                        f"{_BASE}/tracks/search",
                        params={"q": artist, "limit": 1},
                        headers=_headers(),
                        timeout=_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        results = resp.json().get("data", {}).get("results", [])
                        if results:
                            seed_isrc = results[0].get("isrc")
                            if seed_isrc:
                                seed_isrcs = [seed_isrc]
                                logger.info(
                                    "melodata · artist seed '%s' → %s", artist, seed_isrc
                                )
                                break
                except Exception as exc:
                    logger.debug("melodata · artist seed failed for '%s': %s", artist, exc)

        if seed_isrcs:
            recs = _fetch_recommendations(seed_isrcs, profile)
            if recs:
                rec_isrcs = [r.get("isrc") for r in recs if r.get("isrc")]
                full_features = _batch_full_features(rec_isrcs) if rec_isrcs else {}
                melodata_songs = _recs_to_songfeatures(recs, full_features)
                enriched.extend(melodata_songs)
                melodata_count = len(melodata_songs)
                logger.info(
                    "melodata · added %d recommendation tracks to catalog",
                    melodata_count,
                )
        else:
            logger.info("melodata · no seed ISRCs found — skipping recommendations")

    return enriched, bpm_hits
