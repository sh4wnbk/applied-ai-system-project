"""
Edge case and unit tests for MasterMix BPM features.

All tests are pure Python — no API calls, no LangGraph execution.
Run with: pytest tests/test_mastermix.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ExplainedSong, ScoredSong, SongFeature, TasteProfile
from nodes.rerank import _mastermix_filter
from nodes.score import (
    _DIMENSIONS_BASE,
    _bpm_range,
    _cosine_similarity,
    _normalize_bpm,
    _profile_vector,
    _song_vector,
    _vector_breakdown,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _song(title: str, bpm: float | None = None) -> SongFeature:
    return SongFeature(
        title=title,
        artist="Test Artist",
        source="lastfm",
        energy=0.5,
        valence=0.5,
        danceability=0.5,
        acousticness=0.5,
        tags=["test"],
        bpm=bpm,
    )


def _explained(title: str, bpm: float | None = None, confidence: float = 0.8) -> ExplainedSong:
    song = _song(title, bpm)
    scored = ScoredSong(song=song, similarity_score=0.9, vector_breakdown={})
    return ExplainedSong(
        scored_song=scored,
        explanation="Test explanation.",
        tag_overlap=[],
        confidence=confidence,
    )


def _profile(target_bpm: float | None = None) -> TasteProfile:
    return TasteProfile(
        name="Test Profile",
        energy=0.8,
        valence=0.7,
        danceability=0.9,
        acousticness=0.1,
        preferred_tags=["test"],
        target_bpm=target_bpm,
    )


# ─── _bpm_range ───────────────────────────────────────────────────────────────

class TestBpmRange:
    def test_empty_catalog(self):
        assert _bpm_range([]) is None

    def test_all_none(self):
        catalog = [_song("A"), _song("B"), _song("C")]
        assert _bpm_range(catalog) is None

    def test_single_bpm_value(self):
        catalog = [_song("A", bpm=120.0), _song("B"), _song("C")]
        result = _bpm_range(catalog)
        assert result == (120.0, 120.0)

    def test_multiple_bpm_values(self):
        catalog = [_song("A", bpm=100.0), _song("B", bpm=140.0), _song("C", bpm=120.0)]
        result = _bpm_range(catalog)
        assert result == (100.0, 140.0)

    def test_mixed_none_and_values(self):
        catalog = [_song("A", bpm=90.0), _song("B"), _song("C", bpm=150.0)]
        result = _bpm_range(catalog)
        assert result == (90.0, 150.0)


# ─── _normalize_bpm ───────────────────────────────────────────────────────────

class TestNormalizeBpm:
    def test_min_maps_to_zero(self):
        assert _normalize_bpm(100.0, 100.0, 140.0) == pytest.approx(0.0)

    def test_max_maps_to_one(self):
        assert _normalize_bpm(140.0, 100.0, 140.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert _normalize_bpm(120.0, 100.0, 140.0) == pytest.approx(0.5)

    def test_degenerate_min_equals_max_returns_half(self):
        # Single BPM value in catalog — no range to scale against.
        assert _normalize_bpm(120.0, 120.0, 120.0) == pytest.approx(0.5)

    def test_value_outside_range_clamps_beyond_bounds(self):
        # Values outside [min, max] are not clamped — normalization is linear.
        result = _normalize_bpm(160.0, 100.0, 140.0)
        assert result > 1.0


# ─── _profile_vector / _song_vector dimension count ──────────────────────────

class TestVectorDimensions:
    def test_four_dimensions_without_bpm(self):
        p = _profile()
        vec = _profile_vector(p, normalized_bpm=None)
        assert vec.shape == (4,)

    def test_five_dimensions_with_bpm(self):
        p = _profile(target_bpm=120.0)
        vec = _profile_vector(p, normalized_bpm=0.5)
        assert vec.shape == (5,)

    def test_song_vector_four_without_bpm(self):
        s = _song("A", bpm=120.0)
        vec = _song_vector(s, normalized_bpm=None)
        assert vec.shape == (4,)

    def test_song_vector_five_with_bpm(self):
        s = _song("A", bpm=120.0)
        vec = _song_vector(s, normalized_bpm=0.5)
        assert vec.shape == (5,)

    def test_profile_and_song_vectors_same_length(self):
        p = _profile(target_bpm=120.0)
        s = _song("A", bpm=120.0)
        pv = _profile_vector(p, normalized_bpm=0.5)
        sv = _song_vector(s, normalized_bpm=0.5)
        assert pv.shape == sv.shape


# ─── _vector_breakdown keys ───────────────────────────────────────────────────

class TestVectorBreakdown:
    def test_four_dim_breakdown_has_base_keys(self):
        p = _profile()
        s = _song("A")
        pv = _profile_vector(p)
        sv = _song_vector(s)
        norm_p = float(np.linalg.norm(pv))
        norm_s = float(np.linalg.norm(sv))
        bd = _vector_breakdown(pv, sv, norm_p, norm_s, _DIMENSIONS_BASE)
        assert set(bd.keys()) == {"energy", "valence", "danceability", "acousticness"}

    def test_five_dim_breakdown_includes_bpm(self):
        dims = _DIMENSIONS_BASE + ["bpm"]
        p = _profile(target_bpm=120.0)
        s = _song("A", bpm=120.0)
        pv = _profile_vector(p, normalized_bpm=0.5)
        sv = _song_vector(s, normalized_bpm=0.5)
        norm_p = float(np.linalg.norm(pv))
        norm_s = float(np.linalg.norm(sv))
        bd = _vector_breakdown(pv, sv, norm_p, norm_s, dims)
        assert "bpm" in bd
        assert set(bd.keys()) == set(dims)

    def test_contributions_sum_to_cosine_similarity(self):
        dims = _DIMENSIONS_BASE + ["bpm"]
        p = _profile(target_bpm=120.0)
        s = _song("A", bpm=100.0)
        pv = _profile_vector(p, normalized_bpm=0.5)
        sv = _song_vector(s, normalized_bpm=0.0)
        norm_p = float(np.linalg.norm(pv))
        norm_s = float(np.linalg.norm(sv))
        similarity = _cosine_similarity(pv, sv)
        bd = _vector_breakdown(pv, sv, norm_p, norm_s, dims)
        assert sum(bd.values()) == pytest.approx(similarity, abs=1e-6)

    def test_zero_vector_returns_all_zeros(self):
        dims = _DIMENSIONS_BASE
        zero = np.zeros(4)
        pv = np.array([0.8, 0.7, 0.9, 0.1])
        norm_p = float(np.linalg.norm(pv))
        bd = _vector_breakdown(pv, zero, norm_p, 0.0, dims)
        assert all(v == 0.0 for v in bd.values())


# ─── _mastermix_filter ────────────────────────────────────────────────────────

class TestMastermixFilter:
    def test_songs_in_window_pass(self):
        songs = [
            _explained("In 1", bpm=98.0),
            _explained("In 2", bpm=102.0),
            _explained("Out", bpm=130.0),
        ]
        result = _mastermix_filter(songs, target_bpm=100.0)
        titles = {s.scored_song.song.title for s in result}
        assert "In 1" in titles
        assert "In 2" in titles
        assert "Out" not in titles

    def test_neutral_songs_always_included(self):
        songs = [
            _explained("In window", bpm=100.0),
            _explained("Also in", bpm=103.0),
            _explained("Neutral", bpm=None),
        ]
        result = _mastermix_filter(songs, target_bpm=100.0)
        titles = {s.scored_song.song.title for s in result}
        assert "Neutral" in titles

    def test_exact_boundary_included(self):
        songs = [
            _explained("Exactly +5", bpm=105.0),
            _explained("Exactly -5", bpm=95.0),
            _explained("Just outside", bpm=106.0),
            _explained("Also in", bpm=100.0),
        ]
        result = _mastermix_filter(songs, target_bpm=100.0)
        titles = {s.scored_song.song.title for s in result}
        assert "Exactly +5" in titles
        assert "Exactly -5" in titles
        assert "Just outside" not in titles

    def test_fewer_than_two_in_window_disables_filter(self):
        songs = [
            _explained("Only one", bpm=100.0),
            _explained("Outside 1", bpm=150.0),
            _explained("Outside 2", bpm=160.0),
        ]
        # Only one song in window — filter should be disabled, all returned.
        result = _mastermix_filter(songs, target_bpm=100.0)
        assert len(result) == 3

    def test_all_bpm_none_disables_filter(self):
        songs = [_explained("A"), _explained("B"), _explained("C")]
        result = _mastermix_filter(songs, target_bpm=100.0)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        result = _mastermix_filter([], target_bpm=100.0)
        assert result == []

    def test_order_preserved_in_window_before_neutral(self):
        songs = [
            _explained("In 1", bpm=99.0),
            _explained("In 2", bpm=101.0),
            _explained("Neutral", bpm=None),
        ]
        result = _mastermix_filter(songs, target_bpm=100.0)
        # in-window songs come before neutral songs
        titles = [s.scored_song.song.title for s in result]
        assert titles.index("Neutral") > titles.index("In 1")
        assert titles.index("Neutral") > titles.index("In 2")


# ─── TasteProfile.target_bpm field constraints ───────────────────────────────

class TestTasteProfileTargetBpm:
    def test_target_bpm_none_by_default(self):
        p = _profile()
        assert p.target_bpm is None

    def test_target_bpm_accepts_valid_value(self):
        p = _profile(target_bpm=120.0)
        assert p.target_bpm == 120.0

    def test_target_bpm_rejects_negative(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TasteProfile(
                name="T", energy=0.5, valence=0.5, danceability=0.5, acousticness=0.5,
                preferred_tags=["test"], target_bpm=-1.0,
            )

    def test_target_bpm_rejects_above_300(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TasteProfile(
                name="T", energy=0.5, valence=0.5, danceability=0.5, acousticness=0.5,
                preferred_tags=["test"], target_bpm=301.0,
            )

    def test_target_bpm_accepts_boundary_values(self):
        low = _profile(target_bpm=0.0)
        high = _profile(target_bpm=300.0)
        assert low.target_bpm == 0.0
        assert high.target_bpm == 300.0


# ─── SongFeature.bpm field ────────────────────────────────────────────────────

class TestSongFeatureBpm:
    def test_bpm_none_by_default(self):
        s = _song("A")
        assert s.bpm is None

    def test_bpm_accepts_float(self):
        s = _song("A", bpm=128.5)
        assert s.bpm == 128.5

    def test_bpm_accepts_zero(self):
        s = _song("A", bpm=0.0)
        assert s.bpm == 0.0
