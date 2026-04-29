"""
Tempo — Score node for Music Recommender: Music Theory.

Tempo computes cosine similarity between the user's TasteProfile vector and
every song in the catalog. No LLM is used. This is an explicit architectural
decision: cosine similarity produces a deterministic, correct answer for a
given input pair. Delegating a calculation with a correct answer to a language
model would introduce unnecessary variance and cost.

Character: Tempo / Metronome
Model: None — pure numpy cosine similarity
"""

import logging
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from models import AgentState, ScoredSong, SongFeature, TasteProfile

logger = logging.getLogger(__name__)
console = Console()

_DIMENSIONS_BASE = ["energy", "valence", "danceability", "acousticness"]


def _bpm_range(catalog: list[SongFeature]) -> tuple[float, float] | None:
    """Return (min_bpm, max_bpm) for catalog songs that have BPM data, or None if none do."""
    bpms = [s.bpm for s in catalog if s.bpm is not None]
    if not bpms:
        return None
    return float(min(bpms)), float(max(bpms))


def _normalize_bpm(bpm: float, bpm_min: float, bpm_max: float) -> float:
    """Min-Max normalize a BPM value to [0.0, 1.0]. Returns 0.5 when range is zero."""
    if bpm_max == bpm_min:
        return 0.5
    return (bpm - bpm_min) / (bpm_max - bpm_min)


def _profile_vector(profile: TasteProfile, normalized_bpm: float | None = None) -> np.ndarray:
    """Extract the audio feature vector from a TasteProfile. BPM appended when normalized_bpm is provided."""
    values = [profile.energy, profile.valence, profile.danceability, profile.acousticness]
    if normalized_bpm is not None:
        values.append(normalized_bpm)
    return np.array(values, dtype=float)


def _song_vector(song: SongFeature, normalized_bpm: float | None = None) -> np.ndarray:
    """Extract the audio feature vector from a SongFeature. BPM appended when normalized_bpm is provided."""
    values = [song.energy, song.valence, song.danceability, song.acousticness]
    if normalized_bpm is not None:
        values.append(normalized_bpm)
    return np.array(values, dtype=float)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns 0.0 for zero vectors rather than raising — a song or profile
    with all-zero audio features is technically valid input from Pydantic's
    perspective (ge=0.0 allows it), so the node must handle it gracefully.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _vector_breakdown(
    profile_vec: np.ndarray,
    song_vec: np.ndarray,
    norm_a: float,
    norm_b: float,
    dimensions: list[str],
) -> dict[str, float]:
    """
    Compute each dimension's contribution to the total cosine similarity score.

    contribution_i = (profile_i * song_i) / (||profile|| * ||song||)

    The contributions sum to the total similarity score, making the ranking
    fully auditable without an LLM.
    """
    if norm_a == 0.0 or norm_b == 0.0:
        return {dim: 0.0 for dim in dimensions}
    denominator = norm_a * norm_b
    return {
        dim: float((profile_vec[i] * song_vec[i]) / denominator)
        for i, dim in enumerate(dimensions)
    }


def score(state: AgentState) -> AgentState:
    """
    Score every song in raw_catalog against the TasteProfile using cosine similarity.

    When TasteProfile.target_bpm is set and catalog songs carry BPM data, a 5th
    dimension (bpm) is added to both vectors after Min-Max normalization. Songs
    with no BPM value are scored with a neutral midpoint (0.5) so they are not
    excluded, consistent with the MasterMix "neutral track" policy.

    Returns the catalog as a ranked list of ScoredSong objects, highest
    similarity first. The vector_breakdown field on each ScoredSong shows
    exactly how much each audio dimension contributed to the score.

    Writes scored_songs and an agent_log entry to AgentState.
    """
    profile = state["taste_profile"]
    catalog = state["raw_catalog"]

    # Activate the BPM dimension when the profile declares a target and catalog has BPM data.
    bpm_bounds = _bpm_range(catalog) if profile.target_bpm is not None else None
    if bpm_bounds is not None:
        bpm_min, bpm_max = bpm_bounds
        profile_bpm_norm: float | None = _normalize_bpm(profile.target_bpm, bpm_min, bpm_max)
        dimensions = _DIMENSIONS_BASE + ["bpm"]
    else:
        profile_bpm_norm = None
        dimensions = _DIMENSIONS_BASE

    _print_tempo_panel(len(catalog), dimensions)
    logger.info("tempo · node fired · scoring %d songs · dimensions: %s", len(catalog), dimensions)

    profile_vec = _profile_vector(profile, profile_bpm_norm)
    norm_p = float(np.linalg.norm(profile_vec))

    scored: list[ScoredSong] = []

    for song in catalog:
        if bpm_bounds is not None:
            song_bpm_norm: float | None = (
                _normalize_bpm(song.bpm, bpm_min, bpm_max)
                if song.bpm is not None
                else 0.5
            )
        else:
            song_bpm_norm = None

        song_vec = _song_vector(song, song_bpm_norm)
        norm_s = float(np.linalg.norm(song_vec))
        similarity = _cosine_similarity(profile_vec, song_vec)
        breakdown = _vector_breakdown(profile_vec, song_vec, norm_p, norm_s, dimensions)

        scored.append(
            ScoredSong(
                song=song,
                similarity_score=round(similarity, 4),
                vector_breakdown={k: round(v, 4) for k, v in breakdown.items()},
            )
        )

    scored.sort(key=lambda s: s.similarity_score, reverse=True)

    _print_score_table(scored[:10])
    logger.info("tempo · scoring complete · top score: %.4f", scored[0].similarity_score if scored else 0)

    log_entry = (
        f"[{datetime.now().isoformat()}] tempo · scored {len(scored)} songs · "
        f"top score: {scored[0].similarity_score:.4f}" if scored else
        f"[{datetime.now().isoformat()}] tempo · scored 0 songs"
    )

    return {
        **state,
        "scored_songs": scored,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def _print_tempo_panel(catalog_size: int, dimensions: list[str]) -> None:
    """Render Tempo's character panel showing his internal monologue."""
    dim_str = " · ".join(dimensions)
    console.print(
        Panel(
            f"[bold magenta]Tempo[/bold magenta]\n\n"
            f"[italic]Measuring the distance between your taste and the catalog...[/italic]\n\n"
            f"Scoring [green]{catalog_size}[/green] songs via cosine similarity.\n"
            f"Dimensions: {dim_str}\n"
            f"[dim]No language model — cosine math has a correct answer.[/dim]",
            title="[magenta]— Score —[/magenta]",
            border_style="magenta",
        )
    )


def _print_score_table(top_songs: list[ScoredSong]) -> None:
    """Print a Rich table of the top-scored songs with per-dimension breakdown."""
    show_bpm = any(s.song.bpm is not None for s in top_songs)

    table = Table(title="Top Scored Songs", show_header=True, header_style="bold magenta")
    table.add_column("Title", style="white", no_wrap=True)
    table.add_column("Artist", style="dim")
    table.add_column("Src", justify="center")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Energy", justify="right")
    table.add_column("Valence", justify="right")
    table.add_column("Dance", justify="right")
    table.add_column("Acoustic", justify="right")
    if show_bpm:
        table.add_column("BPM", justify="right", style="yellow")

    for s in top_songs:
        bd = s.vector_breakdown
        row = [
            s.song.title[:30],
            s.song.artist[:20],
            s.song.source,
            f"{s.similarity_score:.4f}",
            f"{bd.get('energy', 0):.3f}",
            f"{bd.get('valence', 0):.3f}",
            f"{bd.get('danceability', 0):.3f}",
            f"{bd.get('acousticness', 0):.3f}",
        ]
        if show_bpm:
            row.append(f"{s.song.bpm:.0f}" if s.song.bpm is not None else "—")
        table.add_row(*row)

    console.print(table)
