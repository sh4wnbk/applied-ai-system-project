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

_DIMENSIONS = ["energy", "valence", "danceability", "acousticness"]


def _profile_vector(profile: TasteProfile) -> np.ndarray:
    """Extract the four-dimensional audio feature vector from a TasteProfile."""
    return np.array([
        profile.energy,
        profile.valence,
        profile.danceability,
        profile.acousticness,
    ], dtype=float)


def _song_vector(song: SongFeature) -> np.ndarray:
    """Extract the four-dimensional audio feature vector from a SongFeature."""
    return np.array([
        song.energy,
        song.valence,
        song.danceability,
        song.acousticness,
    ], dtype=float)


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
) -> dict[str, float]:
    """
    Compute each dimension's contribution to the total cosine similarity score.

    contribution_i = (profile_i * song_i) / (||profile|| * ||song||)

    The contributions sum to the total similarity score, making the ranking
    fully auditable without an LLM.
    """
    if norm_a == 0.0 or norm_b == 0.0:
        return {dim: 0.0 for dim in _DIMENSIONS}
    denominator = norm_a * norm_b
    return {
        dim: float((profile_vec[i] * song_vec[i]) / denominator)
        for i, dim in enumerate(_DIMENSIONS)
    }


def score(state: AgentState) -> AgentState:
    """
    Score every song in raw_catalog against the TasteProfile using cosine similarity.

    Returns the catalog as a ranked list of ScoredSong objects, highest
    similarity first. The vector_breakdown field on each ScoredSong shows
    exactly how much each audio dimension contributed to the score.

    Writes scored_songs and an agent_log entry to AgentState.
    """
    profile = state["taste_profile"]
    catalog = state["raw_catalog"]

    _print_tempo_panel(len(catalog))
    logger.info("tempo · node fired · scoring %d songs", len(catalog))

    profile_vec = _profile_vector(profile)
    norm_p = float(np.linalg.norm(profile_vec))

    scored: list[ScoredSong] = []

    for song in catalog:
        song_vec = _song_vector(song)
        norm_s = float(np.linalg.norm(song_vec))
        similarity = _cosine_similarity(profile_vec, song_vec)
        breakdown = _vector_breakdown(profile_vec, song_vec, norm_p, norm_s)

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


def _print_tempo_panel(catalog_size: int) -> None:
    """Render Tempo's character panel showing his internal monologue."""
    console.print(
        Panel(
            f"[bold magenta]Tempo[/bold magenta]\n\n"
            f"[italic]Measuring the distance between your taste and the catalog...[/italic]\n\n"
            f"Scoring [green]{catalog_size}[/green] songs via cosine similarity.\n"
            f"Dimensions: energy · valence · danceability · acousticness\n"
            f"[dim]No language model — cosine math has a correct answer.[/dim]",
            title="[magenta]— Score —[/magenta]",
            border_style="magenta",
        )
    )


def _print_score_table(top_songs: list[ScoredSong]) -> None:
    """Print a Rich table of the top-scored songs with per-dimension breakdown."""
    table = Table(title="Top Scored Songs", show_header=True, header_style="bold magenta")
    table.add_column("Title", style="white", no_wrap=True)
    table.add_column("Artist", style="dim")
    table.add_column("Src", justify="center")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Energy", justify="right")
    table.add_column("Valence", justify="right")
    table.add_column("Dance", justify="right")
    table.add_column("Acoustic", justify="right")

    for s in top_songs:
        bd = s.vector_breakdown
        table.add_row(
            s.song.title[:30],
            s.song.artist[:20],
            s.song.source,
            f"{s.similarity_score:.4f}",
            f"{bd.get('energy', 0):.3f}",
            f"{bd.get('valence', 0):.3f}",
            f"{bd.get('danceability', 0):.3f}",
            f"{bd.get('acousticness', 0):.3f}",
        )

    console.print(table)
