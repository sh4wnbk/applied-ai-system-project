"""
Evaluation harness for Music Recommender: Music Theory.

Runs 10 predefined TasteProfiles through the full system and prints a
summary table. Six profiles are edge cases that stress boundary conditions;
four are normal diverse profiles that represent real listener types.

Usage:
    python eval/harness.py
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Logging to file only — the harness output is the Rich table, not log lines.
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(name)s · %(levelname)s · %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "session.log", encoding="utf-8"),
    ],
)

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from display.gatekeeper import run as gatekeeper_run
from graph import compiled_graph
from models import AgentState, TasteProfile

console = Console()
logger = logging.getLogger(__name__)

# ─── Profile definitions ───────────────────────────────────────────────────────

_200_CHAR_CONTEXT = (
    "Boundary test: verifying the 200-character context field limit is accepted "
    "exactly at the limit without a validation error. Padding to hit the mark: "
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
)
assert len(_200_CHAR_CONTEXT) == 200, f"Context length is {len(_200_CHAR_CONTEXT)}, expected 200"

PROFILES: list[dict] = [
    # ── Edge cases ────────────────────────────────────────────────────────────
    {
        "id": 1,
        "label": "All zeros",
        "description": "All float values at 0.0 — zero-vector cosine similarity edge case",
        "profile": {
            "name": "Zero Vector Profile",
            "energy": 0.0,
            "valence": 0.0,
            "danceability": 0.0,
            "acousticness": 0.0,
            "preferred_tags": ["ambient"],
        },
    },
    {
        "id": 2,
        "label": "All ones",
        "description": "All float values at 1.0 — maximum vector boundary test",
        "profile": {
            "name": "Max Vector Profile",
            "energy": 1.0,
            "valence": 1.0,
            "danceability": 1.0,
            "acousticness": 1.0,
            "preferred_tags": ["dance", "energy"],
        },
    },
    {
        "id": 3,
        "label": "Empty tags",
        "description": "Empty preferred_tags — must be caught by TasteProfile validator, never reach graph",
        "profile": {
            "name": "Empty Tags Profile",
            "energy": 0.5,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.5,
            "preferred_tags": [],
        },
        "expect_validation_error": True,
    },
    {
        "id": 4,
        "label": "Non-English context",
        "description": "Non-English context string — moderation API must handle multilingual input",
        "profile": {
            "name": "Noche Tranquila",
            "energy": 0.3,
            "valence": 0.6,
            "danceability": 0.2,
            "acousticness": 0.8,
            "preferred_tags": ["latin", "acoustic", "chill"],
            "context": "noche tranquila, musica suave para relajarse",
        },
    },
    {
        "id": 5,
        "label": "Max context length",
        "description": "Context at exactly 200 characters — boundary accepts the value",
        "profile": {
            "name": "Boundary Context Profile",
            "energy": 0.5,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.5,
            "preferred_tags": ["pop"],
            "context": _200_CHAR_CONTEXT,
        },
    },
    {
        "id": 6,
        "label": "Minimal valid",
        "description": "Only required fields, no optional fields — context is None",
        "profile": {
            "name": "Minimal Profile",
            "energy": 0.5,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.5,
            "preferred_tags": ["indie"],
        },
    },
    # ── Normal diverse profiles ───────────────────────────────────────────────
    {
        "id": 7,
        "label": "Afrobeats",
        "description": "High energy, high danceability, afrobeats tags",
        "profile": {
            "name": "Afrobeats Session",
            "energy": 0.88,
            "valence": 0.82,
            "danceability": 0.92,
            "acousticness": 0.12,
            "preferred_tags": ["afrobeats", "dance", "african", "pop"],
            "context": "High-energy party playlist.",
        },
    },
    {
        "id": 8,
        "label": "Ambient",
        "description": "Low energy, high acousticness, ambient tags",
        "profile": {
            "name": "Late Night Ambient",
            "energy": 0.18,
            "valence": 0.42,
            "danceability": 0.14,
            "acousticness": 0.88,
            "preferred_tags": ["ambient", "chill", "meditation", "acoustic"],
            "context": "Wind-down session. Need quiet.",
        },
    },
    {
        "id": 9,
        "label": "Jazz",
        "description": "High valence, moderate energy, jazz tags",
        "profile": {
            "name": "Sunday Jazz",
            "energy": 0.52,
            "valence": 0.78,
            "danceability": 0.42,
            "acousticness": 0.64,
            "preferred_tags": ["jazz", "soul", "blues", "classic"],
            "context": "Sunday morning coffee.",
        },
    },
    {
        "id": 10,
        "label": "Industrial",
        "description": "Low valence, high energy, industrial tags",
        "profile": {
            "name": "Industrial Hour",
            "energy": 0.91,
            "valence": 0.18,
            "danceability": 0.55,
            "acousticness": 0.08,
            "preferred_tags": ["industrial", "metal", "noise", "dark"],
            "context": "High-intensity focus session.",
        },
    },
]


# ─── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class HarnessResult:
    profile_id: int
    label: str
    songs_returned: int = 0
    confidence: float = 0.0
    loops: int = 0
    lastfm_count: int = 0
    radio_count: int = 0
    passed: bool = False
    notes: str = ""
    exception: Optional[str] = None


# ─── Runner ────────────────────────────────────────────────────────────────────

def _source_mix(trajectory: list) -> tuple[int, int]:
    """Return (lastfm_count, radio_count) from a final trajectory."""
    lastfm = sum(1 for e in trajectory if e.scored_song.song.source == "lastfm")
    radio = sum(1 for e in trajectory if e.scored_song.song.source == "radio")
    return lastfm, radio


def _run_one(entry: dict) -> HarnessResult:
    """
    Run a single profile through the full pipeline and return a HarnessResult.

    Catches ValidationError (profile 3), gatekeeper blocks, and any graph
    exception. Each case is recorded in the result rather than crashing the
    harness.
    """
    result = HarnessResult(profile_id=entry["id"], label=entry["label"])
    expect_validation_error = entry.get("expect_validation_error", False)

    # ── Pydantic validation ────────────────────────────────────────────────────
    try:
        profile = TasteProfile(**entry["profile"])
    except ValidationError as exc:
        if expect_validation_error:
            result.passed = True
            result.notes = "Correctly rejected by Pydantic validator before reaching graph"
            logger.info("harness · profile %d · validation error caught as expected", entry["id"])
        else:
            result.passed = False
            result.notes = f"Unexpected ValidationError: {exc}"
            result.exception = str(exc)
            logger.error("harness · profile %d · unexpected ValidationError: %s", entry["id"], exc)
        return result

    if expect_validation_error:
        result.passed = False
        result.notes = "Expected ValidationError but profile passed Pydantic — check validator"
        return result

    # ── Gatekeeper ────────────────────────────────────────────────────────────
    try:
        safe = gatekeeper_run(profile)
    except Exception as exc:
        result.passed = False
        result.notes = f"Gatekeeper exception: {exc}"
        result.exception = str(exc)
        return result

    if not safe:
        result.passed = False
        result.notes = "Blocked by Gatekeeper"
        return result

    # ── Graph ─────────────────────────────────────────────────────────────────
    initial_state: AgentState = {
        "taste_profile": profile,
        "raw_catalog": [],
        "scored_songs": [],
        "explained_songs": [],
        "critique_result": None,
        "final_trajectory": [],
        "confidence": 0.0,
        "loop_count": 0,
        "agent_log": [
            f"[{datetime.now().isoformat()}] harness · profile {entry['id']} · start"
        ],
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
    except SystemExit:
        # nodes/retrieve.py calls sys.exit(1) on catalog failures — treat as fail
        result.passed = False
        result.notes = "System exit during graph execution (catalog or source failure)"
        return result
    except Exception as exc:
        result.passed = False
        result.notes = f"Unhandled exception: {type(exc).__name__}"
        result.exception = str(exc)
        logger.error("harness · profile %d · exception: %s", entry["id"], exc)
        return result

    # ── Evaluate results ──────────────────────────────────────────────────────
    trajectory = final_state.get("final_trajectory", [])
    result.songs_returned = len(trajectory)
    result.confidence = final_state.get("confidence", 0.0)
    result.loops = final_state.get("loop_count", 0)
    result.lastfm_count, result.radio_count = _source_mix(trajectory)

    # Check for empty explanations (fail criterion)
    empty_explanations = any(
        not e.explanation.strip() for e in trajectory
    )

    source_mix_ok = result.lastfm_count >= 1 and result.radio_count >= 1
    songs_ok = result.songs_returned == 5
    confidence_ok = result.confidence >= 0.7 or result.loops >= 3

    if empty_explanations:
        result.passed = False
        result.notes = "Empty explanation on one or more ExplainedSong"
    elif not songs_ok:
        result.passed = False
        result.notes = f"Only {result.songs_returned} songs returned"
    elif not confidence_ok:
        result.passed = False
        result.notes = f"Low confidence ({result.confidence:.2f}) without reaching loop ceiling"
    else:
        result.passed = True
        if not source_mix_ok:
            result.notes = f"Pass — source mix incomplete (lastfm:{result.lastfm_count} radio:{result.radio_count})"
        else:
            result.notes = "Pass"

    return result


# ─── Table rendering ──────────────────────────────────────────────────────────

def _render_table(results: list[HarnessResult]) -> None:
    """Print the harness results as a Rich table."""
    table = Table(
        title="Eval Harness Results — Music Recommender: Music Theory",
        show_header=True,
        header_style="bold white",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Profile Name", style="white")
    table.add_column("Source Mix", justify="center")
    table.add_column("Songs", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Loops", justify="right")
    table.add_column("Pass/Fail", justify="center")
    table.add_column("Notes", style="dim")

    for r in results:
        pass_str = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        source_str = f"LF:{r.lastfm_count} RB:{r.radio_count}"
        table.add_row(
            str(r.profile_id),
            r.label,
            source_str,
            str(r.songs_returned),
            f"{r.confidence:.2f}",
            str(r.loops),
            pass_str,
            r.notes[:60],
        )

    console.print(table)


def _render_summary(results: list[HarnessResult]) -> None:
    """Print the summary line after the table."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_conf = (
        sum(r.confidence for r in results if r.songs_returned > 0)
        / max(1, sum(1 for r in results if r.songs_returned > 0))
    )
    source_mix_runs = sum(
        1 for r in results if r.lastfm_count >= 1 and r.radio_count >= 1
    )

    console.print(
        Panel(
            f"[bold]{passed} of {total} profiles passed.[/bold]\n"
            f"Average confidence: [cyan]{avg_conf:.2f}[/cyan]\n"
            f"Source mix achieved in [green]{source_mix_runs}[/green] of {total} runs.",
            title="[white]Harness Summary[/white]",
            border_style="white",
        )
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run all 10 profiles and print the results table."""
    console.print(Rule("[bold white]Music Theory — Eval Harness[/bold white]"))
    console.print(f"[dim]Running {len(PROFILES)} profiles...[/dim]\n")

    results: list[HarnessResult] = []

    for entry in PROFILES:
        console.print(
            f"[dim]→ Profile {entry['id']:02d} / {len(PROFILES)}: "
            f"{entry['label']} — {entry['description']}[/dim]"
        )
        result = _run_one(entry)
        results.append(result)
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"  {status}  {result.notes[:80]}\n")

    console.print(Rule())
    _render_table(results)
    _render_summary(results)


if __name__ == "__main__":
    main()
