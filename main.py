"""
Music Recommender: Music Theory — CLI entry point.

Cass handles input and output. The session begins when a TasteProfile is
submitted, passes through the Gatekeeper, enters the LangGraph graph, and
ends when Cass renders the final five-song trajectory with Glass Box
explanations.

Usage:
    python main.py

To change the TasteProfile, edit the PROFILE constant below or pass
--profile <name> to select from the built-in examples.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

load_dotenv()

# Configure logging before any other imports that create loggers.
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(name)s · %(levelname)s · %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "session.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from display.agents import render_character_panel, render_narrator_intro, render_session_end
from display.gatekeeper import run as gatekeeper_run
from graph import compiled_graph
from models import AgentState, TasteProfile

console = Console()

# ─── Built-in example profiles ────────────────────────────────────────────────

EXAMPLE_PROFILES: dict[str, TasteProfile] = {
    "afrobeats": TasteProfile(
        name="Afrobeats Session",
        energy=0.85,
        valence=0.80,
        danceability=0.90,
        acousticness=0.15,
        preferred_tags=["afrobeats", "dance", "african", "pop"],
        context="High-energy party playlist for a Friday night gathering.",
        target_bpm=100.0,
    ),
    "ambient": TasteProfile(
        name="Late Night Ambient",
        energy=0.20,
        valence=0.45,
        danceability=0.15,
        acousticness=0.85,
        preferred_tags=["ambient", "chill", "acoustic", "meditation"],
        context="Wind-down session after a long day. Need something quiet.",
    ),
    "jazz": TasteProfile(
        name="Sunday Jazz",
        energy=0.50,
        valence=0.75,
        danceability=0.40,
        acousticness=0.65,
        preferred_tags=["jazz", "soul", "blues", "classic"],
        context="Sunday morning coffee. Relaxed but engaged.",
    ),
}

# Default profile when no --profile flag is passed.
DEFAULT_PROFILE = "afrobeats"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Music Recommender: Music Theory — Glass Box recommendation engine"
    )
    parser.add_argument(
        "--profile",
        choices=list(EXAMPLE_PROFILES.keys()),
        default=DEFAULT_PROFILE,
        help=f"Built-in taste profile to use (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--mastermix",
        action="store_true",
        default=False,
        help=(
            "Enable MasterMix mode: filter candidates within ±5 BPM of the profile's "
            "target_bpm before final trajectory selection. Requires target_bpm to be "
            "set in the selected profile or supplied via --bpm."
        ),
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        metavar="BPM",
        help=(
            "Override the profile's target BPM (0–300). "
            "Sets or replaces target_bpm on the selected profile."
        ),
    )
    return parser.parse_args()


def _print_cass_input(profile: TasteProfile, mastermix_mode: bool = False) -> None:
    """Render Cass's input panel displaying the active TasteProfile."""
    render_character_panel("Cass")
    tags = ", ".join(profile.preferred_tags)
    bpm_line = f"  Target BPM:   [yellow]{profile.target_bpm}[/yellow]\n" if profile.target_bpm is not None else ""
    mastermix_line = "  MasterMix:    [yellow]ON — ±5 BPM filter active[/yellow]\n" if mastermix_mode else ""
    console.print(
        Panel(
            f"Profile loaded: [cyan]{profile.name}[/cyan]\n\n"
            f"  Energy:       [magenta]{profile.energy}[/magenta]\n"
            f"  Valence:      [magenta]{profile.valence}[/magenta]\n"
            f"  Danceability: [magenta]{profile.danceability}[/magenta]\n"
            f"  Acousticness: [magenta]{profile.acousticness}[/magenta]\n"
            f"{bpm_line}"
            f"  Tags:         [green]{tags}[/green]\n"
            f"  Context:      [dim]{profile.context or 'none'}[/dim]\n"
            f"{mastermix_line}",
            title="[bright_white]— Cass · Input —[/bright_white]",
            border_style="bright_white",
        )
    )


def _print_cass_output(state: AgentState) -> None:
    """Render Cass's output panel with the full Glass Box trajectory."""
    render_character_panel("Cass")
    trajectory = state.get("final_trajectory", [])

    if not trajectory:
        console.print(
            Panel(
                "[red]No trajectory was produced.[/red]\n"
                "Check logs/session.log for details.",
                title="[red]Cass · Output[/red]",
                border_style="red",
            )
        )
        return

    console.print(Rule("[bright_white]Your Trajectory[/bright_white]"))

    for i, es in enumerate(trajectory, 1):
        song = es.scored_song.song
        bd = es.scored_song.vector_breakdown
        overlap_str = ", ".join(es.tag_overlap) if es.tag_overlap else "none"

        source_color = "cyan" if song.source == "lastfm" else ("yellow" if song.source == "melodata" else "blue")
        url_line = f"\n  Stream: [dim]{song.url}[/dim]" if song.url else ""

        bpm_line = f"\n  BPM:          [yellow]{song.bpm:.0f}[/yellow]" if song.bpm is not None else ""

        console.print(
            Panel(
                f"[bold white]{i}. {song.title}[/bold white]  "
                f"[dim]by {song.artist}[/dim]  "
                f"[{source_color}][{song.source}][/{source_color}]\n\n"
                f"[italic]{es.explanation}[/italic]\n\n"
                f"  Similarity:   [magenta]{es.scored_song.similarity_score:.4f}[/magenta]\n"
                f"  Energy:       {bd.get('energy', 0):.4f}  "
                f"Valence: {bd.get('valence', 0):.4f}  "
                f"Dance: {bd.get('danceability', 0):.4f}  "
                f"Acoustic: {bd.get('acousticness', 0):.4f}"
                f"{bpm_line}\n"
                f"  Tag overlap:  [green]{overlap_str}[/green]\n"
                f"  Confidence:   [cyan]{es.confidence:.2f}[/cyan]"
                f"{url_line}",
                title=f"[bright_white]Track {i}[/bright_white]",
                border_style="bright_white",
            )
        )


def run(profile: TasteProfile, mastermix_mode: bool = False) -> None:
    """
    Execute a full recommendation session for the given TasteProfile.

    Sequence:
      1. Log session start
      2. Gatekeeper pre-flight
      3. Cass input panel
      4. LangGraph graph execution
      5. Cass output panel
      6. Base session end narration
      7. Log session end
    """
    session_start = datetime.now().isoformat()
    logger.info(
        "session start · profile: %s · mastermix: %s · time: %s",
        profile.name,
        mastermix_mode,
        session_start,
    )

    # ── Gatekeeper pre-flight ──────────────────────────────────────────────────
    console.print(Rule("[dim]Gatekeeper Pre-Flight[/dim]"))
    safe = gatekeeper_run(profile)
    if not safe:
        logger.warning("session blocked by gatekeeper · profile: %s", profile.name)
        sys.exit(0)

    # ── Session open ───────────────────────────────────────────────────────────
    render_narrator_intro(profile.name)
    _print_cass_input(profile, mastermix_mode)

    # ── Graph execution ────────────────────────────────────────────────────────
    initial_state: AgentState = {
        "taste_profile": profile,
        "raw_catalog": [],
        "scored_songs": [],
        "explained_songs": [],
        "critique_result": None,
        "final_trajectory": [],
        "confidence": 0.0,
        "loop_count": 0,
        "mastermix_mode": mastermix_mode,
        "agent_log": [f"[{session_start}] session start · profile: {profile.name}"],
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as exc:
        exc_str = str(exc).lower()
        if "timeout" in exc_str or "timed out" in exc_str:
            console.print(
                Panel(
                    "[bold red]The reasoning engine timed out.[/bold red]\n\n"
                    "The session has been logged. Please try again.",
                    title="[red]Session Error[/red]",
                    border_style="red",
                )
            )
            logger.error("session error · timeout · %s", exc)
        elif "rate limit" in exc_str or "ratelimit" in exc_str or "429" in exc_str:
            console.print(
                Panel(
                    "[bold yellow]The reasoning engine is temporarily rate-limited.[/bold yellow]\n\n"
                    "Retry in 60 seconds.",
                    title="[yellow]Rate Limited[/yellow]",
                    border_style="yellow",
                )
            )
            logger.error("session error · rate limit · %s", exc)
        else:
            console.print(
                Panel(
                    f"[bold red]An unexpected error occurred.[/bold red]\n\n{exc}\n\n"
                    "The session has been logged. Please try again.",
                    title="[red]Session Error[/red]",
                    border_style="red",
                )
            )
            logger.error("session error · unexpected · %s", exc)
        sys.exit(1)

    # ── Output ─────────────────────────────────────────────────────────────────
    _print_cass_output(final_state)

    song_count = len(final_state.get("final_trajectory", []))
    confidence = final_state.get("confidence", 0.0)
    loop_count = final_state.get("loop_count", 0)

    render_session_end(song_count, confidence, loop_count)

    logger.info(
        "session end · profile: %s · songs: %d · confidence: %.2f · loops: %d",
        profile.name,
        song_count,
        confidence,
        loop_count,
    )


def main() -> None:
    """Parse arguments, select profile, and run the recommendation session."""
    args = _parse_args()
    profile = EXAMPLE_PROFILES[args.profile]

    if args.bpm is not None:
        if not (0.0 <= args.bpm <= 300.0):
            console.print(
                Panel(
                    "[bold red]--bpm must be between 0 and 300.[/bold red]",
                    title="[red]Invalid BPM[/red]",
                    border_style="red",
                )
            )
            sys.exit(1)
        profile = profile.model_copy(update={"target_bpm": args.bpm})

    if args.mastermix and profile.target_bpm is None:
        console.print(
            Panel(
                "[bold red]MasterMix mode requires a target BPM.[/bold red]\n\n"
                f"The [cyan]{args.profile}[/cyan] profile does not declare a "
                f"[cyan]target_bpm[/cyan]. Pass [cyan]--bpm <value>[/cyan] to set one.",
                title="[red]MasterMix Error[/red]",
                border_style="red",
            )
        )
        sys.exit(1)

    run(profile, mastermix_mode=args.mastermix)


if __name__ == "__main__":
    main()
