"""
Character rendering utilities for Music Recommender: Music Theory.

Each of the seven agent characters has a PNG asset in /assets. When a node
fires, the character's image is rendered inline via Kitty's icat protocol,
followed by a Rich panel with the character's name, instrument, and role.

Image rendering requires the Kitty terminal. On any other terminal the image
step is silently skipped — text output continues normally. This is documented
in the README setup instructions.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()

# Resolve assets directory relative to this file so the module works regardless
# of the working directory the user launches main.py from.
_ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Complete character roster. Each entry maps the canonical character name to
# display metadata. The asset filename for Prestige is prestige.png.
CHARACTERS: dict[str, dict[str, str]] = {
    "Cass": {
        "instrument": "Sony Walkman",
        "role": "Input + Output",
        "asset": "cass.png",
        "color": "red",
        "tagline": "First in, last out. Every session starts and ends with Cass.",
    },
    "Misty": {
        "instrument": "Neumann U87",
        "role": "Retrieve",
        "asset": "misty.png",
        "color": "cyan",
        "tagline": "Scanning the airwaves. If it exists, Misty will find it.",
    },
    "Tempo": {
        "instrument": "Metronome",
        "role": "Score",
        "asset": "tempo.png",
        "color": "magenta",
        "tagline": "Measuring distance in four dimensions. No guessing. Only math.",
    },
    "Prestige": {
        "instrument": "Technics SL-1200",
        "role": "Explain + RAG",
        "asset": "prestige.png",
        "color": "yellow",
        "tagline": "Opening the glass box. Every recommendation earns its place.",
    },
    "Hertz": {
        "instrument": "VU Meter",
        "role": "Critique",
        "asset": "hertz.png",
        "color": "green",
        "tagline": "Checking the signal. No weak explanations make it through.",
    },
    "Maestro": {
        "instrument": "Command Desk",
        "role": "Orchestrate + Rank",
        "asset": "maestro.png",
        "color": "blue",
        "tagline": "Five songs. One journey. Maestro arranges the set list.",
    },
    "Base": {
        "instrument": "Upright Bass",
        "role": "Narrator",
        "asset": "base.png",
        "color": "white",
        "tagline": "The voice beneath everything. Base sets the scene.",
    },
}


def _is_kitty() -> bool:
    """
    Return True if the current terminal is Kitty.

    Kitty sets TERM=xterm-kitty. Some embedded environments also set
    TERM_PROGRAM=kitty. Both are checked for compatibility.
    """
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")
    return "kitty" in term.lower() or "kitty" in term_program.lower()


def _kitty_available() -> bool:
    """Return True if the kitty binary is on PATH."""
    return shutil.which("kitty") is not None


def render_image(character_name: str) -> None:
    """
    Render a character's image inline using Kitty's icat protocol.

    Silently skips if:
      - The terminal is not Kitty
      - The kitty binary is not found
      - The asset file does not exist

    This function never raises — image rendering is a display enhancement,
    not a system requirement. Text output continues regardless.
    """
    char = CHARACTERS.get(character_name)
    if char is None:
        logger.warning("agents · unknown character: %s", character_name)
        return

    if not _is_kitty() or not _kitty_available():
        return

    asset_path = _ASSETS_DIR / char["asset"]
    if not asset_path.exists():
        logger.warning("agents · asset not found: %s", asset_path)
        return

    try:
        subprocess.run(
            ["kitty", "+kitten", "icat", str(asset_path)],
            check=False,
            timeout=5,
        )
    except Exception as exc:
        logger.debug("agents · icat render failed for %s: %s", character_name, exc)


def render_character_panel(character_name: str) -> None:
    """
    Render a character's image (Kitty only) followed by a Rich identity panel.

    Called at the start of each node to make every agent firing visible and
    auditable in the terminal. This satisfies the agentic observable intermediate
    steps requirement from the stretch rubric.
    """
    char = CHARACTERS.get(character_name)
    if char is None:
        logger.warning("agents · unknown character for panel: %s", character_name)
        return

    render_image(character_name)

    color = char["color"]
    console.print(
        Panel(
            Text.assemble(
                (f"{character_name}", f"bold {color}"),
            ),
            title=f"[{color}]— {char['role']} —[/{color}]",
            border_style=color,
        )
    )


def render_narrator_intro(profile_name: str) -> None:
    """
    Render Base's opening narration panel at the start of a session.

    Base does not correspond to a graph node — the narrator role exists to
    frame the session for the user before the graph begins.
    """
    render_image("Base")
    console.print(
        Panel(
            f"[bold]Base[/bold]\n\n"
            f"[italic]A new session begins.[/italic]\n\n"
            f"Profile: [cyan]{profile_name}[/cyan]\n"
            f"The graph is initializing. Seven agents are standing by.",
            title="[white]— Session Start —[/white]",
            border_style="white",
        )
    )


def render_session_end(song_count: int, confidence: float, loop_count: int) -> None:
    """
    Render Base's closing narration panel at the end of a session.

    Summarises the trajectory statistics so the user can see what the system
    produced at a glance before reading the full output.
    """
    render_image("Base")
    console.print(
        Panel(
            f"[bold]Base[/bold]\n\n"
            f"[italic]The session is complete.[/italic]\n\n"
            f"Songs delivered:   [green]{song_count}[/green]\n"
            f"Final confidence:  [cyan]{confidence:.2f}[/cyan]\n"
            f"Loop iterations:   [yellow]{loop_count}[/yellow]",
            title="[white]— Session End —[/white]",
            border_style="white",
        )
    )
