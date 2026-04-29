"""
Misty — Retrieve node for Music Recommender: Music Theory.

Misty calls Last.fm and Radio Browser simultaneously using a thread pool,
merges both catalogs into a single list of SongFeature objects, and writes
the result to AgentState. She is the system's ears — if she hears nothing,
no recommendation can be made.

Character: Misty / Neumann U87 microphone
Model: claude-haiku-4-5 (used for tag normalisation if needed in future;
       the retrieval itself is deterministic HTTP — no LLM call in this node)
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from models import AgentState, SongFeature
from sources.lastfm import fetch_songs
from sources.radiobrowser import fetch_stations

logger = logging.getLogger(__name__)
console = Console()

# Wall-clock timeout for each source fetch in seconds.
# Both run in parallel so the effective wait is max(lastfm, radio), not their sum.
_SOURCE_TIMEOUT = 20


def retrieve(state: AgentState) -> AgentState:
    """
    Fetch songs from Last.fm and radio stations from Radio Browser simultaneously.

    Both sources run in a thread pool so neither blocks the other. If one source
    times out or errors, the system continues with whatever the other source
    returned. If both fail, the graph exits cleanly with a plain-language message.

    Writes raw_catalog and an agent_log entry to AgentState.
    """
    profile = state["taste_profile"]
    tags = profile.preferred_tags

    _print_misty_panel(tags)
    logger.info("misty · node fired · tags: %s", tags)

    lastfm_songs: list[SongFeature] = []
    radio_songs: list[SongFeature] = []
    lastfm_ok = False
    radio_ok = False

    with ThreadPoolExecutor(max_workers=2) as pool:
        lastfm_future = pool.submit(fetch_songs, tags)
        radio_future = pool.submit(fetch_stations, tags)

        try:
            for future in as_completed([lastfm_future, radio_future], timeout=_SOURCE_TIMEOUT + 5):
                if future is lastfm_future:
                    try:
                        lastfm_songs = lastfm_future.result(timeout=_SOURCE_TIMEOUT)
                        lastfm_ok = True
                        logger.info("misty · last.fm · %d songs retrieved", len(lastfm_songs))
                    except Exception as exc:
                        logger.warning("misty · last.fm · failed: %s", exc)
                        console.print(
                            "[yellow]Last.fm is currently unavailable. "
                            "Recommendations based on Radio Browser only.[/yellow]"
                        )
                elif future is radio_future:
                    try:
                        radio_songs = radio_future.result(timeout=_SOURCE_TIMEOUT)
                        radio_ok = True
                        logger.info("misty · radio browser · %d stations retrieved", len(radio_songs))
                    except Exception as exc:
                        logger.warning("misty · radio browser · failed: %s", exc)
                        console.print(
                            "[yellow]Radio Browser is currently unavailable. "
                            "Recommendations based on Last.fm only.[/yellow]"
                        )
        except FuturesTimeout:
            if not lastfm_ok:
                logger.warning("misty · last.fm · timed out")
                console.print(
                    "[yellow]Last.fm timed out. "
                    "Recommendations based on Radio Browser only.[/yellow]"
                )
            if not radio_ok:
                logger.warning("misty · radio browser · timed out")
                console.print(
                    "[yellow]Radio Browser timed out. "
                    "Recommendations based on Last.fm only.[/yellow]"
                )

    if not lastfm_ok and not radio_ok:
        console.print(
            Panel(
                "[bold red]All data sources are currently unavailable.[/bold red]\n\n"
                "The system cannot generate recommendations. Please try again.",
                title="[red]Misty[/red]",
                border_style="red",
            )
        )
        logger.error("misty · both sources failed — exiting")
        sys.exit(1)

    catalog = lastfm_songs + radio_songs

    if len(catalog) < 5:
        console.print(
            Panel(
                f"[bold yellow]Only {len(catalog)} song(s) retrieved after filtering.[/bold yellow]\n\n"
                "The system requires at least 5 songs to produce a reliable trajectory. "
                "Try different tags or check that your API keys are valid.",
                title="[yellow]Misty[/yellow]",
                border_style="yellow",
            )
        )
        logger.warning("misty · catalog too small (%d songs) — exiting", len(catalog))
        sys.exit(1)

    _print_catalog_summary(lastfm_songs, radio_songs)
    logger.info("misty · merged catalog · %d total songs", len(catalog))

    log_entry = (
        f"[{datetime.now().isoformat()}] misty · retrieved {len(lastfm_songs)} from last.fm, "
        f"{len(radio_songs)} from radio browser · total: {len(catalog)}"
    )

    return {
        **state,
        "raw_catalog": catalog,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def _print_misty_panel(tags: list[str]) -> None:
    """Render Misty's character panel showing her internal monologue."""
    tag_str = ", ".join(tags)
    console.print(
        Panel(
            f"[bold cyan]Misty[/bold cyan]\n\n"
            f"[italic]Scanning the airwaves for your sound...[/italic]\n\n"
            f"Tags in scope: [green]{tag_str}[/green]\n"
            f"Calling Last.fm and Radio Browser simultaneously.",
            title="[cyan]— Retrieve —[/cyan]",
            border_style="cyan",
        )
    )


def _print_catalog_summary(lastfm: list[SongFeature], radio: list[SongFeature]) -> None:
    """Print a Rich table showing how many songs each source contributed."""
    table = Table(title="Catalog Retrieved", show_header=True, header_style="bold cyan")
    table.add_column("Source", style="cyan")
    table.add_column("Songs", justify="right")
    table.add_row("Last.fm", str(len(lastfm)))
    table.add_row("Radio Browser", str(len(radio)))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(lastfm) + len(radio)}[/bold]")
    console.print(table)
