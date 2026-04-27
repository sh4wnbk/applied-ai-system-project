"""
Gatekeeper — input safety pre-flight for Music Recommender: Music Theory.

Runs before the LangGraph graph initializes. If the Gatekeeper flags or
errors, the graph never starts. This is the fail-closed contract: unmoderated
input does not reach the LLM nodes under any circumstances.

Threat mitigations applied here (from the threat model in HANDOFF.md):
  - Character substitution (leetspeak, symbol replacement)
  - Unicode homoglyphs (Cyrillic lookalikes, etc.)
  - Invisible and zero-width characters
  - RTL override characters
  - Base64-encoded payloads
  - Multilingual hate speech (delegated to Claude Haiku, 100+ languages)
"""

import base64
import logging
import os
import re
import unicodedata
from typing import Optional

import anthropic
from rich.console import Console
from rich.panel import Panel

from llm import HAIKU
from models import TasteProfile

logger = logging.getLogger(__name__)
console = Console()

# RTL and bidirectional override codepoints that can reverse displayed text.
_BIDI_OVERRIDES = {
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # LRE, RLE, PDF, LRO, RLO
    "\u2066", "\u2067", "\u2068", "\u2069",             # LRI, RLI, FSI, PDI
    "\u200f",                                            # Right-to-Left Mark
}

# Common leetspeak and symbol substitutions that evade keyword filters.
_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "+": "t",
})

_MODERATION_TOOL = {
    "name": "submit_moderation_result",
    "description": "Submit the moderation verdict for the provided text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "flagged": {
                "type": "boolean",
                "description": (
                    "True if the text contains harmful content that should be rejected."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Brief reason if flagged, empty string if not flagged.",
            },
        },
        "required": ["flagged", "reason"],
    },
}

_MODERATION_SYSTEM = (
    "You are a content moderation system for a music recommendation application. "
    "Evaluate the text inside <user_input> tags for harmful content including: "
    "hate speech, harassment, threats, sexual content involving minors, "
    "self-harm promotion, and violent extremism.\n\n"
    "Do NOT flag music genres, moods, or listener preferences even if they are "
    "intense — terms like 'metal', 'dark', 'aggressive', 'industrial', or 'noise' "
    "are acceptable music descriptors.\n\n"
    "Respond only by calling the submit_moderation_result tool."
)


def _strip_nonprintable(text: str) -> str:
    """Remove invisible, zero-width, and control characters from text."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def _reject_if_bidi(text: str) -> Optional[str]:
    """
    Return an error message if the text contains RTL override characters,
    otherwise return None.

    Bidirectional overrides can make harmful text appear safe in a terminal
    while the underlying bytes carry different content.
    """
    found = [ch for ch in text if ch in _BIDI_OVERRIDES]
    if found:
        return "Input contains bidirectional text control characters, which are not permitted."
    return None


def _try_decode_base64(text: str) -> str:
    """
    Attempt to decode base64 segments embedded in the text.

    Base64 decoding is best-effort: if a segment decodes to printable ASCII
    it is replaced with its plaintext form before moderation. Failures are
    silently ignored — the original segment is kept.
    """
    pattern = re.compile(r"[A-Za-z0-9+/]{8,}={0,2}")

    def try_replace(match: re.Match) -> str:
        token = match.group(0)
        try:
            decoded = base64.b64decode(token + "==").decode("utf-8", errors="strict")
            if decoded.isprintable():
                return decoded
        except Exception:
            pass
        return token

    return pattern.sub(try_replace, text)


def _normalize(text: str) -> str:
    """
    Apply the full normalization pipeline to a text field before moderation.

    Order matters: strip control chars first so later steps operate on clean
    input, then handle encoding tricks, then transliterate lookalikes.
    """
    text = _strip_nonprintable(text)
    text = _try_decode_base64(text)
    # Normalize Unicode to NFKD then re-encode to ASCII, replacing unmappable
    # characters. This collapses Cyrillic lookalikes to their Latin equivalents.
    text = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")
    text = text.translate(_LEET_MAP)
    return text.strip()


def _fields_to_check(profile: TasteProfile) -> list[tuple[str, str]]:
    """
    Return (field_name, normalized_value) pairs for all text fields in the profile.

    Float fields are excluded — they are bounded by Pydantic constraints and
    cannot carry text payloads. Each tag is checked individually.
    """
    fields = [("name", _normalize(profile.name))]
    if profile.context:
        fields.append(("context", _normalize(profile.context)))
    for i, tag in enumerate(profile.preferred_tags):
        fields.append((f"tag[{i}]", _normalize(tag)))
    return fields


def _moderate_field(client: anthropic.Anthropic, value: str) -> tuple[bool, str]:
    """
    Call Claude Haiku to evaluate a single normalized text field.

    Returns (flagged, reason). Raises on API errors so the caller can
    enforce the fail-closed contract.
    """
    response = client.messages.create(
        model=HAIKU,
        max_tokens=128,
        system=_MODERATION_SYSTEM,
        tools=[_MODERATION_TOOL],
        tool_choice={"type": "tool", "name": "submit_moderation_result"},
        messages=[
            {
                "role": "user",
                "content": f"<user_input>{value}</user_input>",
            }
        ],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_moderation_result":
            return block.input.get("flagged", False), block.input.get("reason", "")
    return False, ""


def run(profile: TasteProfile) -> bool:
    """
    Run the Gatekeeper pre-flight check against the provided TasteProfile.

    Returns True if the profile is clear and safe to pass to the graph.
    Returns False if the profile is flagged or if the moderation API is
    unavailable — the caller must not enter the graph in either case.

    Logs only the profile name on flag events, never the raw content.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("gatekeeper · ANTHROPIC_API_KEY not set — failing closed")
        _print_unavailable()
        return False

    fields = _fields_to_check(profile)

    # Check for RTL override characters before making any API call.
    for field_name, value in fields:
        bidi_error = _reject_if_bidi(value)
        if bidi_error:
            logger.warning(
                "gatekeeper · profile '%s' · %s · bidi character rejected",
                profile.name,
                field_name,
            )
            _print_flagged(bidi_error)
            return False

    try:
        client = anthropic.Anthropic(api_key=api_key)
        for field_name, value in fields:
            if not value:
                continue
            flagged, reason = _moderate_field(client, value)
            if flagged:
                logger.warning(
                    "gatekeeper · profile '%s' · field '%s' · flagged by moderation",
                    profile.name,
                    field_name,
                )
                _print_flagged(
                    "The system detected content that cannot be processed. "
                    "Please review your input and try again."
                )
                return False

        logger.info("gatekeeper · profile '%s' · all fields cleared moderation", profile.name)
        return True

    except Exception as exc:
        # Fail closed on any moderation API error — unmoderated input must
        # never reach the graph regardless of the failure reason.
        logger.error("gatekeeper · moderation API failure: %s", exc)
        _print_unavailable()
        return False


def _print_flagged(reason: str) -> None:
    """Display a plain-language refusal panel via Rich."""
    console.print(
        Panel(
            f"[bold red]Input not accepted.[/bold red]\n\n{reason}\n\n"
            "If you believe this is an error, please revise your input and try again.",
            title="[red]Gatekeeper[/red]",
            border_style="red",
        )
    )


def _print_unavailable() -> None:
    """Display a plain-language service-unavailable panel via Rich."""
    console.print(
        Panel(
            "[bold yellow]Input safety verification is currently unavailable.[/bold yellow]\n\n"
            "The system cannot process this request until the safety check can be completed.\n"
            "Please try again in a moment.",
            title="[yellow]Gatekeeper[/yellow]",
            border_style="yellow",
        )
    )
