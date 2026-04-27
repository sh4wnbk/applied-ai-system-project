"""
Shared Anthropic client and model constants for Music Recommender: Music Theory.

All LLM-enabled nodes import from here so the client is instantiated once
and model strings are defined in a single place. Changing a model assignment
requires editing one line, not hunting across node files.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Single shared client — reads ANTHROPIC_API_KEY from .env at startup.
# The personal API key stored in .env is separate from the Codepath Claude
# Code account used during development. These must never be mixed.
client = Anthropic()

# Model constants — assigned per-node based on task complexity.
# Haiku handles structured extraction tasks where speed matters.
# Sonnet handles Glass Box explanation where reasoning depth matters.
HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-4-6"
