"""
Pydantic v2 data models and LangGraph AgentState for Music Recommender: Music Theory.

Every model enforces field constraints at instantiation time so invalid data
never reaches the agent graph.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict


class SongFeature(BaseModel):
    """A single song retrieved from Last.fm or Radio Browser, with audio feature estimates."""

    title: str
    artist: str
    source: Literal["lastfm", "radio"]
    energy: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=0.0, le=1.0)
    danceability: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)
    tags: list[str]
    url: Optional[str] = None


class TasteProfile(BaseModel):
    """
    User-supplied preference vector and metadata.

    Float fields define the target position in the four-dimensional
    audio feature space. Text fields are checked by the Gatekeeper
    before the profile reaches the graph.
    """

    name: str = Field(max_length=100)
    energy: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=0.0, le=1.0)
    danceability: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)
    preferred_tags: list[str] = Field(max_length=10)
    context: Optional[str] = Field(default=None, max_length=200)

    @field_validator("preferred_tags", mode="before")
    @classmethod
    def tags_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError(
                "preferred_tags must contain at least one tag. "
                "Add one or more genre or mood tags to describe your taste."
            )
        return v

    @field_validator("preferred_tags", mode="before")
    @classmethod
    def tags_max_length(cls, v: list) -> list:
        for tag in v:
            if isinstance(tag, str) and len(tag) > 50:
                raise ValueError(
                    f"Each tag must be 50 characters or fewer. "
                    f"'{tag[:20]}...' exceeds that limit."
                )
        return v


class ScoredSong(BaseModel):
    """
    A song paired with its cosine similarity score against the TasteProfile vector.

    vector_breakdown shows how much each dimension contributed to the total score,
    making the ranking auditable without an LLM.
    """

    song: SongFeature
    similarity_score: float
    # per-dimension contribution, e.g. {"energy": 0.12, "valence": 0.08, ...}
    vector_breakdown: dict[str, float]


class ExplainedSong(BaseModel):
    """
    A scored song with a Glass Box natural-language explanation from Prestige.

    tag_overlap lists the tags the song shares with the TasteProfile.
    confidence measures how well the explanation matches the scoring evidence.
    """

    scored_song: ScoredSong
    explanation: str
    tag_overlap: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class CritiqueResult(BaseModel):
    """
    Hertz's evaluation of the current ExplainedSong list.

    loop_back is True when confidence < 0.7 AND loop_count < 3.
    Once loop_count reaches 3, loop_back is forced False regardless
    of confidence — the hard ceiling prevents infinite recursion.
    """

    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    loop_back: bool


class AgentState(TypedDict):
    """
    Shared mutable state threaded through every node in the LangGraph graph.

    Each node reads what it needs and writes only its own output fields.
    loop_count enforces the three-iteration ceiling on the critique loop.
    agent_log accumulates one entry per node firing for the audit trail.
    """

    taste_profile: TasteProfile
    raw_catalog: list[SongFeature]
    scored_songs: list[ScoredSong]
    explained_songs: list[ExplainedSong]
    critique_result: CritiqueResult
    final_trajectory: list[ExplainedSong]
    confidence: float
    loop_count: int
    agent_log: list[str]
