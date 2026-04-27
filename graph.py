"""
LangGraph state machine for Music Recommender: Music Theory.

Defines the six-node agentic graph and the conditional routing logic that
implements the critique loop. The graph is compiled once at import time and
reused across calls — compilation is the expensive step in LangGraph.

Node order:
  retrieve → score → explain → critique → (loop back to retrieve OR) rerank → END
"""

import logging

from langgraph.graph import END, StateGraph

from models import AgentState
from nodes.critique import critique
from nodes.explain import explain
from nodes.rerank import rerank
from nodes.retrieve import retrieve
from nodes.score import score

logger = logging.getLogger(__name__)


def _route_after_critique(state: AgentState) -> str:
    """
    Determine the next node after Hertz's critique.

    Returns "retrieve" to loop back for a fresh catalog pass when Hertz's
    confidence is below threshold and the loop ceiling has not been reached.
    Returns "rerank" in all other cases — ceiling reached, approved, or
    critique result absent.
    """
    critique_result = state.get("critique_result")
    if critique_result is None:
        logger.warning("graph · critique_result missing — routing to rerank")
        return "rerank"

    if critique_result.loop_back:
        loop_count = state.get("loop_count", 0)
        logger.info("graph · routing to retrieve · loop_count now %d", loop_count)
        return "retrieve"

    logger.info("graph · routing to rerank · approved or ceiling reached")
    return "rerank"


def build_graph() -> StateGraph:
    """
    Construct and compile the Music Theory recommendation graph.

    Each node is a pure function that accepts AgentState and returns an
    updated AgentState. LangGraph merges the returned dict with the existing
    state — nodes only need to return the fields they modify.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("score", score)
    graph.add_node("explain", explain)
    graph.add_node("critique", critique)
    graph.add_node("rerank", rerank)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "score")
    graph.add_edge("score", "explain")
    graph.add_edge("explain", "critique")

    # Conditional routing: Hertz decides whether to loop back or advance.
    graph.add_conditional_edges(
        "critique",
        _route_after_critique,
        {
            "retrieve": "retrieve",
            "rerank": "rerank",
        },
    )

    graph.add_edge("rerank", END)

    logger.info("graph · compiled successfully")
    return graph.compile()


# Module-level compiled graph — import this in main.py and eval/harness.py.
compiled_graph = build_graph()
