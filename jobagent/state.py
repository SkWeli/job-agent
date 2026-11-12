"""The graph's state: what every node can read, and what it returns updates to.

Short-term memory is `messages` (the conversation, merged with the add_messages reducer).
`summary` holds older messages once they've been compressed. The counters feed the guardrails.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str                  # what the human asked for this run
    summary: str               # compressed history of older steps
    steps: int                 # model calls so far
    tool_calls: int            # tool executions so far
    call_counts: dict          # "tool|args" -> times called (loop detection)
    consecutive_errors: int
    approvals: dict            # tool_call_id -> {"approved": bool, "reason": str}, set by the approval node
    stop_reason: str           # set when the run ends: done | max_steps | max_tool_calls | too_many_errors
