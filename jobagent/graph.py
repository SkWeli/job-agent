"""The agent as a LangGraph state machine.

                ┌──────────────────────────────────────────────┐
                ▼                                              │
  START ──► agent ──(no tool calls)──────────────────► finish ──► END
              │                                          ▲
              ├──(budget hit)──► wrap_up ────────────────┘
              ├──(needs approval)──► approval ──► tools ─┐
              └──(tool calls)──────────────────► tools ──┤
                                                         ├──(too many errors)──► wrap_up
                                                         ├──(context too long)──► summarize ──► agent
                                                         └──────────────────────────────────────► agent

agent     : the model decides what to do next (call tools, or answer)
tools     : runs the tool calls with guardrails: validation, loop guard, error capture, truncation
approval  : pauses the graph (interrupt) until a human approves irreversible calls
summarize : compresses old messages into `summary` to keep the context small
wrap_up   : out of budget: one last model call, without tools, to write the report
finish    : saves the report and records why the run stopped
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import ValidationError

from .drafting import build_drafter
from .export import write_report
from .memory import Notes, render_for_summary, split_point
from .prompts import SUMMARY_PROMPT, render_system
from .state import AgentState
from .store import Tracker
from .tools import ToolContext, ToolError, build_tools
from .tracing import Tracer

DEFAULT_TASK = ("Run today's job search: search the configured sources for the candidate's searches, assess "
                "the promising new jobs, draft applications for the best ones, then report.")


def _fmt_validation(e: ValidationError, tool: str) -> str:
    probs = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'input'}: {x['msg']}" for x in e.errors()[:5])
    return f"ERROR: invalid arguments for {tool}: {probs}. Fix the arguments and call again."


def build_graph(cfg, model, ctx: ToolContext, tracer: Tracer, notes_text: str, checkpointer=None):
    lim = cfg.limits
    tools = build_tools(ctx)
    by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # ------------------------------------------------------------------ nodes
    def agent(state: AgentState):
        system = render_system(cfg, state.get("task") or DEFAULT_TASK, notes_text)
        if state.get("summary"):  # one system message: some providers reject several
            system += f"\n\n# Summary of earlier steps in this run\n{state['summary']}"
        msgs = [SystemMessage(system)] + state["messages"]
        step = state.get("steps", 0) + 1
        with tracer.timer() as t:
            ai = model_with_tools.invoke(msgs)
        tracer.log("model_call", step=step, ms=t.ms, n_messages=len(msgs), usage=getattr(ai, "usage_metadata", None),
                   tool_calls=[{"name": c["name"], "args": c["args"]} for c in ai.tool_calls],
                   text=str(ai.content)[:2000])
        return {"messages": [ai], "steps": step}

    def approval(state: AgentState):
        """Human-in-the-loop. interrupt() saves the state to the checkpointer and stops the graph. The CLI shows
        the request, and `Command(resume=...)` restarts this node, with interrupt() returning the human's answer.
        The node holds no side effects before interrupt(), so re-running it on resume is safe."""
        ai: AIMessage = state["messages"][-1]
        pending = [c for c in ai.tool_calls if c["name"] in cfg.approval_required]
        answer = interrupt({"type": "approval", "calls": [{"id": c["id"], "tool": c["name"], "args": c["args"]}
                                                          for c in pending]})
        decisions = answer if isinstance(answer, dict) else {}
        for c in pending:
            tracer.log("approval", tool=c["name"], args=c["args"], decision=decisions.get(c["id"], {"approved": False}))
        return {"approvals": decisions}

    def tools_node(state: AgentState):
        ai: AIMessage = state["messages"][-1]
        counts = dict(state.get("call_counts") or {})
        approvals = state.get("approvals") or {}
        n_calls, errors = state.get("tool_calls", 0), state.get("consecutive_errors", 0)
        out = []
        for call in ai.tool_calls:
            name, args, cid = call["name"], call["args"], call["id"]
            sig = f"{name}|{json.dumps(args, sort_keys=True, default=str)}"
            result, is_error = None, False

            if name not in by_name:
                result, is_error = f"ERROR: no tool named '{name}'. Available: {', '.join(by_name)}.", True
            elif counts.get(sig, 0) >= lim["max_same_call"]:
                result, is_error = ("BLOCKED: you already made this exact call. The result won't change. "
                                    "Use what you have, try something different, or finish."), True
                tracer.log("guard", tool=name, args=args, reason="repeated identical call")
            elif name in cfg.approval_required and not (approvals.get(cid) or {}).get("approved"):
                why = (approvals.get(cid) or {}).get("reason") or "no reason given"
                result, is_error = (f"DECLINED by the human: {why}. Don't retry; mention it in your report."), True
                tracer.log("guard", tool=name, args=args, reason="human declined")

            if result is None:
                counts[sig] = counts.get(sig, 0) + 1
                with tracer.timer() as t:
                    try:
                        result = str(by_name[name].invoke(args))
                    except ValidationError as e:
                        result, is_error = _fmt_validation(e, name), True
                    except ToolError as e:
                        result, is_error = f"ERROR: {e}", True
                    except Exception as e:  # unexpected: show the model a short message, never a stack trace
                        result, is_error = (f"ERROR: {name} failed unexpectedly ({type(e).__name__}: "
                                            f"{str(e)[:150]}). Try a different approach."), True
                cap = lim["tool_result_chars"]
                if len(result) > cap:
                    result = result[:cap] + " …[truncated]"
                tracer.log("tool_call", tool=name, args=args, ms=t.ms, error=is_error, result=result)
            n_calls += 1
            errors = errors + 1 if is_error else 0
            out.append(ToolMessage(content=result, tool_call_id=cid, name=name, status="error" if is_error else "success"))
        return {"messages": out, "call_counts": counts, "tool_calls": n_calls, "consecutive_errors": errors,
                "approvals": {}}

    def summarize(state: AgentState):
        msgs = state["messages"]
        cut = split_point(msgs, lim["keep_recent_messages"])
        if cut <= 0:
            return {}
        old = msgs[:cut]
        prev = f"\nEarlier summary to merge in:\n{state['summary']}" if state.get("summary") else ""
        resp = model.invoke([SystemMessage(SUMMARY_PROMPT.format(previous=prev)),
                             HumanMessage(render_for_summary(old))])
        tracer.log("summarised", removed=len(old), summary=str(resp.content)[:1500])
        return {"summary": str(resp.content), "messages": [RemoveMessage(id=m.id) for m in old]}

    def wrap_up(state: AgentState):
        ai = state["messages"][-1]
        reason = _limit_reason(state) or "too_many_errors"
        msgs = []
        if isinstance(ai, AIMessage) and ai.tool_calls:  # answer pending calls so the history stays valid
            msgs += [ToolMessage(content=f"BLOCKED: run budget reached ({reason}).", tool_call_id=c["id"], name=c["name"])
                     for c in ai.tool_calls]
        msgs.append(HumanMessage(f"Stop now ({reason}). Write your final report from what you have. No tools."))
        system = render_system(cfg, state.get("task") or DEFAULT_TASK, notes_text)
        if state.get("summary"):
            system += f"\n\n# Summary of earlier steps in this run\n{state['summary']}"
        final = model.invoke([SystemMessage(system)] + state["messages"] + msgs)  # NOTE: no tools bound
        text = str(final.content).strip()
        if not text or getattr(final, "tool_calls", None):  # model ignored the instruction: fall back safely
            text = (f"## Run summary\nStopped early ({reason}) before a report was written. Saved progress: "
                    f"{ctx.db.counts()}. See `python -m jobagent list` and the trace.")
        tracer.log("model_call", step=state.get("steps", 0) + 1, wrap_up=True, text=text[:2000],
                   usage=getattr(final, "usage_metadata", None), tool_calls=[])
        return {"messages": msgs + [AIMessage(content=text)], "stop_reason": reason,
                "steps": state.get("steps", 0) + 1}

    def finish(state: AgentState):
        reason = state.get("stop_reason") or "done"
        text = str(state["messages"][-1].content)
        path = write_report(cfg.output_dir, tracer.run_id, text, reason)
        tracer.log("run_end", stop_reason=reason, steps=state.get("steps"), tool_calls=state.get("tool_calls"),
                   report=str(path))
        return {"stop_reason": reason}

    # ------------------------------------------------------------------ routing
    def _limit_reason(state) -> str | None:
        if state.get("steps", 0) >= lim["max_steps"]:
            return "max_steps"
        if state.get("tool_calls", 0) >= lim["max_tool_calls"]:
            return "max_tool_calls"
        if state.get("consecutive_errors", 0) >= lim["max_consecutive_errors"]:
            return "too_many_errors"
        return None

    def after_agent(state: AgentState) -> str:
        ai = state["messages"][-1]
        if not ai.tool_calls:
            return "finish"
        if _limit_reason(state):
            return "wrap_up"
        if any(c["name"] in cfg.approval_required for c in ai.tool_calls):
            return "approval"
        return "tools"

    def after_tools(state: AgentState) -> str:
        if _limit_reason(state):
            return "wrap_up"
        if len(state["messages"]) > lim["max_context_messages"]:
            return "summarize"
        return "agent"

    # ------------------------------------------------------------------ wiring
    retry = RetryPolicy(max_attempts=3, initial_interval=5.0)  # e.g. free-tier 429s
    g = StateGraph(AgentState)
    g.add_node("agent", agent, retry_policy=retry)
    g.add_node("approval", approval)
    g.add_node("tools", tools_node)
    g.add_node("summarize", summarize, retry_policy=retry)
    g.add_node("wrap_up", wrap_up, retry_policy=retry)
    g.add_node("finish", finish)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "approval", "wrap_up", "finish"])
    g.add_edge("approval", "tools")
    g.add_conditional_edges("tools", after_tools, ["agent", "summarize", "wrap_up"])
    g.add_edge("summarize", "agent")
    g.add_edge("wrap_up", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------- running it

def make_context(cfg, http, draft_model, tracer) -> ToolContext:
    drafter = build_drafter(draft_model) if draft_model is not None else None
    return ToolContext(cfg=cfg, db=Tracker(cfg.data_dir / "jobs.db"), http=http,
                       notes=Notes(cfg.data_dir / "notes.md"), tracer=tracer,
                       drafter=(lambda inp: drafter.invoke(inp)) if drafter else None)


def run_agent(cfg, task: str | None = None, *, model, draft_model=None, http, approver,
              thread_id: str | None = None, resume: bool = False) -> dict:
    """Run (or resume) one agent run. `approver(request) -> {call_id: {"approved": bool, "reason": str}}`."""
    thread_id = thread_id or uuid.uuid4().hex[:8]
    tracer = Tracer(cfg.root / "logs", thread_id)
    ctx = make_context(cfg, http, draft_model if draft_model is not None else model, tracer)
    notes_text = ctx.notes.read()
    conn = sqlite3.connect(cfg.data_dir / "checkpoints.db", check_same_thread=False)
    graph = build_graph(cfg, model, ctx, tracer, notes_text, checkpointer=SqliteSaver(conn))
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": cfg.limits["max_steps"] * 4 + 20}

    if resume:
        saved = graph.get_state(config).values
        if not saved:
            raise SystemExit(f"No saved run with id {thread_id}.")
        ctx.task_text = saved.get("task", "")
        tracer.log("run_resumed", next=list(graph.get_state(config).next))
        inp = None  # continue from the last checkpoint of this thread (re-asks any pending approval)
    else:
        ctx.task_text = task or DEFAULT_TASK
        tracer.log("run_start", task=task or DEFAULT_TASK, llm=cfg.llm, limits=cfg.limits)
        inp = {"task": task or DEFAULT_TASK, "messages": [HumanMessage(task or DEFAULT_TASK)], "steps": 0,
               "tool_calls": 0, "call_counts": {}, "consecutive_errors": 0}
    result = graph.invoke(inp, config)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        result = graph.invoke(Command(resume=approver(request)), config)

    final = result["messages"][-1]
    return {"thread_id": thread_id, "stop_reason": result.get("stop_reason"), "steps": result.get("steps"),
            "tool_calls": result.get("tool_calls"), "report": str(final.content), "log": str(tracer.path)}


def draw(cfg, model) -> str:
    """Mermaid diagram of the graph (paste into mermaid.live)."""
    from .tracing import NullTracer
    ctx = make_context(cfg, http=None, draft_model=None, tracer=NullTracer())
    return build_graph(cfg, model, ctx, NullTracer(), "").get_graph().draw_mermaid()
