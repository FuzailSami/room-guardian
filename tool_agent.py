#!/usr/bin/env python3
"""
Tool-calling agent for Room Guardian (small language model + agentic AI for IoT).

LangGraph flow:
    confirm? ──yes/no to a pending action──────────────────────────────┐
       │                                                                ▼
    choose (SLM picks tools) ─► guard (allow-list) ─► run (tools) ─► respond ─► END
       └─ no tool chosen ─► respond (refuse a command, or hand back for the rule-based fallback)

Safety
- The model only *proposes* tool calls. guard() checks every call against the tool's schema:
  unknown tools, unknown values or out-of-range numbers are refused and logged.
- Actions that lower security (disarming, snoozing) are not run until the user replies "yes".
- At most MAX_CALLS tool calls per question; only one action needs confirming at a time.
- dry_run=True validates actions without carrying them out (used by eval_agent.py).

Memory: the last few questions and answers per session ("web", "phone") are sent with each
request, so follow-ups like "what about yesterday?" work.

Every request is appended to an audit log (JSON lines) with the calls, decisions and timings.
"""

import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, TypedDict

import requests
from langgraph.graph import END, START, StateGraph

MAX_CALLS = 3
YES = re.compile(r"^\s*(y|yes|yeah|yep|confirm|confirmed|do it|ok|okay|sure)\W*$", re.I)
NO = re.compile(r"^\s*(n|no|nope|cancel|stop|never ?mind|don'?t)\W*$", re.I)


@dataclass
class Tool:
    name: str
    description: str
    run: Callable  # run(args, question) -> dict with any of: facts, plain, answer, events, image, period
    params: dict = field(default_factory=dict)  # JSON-schema properties
    required: tuple = ()
    action: bool = False  # changes something in the room (vs. only reads)
    confirm: Optional[Callable] = None  # confirm(args) -> True if the user must say "yes" first
    describe: Optional[Callable] = None  # describe(args) -> "switch to home mode (disarm)"
    label: str = ""  # short name for the list of allowed actions, e.g. "sound the buzzer"

    def schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": list(self.required)}}}

    def what(self, args):
        if self.describe:
            return self.describe(args)
        return self.name.replace("_", " ") + (f" ({', '.join(f'{k}={v}' for k, v in args.items())})" if args else "")


def validate(tool, args):
    """Allow-list check. Returns (clean_args, None) or (None, reason). Extra keys are dropped."""
    if not isinstance(args, dict):
        return None, "arguments were not an object"
    clean = {}
    for key, spec in tool.params.items():
        if key not in args or args[key] in (None, ""):
            if key in tool.required:
                return None, f"missing {key}"
            continue
        value = args[key]
        if "enum" in spec:
            match = next((e for e in spec["enum"] if str(e).lower() == str(value).strip().lower()), None)
            if match is None:
                return None, f"{key}={value!r} is not allowed (allowed: {', '.join(map(str, spec['enum']))})"
            value = match
        elif spec.get("type") == "integer":
            try:
                value = int(float(str(value).strip()))
            except ValueError:
                return None, f"{key}={value!r} is not a number"
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                return None, f"{key}={value} is outside {lo}-{hi}"
        elif spec.get("type") == "string":
            value = str(value)[:200]
        clean[key] = value
    return clean, None


def parse_calls(message):
    """Tool calls from an Ollama /api/chat message. Small models sometimes write the call as JSON
    text instead of using tool_calls, so that is accepted too."""
    calls = []
    for c in message.get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"_raw": args}
        calls.append((str(fn.get("name", "")), args))
    if calls:
        return calls
    text = (message.get("content") or "").strip()
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return []
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and "name" in item:
                calls.append((str(item["name"]), item.get("parameters") or item.get("arguments") or {}))
    return calls


class Sessions:
    """Short conversation memory + one pending action per session."""

    def __init__(self, turns=3, pending_seconds=120):
        self.turns, self.pending_seconds = turns, pending_seconds
        self.data = {}
        self.lock = threading.Lock()

    def get(self, sid):
        with self.lock:
            return self.data.setdefault(sid, {"history": deque(maxlen=self.turns), "pending": None})

    def pending(self, sid):
        p = self.get(sid)["pending"]
        if p and time.time() - p["time"] > self.pending_seconds:
            self.get(sid)["pending"] = None
            return None
        return p


class AgentState(TypedDict, total=False):
    question: str
    session: str
    dry_run: bool
    model: str
    calls: list  # [(name, args)] proposed by the model
    decisions: list  # [{tool, args, status, reason}]
    facts: list
    plain: list  # fact sentences used if the model can't write the answer
    answers: list
    events: list
    image: str
    period: str
    answer: str
    engine: str  # tools | none (no tool chosen -> caller falls back to rules)
    pending: dict
    timing: dict
    error: str


class ToolAgent:
    def __init__(self, tools, context, compose, server="http://localhost:11434", model="llama3.2:1b",
                 num_thread=2, log_path=None, command_pattern=None, timeout=180):
        self.tools = {t.name: t for t in tools}
        self.context = context  # () -> dict of live device state for the system prompt
        self.compose = compose  # (question, facts, summary, plain) -> answer text from the facts only
        self.server, self.model, self.num_thread, self.timeout = server, model, num_thread, timeout
        self.log_path = log_path
        self.command_pattern = command_pattern
        self.sessions = Sessions()
        self.graph = self._build()

    # --- model call ---
    def system_prompt(self):
        actions = ", ".join(t.name for t in self.tools.values() if t.action)
        return (
            "You are the assistant for Room Guardian, a home security camera on a Raspberry Pi.\n"
            f"Current state: {json.dumps(self.context(), default=str)}\n"
            "Call the tool(s) needed to answer the user or to do what they ask:\n"
            "- past activity, people, events, summaries -> get_events\n"
            "- temperature or humidity -> get_temperature\n"
            "- mode, snooze, buzzer or system status -> get_status\n"
            "- what is in the room right now -> look_at_room\n"
            f"- Only call an action ({actions}) when the user clearly asks for it.\n"
            "If no tool fits, answer in one short sentence without calling a tool."
        )

    def choose_calls(self, question, history, model):
        messages = [{"role": "system", "content": self.system_prompt()}]
        for q, a in history:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": question})
        resp = requests.post(f"{self.server}/api/chat", json={
            "model": model, "messages": messages, "stream": False, "keep_alive": "30m",
            "tools": [t.schema() for t in self.tools.values()],
            "options": {"temperature": 0, "num_thread": self.num_thread, "num_predict": 128, "num_ctx": 2048},
        }, timeout=(3, self.timeout))
        resp.raise_for_status()
        message = resp.json().get("message", {})
        return parse_calls(message), (message.get("content") or "").strip()

    # --- graph nodes ---
    def n_confirm(self, s: AgentState):
        pending = self.sessions.pending(s["session"])
        self.sessions.get(s["session"])["pending"] = None
        if YES.match(s["question"]):
            return {"decisions": [{**pending, "status": "confirmed"}]}
        return {"decisions": [{**pending, "status": "cancelled"}], "answers": ["Okay, cancelled."]}

    def n_choose(self, s: AgentState):
        started = time.time()
        history = list(self.sessions.get(s["session"])["history"])
        try:
            calls, text = self.choose_calls(s["question"], history, s.get("model") or self.model)
        except Exception as e:  # model offline / bad response: let the caller fall back
            return {"calls": [], "engine": "none", "error": f"model failed: {e}",
                    "timing": {"choose_s": round(time.time() - started, 2)}}
        return {"calls": calls[:MAX_CALLS], "engine": "tools" if calls else "none",
                "answers": [text] if (not calls and text) else [],
                "timing": {"choose_s": round(time.time() - started, 2)}}

    def n_guard(self, s: AgentState):
        decisions, pending = [], None
        is_command = bool(self.command_pattern and self.command_pattern.search(s["question"]))
        if is_command and not any(getattr(self.tools.get(n), "action", False) for n, _ in s["calls"]):
            # e.g. "delete all events" answered with a read tool: refuse rather than pretend
            return {"decisions": [{"tool": n, "args": a, "status": "refused",
                                   "reason": "this is a command and no allowed action matches it"}
                                  for n, a in s["calls"]], "pending": {}}
        for name, args in s["calls"]:
            tool = self.tools.get(name)
            if tool is None:
                decisions.append({"tool": name, "args": args, "status": "refused",
                                  "reason": "not an allowed tool"})
                continue
            clean, error = validate(tool, args)
            if error:
                decisions.append({"tool": name, "args": args, "status": "refused", "reason": error})
            elif tool.action and tool.confirm and tool.confirm(clean):
                if pending is None:
                    pending = {"tool": name, "args": clean, "time": time.time(), "what": tool.what(clean)}
                    decisions.append({"tool": name, "args": clean, "status": "needs_confirmation"})
                else:
                    decisions.append({"tool": name, "args": clean, "status": "refused",
                                      "reason": "only one action can wait for confirmation"})
            else:
                decisions.append({"tool": name, "args": clean, "status": "approved"})
        if pending:
            self.sessions.get(s["session"])["pending"] = pending
        return {"decisions": decisions, "pending": pending or {}}

    def n_run(self, s: AgentState):
        started = time.time()
        out = {"facts": [], "plain": [], "answers": list(s.get("answers") or []), "events": [], "image": "",
               "period": ""}
        decisions = []
        for d in s["decisions"]:
            if d["status"] not in ("approved", "confirmed"):
                decisions.append(d)
                continue
            tool = self.tools[d["tool"]]
            if tool.action and s.get("dry_run"):
                decisions.append({**d, "status": "dry_run"})
                out["answers"].append(f"(Test mode) Would {tool.what(d['args'])}.")
                continue
            try:
                result = tool.run(d["args"], s["question"]) or {}
            except Exception as e:
                decisions.append({**d, "status": "failed", "reason": str(e)})
                out["answers"].append(f"Couldn't {tool.what(d['args'])}: {e}")
                continue
            decisions.append({**d, "status": "done"})
            out["facts"] += [result["facts"]] if result.get("facts") else []
            out["plain"] += [result["plain"]] if result.get("plain") else []
            out["answers"] += [result["answer"]] if result.get("answer") else []
            out["events"] += result.get("events") or []
            out["image"] = out["image"] or result.get("image", "")
            out["period"] = out["period"] or result.get("period", "")
        return {**out, "decisions": decisions,
                "timing": {**s.get("timing", {}), "tools_s": round(time.time() - started, 2)}}

    def n_respond(self, s: AgentState):
        started = time.time()
        decisions = s.get("decisions") or []
        answers = list(s.get("answers") or [])
        if s.get("pending"):
            answers.append(f"Please confirm: {s['pending']['what']}? Reply yes to go ahead, or no to cancel.")
        refused = [d for d in decisions if d["status"] == "refused"]
        if refused and not any(d["status"] in ("done", "dry_run", "needs_confirmation") for d in decisions):
            answers.append(f"I can't do that ({refused[0]['reason']}). Actions I'm allowed to take: "
                           f"{self.allowed_text()}.")
        if s.get("facts"):  # read tools ran: the SLM writes the answer from their facts only
            summary = bool(re.search(r"\b(summar\w*|recap|overview|report)\b", s["question"], re.I))
            answers.insert(0, self.compose(s["question"], s["facts"], summary, s.get("plain") or []))
        engine = s.get("engine") or ("tools" if decisions else "none")
        if engine == "none" and not decisions:
            if self.command_pattern and self.command_pattern.search(s["question"]):
                answers = [f"I can't do that. Actions I'm allowed to take: {self.allowed_text()}."]
                engine = "tools"
            else:
                answers = []  # caller answers with the rule-based agent instead
        answer = " ".join(a for a in answers if a).strip()
        return {"answer": answer, "engine": engine,
                "timing": {**s.get("timing", {}), "respond_s": round(time.time() - started, 2)}}

    def allowed_text(self):
        return ", ".join(t.label or t.name for t in self.tools.values() if t.action)

    def remember(self, session, question, answer):
        self.sessions.get(session)["history"].append((question, answer[:300]))

    # --- graph ---
    def _build(self):
        g = StateGraph(AgentState)
        for name, fn in [("confirm", self.n_confirm), ("choose", self.n_choose), ("guard", self.n_guard),
                         ("run", self.n_run), ("respond", self.n_respond)]:
            g.add_node(name, fn)

        def start(s):
            if self.sessions.pending(s["session"]) and (YES.match(s["question"]) or NO.match(s["question"])):
                return "confirm"
            return "choose"

        g.add_conditional_edges(START, start, ["confirm", "choose"])
        g.add_edge("confirm", "run")
        g.add_conditional_edges("choose", lambda s: "guard" if s["calls"] else "respond", ["guard", "respond"])
        g.add_edge("guard", "run")
        g.add_edge("run", "respond")
        g.add_edge("respond", END)
        return g.compile()

    def ask(self, question, session="web", dry_run=False, model=None):
        started = time.time()
        if (YES.match(question) or NO.match(question)) and not self.sessions.pending(session):
            result = {"answer": "There's nothing waiting for confirmation.", "engine": "tools", "tool_calls": [],
                      "pending": False, "events": [], "image": "", "period": "", "model": model or self.model,
                      "timing": {"total_s": round(time.time() - started, 2)}, "error": ""}
            self.audit(question, session, dry_run, result)
            return result
        s = self.graph.invoke({"question": question, "session": session, "dry_run": dry_run,
                               "model": model or self.model, "calls": [], "decisions": [], "facts": [], "plain": [],
                               "answers": [], "events": [], "image": "", "period": "", "timing": {}})
        seconds = round(time.time() - started, 2)
        result = {"answer": s.get("answer", ""), "engine": s.get("engine", "none"),
                  "tool_calls": [{k: v for k, v in d.items() if k != "time"} for d in s.get("decisions", [])],
                  "pending": bool(s.get("pending")), "events": s.get("events", []), "image": s.get("image", ""),
                  "period": s.get("period", ""), "model": model or self.model,
                  "timing": {**s.get("timing", {}), "total_s": seconds}, "error": s.get("error", "")}
        if result["engine"] == "tools" and result["answer"]:
            self.remember(session, question, result["answer"])
        self.audit(question, session, dry_run, result)
        return result

    def audit(self, question, session, dry_run, result):
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "session": session,
                                    "question": question, "dry_run": dry_run, "engine": result["engine"],
                                    "model": result["model"], "calls": result["tool_calls"],
                                    "answer": result["answer"][:300], "timing": result["timing"],
                                    "error": result["error"]}, default=str) + "\n")
        except OSError:
            pass
