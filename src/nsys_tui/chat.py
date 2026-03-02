"""
chat.py - LLM chat for AI Profiler (Brain + Navigator).

Uses LiteLLM for model-agnostic completion with OpenAI-style tool calling.
Returns { "content": "...", "actions": [...] }; actions are executed by the frontend.
Model can be selected via NSYS_AI_MODEL (env), or per-request "model" in payload.
Text-to-SQL: query_profile_db tool (see tools_profile.py) is registered for agent use.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Callable

_log = logging.getLogger(__name__)
_telemetry_log = logging.getLogger("nsys_tui.telemetry")

from .tools_profile import (
    TOOL_QUERY_PROFILE_DB,
    open_profile_readonly,
    get_profile_schema_cached,
    query_profile_db,
)

# Cap total messages sent to the LLM per request; keeps token budget bounded.
MAX_AGENT_MESSAGES = 100
# Warn when prompt tokens exceed this threshold.
PROMPT_TOKEN_WARNING_THRESHOLD = 30_000
# Consecutive DB errors before injecting a break-cycle hint.
MAX_CONSECUTIVE_DB_ERRORS = 2

# Model options for UI and resolution. Anthropic 4.x, OpenAI 5.x, Gemini 2.5+ and 3.x only.
MODEL_OPTIONS = [
    # Anthropic (Claude 4.x)
    {"id": "anthropic/claude-opus-4-6-20260205", "label": "Claude Opus 4.6"},
    {"id": "anthropic/claude-sonnet-4-6", "label": "Claude Sonnet 4.6"},
    {"id": "anthropic/claude-sonnet-4-5-20250929", "label": "Claude Sonnet 4.5"},
    {"id": "anthropic/claude-opus-4-5-20251101", "label": "Claude Opus 4.5"},
    {"id": "anthropic/claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
    # OpenAI (GPT-5.x)
    {"id": "gpt-5.2", "label": "GPT-5.2"},
    {"id": "gpt-5.2-pro", "label": "GPT-5.2 Pro"},
    {"id": "gpt-5-mini", "label": "GPT-5 Mini"},
    {"id": "gpt-5.3-codex", "label": "GPT-5.3 Codex"},
    {"id": "gpt-4o", "label": "GPT-4o"},
    # Gemini (2.5+ and 3.x). LiteLLM/API may use -preview for some regions.
    {"id": "gemini/gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
    {"id": "gemini/gemini-2.5-pro-preview-05-20", "label": "Gemini 2.5 Pro (preview)"},
    {"id": "gemini/gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
    {"id": "gemini/gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite"},
    {"id": "gemini/gemini-2.0-flash", "label": "Gemini 2.0 Flash"},
    {"id": "gemini/gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro"},
    {"id": "gemini/gemini-3-pro-preview", "label": "Gemini 3 Pro"},
    {"id": "gemini/gemini-3-flash-preview", "label": "Gemini 3 Flash"},
]

# Map model id (or prefix) to env var for API key.
def _model_to_key(model_id: str) -> str | None:
    """Return the env var name for the given model id, or None if unknown."""
    if not model_id:
        return None
    if model_id.startswith("anthropic/"):
        return "ANTHROPIC_API_KEY"
    if model_id.startswith("openai/") or model_id.startswith("gpt-"):
        return "OPENAI_API_KEY"
    if model_id.startswith("gemini/"):
        return "GEMINI_API_KEY"
    return None


def _get_model_and_key(preferred_model: str | None = None) -> tuple[str | None, str | None]:
    """
    Return (model, api_key) for a configured provider, or (None, None).
    preferred_model: optional model id from request or NSYS_AI_MODEL env.
    """
    preferred = preferred_model or os.environ.get("NSYS_AI_MODEL")
    if preferred:
        key_name = _model_to_key(preferred)
        if key_name and os.environ.get(key_name):
            return preferred, os.environ[key_name]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-5-20250929", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-5.2", os.environ["OPENAI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-pro", os.environ["GEMINI_API_KEY"]
    return None, None


def get_available_models() -> list[dict]:
    """Return list of { id, label } for models that have an API key set."""
    out = []
    for opt in MODEL_OPTIONS:
        key_name = _model_to_key(opt["id"])
        if key_name and os.environ.get(key_name):
            out.append({"id": opt["id"], "label": opt["label"]})
    return out


def get_default_model() -> str | None:
    """Return the default model id (from NSYS_AI_MODEL or first available)."""
    if os.environ.get("NSYS_AI_MODEL"):
        key_name = _model_to_key(os.environ["NSYS_AI_MODEL"])
        if key_name and os.environ.get(key_name):
            return os.environ["NSYS_AI_MODEL"]
    models = get_available_models()
    return models[0]["id"] if models else None


def _tools_openai():
    """OpenAI-style tool definitions: navigate, zoom, and query_profile_db (,)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "navigate_to_kernel",
                "description": "Navigate the UI to a specific kernel. Match by EXACT kernel name from visible_kernels_summary or global_top_kernels. The frontend will resolve the exact occurrence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {"type": "string", "description": "The exact name of the kernel to navigate to."},
                        "occurrence_index": {"type": "integer", "description": "Which occurrence to jump to (1-based). Default is 1.", "default": 1},
                        "reason": {"type": "string", "description": "A short, 1-sentence reason why this kernel was chosen, to show the user."},
                    },
                    "required": ["target_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "zoom_to_time_range",
                "description": "Zoom the UI to a specific time range in seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_s": {"type": "number", "description": "Start time in seconds"},
                        "end_s": {"type": "number", "description": "End time in seconds"},
                    },
                    "required": ["start_s", "end_s"],
                },
            },
        },
        TOOL_QUERY_PROFILE_DB,
    ]


def _build_system_prompt(ui_context: dict, profile_schema: str | None = None) -> str:
    """System prompt with ui_context; optional profile_schema for query_profile_db (,)."""
    ctx_json = json.dumps(ui_context, separators=(",", ":"))
    schema_block = ""
    if profile_schema:
        schema_block = f"""
=== PROFILE DATABASE SCHEMA (for query_profile_db) ===
{profile_schema}
NOTE: Write strict SQLite3 SQL only (use strftime() not DATE_TRUNC/EXTRACT; use || for concatenation not CONCAT()).
=====================================================

"""
    return f"""You are an expert GPU performance analyst and UI navigator for an Nsight Systems viewer.
Your goal is to explain CUDA/GPU bottlenecks clearly and help users navigate the timeline.
{schema_block}=== CURRENT UI CONTEXT ===
```json
{ctx_json}
```
==========================

INSTRUCTIONS:
1. When asked to explain a kernel or bottleneck, use the provided context. Be concise, professional, and use Markdown for formatting.
2. If the user asks to go to, find, or locate a specific kernel or time range, YOU MUST use the provided tools (`navigate_to_kernel` or `zoom_to_time_range`).
3. When a PROFILE DATABASE SCHEMA is provided above, you MUST use the `query_profile_db` tool to answer whole-profile questions (e.g. first kernel, slowest kernel, counts, total GPU time, total kernel count). Run a SELECT; the backend returns the result and you answer from it. Never use `SELECT *`; always select only the columns you need. For total GPU time use SUM(duration_ns)/1e6; for kernel count use COUNT(*). Kernel names are stored as IDs referencing StringIds: join with StringIds (e.g. k.shortName = StringIds.id) and use StringIds.value for human-readable names. IMPORTANT: stats.total_gpu_ms, stats.total_kernel_count, and global_top_kernels are intentionally OMITTED from ui_context when the DB agent is enabled. You MUST use `query_profile_db` to answer any whole-profile questions - do NOT guess or say the data is missing.
4. TOOL USE RULES:
   - Match kernel names exactly from `visible_kernels_summary` or `global_top_kernels`.
   - Do NOT explain what you are about to do before calling a tool. Just call the tool.
   - For `navigate_to_kernel` and `zoom_to_time_range`: execution is immediate on the client; you do not wait for a result. For `query_profile_db`: the backend runs the query and returns rows; use them in your answer.
   - Do NOT output code blocks or JSON for navigation - use the actual tool call mechanism only.
5. If a requested kernel is not in the context, politely say it is not visible or does not exist."""


def _parse_tool_call(name: str, arguments: str) -> dict | None:
    """Graceful parsing: extract only needed fields. Return action dict or None."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return None
    if name == "navigate_to_kernel":
        target = args.get("target_name")
        if not target:
            return None
        return {
            "type": "navigate_to_kernel",
            "target_name": target,
            "occurrence_index": args.get("occurrence_index", 1),
            "reason": args.get("reason"),
        }
    if name == "zoom_to_time_range":
        start_s = args.get("start_s")
        end_s = args.get("end_s")
        if start_s is None or end_s is None:
            return None
        return {"type": "zoom_to_time_range", "start_s": float(start_s), "end_s": float(end_s)}
    return None


def _compact_old_tool_results(api_messages: list) -> None:
    """
    Replace large tool-result content from previous agent turns with compact summaries.

    Reduces prompt size when the model makes multiple DB queries per response. Tool results
    from all-but-the-last tool turn are replaced with "[Summary: DB query returned results.]"
    if they exceed 200 chars. The most recent turn's results are left intact so the model can
    use them for its final answer.
    """
    tool_turn_indices = [
        i for i, m in enumerate(api_messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    if len(tool_turn_indices) < 2:
        return  # Only one tool turn; nothing old yet.
    cutoff = tool_turn_indices[-1]
    for m in api_messages[:cutoff]:
        if m.get("role") == "tool" and len(m.get("content", "")) > 200:
            m["content"] = "[Summary: DB query returned results.]"


def distill_history(messages: list) -> list:
    """
    Compress intermediate tool call/result pairs from previous conversation turns
    into short summary messages, keeping final user/assistant messages intact.

    Strategy:
    - Keep system messages (role=system).
    - Keep all user messages.
    - Keep final assistant messages (those without tool_calls, i.e. the actual answers).
    - For each assistant message that has tool_calls + its following tool result messages:
      replace the entire sequence with a single system summary.

    Returns a new list (does not mutate the input).
    """
    if not messages:
        return messages

    result = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role", "")

        # Keep system and user messages as-is.
        if role in ("system", "user"):
            result.append(m)
            i += 1
            continue

        # Assistant message with tool_calls = intermediate turn → compress.
        if role == "assistant" and m.get("tool_calls"):
            tool_names = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                name = fn.get("name", "unknown")
                tool_names.append(name)
            # Skip over all following 'tool' role messages belonging to this turn.
            i += 1
            tool_count = 0
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_count += 1
                i += 1
            # Replace with summary.
            summary = f"[Agent called {', '.join(tool_names)} ({tool_count} result(s) consumed)]"
            result.append({"role": "system", "content": summary})
            continue

        # Regular assistant message (final answer) - keep it.
        result.append(m)
        i += 1

    return result


def run_agent_loop(
    model: str,
    api_messages: list,
    tools: list | None = None,
    query_runner: Callable[[str], str] | None = None,
    max_turns: int = 5,
) -> tuple[str, list]:
    """
    Run multi-turn agent loop until the model returns no tool_calls or max_turns is reached.
    Used by test_agent.py, Web, and TUI (,).
    query_runner(sql_query: str) -> str is used for query_profile_db; pass None to treat DB calls as no-op.
    Returns (final_content, actions) where actions are parsed navigate_to_kernel / zoom_to_time_range dicts.
    """
    try:
        import litellm
    except ImportError:
        return ("LLM not available (install litellm).", [])

    tools = tools if tools is not None else _tools_openai()
    actions: list = []
    consecutive_db_errors = 0

    for _ in range(max_turns):
        _compact_old_tool_results(api_messages)
        if len(api_messages) > MAX_AGENT_MESSAGES:
            api_messages[:] = [api_messages[0]] + api_messages[-(MAX_AGENT_MESSAGES - 1) :]
        response = litellm.completion(
            model=model,
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0] if response.choices else None
        if not choice:
            return ("", actions)
        message = choice.message
        if isinstance(message, dict):
            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []
        else:
            content = (getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return (content, actions)

        tc_list = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function") or {}
            tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
            name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
            args_str = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments") or "{}"
            tc_list.append((tc_id, name, args_str))

        assistant_msg = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": tid, "type": "function", "function": {"name": n, "arguments": a}}
                for tid, n, a in tc_list
            ],
        }
        api_messages.append(assistant_msg)

        has_external = False
        for tc_id, name, args_str in tc_list:
            if not name or not tc_id:
                continue
            action = _parse_tool_call(name, args_str)
            if action:
                # External navigation tool: collect action, exit after all tools processed.
                has_external = True
                actions.append(action)
            elif name == "query_profile_db":
                # DB tool: run if runner available, else report; continue loop so model sees result.
                if query_runner is not None:
                    try:
                        sql = json.loads(args_str).get("sql_query", "")
                        result = query_runner(sql)
                    except Exception as e:
                        result = f"Error: {e}"
                    if result.startswith("Error:"):
                        consecutive_db_errors += 1
                        if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                            result += "\n[System: Repeated SQL errors. Please answer from available context without further queries.]"
                    else:
                        consecutive_db_errors = 0
                else:
                    result = "Not executed (no profile loaded)."
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": result,
                })
            else:
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": "Not executed.",
                })

        if has_external:
            return (content, actions)

    return ("Max turns reached.", actions)


def chat_completion(body_bytes: bytes) -> dict | None:
    """
    Handle POST /api/chat body. Returns { "content": str, "actions": list } or None for 501.
    """
    try:
        import litellm
    except ImportError:
        return None

    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"content": "Invalid request body.", "actions": []}

    preferred_model = payload.get("model")
    model, _ = _get_model_and_key(preferred_model)
    if not model:
        return None

    messages = payload.get("messages") or []
    ui_context = payload.get("ui_context") or {}
    profile_path = payload.get("profile_path")

    # Feature flag: NSYS_AI_DB_AGENT controls whether the Web path uses the DB-backed
    # agent loop when profile_path is present (,).
    db_agent_flag = os.environ.get("NSYS_AI_DB_AGENT", "").strip().lower()
    db_agent_enabled = bool(db_agent_flag) and db_agent_flag not in ("0", "false", "no", "off")

    # When profile_path is provided AND DB agent is enabled, run full agent loop with
    # query_profile_db (,). Otherwise fall back to v1 behavior using only ui_context.
    if profile_path and db_agent_enabled:
        try:
            from .profile import resolve_profile_path
            sqlite_path = resolve_profile_path(profile_path)
        except RuntimeError as e:
            return {"content": f"Profile error: {e}", "actions": []}
        conn = open_profile_readonly(sqlite_path)
        try:
            schema_str = get_profile_schema_cached(conn, sqlite_path)
            system_prompt = _build_system_prompt(ui_context, profile_schema=schema_str)
            api_messages = [{"role": "system", "content": system_prompt}]
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role and content is not None:
                    api_messages.append({"role": role, "content": content})
            query_runner = lambda sql: query_profile_db(conn, sql)
            try:
                content, actions = run_agent_loop(
                    model=model,
                    api_messages=api_messages,
                    tools=_tools_openai(),
                    query_runner=query_runner,
                    max_turns=5,
                )
                return {"content": content, "actions": actions}
            except Exception as e:
                err_msg = str(e)
                print(f"[nsys-tui] Agent loop error (model={model!r}): {err_msg}", file=sys.stderr)
                if "429" in err_msg or "RateLimitError" in type(e).__name__ or "quota" in err_msg.lower():
                    friendly = "Quota exceeded (429). Try a different model or check API billing."
                else:
                    friendly = err_msg
                return {"content": f"LLM error: {friendly}", "actions": []}
        finally:
            conn.close()

    system_prompt = _build_system_prompt(ui_context)
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role and content is not None:
            api_messages.append({"role": role, "content": content})

    try:
        response = litellm.completion(
            model=model,
            messages=api_messages,
            tools=_tools_openai(),
            tool_choice="auto",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"[nsys-tui] LiteLLM error (model={model!r}): {err_msg}", file=sys.stderr)
        if "429" in err_msg or "RateLimitError" in type(e).__name__ or "quota" in err_msg.lower():
            friendly = "Quota exceeded (429). Try a different model (e.g. Gemini 2.5 Flash) or check billing: https://ai.google.dev/gemini-api/docs/rate-limits"
        else:
            friendly = err_msg
        return {"content": f"LLM error: {friendly}", "actions": []}

    choice = response.choices[0] if response.choices else None
    if not choice:
        return {"content": "", "actions": []}
    message = choice.message
    # Support both object and dict (e.g. different LiteLLM/provider shapes)
    if isinstance(message, dict):
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
    else:
        content = (getattr(message, "content", None) or "").strip()
        tool_calls = getattr(message, "tool_calls", None) or []
    actions = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function")
        fn = fn or {}
        name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
        args_str = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
        args_str = args_str or "{}"
        if name:
            action = _parse_tool_call(name, args_str)
            if action:
                actions.append(action)
    return {"content": content, "actions": actions}


def _sse_event(evt: str, data: dict) -> bytes:
    return f"event: {evt}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def stream_agent_loop(
    model: str,
    messages: list,
    ui_context: dict,
    tools: list | None = None,
    profile_path: str | None = None,
    max_turns: int = 5,
):
    """
    UI-agnostic streaming agent loop. Yields events: text, system, action, done.

    - Internal tool query_profile_db: runs in-loop when profile_path is set; appends
      tool messages and continues to next turn.
    - External tools (navigate_to_kernel, zoom_to_time_range): yields action then done.
    - Connection is opened inside this generator (same thread as caller), respecting (worker-owned connection when called from TUI worker).
    """
    try:
        import litellm
    except ImportError:
        yield {"type": "text", "content": "LLM not available (install litellm)."}
        yield {"type": "done", "usage": {}}
        return

    tools = tools if tools is not None else _tools_openai()
    conn = None
    query_runner = None
    if profile_path:
        try:
            from .profile import resolve_profile_path
            sqlite_path = resolve_profile_path(profile_path)
        except RuntimeError as e:
            yield {"type": "text", "content": f"Profile error: {e}"}
            yield {"type": "done", "usage": {}}
            return
        conn = open_profile_readonly(sqlite_path)
        try:
            schema_str = get_profile_schema_cached(conn, sqlite_path)
            query_runner = lambda sql: query_profile_db(conn, sql)
        except Exception:
            if conn:
                conn.close()
            raise
    else:
        schema_str = None

    system_prompt = _build_system_prompt(ui_context, profile_schema=schema_str)
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role and content is not None:
            api_messages.append({"role": role, "content": content})

    try:
        usage = {}
        consecutive_db_errors = 0
        turn_count = 0
        for _ in range(max_turns):
            turn_count += 1
            _compact_old_tool_results(api_messages)
            if len(api_messages) > MAX_AGENT_MESSAGES:
                api_messages[:] = [api_messages[0]] + api_messages[-(MAX_AGENT_MESSAGES - 1) :]
            try:
                stream = litellm.completion(
                    model=model,
                    messages=api_messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                )
            except Exception as e:
                err_msg = str(e)
                print(f"[nsys-tui] LiteLLM stream error (model={model!r}): {err_msg}", file=sys.stderr)
                friendly = "Quota exceeded (429). Try a different model or check billing." if "429" in err_msg or "quota" in err_msg.lower() else err_msg
                yield {"type": "text", "content": f"LLM error: {friendly}"}
                yield {"type": "done", "usage": usage}
                return

            content_parts = []
            tool_calls_by_index = {}

            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = getattr(choice, "delta", None) or (choice.get("delta") if isinstance(choice, dict) else None)
                if not delta:
                    continue
                c = getattr(delta, "content", None) if not isinstance(delta, dict) else delta.get("content")
                if c:
                    content_parts.append(c)
                    yield {"type": "text", "content": c}
                tcs = getattr(delta, "tool_calls", None) if not isinstance(delta, dict) else delta.get("tool_calls")
                tcs = tcs or []
                for tc in tcs:
                    idx = getattr(tc, "index", 0) if not isinstance(tc, dict) else tc.get("index", 0)
                    tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
                    fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function") or {}
                    if isinstance(fn, dict):
                        name, args = fn.get("name"), fn.get("arguments") or ""
                    else:
                        name, args = getattr(fn, "name", None), getattr(fn, "arguments", None) or ""
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {"id": None, "name": None, "arguments": ""}
                    if tc_id:
                        tool_calls_by_index[idx]["id"] = tc_id
                    if name:
                        tool_calls_by_index[idx]["name"] = name
                    tool_calls_by_index[idx]["arguments"] = tool_calls_by_index[idx]["arguments"] + args
                u = getattr(chunk, "usage", None) or (chunk.get("usage") if isinstance(chunk, dict) else None)
                if u:
                    usage = u if isinstance(u, dict) else {"prompt_tokens": getattr(u, "prompt_tokens", 0), "completion_tokens": getattr(u, "completion_tokens", 0)}

            full_content = "".join(content_parts).strip() if content_parts else ""
            tc_list = []
            for idx in sorted(tool_calls_by_index):
                t = tool_calls_by_index[idx]
                tid = t.get("id") or f"call_{idx}"
                name = t.get("name")
                args_str = t.get("arguments") or "{}"
                tc_list.append((tid, name, args_str))

            if usage:
                pt = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
                if isinstance(pt, int) and pt > PROMPT_TOKEN_WARNING_THRESHOLD:
                    _log.warning(
                        "stream_agent_loop: high prompt token usage (%d). model=%s", pt, model
                    )
                    yield {"type": "system", "content": f"⚠ Large context ({pt:,} tokens). Consider starting a new chat to reduce cost."}

            if not tc_list:
                yield {"type": "done", "usage": usage}
                return

            assistant_msg = {
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": [
                    {"id": tid, "type": "function", "function": {"name": n, "arguments": a}}
                    for tid, n, a in tc_list
                ],
            }
            api_messages.append(assistant_msg)

            has_external = False
            for tid, name, args_str in tc_list:
                if not name or not tid:
                    continue
                if name == "query_profile_db" and query_runner is not None:
                    yield {"type": "system", "content": "Running DB query..."}
                    try:
                        args = json.loads(args_str)
                        sql = args.get("sql_query", "")
                        result = query_runner(sql)
                    except Exception as e:
                        result = f"Error: {e}"
                    if result.startswith("Error:"):
                        consecutive_db_errors += 1
                        if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                            result += "\n[System: Repeated SQL errors. Please answer from available context without further queries.]"
                    else:
                        consecutive_db_errors = 0
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "name": name,
                        "content": result,
                    })
                else:
                    if name == "query_profile_db":
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tid,
                            "name": name,
                            "content": "Not executed (no profile loaded).",
                        })
                    action = _parse_tool_call(name, args_str)
                    if action:
                        has_external = True
                        yield {"type": "action", "action": action}

            if has_external:
                yield {"type": "done", "usage": usage}
                return
        yield {"type": "done", "usage": usage}
    finally:
        # Telemetry: log usage data for cost analysis and future rate-limiting.
        if usage:
            _telemetry_log.info(
                "agent_usage model=%s prompt_tokens=%s completion_tokens=%s turns=%d",
                model,
                usage.get("prompt_tokens", "?") if isinstance(usage, dict) else getattr(usage, "prompt_tokens", "?"),
                usage.get("completion_tokens", "?") if isinstance(usage, dict) else getattr(usage, "completion_tokens", "?"),
                turn_count,
            )
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def chat_completion_stream(body_bytes: bytes):
    """
    Generator yielding SSE bytes for streaming mode (,).
    Always delegates to stream_agent_loop; uses profile_path when NSYS_AI_DB_AGENT is set.
    """
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        yield _sse_event("text", {"chunk": "Invalid request body."})
        yield _sse_event("done", {})
        return

    preferred_model = payload.get("model")
    model, _ = _get_model_and_key(preferred_model)
    if not model:
        yield _sse_event("done", {"error": "LLM not configured"})
        return

    messages = payload.get("messages") or []
    ui_context = payload.get("ui_context") or {}
    profile_path = payload.get("profile_path")
    db_agent_flag = os.environ.get("NSYS_AI_DB_AGENT", "").strip().lower()
    db_agent_enabled = bool(db_agent_flag) and db_agent_flag not in ("0", "false", "no", "off")

    # Use DB-backed agent when profile_path is present and NSYS_AI_DB_AGENT is set.
    effective_profile = profile_path if (profile_path and db_agent_enabled) else None

    try:
        for ev in stream_agent_loop(
            model=model,
            messages=messages,
            ui_context=ui_context,
            tools=_tools_openai(),
            profile_path=effective_profile,
            max_turns=5,
        ):
            t = ev.get("type")
            if t == "text":
                yield _sse_event("text", {"chunk": ev.get("content", "")})
            elif t == "system":
                yield _sse_event("system", {"content": ev.get("content", "")})
            elif t == "action":
                yield _sse_event("action", ev.get("action", {}))
            elif t == "done":
                yield _sse_event("done", ev.get("usage") or {})
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    except Exception as e:
        err_msg = str(e)
        print(f"[nsys-tui] stream_agent_loop error (model={model!r}): {err_msg}", file=sys.stderr)
        try:
            yield _sse_event("text", {"chunk": f"Stream error: {err_msg}"})
            yield _sse_event("done", {"error": err_msg})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
