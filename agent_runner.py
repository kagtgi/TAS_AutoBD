"""
TAS AutoBD — Agentic Runner (ReAct Loop)
=========================================
Implements a ReAct (Reason + Act) agent loop using Claude's *native tool-use*
API (not LangChain agents).  The loop runs until:
  • Claude returns stop_reason="end_turn"  → clean finish, return final text
  • max_iterations is reached             → return best available text

Key features
------------
  • Parallel tool execution — when Claude requests multiple tools in one turn,
    all are executed concurrently via ThreadPoolExecutor
  • on_tool_call callback — optional hook invoked *before* each tool fires,
    useful for real-time UI progress reporting
  • Clean message serialisation — SDK content-block objects are converted to
    plain dicts so the conversation history stays portable

This module requires LLM_PROVIDER=anthropic (or at least ANTHROPIC_API_KEY set).
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL
from tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITER = 12  # safety cap on tool-call rounds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blocks_to_dicts(content_blocks) -> List[Dict]:
    """
    Convert Anthropic SDK content-block objects (TextBlock, ToolUseBlock, …)
    to plain dicts compatible with the messages API.
    """
    result = []
    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            result.append({"type": "text", "text": block.text})
        elif btype == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return result


def _extract_text(content_blocks) -> str:
    """Return the first non-empty text block from a list of content objects."""
    for block in content_blocks:
        btype = getattr(block, "type", None)
        text = getattr(block, "text", None) or (
            block.get("text", "") if isinstance(block, dict) else ""
        )
        if btype == "text" and text and text.strip():
            return text
    return ""


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agent(
    system_prompt: str,
    user_message: str,
    tools: Optional[List[Dict]] = None,
    max_iterations: int = _DEFAULT_MAX_ITER,
    on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> str:
    """
    Run an autonomous ReAct agent loop using Claude's native tool-use API.

    The agent reasons about what it needs, calls tools to gather information,
    observes results, and repeats until it is satisfied or the iteration cap
    is reached.  Multiple tool calls within a single response are executed
    in parallel for maximum efficiency.

    Parameters
    ----------
    system_prompt   : Agent role, goal, and behavioural instructions.
    user_message    : The initial task / question string.
    tools           : Tool schemas list (defaults to all tools in TOOL_SCHEMAS).
    max_iterations  : Maximum number of tool-call rounds before forced stop.
    on_tool_call    : Optional callback(tool_name, tool_inputs) called just
                      *before* each tool execution.  Safe to use for UI updates
                      from the main thread; callbacks from parallel workers are
                      collected without blocking.

    Returns
    -------
    str — The agent's final text answer.

    Raises
    ------
    EnvironmentError  if ANTHROPIC_API_KEY is not set.
    """
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is required for the agentic runner. "
            "Set it in your .env file."
        )

    if tools is None:
        tools = TOOL_SCHEMAS

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages: List[Dict] = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        logger.info("Agent iteration %d / %d", iteration + 1, max_iterations)

        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        logger.info("Stop reason: %s | blocks: %d", response.stop_reason, len(response.content))

        # Persist assistant turn (convert SDK objects → plain dicts)
        messages.append(
            {"role": "assistant", "content": _blocks_to_dicts(response.content)}
        )

        # ── Agent is satisfied: return its final answer ────────────────────
        if response.stop_reason == "end_turn":
            return _extract_text(response.content)

        # ── Agent wants to call tools ──────────────────────────────────────
        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

            if not tool_blocks:
                logger.warning("stop_reason=tool_use but no tool_use blocks found; stopping.")
                break

            # --- Execute tool calls (parallel when >1) --------------------
            def _exec_one(block) -> tuple:
                """Execute a single tool; notify callback; return (id, result_json)."""
                if on_tool_call:
                    try:
                        on_tool_call(block.name, block.input)
                    except Exception:
                        pass  # callback errors must never break the agent
                logger.info(
                    "  → %s(%s)",
                    block.name,
                    json.dumps(block.input, ensure_ascii=False)[:120],
                )
                return block.id, execute_tool(block.name, block.input)

            if len(tool_blocks) > 1:
                # Parallel execution via thread pool
                id_to_result: Dict[str, str] = {}
                with ThreadPoolExecutor(max_workers=min(len(tool_blocks), 5)) as pool:
                    future_map = {pool.submit(_exec_one, b): b for b in tool_blocks}
                    for future in as_completed(future_map):
                        tool_id, result_json = future.result()
                        id_to_result[tool_id] = result_json
                # Preserve original block order in results
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": id_to_result[b.id],
                    }
                    for b in tool_blocks
                ]
                logger.info("  Executed %d tools in parallel.", len(tool_blocks))
            else:
                tid, rjson = _exec_one(tool_blocks[0])
                tool_results = [
                    {"type": "tool_result", "tool_use_id": tid, "content": rjson}
                ]

            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason — surface whatever text exists
            text = _extract_text(response.content)
            if text:
                logger.warning("Unexpected stop_reason=%r — returning available text.", response.stop_reason)
                return text
            break

    # ── Iteration cap reached — return the last assistant text found ──────────
    logger.warning("Agent reached max iterations (%d); returning best available text.", max_iterations)
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = _extract_text(msg.get("content", []))
            if text:
                return text
    return "Research complete (iteration limit reached)."
