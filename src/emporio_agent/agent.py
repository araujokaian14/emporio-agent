"""The agent loop: LLM + native function calling (a ReAct-style loop without a framework).

    user message
        → LLM sees system prompt + history + tool schemas
        → LLM either answers, or asks for one or more tool calls
        → we run the tools in Python and send the JSON results back
        → repeat until the LLM answers (max MAX_STEPS rounds)

No LangChain/LlamaIndex on purpose: the loop is short, easy to read, test and debug,
and works with any OpenAI-compatible endpoint (Ollama, OpenAI, Groq, vLLM...).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, now
from .prompts import build_system_prompt
from .tools import StoreTools

MAX_STEPS = 6            # tool-calling rounds per user message
MAX_USER_TURNS = 8       # conversation memory kept in the prompt (fits an 8k context)
MAX_INPUT_CHARS = 2000
FALLBACK_REPLY = ("Desculpe, tive um problema para consultar essa informação agora. "
                  "Pode tentar de novo em instantes? Se preferir, fale com a equipe pelo (67) 3341-4444.")

TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ToolTrace:
    name: str
    arguments: dict
    result: str


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolTrace] = field(default_factory=list)
    seconds: float = 0.0


def _parse_text_tool_calls(content: str | None) -> list[dict]:
    """Some small local models write the tool call as text instead of using the API field.
    Recover those so the loop still works: <tool_call>{"name": ..., "arguments": {...}}</tool_call>."""
    calls = []
    for i, raw in enumerate(TEXT_TOOL_CALL_RE.findall(content or "")):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append({"id": f"call_text_{i}", "name": obj["name"],
                          "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False)})
    return calls


class Agent:
    def __init__(self, tools: StoreTools, client: Any = None, model: str = LLM_MODEL,
                 temperature: float = 0.2):
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.tools = tools
        self.schemas = tools.schemas()
        self.history: list[dict] = []  # user / assistant / tool messages (system prompt is rebuilt each turn)

    def reset(self) -> None:
        self.history = []

    # ------------------------------------------------------------------ memory

    def _trimmed_history(self) -> list[dict]:
        """Keep only the last MAX_USER_TURNS user turns. Cutting at a user message guarantees we
        never separate an assistant tool call from its tool result (that would break the API)."""
        user_idx = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(user_idx) <= MAX_USER_TURNS:
            return self.history
        return self.history[user_idx[-MAX_USER_TURNS]:]

    # ------------------------------------------------------------------ main loop

    def chat(self, user_message: str) -> AgentReply:
        start = time.perf_counter()
        self.history.append({"role": "user", "content": user_message.strip()[:MAX_INPUT_CHARS]})
        traces: list[ToolTrace] = []

        for _ in range(MAX_STEPS):
            messages = [{"role": "system", "content": build_system_prompt(now())}] + self._trimmed_history()
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=self.schemas,
                    temperature=self.temperature,
                )
            except Exception as exc:
                return self._finish(FALLBACK_REPLY + f"\n\n[erro técnico: {type(exc).__name__}]", traces, start)

            msg = response.choices[0].message
            calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                     for tc in (msg.tool_calls or [])]
            if not calls:
                calls = _parse_text_tool_calls(msg.content)

            if not calls:  # final answer
                text = (msg.content or "").strip() or FALLBACK_REPLY
                self.history.append({"role": "assistant", "content": text})
                return self._finish(text, traces, start)

            self.history.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls],
            })
            for c in calls:
                result = self.tools.execute(c["name"], c["arguments"])
                try:
                    args = json.loads(c["arguments"]) if c["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": c["arguments"]}
                traces.append(ToolTrace(c["name"], args, result))
                self.history.append({"role": "tool", "tool_call_id": c["id"], "content": result})

        # The model kept calling tools without answering: stop and hand over to a human.
        text = ("Deixa eu passar sua pergunta para alguém da equipe confirmar, tudo bem? "
                "Retornamos em até 24 horas úteis.")
        self.history.append({"role": "assistant", "content": text})
        return self._finish(text, traces, start)

    @staticmethod
    def _finish(text: str, traces: list[ToolTrace], start: float) -> AgentReply:
        return AgentReply(text=text, tool_calls=traces, seconds=round(time.perf_counter() - start, 2))


def build_agent(use_embeddings: bool = True) -> Agent:
    """Wire everything with the real data, the policy index and the configured LLM."""
    from .data_loader import load_store_data
    from .policies_rag import PolicyIndex

    tools = StoreTools(load_store_data(), policies=PolicyIndex.from_pdf(use_embeddings=use_embeddings))
    return Agent(tools)
