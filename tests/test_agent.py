"""Agent loop tests with a scripted fake LLM (no Ollama needed).

They check the plumbing, not the model's intelligence: tool calls are executed, results go
back to the model, history stays valid, and failures degrade gracefully.
"""

from datetime import date, datetime
from types import SimpleNamespace as NS

import pytest

from emporio_agent import agent as agent_mod
from emporio_agent.agent import Agent
from emporio_agent.data_loader import load_store_data
from emporio_agent.prompts import build_system_prompt, store_status
from emporio_agent.tools import StoreTools


def tool_call(name, args, id_="c1"):
    return NS(id=id_, function=NS(name=name, arguments=args))


def response(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


class FakeLLM:
    """Returns the scripted responses in order and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(scope="module")
def tools():
    return StoreTools(load_store_data(), today=date(2026, 11, 2))


def test_tool_call_then_answer(tools):
    llm = FakeLLM([
        response(tool_calls=[tool_call("get_product", '{"product": "Takamine GD20"}')]),
        response(content="O Takamine GD20 sai por R$ 2.199,00."),
    ])
    reply = Agent(tools, client=llm).chat("Quanto custa o Takamine GD20?")
    assert reply.text.startswith("O Takamine")
    assert [t.name for t in reply.tool_calls] == ["get_product"]
    # the tool result was sent back to the model in the 2nd request
    tool_msgs = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert '"price_brl": 2199.0' in tool_msgs[0]["content"]


def test_tool_call_written_as_text_is_recovered(tools):
    llm = FakeLLM([
        response(content='<tool_call>{"name": "list_active_promotions", "arguments": {}}</tool_call>'),
        response(content="Temos 4 promoções ativas."),
    ])
    reply = Agent(tools, client=llm).chat("Tem promoção?")
    assert [t.name for t in reply.tool_calls] == ["list_active_promotions"]


def test_llm_error_returns_friendly_fallback(tools):
    reply = Agent(tools, client=FakeLLM([ConnectionError("down")])).chat("oi")
    assert "problema" in reply.text


def test_infinite_tool_loop_is_stopped(tools):
    loop = [response(tool_calls=[tool_call("list_active_promotions", "{}", id_=f"c{i}")])
            for i in range(agent_mod.MAX_STEPS)]
    reply = Agent(tools, client=FakeLLM(loop)).chat("?")
    assert "equipe" in reply.text and len(reply.tool_calls) == agent_mod.MAX_STEPS


def test_history_trimming_keeps_tool_pairs_intact(tools):
    a = Agent(tools, client=FakeLLM([]))
    for i in range(agent_mod.MAX_USER_TURNS + 3):
        a.history += [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}"}]},
                      {"role": "tool", "tool_call_id": f"t{i}", "content": "{}"},
                      {"role": "assistant", "content": f"a{i}"}]
    trimmed = a._trimmed_history()
    assert trimmed[0]["role"] == "user"
    assert sum(m["role"] == "user" for m in trimmed) == agent_mod.MAX_USER_TURNS


@pytest.mark.parametrize("moment,expected", [
    (datetime(2026, 11, 4, 10, 0), "ABERTA"),   # Wednesday morning
    (datetime(2026, 11, 7, 14, 0), "FECHADA"),  # Saturday after 13:00
    (datetime(2026, 11, 8, 10, 0), "FECHADA"),  # Sunday
    (datetime(2026, 11, 2, 10, 0), "feriado"),  # Finados
])
def test_store_status(moment, expected):
    assert expected in store_status(moment)


def test_system_prompt_has_date_and_rules():
    prompt = build_system_prompt(datetime(2026, 11, 4, 10, 0))
    assert "quarta-feira, 04/11/2026 10:00" in prompt
    assert "NUNCA informe preço" in prompt
