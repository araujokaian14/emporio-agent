"""Runs the example conversations against the real agent and saves them to examples/*.md.

    python scripts/run_scenarios.py              # all scenarios
    python scripts/run_scenarios.py 3            # only scenario 3

The output files are the raw model answers (not edited), with the tools called in each turn
and the latency, so the reviewer can see what the agent actually did.
The store clock is fixed to a weekday morning (FAKE_NOW) so "store is closed" does not
change the answers depending on when the script runs. Override with FAKE_NOW=... if you want.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("FAKE_NOW", "2026-11-04T10:30")  # Wednesday 10:30 in Campo Grande

from emporio_agent.agent import build_agent  # noqa: E402
from emporio_agent.config import LLM_MODEL, now  # noqa: E402

SCENARIOS = [
    ("01_catalogo_violoes_ate_1000", "Busca no catálogo com filtro de preço e follow-up", [
        "Oi! Quais opções de violões vocês têm custando até R$1000?",
        "Legal! Desses, qual você indica pra quem está começando e quer corda de nylon?",
    ]),
    ("02_preco_takamine_pix_parcelas", "Consulta de preço, PIX e parcelamento", [
        "Quanto custa o Takamine GD20?",
        "Se eu pagar no PIX fica quanto? E dá pra parcelar no cartão?",
    ]),
    ("03_devolucao_pedido_politica", "Caso não trivial: verificação de identidade + dados do pedido + política de devolução", [
        "Me arrependi da minha compra, posso devolver meu pedido?",
        "É o pedido 2, meu e-mail é thiago.barbosa@jmail.com",
        "E se o violão tivesse vindo com defeito, o que eu faria?",
    ]),
    ("04_fora_do_escopo", "Acessórios (fora do catálogo) e assunto sem relação com a loja", [
        "Vocês vendem cordas e amplificador pra guitarra?",
        "Beleza. E qual vai ser a previsão do tempo amanhã em Campo Grande?",
    ]),
    ("05_loja_promocoes_produto_esgotado", "Info da loja, promoção inativa (Black Friday) e produto esgotado", [
        "Qual o endereço da loja e até que horas vocês abrem no sábado?",
        "Tem desconto de Black Friday? E vocês têm o violão Giannini GF-3D em estoque?",
    ]),
]


def run(scenario, agent) -> str:
    slug, description, turns = scenario
    agent.reset()
    out = [f"# {slug.split('_', 1)[1].replace('_', ' ').title()}\n",
           f"*{description}*\n",
           f"> Gerado por `scripts/run_scenarios.py` — resposta real do modelo `{LLM_MODEL}`, sem edição. "
           f"Relógio da loja simulado: {now():%d/%m/%Y %H:%M}.\n"]
    for user in turns:
        reply = agent.chat(user)
        out.append(f"**Cliente:** {user}\n")
        for t in reply.tool_calls:
            out.append(f"> 🔧 `{t.name}({json.dumps(t.arguments, ensure_ascii=False)})`  ")
        out.append(f"> ⏱ {reply.seconds}s\n")
        out.append(f"**Melodia:** {reply.text}\n")
        print(f"[{slug}] {user[:50]}... -> {len(reply.tool_calls)} tool call(s), {reply.seconds}s")
    return "\n".join(out)


def main() -> None:
    only = {int(a) for a in sys.argv[1:]}
    out_dir = Path(__file__).resolve().parents[1] / "examples"
    out_dir.mkdir(exist_ok=True)
    agent = build_agent()
    for i, scenario in enumerate(SCENARIOS, start=1):
        if only and i not in only:
            continue
        (out_dir / f"{scenario[0]}.md").write_text(run(scenario, agent), encoding="utf-8")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
