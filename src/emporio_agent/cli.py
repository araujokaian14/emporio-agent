"""Interactive terminal chat.

    python -m emporio_agent.cli            # normal chat
    python -m emporio_agent.cli --debug    # also shows which tools were called

Commands during the chat: /nova (new conversation), /debug (toggle tool trace),
/salvar <arquivo.md> (export the conversation), /sair (quit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import AgentReply, build_agent
from .config import LLM_MODEL

console = Console()


def show_trace(reply: AgentReply) -> None:
    for t in reply.tool_calls:
        args = json.dumps(t.arguments, ensure_ascii=False)
        console.print(f"  [dim]🔧 {t.name}({args})[/dim]")
    console.print(f"  [dim]⏱ {reply.seconds}s[/dim]")


def to_markdown(transcript: list[tuple[str, AgentReply | str]]) -> str:
    lines = []
    for who, content in transcript:
        if who == "cliente":
            lines.append(f"**Cliente:** {content}\n")
        else:
            for t in content.tool_calls:
                lines.append(f"> 🔧 `{t.name}({json.dumps(t.arguments, ensure_ascii=False)})`")
            if content.tool_calls:
                lines.append("")
            lines.append(f"**Melodia:** {content.text}\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Empório da Música — atendimento virtual")
    parser.add_argument("--debug", action="store_true", help="mostra as ferramentas chamadas")
    parser.add_argument("--no-embeddings", action="store_true", help="busca nas políticas só por palavras-chave")
    args = parser.parse_args()

    with console.status("Carregando dados e políticas..."):
        agent = build_agent(use_embeddings=not args.no_embeddings)
    debug = args.debug
    transcript: list[tuple[str, AgentReply | str]] = []

    console.print(Panel.fit(
        f"🎸 [bold]Empório da Música[/bold] — atendimento virtual\n"
        f"[dim]modelo: {LLM_MODEL} | políticas: {len(agent.tools.policies.chunks)} trechos "
        f"({'híbrido' if agent.tools.policies.vectors is not None else 'palavras-chave'})\n"
        f"comandos: /nova  /debug  /salvar arquivo.md  /sair[/dim]"))

    while True:
        try:
            user = console.input("\n[bold cyan]Você:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("/sair", "/exit", "/quit"):
            break
        if user == "/nova":
            agent.reset(); transcript.clear()
            console.print("[dim]Nova conversa iniciada.[/dim]")
            continue
        if user == "/debug":
            debug = not debug
            console.print(f"[dim]debug {'ligado' if debug else 'desligado'}[/dim]")
            continue
        if user.startswith("/salvar"):
            path = Path(user.split(maxsplit=1)[1] if " " in user else "conversa.md")
            path.write_text(to_markdown(transcript), encoding="utf-8")
            console.print(f"[dim]Conversa salva em {path}[/dim]")
            continue

        with console.status("Melodia está digitando..."):
            reply = agent.chat(user)
        transcript += [("cliente", user), ("agente", reply)]
        if debug:
            show_trace(reply)
        console.print("[bold magenta]Melodia:[/bold magenta]", Markdown(reply.text))

    console.print("[dim]Até logo! 🎶[/dim]")


if __name__ == "__main__":
    main()
