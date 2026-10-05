#!/usr/bin/env python3
"""
model-face-off: ask Claude, GPT and Gemini the same question and
see their answers next to each other.

Usage:
    python faceoff.py "explain transformers to a 10 year old"
    python faceoff.py --only claude gemini "write a haiku about rain"
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from rich.console import Console
from rich.columns import Columns
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import models

console = Console()


def show(results):
    # answers side by side (rich wraps them into columns that fit the terminal)
    panels = []
    for r in results:
        body = Markdown(r["text"]) if "text" in r else f"[red]{r['error']}[/red]"
        panels.append(Panel(body, title=f"[bold]{r['name']}[/bold] · {r['model']}", expand=True))
    console.print(Columns(panels, equal=True, expand=True))

    # and a little scoreboard underneath
    table = Table(title="scoreboard")
    table.add_column("model")
    table.add_column("time (s)", justify="right")
    table.add_column("tokens in", justify="right")
    table.add_column("tokens out", justify="right")
    table.add_column("words", justify="right")

    ok = [r for r in results if "text" in r]
    fastest = min(ok, key=lambda r: r["seconds"])["name"] if ok else None

    for r in results:
        if "error" in r:
            table.add_row(r["name"], str(r["seconds"]), "-", "-", "failed")
            continue
        name = f"{r['name']} ⚡" if r["name"] == fastest else r["name"]
        table.add_row(name, str(r["seconds"]), str(r["tokens_in"]), str(r["tokens_out"]),
                      str(len(r["text"].split())))
    console.print(table)


def save(prompt, results):
    filename = datetime.now().strftime("faceoff-%Y%m%d-%H%M%S.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Face-off\n\n**Prompt:** {prompt}\n\n")
        for r in results:
            f.write(f"## {r['name']} ({r['model']}), {r['seconds']}s\n\n")
            f.write(r.get("text") or f"_error: {r.get('error')}_")
            f.write("\n\n")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Ask several LLMs the same question.")
    parser.add_argument("prompt", nargs="?", help="the question (leave empty to type it in)")
    parser.add_argument("--only", nargs="+", choices=models.ASK.keys(),
                        help="only use these providers")
    parser.add_argument("--system", default="You are a helpful assistant. Keep answers fairly short.",
                        help="system prompt given to every model")
    parser.add_argument("--save", action="store_true", help="also save the results to a markdown file")
    args = parser.parse_args()

    prompt = args.prompt or console.input("[bold]prompt >[/bold] ").strip()
    if not prompt:
        sys.exit("need a prompt")

    names = args.only or models.available()
    missing = [n for n in names if n not in models.available()]
    if missing:
        console.print(f"[yellow]skipping {', '.join(missing)} (no API key set)[/yellow]")
        names = [n for n in names if n not in missing]
    if not names:
        sys.exit("no API keys found. set at least one of: " + ", ".join(models.API_KEYS.values()))

    # ask everyone at the same time, otherwise you'd wait for each one in turn
    with console.status(f"asking {', '.join(names)}..."):
        with ThreadPoolExecutor() as pool:
            results = list(pool.map(lambda n: models.run(n, prompt, args.system), names))

    show(results)

    if args.save:
        console.print(f"saved to {save(prompt, results)}")


if __name__ == "__main__":
    main()
