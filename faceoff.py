#!/usr/bin/env python3
"""
model-face-off: ask Claude, GPT and Gemini the same question and
see their answers next to each other.

Usage:
    python faceoff.py "explain transformers to a 10 year old"
    python faceoff.py --only claude gemini "write a haiku about rain"
    python faceoff.py --file prompts.txt --judge   # a whole list, with a running score
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.columns import Columns
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import models
from judge import judge

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


def save(rounds):
    """rounds is a list of (prompt, results, winner). One prompt or a whole file, same format."""
    filename = datetime.now().strftime("faceoff-%Y%m%d-%H%M%S.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Face-off\n\n")
        for prompt, results, winner in rounds:
            f.write(f"**Prompt:** {prompt}\n\n")
            for r in results:
                f.write(f"## {r['name']} ({r['model']}), {r['seconds']}s\n\n")
                f.write(r.get("text") or f"_error: {r.get('error')}_")
                f.write("\n\n")
            if winner:
                f.write(f"**Winner:** {winner}\n\n")
            f.write("---\n\n")
    return filename


def load_prompts(path):
    """One prompt per line. Blank lines and lines starting with # are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def tally(rounds):
    """Running score over several prompts: judge wins, times fastest, failures."""
    wins, fastest, failed = Counter(), Counter(), Counter()
    for _, results, winner in rounds:
        if winner:
            wins[winner] += 1
        ok = [r for r in results if "text" in r]
        if ok:
            fastest[min(ok, key=lambda r: r["seconds"])["name"]] += 1
        failed.update(r["name"] for r in results if "error" in r)
    return wins, fastest, failed


def show_tally(rounds, names, judged):
    wins, fastest, failed = tally(rounds)
    table = Table(title=f"overall, {len(rounds)} prompts")
    table.add_column("model")
    if judged:
        table.add_column("wins", justify="right")
    table.add_column("fastest", justify="right")
    table.add_column("failed", justify="right")
    for name in sorted(names, key=lambda n: (-wins[n], -fastest[n])):
        row = [name] + ([str(wins[name])] if judged else []) + [str(fastest[name]), str(failed[name])]
        table.add_row(*row)
    console.print(table)


def ask_all(names, prompt, system):
    # ask everyone at the same time, otherwise you'd wait for each one in turn
    with console.status(f"asking {', '.join(names)}..."):
        with ThreadPoolExecutor() as pool:
            return list(pool.map(lambda n: models.run(n, prompt, system), names))


def run_judge(prompt, results, judge_name):
    with console.status(f"{judge_name} is judging..."):
        winner, reason = judge(prompt, results, judge_name)
    if winner:
        console.print(f"\n🏆 [bold]{judge_name}[/bold] picks [bold green]{winner}[/bold green]: {reason}")
    else:
        console.print(f"\n[yellow]no verdict: {reason}[/yellow]")
    return winner


def main():
    parser = argparse.ArgumentParser(description="Ask several LLMs the same question.")
    parser.add_argument("prompt", nargs="?", help="the question (leave empty to type it in)")
    parser.add_argument("--only", nargs="+", choices=models.ASK.keys(),
                        help="only use these providers")
    parser.add_argument("--system", default="You are a helpful assistant. Keep answers fairly short.",
                        help="system prompt given to every model")
    parser.add_argument("--file", help="run every prompt in this file (one per line) and keep score")
    parser.add_argument("--save", action="store_true", help="also save the results to a markdown file")
    parser.add_argument("--judge", action="store_true", help="have a model pick the best answer")
    parser.add_argument("--judge-with", choices=models.ASK.keys(),
                        help="which model judges (default: claude if you have a key for it)")
    args = parser.parse_args()

    if args.file:
        prompts = load_prompts(args.file)
        if not prompts:
            sys.exit(f"no prompts in {args.file}")
    else:
        prompt = args.prompt or console.input("[bold]prompt >[/bold] ").strip()
        if not prompt:
            sys.exit("need a prompt")
        prompts = [prompt]

    names = args.only or models.available()
    missing = [n for n in names if n not in models.available()]
    if missing:
        console.print(f"[yellow]skipping {', '.join(missing)} (no API key set)[/yellow]")
        names = [n for n in names if n not in missing]
    if not names:
        sys.exit("no API keys found. set at least one of: " + ", ".join(models.API_KEYS.values()))

    judged = args.judge or args.judge_with
    if judged:
        available = models.available()
        judge_name = args.judge_with or ("claude" if "claude" in available else available[0])

    rounds = []
    for i, prompt in enumerate(prompts, 1):
        if len(prompts) > 1:
            console.rule(f"[bold]{i}/{len(prompts)}[/bold] {prompt}")
        results = ask_all(names, prompt, args.system)
        show(results)
        winner = run_judge(prompt, results, judge_name) if judged else None
        rounds.append((prompt, results, winner))

    if len(rounds) > 1:
        console.print()
        show_tally(rounds, names, judged)

    if args.save:
        console.print(f"saved to {save(rounds)}")

if __name__ == "__main__":
    main()
