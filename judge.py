"""
Let one model act as the referee and pick the best answer.

The answers are shuffled and labelled A, B, C so the judge doesn't know
which model wrote what. Otherwise it might just favour itself.
"""

import json
import random
import re
import string

import models

JUDGE_SYSTEM = """You are a strict, fair judge comparing answers to the same question.
Judge on correctness first, then how well it answers what was actually asked, then clarity.
Length is not a virtue by itself.
Reply with only JSON like: {"winner": "A", "reason": "one or two sentences"}"""


def build_prompt(question, labelled):
    parts = [f"Question:\n{question}\n"]
    for label, r in labelled:
        parts.append(f"Answer {label}:\n{r['text']}\n")
    return "\n".join(parts)


def parse_verdict(text):
    """Pull the json out even if the model wraps it in extra words or a code block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "winner" not in data:
        return None
    return str(data["winner"]).strip().upper(), data.get("reason", "")


def judge(question, results, judge_name, seed=None):
    """Returns (winner_name, reason), or (None, why it failed)."""
    answers = [r for r in results if "text" in r]
    if len(answers) < 2:
        return None, "need at least two answers to compare"

    random.Random(seed).shuffle(answers)
    labelled = list(zip(string.ascii_uppercase, answers))

    verdict = models.run(judge_name, build_prompt(question, labelled), JUDGE_SYSTEM)
    if "error" in verdict:
        return None, f"judge failed: {verdict['error']}"

    parsed = parse_verdict(verdict["text"])
    if not parsed:
        return None, "couldn't understand the judge's reply"
    label, reason = parsed
    for l, r in labelled:
        if l == label:
            return r["name"], reason
    return None, f"judge picked '{label}', which isn't one of the answers"
