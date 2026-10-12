import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faceoff


def result(name, seconds, text="hi"):
    r = {"name": name, "model": name + "-model", "seconds": seconds}
    if text is None:
        r["error"] = "boom"
    else:
        r.update(text=text, tokens_in=1, tokens_out=1)
    return r


def test_load_prompts_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# header\n\nfirst one\n   \n  # indented comment\nsecond one  \n")
    assert faceoff.load_prompts(f) == ["first one", "second one"]


def test_tally_counts_wins_fastest_and_failures():
    rounds = [
        ("q1", [result("claude", 2.0), result("gpt", 1.0)], "claude"),
        ("q2", [result("claude", 1.5), result("gpt", 3.0)], "claude"),
        ("q3", [result("claude", 1.0), result("gpt", 0.5, text=None)], None),
    ]
    wins, fastest, failed = faceoff.tally(rounds)
    assert wins == {"claude": 2}
    assert fastest == {"gpt": 1, "claude": 2}  # a failed answer can't be the fastest
    assert failed == {"gpt": 1}


def test_file_mode_runs_every_prompt(tmp_path, monkeypatch):
    f = tmp_path / "p.txt"
    f.write_text("one\ntwo\n")
    asked = []
    monkeypatch.setattr(faceoff.models, "available", lambda: ["claude", "gpt"])
    monkeypatch.setattr(faceoff.models, "run",
                        lambda name, prompt, system: asked.append(prompt) or result(name, 1.0))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["faceoff.py", "--file", str(f), "--save"])
    faceoff.main()
    assert sorted(asked) == ["one", "one", "two", "two"]
    saved = next(tmp_path.glob("faceoff-*.md")).read_text()
    assert "**Prompt:** one" in saved and "**Prompt:** two" in saved
