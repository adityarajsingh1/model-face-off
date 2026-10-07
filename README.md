# model-face-off 🥊

Ask **Claude**, **GPT** and **Gemini** the same question and see their answers side by side, along with how fast each one was and how many tokens it used.

I kept switching between three browser tabs to compare models, so I wrote this to do it in one go from the terminal. It's handy for figuring out which model is better at a specific kind of task (coding, summarizing, explaining stuff) instead of just going by benchmarks.

## What it does

- sends your prompt to all three models **at the same time**
- shows the answers in columns next to each other
- prints a little scoreboard: response time, tokens in/out, word count (the fastest one gets a ⚡)
- skips any provider you don't have an API key for, so it works even with just one or two
- `--judge` lets one model play referee and pick the best answer. The answers are shuffled and anonymised first, so it can't just vote for itself
- `--save` dumps everything to a markdown file if you want to keep it

## Setup

Python 3.9+ needed.

```bash
git clone https://github.com/adityarajsingh1/model-face-off.git
cd model-face-off
python -m venv .venv
source .venv/bin/activate      # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add whichever keys you have:

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."     # free key from Google AI Studio
```

## Usage

```bash
python faceoff.py "explain transformers to a 10 year old"

# only compare two of them
python faceoff.py --only claude gemini "write a haiku about rain"

# same system prompt for everyone
python faceoff.py --system "Answer like a pirate." "what is an API?"

# let claude (or any of them) pick a winner
python faceoff.py --judge "why is the sky blue?"
python faceoff.py --judge-with gemini "explain big-O notation"

# keep the results
python faceoff.py --save "what's the difference between a list and a tuple in python?"
```

You can also run it with no prompt and it'll ask you for one.

Want a different model? Set `CLAUDE_MODEL`, `OPENAI_MODEL` or `GEMINI_MODEL` (see `.env.example`).

## How it works

- `models.py` has one tiny function per provider. They all return the same thing (text, tokens in, tokens out), which keeps the comparison fair and makes adding a new provider easy.
- `faceoff.py` handles the command line, runs all the requests in parallel with a thread pool, and uses [rich](https://github.com/Textualize/rich) to draw the panels and the table.

## Ideas for later

- [ ] estimate cost per answer
- [x] let a model judge the answers and pick a winner
- [ ] run a whole list of prompts from a file and keep a running score
- [ ] add a local model through Ollama
- [ ] a simple web version with Streamlit
