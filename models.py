"""
One small function per provider. Each one takes a prompt and returns
the answer plus a few numbers (time taken, tokens in/out) so we can
compare them fairly.
"""

import os
import time

DEFAULT_MODELS = {
    "claude": os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
    "gpt": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
}

# which env var each provider needs, so we can skip the ones you haven't set up
API_KEYS = {
    "claude": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def ask_claude(prompt, system, model):
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def ask_gpt(prompt, system, model):
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def ask_gemini(prompt, system, model):
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    usage = resp.usage_metadata
    return resp.text, usage.prompt_token_count, usage.candidates_token_count


ASK = {
    "claude": ask_claude,
    "gpt": ask_gpt,
    "gemini": ask_gemini,
}


def run(name, prompt, system, model=None):
    """Ask one model and time it. Never raises, errors come back in the result."""
    model = model or DEFAULT_MODELS[name]
    result = {"name": name, "model": model}
    start = time.perf_counter()
    try:
        text, tokens_in, tokens_out = ASK[name](prompt, system, model)
        result.update(text=text, tokens_in=tokens_in, tokens_out=tokens_out)
    except Exception as e:
        result["error"] = str(e)
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result


def available():
    """Providers that have an API key set."""
    return [name for name, key in API_KEYS.items() if os.getenv(key)]
