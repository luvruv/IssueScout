"""One LLM entry point.  LLM_PROVIDER = openai | anthropic | mock | none   (mock = offline, for tests and demos)
Failures never crash the app: complete() returns None and the error goes to data/llm_errors.log."""
import json, os, re, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

_LAST = {"error": None}


def _log(msg):
    _LAST["error"] = msg
    os.makedirs("data", exist_ok=True)
    with open("data/llm_errors.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")


def last_error():
    return _LAST["error"]


def provider():
    return os.getenv("LLM_PROVIDER", "none").lower()


def _mock(prompt):
    if "Return ONLY JSON" in prompt:                                   # skill extraction
        m = re.search(r"from this list only: \[(.*?)\]", prompt, re.S)
        allowed = [a.strip(" '\"") for a in m.group(1).split(",")] if m else []
        body = prompt.split("Title:", 1)[-1].lower()
        return json.dumps({"skills": [a for a in allowed if a and re.search(rf"\b{re.escape(a)}\b", body)][:5]})
    if "FACTS:" in prompt:                                             # explanation
        f = json.loads(prompt.split("FACTS:", 1)[1].split("ISSUE TEXT:", 1)[0].strip())
        return (f"This looks like a {f['difficulty']} task with about {f['est_hours']} hours of work and a "
                f"{round(f['p_merge'] * 100)}% chance of a merged PR. It needs {', '.join(f['skills']) or 'no special skills'}.\n"
                "Starter steps:\n- Read the issue and reproduce it.\n- Find the code it mentions.\n- Open a small PR.")
    if "excerpts" in prompt.lower():                                   # RAG: extractive echo of the first excerpt
        ctx = prompt.split("EXCERPTS:", 1)[-1].split("QUESTION:", 1)[0]
        return " ".join(ctx.strip().split("\n")[1:4])[:600] or "NOT FOUND IN DOCS"
    return "OK"


def complete(prompt, max_tokens=500, system=None):
    p = provider()
    t0 = time.time()
    try:
        if p == "mock":
            return _mock(prompt)
        if p == "ollama":
            from openai import OpenAI
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            client = OpenAI(base_url=base_url, api_key="ollama", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            model = os.getenv("LLM_MODEL", "llama3.2:1b")
            r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens)
            return r.choices[0].message.content
        if p == "openai":
            from openai import OpenAI
            client = OpenAI(timeout=float(os.getenv("LLM_TIMEOUT", "60")))
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
            try:     # newer models want max_completion_tokens (and spend some of it on reasoning), older ones max_tokens
                r = client.chat.completions.create(model=model, messages=msgs, max_completion_tokens=max(max_tokens, 800))
            except Exception as e:
                if "max_completion_tokens" not in str(e) and "Unsupported parameter" not in str(e):
                    raise
                r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens)
            return r.choices[0].message.content
        if p == "anthropic":
            import anthropic
            kw = {"system": system} if system else {}
            m = anthropic.Anthropic(timeout=float(os.getenv("LLM_TIMEOUT", "60"))).messages.create(
                model=os.getenv("LLM_MODEL") or "claude-haiku-5-5", max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **kw)
            return m.content[0].text
    except Exception as e:
        _log(f"{p} failed after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:300]}")
    return None


def ping():
    """(ok, message, seconds). Use `python cli.py llm-check` to find a wrong model name or key BEFORE a demo."""
    p = provider()
    if p == "none":
        return False, "LLM_PROVIDER=none (set openai, anthropic or mock in .env)", 0.0
    t = time.time(); out = complete("Reply with exactly: OK", 20)
    return (out is not None and len(out) > 0), (out if out else f"FAILED: {last_error()}"), round(time.time() - t, 2)