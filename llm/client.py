import os


def complete(prompt, max_tokens=500):
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    try:
        if provider == "anthropic":
            import anthropic
            m = anthropic.Anthropic().messages.create(
                model=os.getenv("LLM_MODEL", "claude-sonnet-5"), max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return m.content[0].text
        if provider == "openai":
            from openai import OpenAI                     # pip install openai
            r = OpenAI().chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"), max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
    except Exception as e:
        print("LLM failed:", e)
    return None