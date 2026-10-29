"""
Autonomous Embedded AI Issue Agent:
- Zero external API keys (no OpenAI/Claude/Gemini quota limits)
- Zero external daemon software (no Ollama server required)
- Uses in-process llama-cpp-python with a local GGUF model
- Intelligent text processing: extracts stack traces & code blocks, smart-chunks context
- Outputs structured, GitHub-ready Pull Request markdown with git diffs and test plans
"""
import os, re
from llama_cpp import Llama
from llm.rag import RepoRAG

DEFAULT_MODEL = "models/qwen2.5-coder-0.5b.Q4_K_M.gguf"


def extract_critical_snippets(text: str) -> dict:
    """
    Intelligently extracts high-value signals from GitHub issue text:
    - Code blocks (```...```)
    - Stack traces / exceptions (Traceback (most recent call last)...)
    - File path and line number references
    """
    if not text:
        return {"code_blocks": [], "stack_traces": [], "clean_text": ""}

    # 1. Extract stack traces
    trace_pattern = re.compile(
        r"(Traceback \(most recent call last\):[\s\S]*?(?:Error|Exception):[^\n]+)",
        re.MULTILINE
    )
    stack_traces = [m.strip() for m in trace_pattern.findall(text)]

    # 2. Extract markdown code blocks
    code_pattern = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)```")
    code_blocks = [m.strip() for m in code_pattern.findall(text)]

    # 3. Clean remaining prose text by removing code fences to avoid duplicate context
    clean_prose = code_pattern.sub("[code snippet]", text)
    clean_prose = re.sub(r"\s+", " ", clean_prose).strip()

    return {
        "code_blocks": code_blocks[:3],          # prioritize top 3 code snippets
        "stack_traces": stack_traces[:2],        # prioritize top 2 traces
        "clean_text": clean_prose
    }


def smart_chunk_context(clean_text: str, stack_traces: list, code_blocks: list, max_chars: int = 900) -> str:
    """
    Builds a budget-constrained context representation preserving critical code signals first.
    """
    sections = []
    budget = max_chars

    # 1. Stack traces have highest triage priority
    if stack_traces:
        trace_str = "STACK TRACE:\n" + stack_traces[0][:350]
        sections.append(trace_str)
        budget -= len(trace_str)

    # 2. Code blocks provide exact error reproduction
    if code_blocks and budget > 200:
        code_str = "RELEVANT SNIPPET:\n" + code_blocks[0][:250]
        sections.append(code_str)
        budget -= len(code_str)

    # 3. Remaining budget allocated to issue description prose
    if budget > 100:
        sections.append("DESCRIPTION: " + clean_text[:budget])

    return "\n\n".join(sections)


class LocalIssueAgent:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load GGUF model directly in Python memory (4 CPU threads, 2048 token context)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )

    def analyze(self, repo: str, issue_title: str, issue_body: str = "") -> str:
        """
        Executes end-to-end issue triage, RAG retrieval, and generates a structured
        GitHub-ready Pull Request plan with code diffs.
        """
        # 1. Signal extraction and smart chunking
        signals = extract_critical_snippets(issue_body)
        optimized_issue_context = smart_chunk_context(
            signals["clean_text"],
            signals["stack_traces"],
            signals["code_blocks"],
            max_chars=900
        )

        # 2. RAG Retrieval from repository docs
        rag = RepoRAG(repo, backend="auto")
        query = f"{issue_title} {' '.join(signals['clean_text'].split()[:15])}"
        context_chunks = rag.retrieve(query, k=2)
        guidelines = "\n---\n".join([c["text"] for c in context_chunks])[:600]

        # 3. Structured system prompt enforcing GitHub PR format with git diff
        prompt = f"""<|im_start|>system
You are a Principal Open-Source Software Engineer.
Given an issue, stack traces, and repo contributing rules, provide a structured GitHub PR response containing:
1. ### 🔍 Root Cause Analysis
2. ### 📋 Step-by-Step Resolution Roadmap
3. ### 💻 Proposed Code Fix / Git Diff (Use ```diff or ```python)
4. ### 🧪 Verification & Unit Test Plan
<|im_end|>
<|im_start|>user
[REPO GUIDELINES]
{guidelines}

[ISSUE DETAILS]
Title: {issue_title}
{optimized_issue_context}

Please generate the complete Pull Request analysis and code solution:
<|im_end|>
<|im_start|>assistant
"""
        response = self.llm(
            prompt,
            max_tokens=650,
            stop=["<|im_end|>"],
            temperature=0.2
        )
        
        solution_text = response["choices"][0]["text"].strip()
        
        # Format with GitHub PR metadata header if not already present
        header = f"""## 🚀 GitHub Issue Resolution & PR Kit: `{repo}`
**Target Issue:** {issue_title}

"""
        return header + solution_text
