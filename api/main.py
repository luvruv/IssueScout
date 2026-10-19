"""Deployment layer.  uvicorn api.main:app --port 8000   then open http://localhost:8000/docs"""
import json, os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.profile import Profile
from core.recommend import recommend, learning_path
from core import outcomes

app = FastAPI(title="IssueScout", version="2.0")


class Feedback(BaseModel):
    issue_id: str
    verdict: str            # "tried" | "too_hard" | "good"


class Ref(BaseModel):
    ref: str                # owner/repo#123 or an issue URL
    use_llm: bool = False


class Question(BaseModel):
    repo: str
    question: str
    use_llm: bool = True


class Outcome(BaseModel):
    issue_id: str
    status: str
    hours: float | None = None
    notes: str = ""


@app.get("/health")
def health():
    return {"ok": True, "difficulty_model": os.path.exists("models/difficulty.joblib")}


@app.post("/recommend")
def rec(p: Profile):
    return recommend(p)


@app.post("/path")
def path(p: Profile, target: str):
    return learning_path(p, target)


@app.post("/predict")
def predict(r: Ref):
    from ml.predict import predict_issue
    try:
        return predict_issue(r.ref)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/prepare")
def prepare(r: Ref):
    from llm.first_pr_kit import build_kit
    try:
        return build_kit(r.ref, use_llm=r.use_llm)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/ask")
def ask(q: Question):
    from llm.rag import ask as rag_ask
    return rag_ask(q.repo, q.question, use_llm=q.use_llm)


@app.post("/feedback")
def feedback(f: Feedback):
    os.makedirs("data", exist_ok=True)
    with open("data/feedback.jsonl", "a") as fh:
        fh.write(json.dumps({**f.model_dump(), "at": datetime.utcnow().isoformat()}) + "\n")
    return {"saved": True}


@app.post("/outcome")
def outcome(o: Outcome):
    try:
        return outcomes.add(o.issue_id, o.status, o.hours, o.notes)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/outcomes")
def outcome_summary():
    return outcomes.summary()