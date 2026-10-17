import json, os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from core.profile import Profile
from core.recommend import recommend, learning_path

app = FastAPI(title="IssueScout")


class Feedback(BaseModel):
    issue_id: str
    verdict: str            # "tried" | "too_hard" | "good"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/recommend")
def rec(p: Profile):
    return recommend(p)


@app.post("/path")
def path(p: Profile, target: str):
    return learning_path(p, target)


@app.post("/feedback")
def feedback(f: Feedback):
    os.makedirs("data", exist_ok=True)
    with open("data/feedback.jsonl", "a") as fh:
        fh.write(json.dumps({**f.model_dump(), "at": datetime.utcnow().isoformat()}) + "\n")
    return {"saved": True}