"""
OpenEnv-compatible HTTP server for AI_Runbook.
Place this file at: server/app.py
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import RunbookEnv
from tasks import list_tasks
from grader import grade

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Incident Response Env",
    description="OpenEnv-compatible benchmark environment",
    version="1.0.0",
)

# ── Global session state ──────────────────────────────────────────────────────
_env: RunbookEnv | None = None
_current_task = None
_tasks = {t.id: t for t in list_tasks()}
_action_history: list[str] = []

_difficulty_map = {
    "easy":   next((tid for tid in _tasks if "easy"   in tid), None),
    "medium": next((tid for tid in _tasks if "medium" in tid), None),
    "hard":   next((tid for tid in _tasks if "hard"   in tid), None),
}


# ── Schemas ───────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: str = "easy"

class StepRequest(BaseModel):
    action: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_task_id(task_label: str) -> str:
    if task_label in _tasks:
        return task_label
    mapped = _difficulty_map.get(task_label.lower())
    if mapped:
        return mapped
    raise HTTPException(
        status_code=400,
        detail=f"Unknown task '{task_label}'. Available: {list(_tasks.keys())}"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "autonomous-incident-response-env",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/reset", "/step", "/validate", "/state"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    global _env, _current_task, _action_history
    task_id = _resolve_task_id(body.task)
    _current_task = _tasks[task_id]
    _env = RunbookEnv(_current_task)
    obs = _env.reset()
    _action_history = []
    return {
        "task_id":         task_id,
        "description":     obs.get("description", ""),
        "allowed_actions": obs.get("allowed_actions", []),
        "action_map":      obs.get("action_map", {}),
        "current_step":    obs.get("current_step", 1),
        "remaining_steps": obs.get("remaining_steps", _current_task.max_steps),
        "progress_ratio":  obs.get("progress_ratio", 0.0),
        "done":            False,
    }

@app.post("/step")
def step(body: StepRequest):
    global _action_history
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    obs, reward, done, info = _env.step(body.action)
    _action_history.append(body.action)
    return {
        "action":          body.action,
        "reward":          reward,
        "done":            done,
        "progress_ratio":  obs.get("progress_ratio", 0.0),
        "allowed_actions": obs.get("allowed_actions", []),
        "action_map":      obs.get("action_map", {}),
        "current_step":    obs.get("current_step", 0),
        "remaining_steps": obs.get("remaining_steps", 0),
        "history":         info.get("history", []),
    }

@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _env.state()

@app.get("/validate")
@app.post("/validate")
def validate():
    if _env is None or _current_task is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    final_state = _env.state()
    action_history = [
        item[0] if isinstance(item, tuple) else item
        for item in final_state.get("history", [])
    ]
    result = grade(actions=action_history, correct_steps=_current_task.steps)
    return {
        "task_id":             _current_task.id,
        "score":               result["score"],
        "accuracy_percentage": result["accuracy_percentage"],
        "correct_matches":     result["correct_matches"],
        "total_steps":         result["total_steps"],
        "action_history":      action_history,
    }


# ── Entry point (required by openenv validate checker) ────────────────────────
def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
