"""
OpenEnv-compatible HTTP server for AI_Runbook (autonomous-incident-response-env).
Exposes /reset, /step, /validate endpoints expected by the hackathon checker.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import RunbookEnv
from tasks import list_tasks
from grader import grade

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Incident Response Env",
    description="OpenEnv-compatible benchmark environment",
    version="1.0.0",
)

# ── Global session state (one active episode at a time) ───────────────────────
_env: RunbookEnv | None = None
_current_task = None
_tasks = {t.id: t for t in list_tasks()}   # e.g. {"cpu_spike_easy": Task, ...}
_action_history: list[str] = []

# Map difficulty label → task id (openenv.yaml lists easy/medium/hard)
_difficulty_map = {
    "easy":   next((tid for tid in _tasks if "easy"   in tid), None),
    "medium": next((tid for tid in _tasks if "medium" in tid), None),
    "hard":   next((tid for tid in _tasks if "hard"   in tid), None),
}


# ── Request / Response schemas ────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: str = "easy"          # accepts difficulty label OR full task id


class StepRequest(BaseModel):
    action: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_task_id(task_label: str) -> str:
    """Accept 'easy' / 'medium' / 'hard' OR a full task id."""
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
    """Start a new episode. POST body: {"task": "easy"|"medium"|"hard"}"""
    global _env, _current_task, _action_history

    task_id = _resolve_task_id(body.task)
    _current_task = _tasks[task_id]
    _env = RunbookEnv(_current_task)
    obs = _env.reset()
    _action_history = []

    return {
        "task_id":        task_id,
        "description":    obs.get("description", ""),
        "allowed_actions": obs.get("allowed_actions", []),
        "action_map":     obs.get("action_map", {}),
        "current_step":   obs.get("current_step", 1),
        "remaining_steps": obs.get("remaining_steps", _current_task.max_steps),
        "progress_ratio": obs.get("progress_ratio", 0.0),
        "done":           False,
    }


@app.post("/step")
def step(body: StepRequest):
    """Advance the episode by one action. POST body: {"action": "<action_token>"}"""
    global _action_history

    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    obs, reward, done, info = _env.step(body.action)
    _action_history.append(body.action)

    return {
        "action":         body.action,
        "reward":         reward,
        "done":           done,
        "progress_ratio": obs.get("progress_ratio", 0.0),
        "allowed_actions": obs.get("allowed_actions", []),
        "action_map":     obs.get("action_map", {}),
        "current_step":   obs.get("current_step", 0),
        "remaining_steps": obs.get("remaining_steps", 0),
        "history":        info.get("history", []),
    }


@app.get("/state")
def state():
    """Return current episode state without advancing it."""
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _env.state()


@app.get("/validate")
@app.post("/validate")
def validate():
    """Grade the current episode and return the score."""
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)= response.choices[0].message.content or ""
			
			# Parse: lowercase, strip punct/spaces, get first word
			import string
			clean_output = raw_output.lower().strip(string.punctuation + " \n\r\t")
			parsed_token = clean_output.split()[0] if clean_output else ""
			
			print(f"Raw Output: '{raw_output}' | Parsed: '{parsed_token}'")
			
			if parsed_token in allowed_actions:
				print(f"Final Action: '{parsed_token}'")
				return parsed_token
			else:
				print(f"[Warning] Parsed token '{parsed_token}' not in allowed actions. Retrying...")
				
		except Exception as e:
			print(f"[Warning] OpenAI API Error: {e}")
	
	print("[Warning] Invalid model output, using fallback")
	print(f"Final Action: '{fallback_action}'")
	return fallback_action


def run_inference(task: Task) -> float:
	print(f"\n--- Starting Inference for Task: {task.id} ---")
	
	env = RunbookEnv(task)
	obs = env.reset()
	done = False
	
	max_iter = task.max_steps + 5
	iter_count = 0
	
	while not done and iter_count < max_iter:
		iter_count += 1
		allowed_actions = obs["allowed_actions"]
		action_map = obs["action_map"]
		
		# Exact signature as requested
		action = get_action(obs, allowed_actions)
		
		if action == "STOP_EXECUTION":
			print("Fallback triggered. Stopping sequence.")
			break
			
		obs, reward, done, info = env.step(action)
		
		action_desc = action_map.get(action, "Unknown Action")
		progress = obs["progress_ratio"]
		
		# e.g. Step: 1 | Action: check_cpu (Check CPU usage) | Reward: 0.33
		step_num = len(info['history'])
		print(f"Step: {step_num} | Action: {action} ({action_desc}) | Reward: {reward:.2f} | Progress: {progress:.2f}")

	final_state = env.state()
	
	# Extract purely the actions, since history is now a list of tuples: (action, reason)
	action_history = [item[0] if isinstance(item, tuple) else item for item in final_state["history"]]
	grading_result = grade(actions=action_history, correct_steps=task.steps)
	final_score = grading_result["score"]
	
	print(f"\n--- Final Score for Task '{task.id}' ---")
	print(f"Score: {final_score:.2f} ({grading_result['accuracy_percentage']:.1f}%)")
	print(f"Correct Matches: {grading_result['correct_matches']} / {grading_result['total_steps']}")
	
	return final_score


def main():
	if not api_key:
		print("Warning: OPENAI_API_KEY environment variable not set.")
		print("Please set it in your .env file or environment.")
		# We don't return here so that the script can still be run to see the error,
		# or user can set the env var and try again.
	
	tasks = list_tasks()
	total_score = 0.0
	
	for task in tasks:
		score = run_inference(task)
		total_score += score
		
	avg_score = total_score / len(tasks) if tasks else 0.0
	print("\n" + "="*50)
	print(f"=== OVERALL AVERAGE SCORE: {avg_score:.2f} ===")
	print("="*50 + "\n")


if __name__ == "__main__":
	main()
