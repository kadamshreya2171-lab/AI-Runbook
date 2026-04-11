import os
import random
import json
from openai import OpenAI

from env import RunbookEnv
from tasks import Task, list_tasks
from grader import grade

# ── LiteLLM proxy client (injected by hackathon checker) ─────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")

client = None
if API_BASE_URL and API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

USED_ACTIONS: dict[str, list[str]] = {}


def smart_fallback(task_id: str, allowed_actions: list[str]) -> str:
    if task_id not in USED_ACTIONS:
        USED_ACTIONS[task_id] = []
    used = USED_ACTIONS[task_id]
    remaining = [a for a in allowed_actions if a not in used]
    action = remaining[0] if remaining else random.choice(allowed_actions)
    used.append(action)
    return action


def get_action_from_llm(observation: dict, allowed_actions: list[str]) -> str:
    """Call the hackathon LiteLLM proxy to select an action."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a DevOps incident response agent. Return ONLY one valid action token from the allowed list. No explanation."
                },
                {
                    "role": "user",
                    "content": (
                        f"Incident: {observation.get('description', '')}\n"
                        f"Step: {observation.get('current_step', 1)}\n"
                        f"Allowed actions: {json.dumps(allowed_actions)}\n"
                        f"Reply with exactly one action token."
                    )
                }
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        token = raw.strip().split()[0].lower().strip(".,;:")
        if token in allowed_actions:
            return token
    except Exception as e:
        print(f"LLM call failed: {e}, using fallback", flush=True)

    return smart_fallback(observation.get("task_id", "default"), allowed_actions)


def get_action(observation: dict, allowed_actions: list[str]) -> str:
    if client:
        return get_action_from_llm(observation, allowed_actions)
    return smart_fallback(observation.get("task_id", "default"), allowed_actions)


def run_inference(task: Task) -> float:
    print(f"[START] task={task.id}", flush=True)

    USED_ACTIONS[task.id] = []

    env = RunbookEnv(task)
    obs = env.reset()
    obs["task_id"] = task.id

    done = False
    max_iter = task.max_steps + 5
    iter_count = 0
    step_num = 0

    while not done and iter_count < max_iter:
        iter_count += 1
        allowed_actions = obs["allowed_actions"]
        action_map = obs["action_map"]
        action = get_action(obs, allowed_actions)

        if action == "STOP_EXECUTION":
            break

        obs, reward, done, info = env.step(action)
        obs["task_id"] = task.id
        step_num = len(info["history"])

        print(f"[STEP] step={step_num} action={action} reward={reward:.4f}", flush=True)

    final_state = env.state()
    action_history = [
        item[0] if isinstance(item, tuple) else item
        for item in final_state["history"]
    ]

    grading_result = grade(actions=action_history, correct_steps=task.steps)
    final_score = grading_result["score"]

    print(f"[END] task={task.id} score={final_score:.4f} steps={step_num}", flush=True)

    return final_score


def main():
    print("🚀 Starting Autonomous Incident Response Environment...", flush=True)
    if client:
        print("✅ LLM proxy connected", flush=True)
    else:
        print("⚠️ No LLM proxy — using smart fallback", flush=True)

    tasks = list_tasks()
    total_score = 0.0

    for task in tasks:
        score = run_inference(task)
        total_score += score

    avg_score = total_score / len(tasks) if tasks else 0.0
    print(f"\n=== OVERALL AVERAGE SCORE: {avg_score:.2f} ===", flush=True)


if __name__ == "__main__":
    main()
