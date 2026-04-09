import random
from typing import Any

from env import RunbookEnv
from tasks import Task, list_tasks
from grader import grade

# Global per-task action memory to avoid repeats
USED_ACTIONS: dict[str, list[str]] = {}


def smart_fallback(task_id: str, allowed_actions: list[str]) -> str:
    if task_id not in USED_ACTIONS:
        USED_ACTIONS[task_id] = []
    used = USED_ACTIONS[task_id]
    remaining = [a for a in allowed_actions if a not in used]
    action = remaining[0] if remaining else random.choice(allowed_actions)
    used.append(action)
    return action


def get_action(observation: dict, allowed_actions: list[str]) -> str:
    task_id = observation.get("task_id", "default")
    return smart_fallback(task_id, allowed_actions)


def run_inference(task: Task) -> float:
    print(f"\n--- Starting Inference for Task: {task.id} ---")
    USED_ACTIONS[task.id] = []

    env = RunbookEnv(task)
    obs = env.reset()
    obs["task_id"] = task.id

    done = False
    max_iter = task.max_steps + 5
    iter_count = 0

    while not done and iter_count < max_iter:
        iter_count += 1
        allowed_actions = obs["allowed_actions"]
        action_map = obs["action_map"]
        action = get_action(obs, allowed_actions)

        if action == "STOP_EXECUTION":
            print("Stopping execution.")
            break

        obs, reward, done, info = env.step(action)
        obs["task_id"] = task.id

        action_desc = action_map.get(action, "Unknown Action")
        progress = obs["progress_ratio"]
        step_num = len(info["history"])
        print(f"Step: {step_num} | Action: {action} ({action_desc}) | Reward: {reward:.2f} | Progress: {progress:.2f}")

    final_state = env.state()
    action_history = [
        item[0] if isinstance(item, tuple) else item
        for item in final_state["history"]
    ]

    grading_result = grade(actions=action_history, correct_steps=task.steps)
    final_score = grading_result["score"]

    print(f"\n--- Final Score for Task '{task.id}' ---")
    print(f"Score: {final_score:.2f} ({grading_result['accuracy_percentage']:.1f}%)")
    print(f"Correct Matches: {grading_result['correct_matches']} / {grading_result['total_steps']}")

    return final_score


def main():
    print("🚀 Starting Autonomous Incident Response Environment (Offline Mode)...")

    tasks = list_tasks()
    total_score = 0.0

    for task in tasks:
        score = run_inference(task)
        total_score += score

    avg_score = total_score / len(tasks) if tasks else 0.0

    print("\n" + "=" * 50)
    print(f"=== OVERALL AVERAGE SCORE: {avg_score:.2f} ===")
    print("=" * 50)


if __name__ == "__main__":
    main()
