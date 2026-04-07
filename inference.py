import os
import json
from typing import Any
try:
	from dotenv import load_dotenv
	load_dotenv()
except ImportError:
	pass

import openai

from env import RunbookEnv
from tasks import Task, list_tasks
from grader import grade

# Initialize globally for get_action signature matching
api_key = os.environ.get("OPENAI_API_KEY")
client = openai.Client(api_key=api_key) if api_key else None


def get_action(observation: dict, allowed_actions: list[str]) -> str:
	if not client:
		print("Error: OpenAI client not initialized.")
		return allowed_actions[0] if allowed_actions else "STOP_EXECUTION"
		
	system_prompt = (
		"You are a DevOps incident response agent. You MUST return ONLY one valid action token. "
		"No explanation, no extra words."
	)
	
	user_prompt = (
		f"Incident Description: {observation['description']}\n"
		f"Current Step: {observation['current_step']}\n"
		f"Remaining Steps: {observation['remaining_steps']}\n"
		f"Allowed Actions:\n{json.dumps(allowed_actions, indent=2)}\n"
	)
	
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt}
	]
	
	max_retries = 2
	fallback_action = allowed_actions[0] if allowed_actions else "STOP_EXECUTION"

	for attempt in range(max_retries + 1):
		try:
			response = client.chat.completions.create(
				model="gpt-4o-mini",
				messages=messages,
				temperature=0.0,
				max_tokens=20,
				timeout=10.0
			)
			
			raw_output = response.choices[0].message.content or ""
			
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
