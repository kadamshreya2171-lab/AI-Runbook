from __future__ import annotations

from typing import Any


def _build_mistakes(actions: list[str], correct_steps: list[str]) -> list[dict[str, Any]]:
	mistakes: list[dict[str, Any]] = []
	comparison_length = min(len(actions), len(correct_steps))

	for index in range(comparison_length):
		expected = correct_steps[index]
		actual = actions[index]
		if actual != expected:
			mistakes.append({"index": index, "expected": expected, "actual": actual})

	for index in range(comparison_length, len(correct_steps)):
		mistakes.append({"index": index, "expected": correct_steps[index], "actual": None})

	return mistakes


def grade(actions: list[str], correct_steps: list[str]) -> dict[str, Any]:
	total_steps = len(correct_steps)
	comparison_length = min(len(actions), total_steps)

	correct_matches = 0
	incorrect_matches = 0

	for index in range(comparison_length):
		if actions[index] == correct_steps[index]:
			correct_matches += 1
		else:
			incorrect_matches += 1

	incorrect_matches += total_steps - comparison_length

	if total_steps == 0:
		score = 1.0
	elif correct_matches == total_steps:
		score = 1.0
	else:
		score = correct_matches / total_steps

	accuracy_percentage = score * 100.0
	mistakes = _build_mistakes(actions=actions, correct_steps=correct_steps)

	return {
		"score": score,
		"correct_steps_count": correct_matches,
		"total_steps": total_steps,
		"accuracy_percentage": accuracy_percentage,
		"correct_matches": correct_matches,
		"incorrect_matches": incorrect_matches,
		"mistakes": mistakes,
	}
