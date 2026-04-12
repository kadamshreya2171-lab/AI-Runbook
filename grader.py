@'
from __future__ import annotations
from typing import Any

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def _clamp(score):
    return max(_SCORE_MIN, min(_SCORE_MAX, score))

def _build_mistakes(actions, correct_steps):
    mistakes = []
    n = min(len(actions), len(correct_steps))
    for i in range(n):
        if actions[i] != correct_steps[i]:
            mistakes.append({"index": i, "expected": correct_steps[i], "actual": actions[i]})
    for i in range(n, len(correct_steps)):
        mistakes.append({"index": i, "expected": correct_steps[i], "actual": None})
    return mistakes

def grade(actions, correct_steps):
    total = len(correct_steps)
    n = min(len(actions), total)
    correct = sum(1 for i in range(n) if actions[i] == correct_steps[i])
    raw = correct / total if total > 0 else 1.0
    score = _clamp(raw)
    return {
        "score": score,
        "correct_matches": correct,
        "incorrect_matches": total - correct,
        "total_steps": total,
        "accuracy_percentage": score * 100.0,
        "correct_steps_count": correct,
        "mistakes": _build_mistakes(actions, correct_steps),
    }
'@ | Set-Content grader.py
