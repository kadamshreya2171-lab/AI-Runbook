---
title: AI Runbook - Autonomous Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
pinned: false
---
# 🚨 AI Runbook — Autonomous Incident Response Environment

An **OpenEnv-compatible reinforcement learning environment** for training and evaluating AI agents on real-world DevOps incident response tasks.

Agents must follow structured runbooks — diagnosing incidents, mitigating failures, and restoring system health — step by step, in the correct order.

---

## 🎯 What This Environment Models

Modern SRE and DevOps teams rely on runbooks: ordered sequences of diagnostic and mitigation actions for known failure patterns. This environment simulates that workflow, requiring an agent to:

1. Observe an incident description and available actions
2. Select the correct next step from the runbook
3. Receive reward proportional to progress through the correct sequence
4. Complete all steps to resolve the incident

This is a **genuine training signal** — agents that score well here have learned real incident response reasoning, not just pattern matching.

---

## 📋 Tasks

| Task ID | Name | Difficulty | Steps |
|---|---|---|---|
| `cpu_spike_easy` | Investigate CPU Spike on API Node | Easy | 3 |
| `db_connection_pool_medium` | Stabilize DB Connection Pool Exhaustion | Medium | 5 |
| `k8s_region_outage_hard` | Handle Regional K8s Control Plane Outage | Hard | 7 |

### Difficulty Progression
- **Easy**: Single-node CPU spike — diagnose and scale
- **Medium**: Multi-step DB triage — inspect, identify, mitigate, throttle, track
- **Hard**: Full regional failover — declare, verify, promote, reroute, validate, communicate, plan recovery

---

## 🏗️ Environment Design

### Observation Space
Each step, the agent receives:
```json
{
  "description": "Incident description text",
  "current_step": 2,
  "remaining_steps": 3,
  "allowed_actions": ["action_a", "action_b", ...],
  "action_map": {"action_a": "Human-readable description"},
  "progress_ratio": 0.4,
  "incident_state": "Mitigation in progress. Step 2 completed."
}
```

### Action Space
Discrete set of named action tokens per task (e.g. `check_cpu`, `declare_incident`). Actions outside the allowed set are penalized.

### Reward Shaping
| Outcome | Reward |
|---|---|
| Correct action | `0.2 + 0.6 × progress_ratio` (grows as episode progresses) |
| Wrong action | `-0.4` |
| Invalid action | `-1.0` |
| Episode completion bonus | `+0.5` |
| Too many wrong steps (≥3) | `-0.5` penalty + episode ends |

This reward structure encourages agents to act correctly early and maintain accuracy throughout.

### Episode Boundaries
- Episode ends when: all steps completed ✅, max wrong steps reached ❌, or max_steps exceeded ⏱️
---
## 🔌 API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode. Body: `{"task": "easy\|medium\|hard"}` |
| `POST` | `/step` | Take action. Body: `{"action": "action_token"}` |
| `GET` | `/validate` | Grade current episode |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check |

---

## 📁 Project Structure

```
AI_Runbook/
├── env.py          # RunbookEnv — core RL environment
├── tasks.py        # Task definitions, ACTION_MAP, Pydantic models
├── grader.py       # Scoring logic with partial credit
├── inference.py    # Agent runner with LiteLLM proxy support
├── server/
│   └── app.py      # FastAPI server (OpenEnv HTTP spec)
├── Dockerfile      # Container definition
└── openenv.yaml    # OpenEnv specification
```

---

## 🤖 Agent Integration

```python
from openai import OpenAI
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)
# POST /reset → get observation
# Feed observation to LLM → get action token
# POST /step with action → get next observation + reward
# Repeat until done=True
# GET /validate → get final score
```

---

## 🧪 Grading

Scores are computed as `correct_steps / total_steps`, clamped to `(0.01, 0.99)`.
Partial credit is awarded — an agent completing 4/5 steps correctly scores ~0.80.

---

## 👥 Authors

Built for the **OpenEnv Hackathon** by Scaler × Meta.
