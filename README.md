# 🚀 Autonomous Incident Response Benchmark Environment
*The ultimate proving ground for AI SRE agents.*

---

## 🧠 The Problem
Production incidents are stressful, expensive, and rely heavily on manual human intervention. Site Reliability Engineers (SREs) follow strict runbooks to stabilize systems, but executing these steps takes time—minutes that cost companies thousands of dollars. We need autonomous agents that can seamlessly execute runbooks when alerts fire, but how do we know we can trust them?

## 💡 Our Solution
Built by team **ByteBrains**, this OpenEnv-powered benchmark is a deterministic sandbox specifically designed to test, train, and evaluate AI agents on executing DevOps incident runbooks. By feeding the agent realistic system alerts and a strict set of allowed actions, we create an environment that mathematically grades an AI's ability to logically resolve outages without hallucination.

## ⚙️ How It Works 
The environment loops exactly like a real incident console:
1. `reset()` → Incident fires off (e.g., CPU spiked to 99%).
2. `observation` → The AI receives context, incident state, and allowed action tokens.
3. `AI action` → The model strictly selects a specialized action token.
4. `step()` → The environment processes the action.
5. `reward` → The agent is rewarded for correct execution or heavily penalized for straying off the runbook.
6. **Repeat** until the system stabilizes or the agent fails.

## 🧩 Key Features
- **Action Token System**: Instead of bloated descriptive strings, agents choose from concise action tokens (e.g., `check_cpu`). This dramatically drops AI hallucination rates and forces structural reliability.
- **Deterministic Grading (0.0 to 1.0)**: Zero randomness. An agent's final score is a hard mathematical reflection of its adherence to the exact sequence.
- **Context-Aware Observations**: The `incident_state` dynamically updates based on mitigation progress, giving the AI realistic, contextual feedback as it works.
- **Real-World Failure Consequences**: Tripping over wrong steps increments an internal failure counter. Hit the threshold, and the episode terminates early with a massive penalty—mimicking real production catastrophes.
- **Action Reasoning Tracking**: Every step taken natively supports reasoning memory, providing full explainability into *why* the AI chose its action.

## 📊 Tasks Overview
The benchmark comes pre-loaded with escalating incident complexities:
- 🟢 **Easy (CPU Spike)**: 3 steps. An API node is burning CPU. Diagnose, check logs, scale out.
- 🟡 **Medium (DB Connection Exhaustion)**: 5 steps. The database pool is maxed out. Inspect metrics, identify queries, throttle endpoints, open optimization tickets.
- 🔴 **Hard (K8s Regional Outage)**: 7+ steps. Total control plane failure in the primary region. Declare severity, verify secondary clusters, route global traffic, run synthetics.

## 🤖 AI Agent Integration
We built a robust, production-ready inference execution loop (`inference.py`) interacting with the OpenAI API. 
- **Strict Parsing**: The agent is forced by aggressive prompting and temperature 0.0 to return strictly formatted action tokens. 
- **Resilience**: It dynamically validates outputs against allowed actions, utilizing auto-retry mechanisms for parsing failures.
- **Safe Fallbacks**: If the AI critically hallucinates past retries, the loop executes a safe fallback default action, ensuring the engine never crashes.

## 📦 Project Structure
```text
ai-runbook-env/
├── env.py              # Core RunbookEnv logic & contextual state
├── tasks.py            # Task definitions & Action Token mapping
├── models.py           # Pydantic validation schemas
├── grader.py           # Mathematical correctness evaluation
├── inference.py        # OpenAI Agent execution loop
├── test_env.py         # Deterministic offline suite
├── Dockerfile          # HF Spaces deployment blueprint
├── openenv.yaml        # OpenEnv core registration
└── README.md
```

## ▶️ How to Run
It's incredibly simple to start testing locally:
```bash
# 1. Install dependencies
pip install openenv-core openai pydantic python-dotenv

# 2. Add your API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 3. Run the complete AI benchmark loop
python inference.py
```

## 🐳 Docker Usage
Ready for production evaluation or Hugging Face Spaces. It uses a hyper-optimized `python:3.12-slim` image:
```bash
# Build the clean image
docker build -t ai-runbook-env .

# Run the containerized benchmark
docker run --env-file .env ai-runbook-env
```

## ✅ Testing
We believe in unbreakable foundations. Run `python test_env.py` to fire our robust manual test suite—validating everything from happy paths to edge-case boundary faults and invalid-action penalties. *(All tests currently pass with 100% success).*

## 🎯 Why This Project Stands Out
This isn't a toy project or a generic chatbot wrapper. It is a highly opinionated, cleanly modular, strictly evaluated reinforcement-learning environment modeled after real SRE on-call trauma. It evaluates the most crucial skill an AI agent must master before touching production: following the rules accurately and deterministically.

## 📌 Final Note
We built this because AI DevOps agents are the future, and the future needs a rigorous testing ground. We hope you enjoy breaking—and saving—it.  
Built with ❤️ by **ByteBrains**.
