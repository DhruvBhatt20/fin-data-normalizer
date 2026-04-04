---
title: Financial Data Normalizer
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Financial Data Normalizer — OpenEnv Environment

A real-world reinforcement learning environment where an AI agent acts as a **financial data analyst**, processing messy, inconsistent financial data across three tasks of increasing difficulty.

Built for the [Meta PyTorch OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv).

---

## Motivation

Every financial institution, fintech firm, and research team deals with the same problem: financial data arrives from multiple sources in inconsistent formats — different units, currencies, date conventions, and conflicting values. Normalizing this data is a critical, time-consuming task that is an ideal candidate for AI agent automation.

This environment simulates exactly that workflow, forcing an agent to:
- Parse and convert financial figures across units and currencies
- Extract structured metrics from unstructured central bank text
- Resolve conflicts across multiple data sources using explicit priority rules

---

## Environment Description

**Name:** `fin_data_normalizer`  
**Type:** Single-turn task environment (one `step()` per episode)  
**Interface:** OpenEnv standard (`reset()` / `step()` / `state()`)  
**Deployment:** HuggingFace Spaces (Docker, FastAPI, WebSocket)

---

## Action Space

The agent submits a single structured JSON object as its action:

```python
class FinDataNormalizerAction(Action):
    result: Dict[str, Any]  # Task-specific answer dictionary
```

The structure of `result` depends on the active task (see Task Descriptions below).

---

## Observation Space

```python
class FinDataNormalizerObservation(Observation):
    task_name: str           # Active task identifier
    task_description: str    # Full instructions for the agent
    task_data: Dict          # Raw input data to process
    difficulty: str          # "easy" | "medium" | "hard"
    score: float             # Score awarded (0.0–1.0), 0.0 on reset
    feedback: str            # Human-readable explanation of score
    fields_correct: List[str]  # Fields the agent got right
    fields_wrong: List[str]    # Fields the agent got wrong
    done: bool               # True after step() is called
    reward: float            # Same as score
```

---

## Task Descriptions

### Task 1: Unit Normalization (Easy)

**Objective:** Normalize a list of company revenue figures to Crores INR from mixed units and currencies.

**Input:** List of companies with revenues in formats like `"1,23,456 Cr"`, `"12.34 Billion USD"`, `"1234560 Lakhs"`, `"~890 Cr"`, `"Data not available"`.

**Expected Output:**
```json
{
  "normalized": [
    {"company": "Reliance", "revenue_cr": 123456.0},
    {"company": "Infosys", "revenue_cr": 103039.0},
    {"company": "TCS", "revenue_cr": 123456.0},
    {"company": "Wipro", "revenue_cr": 890.0},
    {"company": "HCL", "revenue_cr": null}
  ]
}
```

**Grader:** Each company is scored independently (partial credit). Approximate values (`~`) must be parsed. Unavailable data must return `null` not `0`.

| Company | Weight | Challenge |
|---------|--------|-----------|
| Reliance | 0.15 | Standard Cr format |
| Infosys | 0.25 | Billion USD → Cr conversion |
| TCS | 0.20 | Lakhs → Cr conversion |
| Wipro | 0.20 | Strip `~` prefix |
| HCL | 0.20 | Return `null` for unavailable |

---

### Task 2: RBI Metric Extraction (Medium)

**Objective:** Extract six structured metrics from an RBI Monetary Policy Committee (MPC) statement written in natural language (numbers as words).

**Input:** Paragraph of MPC statement text including phrases like *"six point five percent"*, *"thirty basis points"*, *"five to one"*.

**Expected Output:**
```json
{
  "repo_rate_pct": 6.5,
  "sdf_rate_pct": 6.25,
  "cpi_inflation_pct": 5.1,
  "core_inflation_pct": 4.2,
  "core_inflation_change_bps": -30,
  "mpc_vote": "5-1"
}
```

**Grader:** Each of the 6 fields is worth 1/6 of the total score. Numeric tolerance: ±0.01. `core_inflation_change_bps` must be **negative** (easing = reduction).

---

### Task 3: Conflict Resolution (Hard)

**Objective:** Given the same financial metric reported by 4 sources with different reliability tiers and publication dates, apply explicit priority rules to resolve the conflict.

**Rules:**
- `RULE_1`: Audited sources take highest priority
- `RULE_2`: Among same reliability tier, most recent `date_published` wins
- `RULE_3`: Normalize all values to Crores INR before comparison
- `RULE_4`: Flag `conflicts_detected=true` if any source differs by >0.5 Cr

**Expected Output:**
```json
{
  "resolved_value_cr": 234.0,
  "chosen_source": "Annual Report",
  "rule_applied": "RULE_1",
  "conflicts_detected": true,
  "conflict_detail": "sources differ by up to 0.7 Cr after normalization"
}
```

**Grader:** Field-level partial credit (resolved_value: 0.35, chosen_source: 0.25, rule_applied: 0.25, conflicts_detected: 0.15).

---

## Reward Function

All rewards are in `[0.0, 1.0]`. The environment provides **dense partial credit** — agents receive signal proportional to how many fields they get correct, not just binary win/lose. This makes the reward function useful for RL training signal across the full trajectory.

| Task | Scoring Method |
|------|---------------|
| Unit Normalization | Per-company weighted score |
| Metric Extraction | Per-field uniform score (1/6 each) |
| Conflict Resolution | Per-field weighted score |

---

## Setup & Usage

### Prerequisites
- Docker
- Python 3.10+
- `openenv-core>=0.2.3`

### Local Development

```bash
# Clone the repo
git clone https://github.com/DhruvBhatt20/fin-data-normalizer.git
cd fin-data-normalizer

# Build Docker image
docker build -t fin_data_normalizer_env:latest -f server/Dockerfile .

# Run the server
docker run -d -p 8000:8000 fin_data_normalizer_env:latest

# Test health
curl http://localhost:8000/health
```

### Python Client

```python
import asyncio
from client import FinDataNormalizerEnv
from models import FinDataNormalizerAction

async def main():
    async with FinDataNormalizerEnv(base_url="http://localhost:8000") as env:
        # Reset to a specific task
        result = await env.reset(task_name="unit_normalization")
        print(result.observation.task_description)

        # Submit answer
        action = FinDataNormalizerAction(result={
            "normalized": [
                {"company": "Reliance", "revenue_cr": 123456.0},
                # ...
            ]
        })
        result = await env.step(action)
        print(f"Score: {result.observation.score}")
        print(f"Feedback: {result.observation.feedback}")

asyncio.run(main())
```

### Running the Baseline Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
export LOCAL_IMAGE_NAME="fin_data_normalizer_env:latest"

python3 inference.py
```

---

## Baseline Scores

Scores achieved by `Qwen/Qwen2.5-72B-Instruct` on the hosted environment:

| Task | Difficulty | Baseline Score |
|------|-----------|----------------|
| unit_normalization | Easy | 0.550 |
| metric_extraction | Medium | 1.000 |
| conflict_resolution | Hard | 1.000 |
| **Average** | | **0.850** |

*Note: Scores are approximate and may vary across runs due to model temperature.*

---

## Deployment

The environment is deployed as a HuggingFace Space:

**Space URL:** https://huggingface.co/spaces/DhruvBhatt20/fin-data-normalizer-env  
**API Base:** https://dhruvbhatt20-fin-data-normalizer-env.hf.space  
**Health:** https://dhruvbhatt20-fin-data-normalizer-env.hf.space/health  
**Docs:** https://dhruvbhatt20-fin-data-normalizer-env.hf.space/docs  

---

## Project Structure

```
fin_data_normalizer/
├── models.py                          # Action/Observation/State types (Pydantic)
├── client.py                          # WebSocket client for training code
├── inference.py                       # Baseline inference script (OpenAI client)
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Project dependencies
├── uv.lock                            # Locked dependencies
├── README.md                          # This file
└── server/
    ├── fin_data_normalizer_environment.py  # Core environment + graders
    ├── app.py                         # FastAPI server
    ├── Dockerfile                     # Container definition
    └── requirements.txt               # Server dependencies
```

---

## Author

**Dhruv Bhatt**  
B.E. EEE + M.Sc. Economics, BITS Pilani  
GitHub: [@DhruvBhatt20](https://github.com/DhruvBhatt20)