---
title: Financial Data Normalizer
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
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

### Instance Pool Design

Each task has a **pool of multiple instances** randomly selected at `reset()`. This is a deliberate design choice that makes the environment a genuine RL problem rather than a static benchmark:

- Agents cannot memorize a single hardcoded answer — they must **generalize** across varied inputs
- Instances vary in unit formats, number representations, source configurations, and rule complexity
- Harder instances (e.g., two competing audited sources, implicit SDF calculations, mixed numeric/word forms) are included to ensure the reward signal is meaningful across a range of model capabilities

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
    score: float             # Score awarded, strictly in (0, 1) exclusive after step()
    feedback: str            # Human-readable explanation of score
    fields_correct: List[str]  # Fields the agent got right
    fields_wrong: List[str]    # Fields the agent got wrong
    done: bool               # True after step() is called
    reward: float            # Same as score (None at reset, float after step)
```

---

## Task Descriptions

### Task 1: Unit Normalization (Easy)

**Objective:** Normalize a list of company revenue figures to Crores INR from mixed units and currencies.

**Instance A input formats:** `"1,23,456 Cr"`, `"12.34 Billion USD"`, `"1234560 Lakhs"`, `"~890 Cr"`, `"Data not available"`  
**Instance B input formats:** `"₹8,732 Cr"`, `"2.1 Billion INR"`, `"USD 4.6 Million"`, `"~62,500 Lakhs"`, `"Figure not disclosed"`

**Expected Output format:**
```json
{
  "normalized": [
    {"company": "CompanyName", "revenue_cr": 12345.6},
    {"company": "OtherCompany", "revenue_cr": null}
  ]
}
```

**Grader:** Each company is scored independently (partial credit). Approximate values (`~`) must be parsed. The `₹` prefix and commas must be stripped. Unavailable data must return `null`. Tolerance: ±1% of expected value.

| Challenge type | Examples |
|----------------|---------|
| Direct Cr/INR | `"1,23,456 Cr"`, `"₹8,732 Cr"` |
| Billion conversions | `"12.34 Billion USD"`, `"2.1 Billion INR"` |
| Lakh conversions | `"1234560 Lakhs"`, `"~62,500 Lakhs"` |
| USD Million | `"USD 4.6 Million"` |
| Null entries | `"Data not available"`, `"Figure not disclosed"` |

---

### Task 2: RBI Metric Extraction (Medium)

**Objective:** Extract six structured metrics from an RBI Monetary Policy Committee (MPC) statement written in natural language.

**Instance A:** All numbers in word form (*"six point five percent"*, *"thirty basis points"*, *"five to one"*).  
**Instance B:** Mixed numeric and word forms; SDF rate derivable implicitly (*"25 basis points below repo"*); vote expressed as *"4 members in favour, 2 against"*; easing described as *"softened"*.

**Expected Output format:**
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

**Grader:** Each of the 6 fields is worth 1/6 of the total score. Numeric tolerance: ±0.01. `core_inflation_change_bps` is **negative** for easing/softening and **positive** for tightening. `mpc_vote` must be in `"X-Y"` format.

---

### Task 3: Conflict Resolution (Hard)

**Objective:** Given the same financial metric reported by 4 sources with different reliability tiers and publication dates, apply explicit priority rules to resolve the conflict.

**Rules:**
- `RULE_1`: Audited sources take highest priority
- `RULE_2`: Among same reliability tier, most recent `date_published` wins
- `RULE_3`: Normalize all values to Crores INR before comparison (e.g. `₹5120 Mn` → `512 Cr`)
- `RULE_4`: Flag `conflicts_detected=true` if any source differs by >0.5 Cr after normalization

**Instance A:** Single audited source — `RULE_1` directly identifies the winner.  
**Instance B (harder):** Two audited sources with different publication dates — `RULE_1` narrows to audited tier, then `RULE_2` (recency) decides. One non-Cr source requires `RULE_3` normalization before comparison.

**Expected Output format:**
```json
{
  "resolved_value_cr": 234.0,
  "chosen_source": "Annual Report",
  "rule_applied": "RULE_1",
  "conflicts_detected": true,
  "conflict_detail": "sources differ by up to 0.7 Cr after normalization"
}
```

**Grader:** Field-level partial credit (resolved_value_cr: 0.35, chosen_source: 0.25, rule_applied: 0.25, conflicts_detected: 0.15).

---

## Reward Function

All rewards are strictly in `(0, 1)` exclusive. The environment provides **dense partial credit** — agents receive signal proportional to how many fields they get correct, not just binary win/lose. This makes the reward function useful for RL training signal across the full trajectory.

| Task | Scoring Method |
|------|---------------|
| Unit Normalization | Per-company weighted score (weights vary by conversion difficulty) |
| Metric Extraction | Per-field uniform score (1/6 each) |
| Conflict Resolution | Per-field weighted score (value: 0.35, source: 0.25, rule: 0.25, conflict flag: 0.15) |

Because each episode randomly draws from the instance pool, the reward landscape is non-trivial: a model that correctly applies general financial reasoning will consistently outperform one that pattern-matches to a single example.

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

Scores achieved by `Qwen/Qwen2.5-72B-Instruct` on the hosted environment. Since instances are drawn randomly, scores vary across runs — this spread is intentional and demonstrates the environment's discriminative power.

| Task | Difficulty | Instance A | Instance B |
|------|-----------|-----------|-----------|
| unit_normalization | Easy | ~0.75 | ~0.55 |
| metric_extraction | Medium | ~1.00 | ~0.80 |
| conflict_resolution | Hard | ~1.00 | ~0.65 |

*Instance B tasks are deliberately harder: mixed number formats, implicit calculations, and multi-rule chaining. A stronger RL-trained agent should outperform the zero-shot baseline on Instance B.*

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