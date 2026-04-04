"""
Inference Script — Financial Data Normalizer Environment (FIXED)
=========================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from fin_data_normalizer import FinDataNormalizerAction, FinDataNormalizerEnv
except ImportError:
    from client import FinDataNormalizerEnv
    from models import FinDataNormalizerAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "fin_data_normalizer_env:latest")

BENCHMARK = "fin_data_normalizer"
MAX_STEPS = 1  # Single-turn environment — one step per episode
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["unit_normalization", "metric_extraction", "conflict_resolution"]

# ---------------------------------------------------------------------------
# Logging helpers (strictly follow required format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert financial data analyst. You will be given a financial data task
    and must return a precise JSON object as your answer.

    Rules:
    - Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.
    - For unit normalization: normalize all values to Crores INR. Use null for unavailable data.
    - For metric extraction: extract numbers written as words (e.g. "six point five" = 6.5).
      core_inflation_change_bps must be negative if the text says "easing".
    - For conflict resolution: apply rules strictly in order. Return the rule ID (e.g. RULE_1).

    Your response must be parseable by json.loads().
""").strip()


# ---------------------------------------------------------------------------
# Build user prompt from observation
# ---------------------------------------------------------------------------

def build_user_prompt(observation) -> str:
    return textwrap.dedent(f"""
        Task: {observation.task_name}
        Difficulty: {observation.difficulty}

        Instructions:
        {observation.task_description}

        Input Data:
        {json.dumps(observation.task_data, indent=2, ensure_ascii=False)}

        Return your answer as a JSON object.
    """).strip()


# ---------------------------------------------------------------------------
# Call LLM and parse JSON result
# ---------------------------------------------------------------------------

def get_model_result(client: OpenAI, observation) -> Dict[str, Any]:
    user_prompt = build_user_prompt(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        return {}
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        return {}


# ---------------------------------------------------------------------------
# Run one episode for a given task
# ---------------------------------------------------------------------------

async def run_episode(env, client: OpenAI, task_name: str) -> float:
    """Run one full episode for the given task. Returns the score (0.0–1.0)."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            agent_result = get_model_result(client, observation)
            action = FinDataNormalizerAction(result=agent_result)
            action_str = json.dumps(agent_result, ensure_ascii=False)

            result = await env.step(action)
            observation = result.observation

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str[:120],  # truncate for readability
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main — run all 3 tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await FinDataNormalizerEnv.from_docker_image(IMAGE_NAME)

    all_scores: Dict[str, float] = {}

    try:
        for task_name in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"Running task: {task_name}", flush=True)
            print(f"{'='*60}", flush=True)
            score = await run_episode(env, client, task_name)
            all_scores[task_name] = score

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in all_scores.items():
        print(f"  {task}: {score:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"  AVERAGE: {avg:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())