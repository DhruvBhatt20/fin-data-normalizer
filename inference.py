"""
Inference Script - Financial Data Normalizer Environment
=========================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    SPACE_URL      The HF Space URL (defaults to deployed space).
"""

import asyncio
import json
import os
import textwrap
import requests
from typing import Any, Dict, List, Optional

from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL = os.getenv("SPACE_URL", "https://dhruvbhatt20-fin-data-normalizer-env.hf.space").rstrip("/")

BENCHMARK = "fin_data_normalizer"
MAX_STEPS = 1
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5
TASKS = ["unit_normalization", "metric_extraction", "conflict_resolution"]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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

def env_reset(task_name: str) -> Dict[str, Any]:
    """Call /reset HTTP endpoint."""
    resp = requests.post(
        f"{SPACE_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def env_step(result: Dict[str, Any], task_name: str = "") -> Dict[str, Any]:
    resp = requests.post(
        f"{SPACE_URL}/step",
        json={"action": {"result": result, "task_name": task_name}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def build_user_prompt(task_name: str, task_description: str, task_data: Dict) -> str:
    return textwrap.dedent(f"""
        Task: {task_name}

        Instructions:
        {task_description}

        Input Data:
        {json.dumps(task_data, indent=2, ensure_ascii=False)}

        Return your answer as a JSON object.
    """).strip()

def get_model_result(client: OpenAI, task_name: str, task_description: str, task_data: Dict) -> Dict[str, Any]:
    user_prompt = build_user_prompt(task_name, task_description, task_data)
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
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        return {}
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        return {}

def run_episode(client: OpenAI, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_response = env_reset(task_name)
        obs = reset_response.get("observation", {})
        task_description = obs.get("task_description", "")
        task_data = obs.get("task_data", {})
        done = reset_response.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            agent_result = get_model_result(client, task_name, task_description, task_data)
            action_str = json.dumps(agent_result, ensure_ascii=False)

            step_response = env_step(agent_result, task_name)
            reward = float(step_response.get("reward") or 0.0)
            done = step_response.get("done", True)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str[:120], reward=reward, done=done, error=error_msg)

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

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] Connecting to: {SPACE_URL}", flush=True)
    print(f"[DEBUG] Using model: {MODEL_NAME}", flush=True)

    all_scores: Dict[str, float] = {}

    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_episode(client, task_name)
        all_scores[task_name] = score

    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in all_scores.items():
        print(f"  {task}: {score:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"  AVERAGE: {avg:.3f}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
