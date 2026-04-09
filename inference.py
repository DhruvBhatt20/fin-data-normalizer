"""
Inference Script - Financial Data Normalizer Environment
=========================================================
MANDATORY environment variables:
    API_BASE_URL       The API endpoint for the LLM (default: HF router).
    MODEL_NAME         The model identifier (default: Qwen/Qwen2.5-72B-Instruct).
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Local Docker image name built from this repo
                       (e.g. fin_data_normalizer_env:latest).
                       Also accepted as IMAGE_NAME.
"""

import asyncio
import json
import os
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from client import FinDataNormalizerEnv
    from models import FinDataNormalizerAction
except ImportError:
    from fin_data_normalizer.client import FinDataNormalizerEnv
    from fin_data_normalizer.models import FinDataNormalizerAction

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "fin_data_normalizer"
MAX_STEPS = 1
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5
TASKS = ["unit_normalization", "metric_extraction", "conflict_resolution"]

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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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


async def run_episode(env: "FinDataNormalizerEnv", client: OpenAI, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False
    error_msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset(task_name=task_name)
        obs = reset_result.observation
        task_description = obs.task_description
        task_data = obs.task_data

        agent_result = get_model_result(client, task_name, task_description, task_data)
        action_str = json.dumps(agent_result, ensure_ascii=False)

        step_result = await env.step(FinDataNormalizerAction(result=agent_result))
        reward = float(step_result.reward or 0.0)
        reward = max(0.01, min(0.99, reward))  # must be strictly in (0, 1)
        done = step_result.done

        rewards.append(reward)
        steps_taken = 1

        log_step(step=1, action=action_str[:120], reward=reward, done=done, error=None)

        score = min(max(reward, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        if not rewards:
            rewards = [0.01]  # ensure never empty — evaluator requires a valid score
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def _connect_with_retry(base_url: str, retries: int = 10, delay: float = 5.0) -> "FinDataNormalizerEnv":
    """Connect to an already-running environment server, retrying if not yet ready."""
    env = FinDataNormalizerEnv(base_url=base_url)
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            await env.connect()
            print(f"[DEBUG] Connected to {base_url} on attempt {attempt}", flush=True)
            return env
        except Exception as exc:
            last_exc = exc
            print(f"[DEBUG] Connection attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt < retries:
                await asyncio.sleep(delay)
    raise ConnectionError(f"Failed to connect to {base_url} after {retries} attempts: {last_exc}") from last_exc


async def get_env() -> "FinDataNormalizerEnv":
    """
    Connect to the environment.

    Priority order:
    1. If OPENENV_BASE_URL or ENV_BASE_URL is explicitly set in the environment,
       connect directly to that URL (the evaluator has already started the server).
    2. If LOCAL_IMAGE_NAME / IMAGE_NAME is set and docker is available,
       spin up a fresh container via from_docker_image().
    3. Fall back to localhost:8000 with retries.
    """
    import shutil

    explicit_url = os.environ.get("OPENENV_BASE_URL") or os.environ.get("ENV_BASE_URL")
    if explicit_url:
        print(f"[DEBUG] Using evaluator-provided URL: {explicit_url}", flush=True)
        return await _connect_with_retry(explicit_url)

    if IMAGE_NAME and shutil.which("docker"):
        print(f"[DEBUG] Starting container from image: {IMAGE_NAME}", flush=True)
        try:
            return await FinDataNormalizerEnv.from_docker_image(IMAGE_NAME)
        except Exception as exc:
            print(f"[DEBUG] from_docker_image failed ({exc}), falling back to localhost", flush=True)

    print(f"[DEBUG] Connecting to localhost:8000", flush=True)
    return await _connect_with_retry("http://localhost:8000")


async def main() -> None:
    print(f"[DEBUG] Using model: {MODEL_NAME}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        env = await get_env()
    except Exception as exc:
        print(f"[DEBUG] Fatal: could not connect to environment: {exc}", flush=True)
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[0.01])
        return

    all_scores: Dict[str, float] = {}
    try:
        for task_name in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"Running task: {task_name}", flush=True)
            print(f"{'='*60}", flush=True)
            score = await run_episode(env, client, task_name)
            all_scores[task_name] = score
    finally:
        await env.close()

    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in all_scores.items():
        print(f"  {task}: {score:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"  AVERAGE: {avg:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as _top_exc:
        print(f"[DEBUG] TOP-LEVEL EXCEPTION: {type(_top_exc).__name__}: {_top_exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(0)
