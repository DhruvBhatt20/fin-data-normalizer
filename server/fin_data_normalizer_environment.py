# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Financial Data Normalizer Environment Implementation.

Simulates a real-world financial analyst's workspace where an AI agent must:
  Task 1 (easy)   - Normalize financial figures to a common unit (Crores INR)
  Task 2 (medium) - Extract key metrics from RBI Monetary Policy Statement text
  Task 3 (hard)   - Resolve conflicts across multiple financial data sources

All graders are fully deterministic and produce scores in (0.0, 1.0) exclusive.
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FinDataNormalizerAction, FinDataNormalizerObservation
except ImportError:
    from models import FinDataNormalizerAction, FinDataNormalizerObservation


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = ["unit_normalization", "metric_extraction", "conflict_resolution"]

TASK_DIFFICULTIES = {
    "unit_normalization": "easy",
    "metric_extraction": "medium",
    "conflict_resolution": "hard",
}

TASK_DESCRIPTIONS = {
    "unit_normalization": (
        "You are given a list of companies with revenue figures expressed in "
        "different units (Crores, Lakhs, Millions, Billions) and currencies "
        "(INR, USD). Normalize all values to Crores INR using the provided "
        "exchange rate. Return null for entries that contain no parseable value "
        "(e.g. 'Data not available'). Approximate values prefixed with '~' "
        "should be parsed by stripping the '~' symbol. "
        "Return: {\"normalized\": [{\"company\": str, \"revenue_cr\": float | null}, ...]}"
    ),
    "metric_extraction": (
        "You are given an excerpt from an RBI Monetary Policy Committee (MPC) "
        "statement. Extract the following six fields exactly: "
        "repo_rate_pct (float), sdf_rate_pct (float), cpi_inflation_pct (float), "
        "core_inflation_pct (float), core_inflation_change_bps (int, negative if easing), "
        "mpc_vote (str, format 'X-Y'). Numbers may be written as words "
        "(e.g. 'six point five' = 6.5, 'thirty basis points' = 30). "
        "Return all six fields as a flat JSON object."
    ),
    "conflict_resolution": (
        "You are given the same financial metric reported by multiple sources "
        "with different reliability tiers and publication dates. "
        "Apply the provided rules to resolve the conflict and identify the "
        "canonical value. Rules: RULE_1 = audited sources take highest priority, "
        "RULE_2 = among same reliability_tier, most recent date_published wins, "
        "RULE_3 = normalize all values to Crores INR before comparison, "
        "RULE_4 = flag conflicts_detected=true if any source differs by more "
        "than 0.5 Cr after normalization. "
        "Return: {\"resolved_value_cr\": float, \"chosen_source\": str, "
        "\"rule_applied\": str, \"conflicts_detected\": bool, \"conflict_detail\": str}"
    ),
}

# ---------------------------------------------------------------------------
# Task input data (static — deterministic grading)
# ---------------------------------------------------------------------------

TASK_DATA = {
    "unit_normalization": {
        "data": [
            {"company": "Reliance", "revenue": "1,23,456 Cr"},
            {"company": "Infosys", "revenue": "12.34 Billion USD"},
            {"company": "TCS", "revenue": "1234560 Lakhs"},
            {"company": "Wipro", "revenue": "~890 Cr"},
            {"company": "HCL", "revenue": "Data not available"},
        ],
        "target_unit": "Crores INR",
        "usd_to_inr": 83.5,
    },
    "metric_extraction": {
        "text": (
            "The Monetary Policy Committee voted five to one to keep the policy "
            "repo rate unchanged at six point five percent. The Standing Deposit "
            "Facility rate remains at six point two five percent. Headline CPI "
            "inflation is projected at five point one percent for Q3 FY24, with "
            "core inflation easing by thirty basis points to four point two percent "
            "compared to the previous quarter."
        ),
        "extract": [
            "repo_rate_pct",
            "sdf_rate_pct",
            "cpi_inflation_pct",
            "core_inflation_pct",
            "core_inflation_change_bps",
            "mpc_vote",
        ],
    },
    "conflict_resolution": {
        "metric": "Q2_FY24_revenue",
        "sources": [
            {
                "source": "Annual Report",
                "value": "₹234 Cr",
                "date_published": "2024-09-30",
                "reliability_tier": "audited",
            },
            {
                "source": "Press Release",
                "value": "₹234.5 Cr",
                "date_published": "2024-07-15",
                "reliability_tier": "unaudited",
            },
            {
                "source": "Exchange Filing",
                "value": "₹2340 Mn",
                "date_published": "2024-07-20",
                "reliability_tier": "regulatory",
            },
            {
                "source": "Analyst Report",
                "value": "₹233.8 Cr",
                "date_published": "2024-08-01",
                "reliability_tier": "regulatory",
            },
        ],
        "rules": {
            "RULE_1": "audited sources take highest priority",
            "RULE_2": "among same reliability_tier, most recent date_published wins",
            "RULE_3": "normalize all values to Crores INR before comparison",
            "RULE_4": "flag conflicts_detected=true if any source differs by more than 0.5 Cr",
        },
    },
}

# ---------------------------------------------------------------------------
# Ground-truth answers (used by graders)
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "unit_normalization": {
        "normalized": [
            {"company": "Reliance", "revenue_cr": 123456.0},
            {"company": "Infosys", "revenue_cr": round(12.34 * 1e9 * 83.5 / 1e7, 2)},  # ~103039.0
            {"company": "TCS", "revenue_cr": 123456.0},
            {"company": "Wipro", "revenue_cr": 890.0},
            {"company": "HCL", "revenue_cr": None},
        ]
    },
    "metric_extraction": {
        "repo_rate_pct": 6.5,
        "sdf_rate_pct": 6.25,
        "cpi_inflation_pct": 5.1,
        "core_inflation_pct": 4.2,
        "core_inflation_change_bps": -30,
        "mpc_vote": "5-1",
    },
    "conflict_resolution": {
        "resolved_value_cr": 234.0,
        "chosen_source": "Annual Report",
        "rule_applied": "RULE_1",
        "conflicts_detected": True,
    },
}


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _approx_equal(a: Optional[float], b: Optional[float], tol: float = 0.01) -> bool:
    """Return True if both are None, or both are floats within tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(b), 1.0)


# ---------------------------------------------------------------------------
# Grader: Task 1 — Unit Normalization
# ---------------------------------------------------------------------------

def grade_unit_normalization(result: Dict[str, Any]) -> Tuple[float, str, List[str], List[str]]:
    """
    Score the agent's unit normalization result.

    Scoring:
      - Reliance (clean Cr):        0.15
      - Infosys (Billion USD→Cr):   0.25  (hardest conversion)
      - TCS (Lakhs→Cr):             0.20
      - Wipro (~890 Cr, strip ~):   0.20
      - HCL (null for unavailable): 0.20
    Total: 1.0
    """
    weights = {
        "Reliance": 0.15,
        "Infosys": 0.25,
        "TCS": 0.20,
        "Wipro": 0.20,
        "HCL": 0.20,
    }
    gt = {row["company"]: row["revenue_cr"] for row in GROUND_TRUTH["unit_normalization"]["normalized"]}

    normalized = result.get("normalized", [])
    if not isinstance(normalized, list):
        return 0.0, "Result must contain a 'normalized' list.", [], list(weights.keys())

    agent_map = {}
    for row in normalized:
        if isinstance(row, dict) and "company" in row:
            agent_map[row["company"]] = row.get("revenue_cr")

    correct, wrong = [], []
    score = 0.0
    for company, weight in weights.items():
        gt_val = gt.get(company)
        agent_val = agent_map.get(company, "MISSING")
        if agent_val == "MISSING":
            wrong.append(company)
            continue
        if gt_val is None:
            if agent_val is None:
                score += weight
                correct.append(company)
            else:
                wrong.append(company)
        else:
            if _approx_equal(agent_val, gt_val, tol=0.01):
                score += weight
                correct.append(company)
            else:
                wrong.append(company)

    score = round(min(max(score, 0.01), 0.99), 4)
    feedback = (
        f"Score: {score:.2f}. Correct: {correct}. Wrong/missing: {wrong}. "
        "Tolerance: ±1% of expected value. 'Data not available' must map to null."
    )
    return score, feedback, correct, wrong


# ---------------------------------------------------------------------------
# Grader: Task 2 — RBI Metric Extraction
# ---------------------------------------------------------------------------

def grade_metric_extraction(result: Dict[str, Any]) -> Tuple[float, str, List[str], List[str]]:
    """
    Score the agent's RBI metric extraction.

    Each of the 6 fields is worth 1/6 ≈ 0.1667.
    Numeric fields: ±0.01 absolute tolerance.
    mpc_vote: exact string match after stripping whitespace.
    core_inflation_change_bps: must be negative (easing).
    """
    gt = GROUND_TRUTH["metric_extraction"]
    fields = list(gt.keys())
    per_field = round(1.0 / len(fields), 6)

    correct, wrong = [], []
    score = 0.0

    for field in fields:
        gt_val = gt[field]
        agent_val = result.get(field)
        if agent_val is None:
            wrong.append(field)
            continue

        if field == "mpc_vote":
            if str(agent_val).strip() == str(gt_val).strip():
                score += per_field
                correct.append(field)
            else:
                wrong.append(field)
        elif field == "core_inflation_change_bps":
            try:
                agent_int = int(agent_val)
                if agent_int == int(gt_val):
                    score += per_field
                    correct.append(field)
                else:
                    wrong.append(field)
            except (TypeError, ValueError):
                wrong.append(field)
        else:
            try:
                agent_float = float(agent_val)
                if abs(agent_float - float(gt_val)) <= 0.01:
                    score += per_field
                    correct.append(field)
                else:
                    wrong.append(field)
            except (TypeError, ValueError):
                wrong.append(field)

    score = round(min(max(score, 0.01), 0.99), 4)
    feedback = (
        f"Score: {score:.2f}. Correct fields: {correct}. "
        f"Wrong/missing fields: {wrong}. "
        "Note: core_inflation_change_bps must be -30 (negative = easing). "
        "mpc_vote must be '5-1'."
    )
    return score, feedback, correct, wrong


# ---------------------------------------------------------------------------
# Grader: Task 3 — Conflict Resolution
# ---------------------------------------------------------------------------

def grade_conflict_resolution(result: Dict[str, Any]) -> Tuple[float, str, List[str], List[str]]:
    """
    Score the agent's conflict resolution result.

    Scoring breakdown:
      resolved_value_cr  0.35  (±0.5 Cr tolerance)
      chosen_source      0.25  (exact string match)
      rule_applied       0.25  (must be exactly 'RULE_1')
      conflicts_detected 0.15  (must be True)
    Total: 1.0
    """
    gt = GROUND_TRUTH["conflict_resolution"]
    scoring = {
        "resolved_value_cr": 0.35,
        "chosen_source": 0.25,
        "rule_applied": 0.25,
        "conflicts_detected": 0.15,
    }

    correct, wrong = [], []
    score = 0.0

    # resolved_value_cr
    agent_val = result.get("resolved_value_cr")
    try:
        if _approx_equal(float(agent_val), gt["resolved_value_cr"], tol=0.002):
            score += scoring["resolved_value_cr"]
            correct.append("resolved_value_cr")
        else:
            wrong.append("resolved_value_cr")
    except (TypeError, ValueError):
        wrong.append("resolved_value_cr")

    # chosen_source
    if str(result.get("chosen_source", "")).strip() == gt["chosen_source"]:
        score += scoring["chosen_source"]
        correct.append("chosen_source")
    else:
        wrong.append("chosen_source")

    # rule_applied
    if str(result.get("rule_applied", "")).strip() == gt["rule_applied"]:
        score += scoring["rule_applied"]
        correct.append("rule_applied")
    else:
        wrong.append("rule_applied")

    # conflicts_detected
    agent_cd = result.get("conflicts_detected")
    if isinstance(agent_cd, bool) and agent_cd == gt["conflicts_detected"]:
        score += scoring["conflicts_detected"]
        correct.append("conflicts_detected")
    elif str(agent_cd).lower() == "true" and gt["conflicts_detected"]:
        score += scoring["conflicts_detected"]
        correct.append("conflicts_detected")
    else:
        wrong.append("conflicts_detected")

    score = round(min(max(score, 0.01), 0.99), 4)
    feedback = (
        f"Score: {score:.2f}. Correct: {correct}. Wrong: {wrong}. "
        "RULE_1 (audited priority) must be applied. "
        "resolved_value_cr must be 234.0 (Annual Report, audited). "
        "conflicts_detected must be True (sources differ by >0.5 Cr)."
    )
    return score, feedback, correct, wrong


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "unit_normalization": grade_unit_normalization,
    "metric_extraction": grade_metric_extraction,
    "conflict_resolution": grade_conflict_resolution,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FinDataNormalizerEnvironment(Environment):
    """
    Financial Data Normalizer RL Environment.

    An AI agent is presented with one of three real-world financial data tasks
    (unit normalization, RBI metric extraction, conflict resolution) and must
    return a structured JSON result. A deterministic grader scores the response
    on a 0.0–1.0 scale with partial credit and field-level feedback.

    Episode structure:
      - reset(task_name=None) → randomly selects a task (or use provided name)
      - step(action) → grades the agent's result, returns score + feedback
      - One step per episode (single-turn task environment)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[str] = None
        self._done: bool = False

    def reset(self, task_name: Optional[str] = None) -> FinDataNormalizerObservation:
        """
        Start a new episode.

        Args:
            task_name: One of 'unit_normalization', 'metric_extraction',
                       'conflict_resolution'. If None, cycles through tasks
                       in order for reproducibility.
        """
        import random
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False

        if task_name and task_name in TASKS:
            self._current_task = task_name
        else:
            self._current_task = random.choice(TASKS)

        return FinDataNormalizerObservation(
            task_name=self._current_task,
            task_description=TASK_DESCRIPTIONS[self._current_task],
            task_data=TASK_DATA[self._current_task],
            difficulty=TASK_DIFFICULTIES[self._current_task],
            score=0.0,
            feedback="Episode started. Submit your result via step().",
            fields_correct=[],
            fields_wrong=[],
            done=False,
            reward=0.0,
        )

    def step(self, action: FinDataNormalizerAction) -> FinDataNormalizerObservation:  # type: ignore[override]
        """
        Grade the agent's result for the current task.

        Args:
            action: FinDataNormalizerAction with agent's result dict.

        Returns:
            FinDataNormalizerObservation with score, feedback, and done=True.
        """
        self._state.step_count += 1

        if self._done:
            return FinDataNormalizerObservation(
                task_name=self._current_task or "",
                task_description="Episode already complete. Call reset() to start a new episode.",
                task_data={},
                difficulty="easy",
                score=0.0,
                feedback="Episode already complete.",
                fields_correct=[],
                fields_wrong=[],
                done=True,
                reward=0.0,
            )

        # Support stateless mode: use task_name from action if no active episode
        if self._current_task is None:
            if action.task_name and action.task_name in TASKS:
                self._current_task = action.task_name
            else:
                return FinDataNormalizerObservation(
                    task_name="",
                    task_description="No active episode. Call reset() or provide task_name in action.",
                    task_data={},
                    difficulty="easy",
                    score=0.0,
                    feedback="No active task.",
                    fields_correct=[],
                    fields_wrong=[],
                    done=True,
                    reward=0.0,
                )

        grader = GRADERS[self._current_task]
        score, feedback, correct, wrong = grader(action.result)

        self._done = True

        return FinDataNormalizerObservation(
            task_name=self._current_task,
            task_description=TASK_DESCRIPTIONS[self._current_task],
            task_data=TASK_DATA[self._current_task],
            difficulty=TASK_DIFFICULTIES[self._current_task],
            score=score,
            feedback=feedback,
            fields_correct=correct,
            fields_wrong=wrong,
            done=True,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state