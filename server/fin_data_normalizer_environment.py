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
Each task has multiple instances drawn randomly at reset() to prevent memorization
and create a proper RL environment with non-trivial generalization requirements.
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
        "(e.g. 'Data not available', 'Figure not disclosed', 'N/A'). "
        "Approximate values prefixed with '~' should be parsed by stripping the '~' symbol. "
        "The '₹' prefix and commas in numbers should be ignored during parsing. "
        "Return: {\"normalized\": [{\"company\": str, \"revenue_cr\": float | null}, ...]}"
    ),
    "metric_extraction": (
        "You are given an excerpt from an RBI Monetary Policy Committee (MPC) "
        "statement. Extract the following six fields exactly: "
        "repo_rate_pct (float), sdf_rate_pct (float), cpi_inflation_pct (float), "
        "core_inflation_pct (float), core_inflation_change_bps (int, negative if easing/softening, "
        "positive if tightening/rising), "
        "mpc_vote (str, format 'X-Y' where X voted in favour and Y against). "
        "Numbers may be written as words (e.g. 'six point five' = 6.5, 'thirty basis points' = 30). "
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
# Task instance pools — randomly selected at reset() for each episode.
# Multiple instances prevent memorization and make the environment a proper
# RL problem requiring generalization.
# ---------------------------------------------------------------------------

TASK_INSTANCES: Dict[str, List[Dict[str, Any]]] = {

    # -----------------------------------------------------------------------
    # Unit Normalization instances
    # -----------------------------------------------------------------------
    "unit_normalization": [
        # Instance A — standard Indian large-cap revenues, mixed units
        {
            "data": {
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
            "ground_truth": {
                "normalized": [
                    {"company": "Reliance", "revenue_cr": 123456.0,                               "weight": 0.15},
                    {"company": "Infosys",  "revenue_cr": round(12.34 * 1e9 * 83.5 / 1e7, 2),    "weight": 0.25},
                    {"company": "TCS",      "revenue_cr": round(1234560 / 100, 2),                "weight": 0.20},
                    {"company": "Wipro",    "revenue_cr": 890.0,                                  "weight": 0.20},
                    {"company": "HCL",      "revenue_cr": None,                                   "weight": 0.20},
                ],
            },
        },
        # Instance B — mid-cap mix, ₹ prefix, Billion INR, USD Million, different exchange rate
        {
            "data": {
                "data": [
                    {"company": "HDFC Bank",      "revenue": "₹8,732 Cr"},
                    {"company": "Bajaj Finance",  "revenue": "2.1 Billion INR"},
                    {"company": "Maruti Suzuki",  "revenue": "USD 4.6 Million"},
                    {"company": "Asian Paints",   "revenue": "~62,500 Lakhs"},
                    {"company": "Zomato",         "revenue": "Figure not disclosed"},
                ],
                "target_unit": "Crores INR",
                "usd_to_inr": 84.0,
            },
            "ground_truth": {
                "normalized": [
                    {"company": "HDFC Bank",     "revenue_cr": 8732.0,                               "weight": 0.15},
                    {"company": "Bajaj Finance", "revenue_cr": round(2.1 * 1e9 / 1e7, 2),            "weight": 0.20},
                    {"company": "Maruti Suzuki", "revenue_cr": round(4.6 * 1e6 * 84.0 / 1e7, 2),    "weight": 0.25},
                    {"company": "Asian Paints",  "revenue_cr": round(62500 / 100, 2),                "weight": 0.20},
                    {"company": "Zomato",        "revenue_cr": None,                                 "weight": 0.20},
                ],
            },
        },
    ],

    # -----------------------------------------------------------------------
    # Metric Extraction instances
    # -----------------------------------------------------------------------
    "metric_extraction": [
        # Instance A — mostly word-form numbers, clear "easing" language
        {
            "data": {
                "text": (
                    "The Monetary Policy Committee voted five to one to keep the policy "
                    "repo rate unchanged at six point five percent. The Standing Deposit "
                    "Facility rate remains at six point two five percent. Headline CPI "
                    "inflation is projected at five point one percent for Q3 FY24, with "
                    "core inflation easing by thirty basis points to four point two percent "
                    "compared to the previous quarter."
                ),
                "extract": [
                    "repo_rate_pct", "sdf_rate_pct", "cpi_inflation_pct",
                    "core_inflation_pct", "core_inflation_change_bps", "mpc_vote",
                ],
            },
            "ground_truth": {
                "repo_rate_pct": 6.5,
                "sdf_rate_pct": 6.25,
                "cpi_inflation_pct": 5.1,
                "core_inflation_pct": 4.2,
                "core_inflation_change_bps": -30,
                "mpc_vote": "5-1",
            },
        },
        # Instance B — mixed numeric/word forms, implicit SDF calculation,
        #              vote in words ("four yeas, two nays"), "softened" for easing
        {
            "data": {
                "text": (
                    "The six-member Monetary Policy Committee, with 4 members voting in "
                    "favour and 2 against, resolved to cut the policy repo rate by 50 basis "
                    "points to six percent per annum. The Standing Deposit Facility rate, "
                    "set 25 basis points below the repo rate, now stands at 5.75 percent. "
                    "Headline CPI inflation for Q3 FY25 is projected at five point eight "
                    "percent. Core inflation has softened by 20 basis points, declining to "
                    "4.5 percent from the previous quarter's reading."
                ),
                "extract": [
                    "repo_rate_pct", "sdf_rate_pct", "cpi_inflation_pct",
                    "core_inflation_pct", "core_inflation_change_bps", "mpc_vote",
                ],
            },
            "ground_truth": {
                "repo_rate_pct": 6.0,
                "sdf_rate_pct": 5.75,
                "cpi_inflation_pct": 5.8,
                "core_inflation_pct": 4.5,
                "core_inflation_change_bps": -20,
                "mpc_vote": "4-2",
            },
        },
    ],

    # -----------------------------------------------------------------------
    # Conflict Resolution instances
    # -----------------------------------------------------------------------
    "conflict_resolution": [
        # Instance A — single audited source wins by RULE_1 directly
        {
            "data": {
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
            "ground_truth": {
                "resolved_value_cr": 234.0,
                "chosen_source": "Annual Report",
                "rule_applied": "RULE_1",
                "conflicts_detected": True,
            },
        },
        # Instance B — two audited sources; RULE_1 narrows to audited tier,
        #              then RULE_2 (recency) decides the winner.
        #              Tests whether the model can correctly chain rules.
        {
            "data": {
                "metric": "Q3_FY24_revenue",
                "sources": [
                    {
                        "source": "Audited Annual Report",
                        "value": "₹512 Cr",
                        "date_published": "2024-12-15",
                        "reliability_tier": "audited",
                    },
                    {
                        "source": "Audited Interim Report",
                        "value": "₹509.8 Cr",
                        "date_published": "2024-10-01",
                        "reliability_tier": "audited",
                    },
                    {
                        "source": "Exchange Filing",
                        "value": "₹5120 Mn",
                        "date_published": "2024-11-01",
                        "reliability_tier": "regulatory",
                    },
                    {
                        "source": "Analyst Note",
                        "value": "₹511.5 Cr",
                        "date_published": "2024-11-20",
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
            "ground_truth": {
                # RULE_1 → only audited sources qualify
                # RULE_2 → Annual Report (2024-12-15) is more recent than Interim (2024-10-01)
                # Exchange Filing: ₹5120 Mn = 512.0 Cr (RULE_3, matches winner)
                # Audited Interim differs by 2.2 Cr > 0.5 → conflicts_detected = True
                "resolved_value_cr": 512.0,
                "chosen_source": "Audited Annual Report",
                "rule_applied": "RULE_2",
                "conflicts_detected": True,
            },
        },
    ],
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

def grade_unit_normalization(
    result: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Tuple[float, str, List[str], List[str]]:
    """
    Score the agent's unit normalization result against the instance ground truth.

    Weights and expected values are embedded in ground_truth["normalized"] per row.
    Each row: {"company": str, "revenue_cr": float|None, "weight": float}
    """
    gt_rows = ground_truth.get("normalized", [])
    gt_map: Dict[str, Optional[float]] = {}
    weight_map: Dict[str, float] = {}
    n = len(gt_rows) or 1
    for row in gt_rows:
        company = row["company"]
        gt_map[company] = row["revenue_cr"]
        weight_map[company] = row.get("weight", 1.0 / n)

    normalized = result.get("normalized", [])
    if not isinstance(normalized, list):
        return 0.01, "Result must contain a 'normalized' list.", [], list(weight_map.keys())

    agent_map: Dict[str, Any] = {}
    for row in normalized:
        if isinstance(row, dict) and "company" in row:
            agent_map[row["company"]] = row.get("revenue_cr")

    correct, wrong = [], []
    score = 0.0
    for company, weight in weight_map.items():
        gt_val = gt_map.get(company)
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
            try:
                if _approx_equal(float(agent_val), gt_val, tol=0.01):
                    score += weight
                    correct.append(company)
                else:
                    wrong.append(company)
            except (TypeError, ValueError):
                wrong.append(company)

    score = round(min(max(score, 0.01), 0.99), 4)
    feedback = (
        f"Score: {score:.2f}. Correct: {correct}. Wrong/missing: {wrong}. "
        "Tolerance: ±1% of expected value. Unparseable entries must map to null."
    )
    return score, feedback, correct, wrong


# ---------------------------------------------------------------------------
# Grader: Task 2 — RBI Metric Extraction
# ---------------------------------------------------------------------------

def grade_metric_extraction(
    result: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Tuple[float, str, List[str], List[str]]:
    """
    Score the agent's RBI metric extraction against the instance ground truth.

    ground_truth is a flat dict: {field_name: expected_value, ...}
    Each of the 6 fields is worth 1/6.
    """
    gt = ground_truth
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
                if int(agent_val) == int(gt_val):
                    score += per_field
                    correct.append(field)
                else:
                    wrong.append(field)
            except (TypeError, ValueError):
                wrong.append(field)
        else:
            try:
                if abs(float(agent_val) - float(gt_val)) <= 0.01:
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
        f"Expected: {gt}."
    )
    return score, feedback, correct, wrong


# ---------------------------------------------------------------------------
# Grader: Task 3 — Conflict Resolution
# ---------------------------------------------------------------------------

def grade_conflict_resolution(
    result: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Tuple[float, str, List[str], List[str]]:
    """
    Score the agent's conflict resolution result against the instance ground truth.

    Scoring breakdown:
      resolved_value_cr  0.35  (±0.2% tolerance)
      chosen_source      0.25  (exact string match)
      rule_applied       0.25  (exact string match, e.g. 'RULE_1' or 'RULE_2')
      conflicts_detected 0.15  (bool match)
    Total: 1.0
    """
    gt = ground_truth
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
    elif str(agent_cd).lower() == str(gt["conflicts_detected"]).lower():
        score += scoring["conflicts_detected"]
        correct.append("conflicts_detected")
    else:
        wrong.append("conflicts_detected")

    score = round(min(max(score, 0.01), 0.99), 4)
    feedback = (
        f"Score: {score:.2f}. Correct: {correct}. Wrong: {wrong}. "
        f"Expected: resolved_value_cr={gt['resolved_value_cr']}, "
        f"chosen_source={gt['chosen_source']!r}, "
        f"rule_applied={gt['rule_applied']!r}, "
        f"conflicts_detected={gt['conflicts_detected']}."
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
    on a (0, 1) exclusive scale with partial credit and field-level feedback.

    Each reset() randomly selects one instance from the task's instance pool,
    requiring the agent to generalize across varied inputs rather than memorize
    a single example — making this a genuine RL environment.

    Episode structure:
      - reset(task_name=None) → randomly selects a task (or use provided name)
                                then randomly selects an instance from the pool
      - step(action)          → grades the agent's result, returns score + feedback
      - One step per episode (single-turn task environment)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[str] = None
        self._current_task_data: Dict[str, Any] = {}
        self._current_ground_truth: Dict[str, Any] = {}
        self._done: bool = False

    def reset(self, task_name: Optional[str] = None) -> FinDataNormalizerObservation:
        """
        Start a new episode, randomly selecting an instance from the task's pool.

        Args:
            task_name: One of 'unit_normalization', 'metric_extraction',
                       'conflict_resolution'. If None, chosen randomly.
        """
        import random

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False

        if task_name and task_name in TASKS:
            self._current_task = task_name
        else:
            self._current_task = random.choice(TASKS)

        # Randomly select one instance from the pool for this task
        instance = random.choice(TASK_INSTANCES[self._current_task])
        self._current_task_data = instance["data"]
        self._current_ground_truth = instance["ground_truth"]

        return FinDataNormalizerObservation(
            task_name=self._current_task,
            task_description=TASK_DESCRIPTIONS[self._current_task],
            task_data=self._current_task_data,
            difficulty=TASK_DIFFICULTIES[self._current_task],
            score=0.01,
            feedback="Episode started. Submit your result via step().",
            fields_correct=[],
            fields_wrong=[],
            done=False,
            reward=None,
        )

    def step(self, action: FinDataNormalizerAction) -> FinDataNormalizerObservation:  # type: ignore[override]
        """
        Grade the agent's result for the current task instance.

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
                score=0.01,
                feedback="Episode already complete.",
                fields_correct=[],
                fields_wrong=[],
                done=True,
                reward=0.01,
            )

        # Support stateless mode: use task_name from action if no active episode
        if self._current_task is None:
            if action.task_name and action.task_name in TASKS:
                self._current_task = action.task_name
                # No instance selected yet in stateless mode — pick randomly
                import random
                instance = random.choice(TASK_INSTANCES[self._current_task])
                self._current_task_data = instance["data"]
                self._current_ground_truth = instance["ground_truth"]
            else:
                return FinDataNormalizerObservation(
                    task_name="",
                    task_description="No active episode. Call reset() or provide task_name in action.",
                    task_data={},
                    difficulty="easy",
                    score=0.01,
                    feedback="No active task.",
                    fields_correct=[],
                    fields_wrong=[],
                    done=True,
                    reward=0.01,
                )

        grader = GRADERS[self._current_task]
        score, feedback, correct, wrong = grader(action.result, self._current_ground_truth)
        score = round(min(max(score, 0.01), 0.99), 4)  # safety clamp

        self._done = True

        return FinDataNormalizerObservation(
            task_name=self._current_task,
            task_description=TASK_DESCRIPTIONS[self._current_task],
            task_data=self._current_task_data,
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
