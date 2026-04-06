# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Financial Data Normalizer Environment.

This environment simulates a real-world financial analyst's workspace where
an AI agent must clean, extract, and reconcile messy financial data.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FinDataNormalizerObservation(Observation):
    """
    Observation returned to the agent after reset() or step().

    Contains the current task specification and feedback from the last action.
    """

    task_name: str = Field(
        default="",
        description="Name of the active task: unit_normalization | metric_extraction | conflict_resolution",
    )
    task_description: str = Field(
        default="",
        description="Human-readable description of what the agent must accomplish.",
    )
    task_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="The raw input data the agent must process.",
    )
    difficulty: str = Field(
        default="easy",
        description="Difficulty level: easy | medium | hard",
    )
    score: float = Field(
        default=0.0,
        description="Score awarded for the last action (0.0 to 1.0). 0.0 on reset.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Human-readable feedback explaining the score.",
    )
    fields_correct: Optional[List[str]] = Field(
        default=None,
        description="List of fields the agent got correct (for partial credit transparency).",
    )
    fields_wrong: Optional[List[str]] = Field(
        default=None,
        description="List of fields the agent got wrong.",
    )


class FinDataNormalizerAction(Action):
    """
    Action submitted by the agent to the Financial Data Normalizer environment.

    The agent returns a JSON-serializable dictionary containing its answer
    for the current task. The structure of 'result' depends on the active task:

    Task 1 (unit_normalization):
        {"normalized": [{"company": str, "revenue_cr": float | null}, ...]}

    Task 2 (metric_extraction):
        {"repo_rate_pct": float, "sdf_rate_pct": float,
         "cpi_inflation_pct": float, "core_inflation_pct": float,
         "core_inflation_change_bps": int, "mpc_vote": str}

    Task 3 (conflict_resolution):
        {"resolved_value_cr": float, "chosen_source": str,
         "rule_applied": str, "conflicts_detected": bool,
         "conflict_detail": str}
    """

    task_name: Optional[str] = Field(
        default=None,
        description="Task to grade. Required if reset() was not called in this session.",
    )
    result: Dict[str, Any] = Field(
        ...,
        description="Agent's answer for the current task as a structured dictionary.",
    )