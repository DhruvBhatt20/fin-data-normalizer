# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Financial Data Normalizer Environment Client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import FinDataNormalizerAction, FinDataNormalizerObservation
except ImportError:
    from models import FinDataNormalizerAction, FinDataNormalizerObservation


class FinDataNormalizerEnv(
    EnvClient[FinDataNormalizerAction, FinDataNormalizerObservation, State]
):
    """
    Client for the Financial Data Normalizer Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example (async):
        >>> async with FinDataNormalizerEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.task_name)
        ...     action = FinDataNormalizerAction(result={"normalized": [...]})
        ...     result = await client.step(action)
        ...     print(result.observation.score)

    Example (Docker):
        >>> client = await FinDataNormalizerEnv.from_docker_image("fin_data_normalizer_env:latest")
        >>> try:
        ...     result = await client.reset()
        ...     result = await client.step(FinDataNormalizerAction(result={}))
        ... finally:
        ...     await client.close()
    """

    def _step_payload(self, action: FinDataNormalizerAction) -> Dict:
        """Convert FinDataNormalizerAction to JSON payload for WebSocket step message."""
        return {
            "result": action.result,
        }

    def _parse_result(self, payload: Dict) -> StepResult[FinDataNormalizerObservation]:
        """Parse server response into StepResult[FinDataNormalizerObservation]."""
        obs_data = payload.get("observation", {})
        observation = FinDataNormalizerObservation(
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            task_data=obs_data.get("task_data", {}),
            difficulty=obs_data.get("difficulty", "easy"),
            score=obs_data.get("score", 0.0),
            feedback=obs_data.get("feedback"),
            fields_correct=obs_data.get("fields_correct", []),
            fields_wrong=obs_data.get("fields_wrong", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )