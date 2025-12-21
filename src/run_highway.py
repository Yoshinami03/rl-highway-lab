from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

from pettingzoo.utils import ParallelEnv
import gymnasium as gym
import numpy as np
from gymnasium import spaces

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_HE_PATH = os.path.join(_SRC_DIR, "highway_env")
for _p in (_LOCAL_HE_PATH, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import highway_env

from env_config import HighwayEnvConfig, default_config
from components.fleet import ControlledFleet
from components.spawn import VehicleSpawner
from components.action import EnvStepper, VehicleActionApplier
from components.metrics import AgentVehicleSelector, ObservationBuilder, RewardCalculator


class HighwayMultiEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "highway_multi_merge", "is_parallel": True}

    def __init__(
        self,
        config: Optional[HighwayEnvConfig] = None,
        num_agents: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        if config is None:
            config = default_config

        if num_agents is not None:
            config.num_agents = num_agents
        if render_mode is not None:
            config.render_mode = render_mode

        self.config = config
        self.render_mode = config.get_render_mode()

        self.env = gym.make(
            config.env_name,
            render_mode=self.render_mode,
            config=config.get_gym_config(),
        )
        self.core_env = self.env.unwrapped

        num_controlled = config.controlled_vehicles if config.controlled_vehicles is not None else config.num_agents
        self._num_agents = int(num_controlled)
        self.possible_agents = [f"car_{i}" for i in range(self._num_agents)]
        self.agents: List[str] = []

        self.obs_dim = int(config.obs_shape[0] if len(config.obs_shape) > 0 else 4)

        self.observation_spaces = {a: self._build_obs_space() for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(config.action_space_size) for a in self.possible_agents}

        self._current_step = 0

        self._fleet = ControlledFleet(max_agents=self._num_agents)
        self._spawner = VehicleSpawner(
            core_env=self.core_env,
            spawn_probability=float(config.spawn_probability),
            spawn_cooldown_steps=int(config.spawn_cooldown_steps),
        )
        self._action_applier = VehicleActionApplier(core_env=self.core_env)
        self._env_stepper = EnvStepper(env=self.env)
        self._selector = AgentVehicleSelector(core_env=self.core_env)
        self._obs_builder = ObservationBuilder(core_env=self.core_env, obs_dim=self.obs_dim, obs_dtype=str(config.obs_dtype))
        self._reward_calc = RewardCalculator(
            core_env=self.core_env,
            speed_normalization=float(config.speed_normalization),
            crash_penalty=float(config.crash_penalty),
        )

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    @property
    def unwrapped(self):
        return self

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        self.env.reset(seed=seed, options=options)

        self._current_step = 0
        self._fleet.reset()
        self._spawner.reset()

        for v in self._spawner.spawn_initial(step=self._current_step):
            self._fleet.append(v)

        self.agents = self.possible_agents[:]

        obs = {a: self._obs_for_agent_index(i) for i, a in enumerate(self.agents)}
        info = {a: {} for a in self.agents}
        return obs, info

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        self._current_step += 1

        for v in self._spawner.spawn_continuous(step=self._current_step):
            self._fleet.append(v)

        controlled = self._fleet.primary()
        if not controlled:
            controlled = list(self.core_env.road.vehicles[: len(self.agents)])

        self._action_applier.apply(actions=actions, agents=self.agents, vehicles=controlled)

        _, _, term, trunc, base_info = self._env_stepper.step(
            actions=actions,
            agents=self.agents,
            controlled_count=len(controlled),
        )

        obs = {a: self._obs_for_agent_index(i) for i, a in enumerate(self.agents)}
        rewards = {a: self._reward_for_agent_index(i) for i, a in enumerate(self.agents)}

        terminations = {a: bool(term) for a in self.agents}
        truncations = {a: bool(trunc) for a in self.agents}
        infos = {a: dict(base_info) if base_info else {} for a in self.agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _build_obs_space(self) -> spaces.Box:
        obs_dim = self.obs_dim
        obs_low = np.array([-np.inf] * obs_dim, dtype=np.float32)
        obs_high = np.array([np.inf] * obs_dim, dtype=np.float32)

        if obs_dim >= 4:
            obs_low[2] = 0.0
            obs_high[2] = 200.0
            obs_low[3] = -50.0
            obs_high[3] = 50.0

        return spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_dim,),
            dtype=getattr(np, self.config.obs_dtype),
        )

    def _obs_for_agent_index(self, index: int) -> np.ndarray:
        controlled = self._fleet.primary()
        vehicle = self._selector.select(index=index, controlled=controlled)
        return self._obs_builder.build(vehicle)

    def _reward_for_agent_index(self, index: int) -> float:
        controlled = self._fleet.primary()
        vehicle = self._selector.select(index=index, controlled=controlled)
        return self._reward_calc.calc(vehicle)
