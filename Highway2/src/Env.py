from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils import ParallelEnv


@dataclass
class CoopMergeConfig:
    pre_merge: float = 400.0
    merge_length: float = 200.0
    post_merge: float = 400.0

    episode_horizon: int = 500
    dt: float = 1.0

    lane_width: float = 4.0
    main_lane_count: int = 3

    vmin_kmh: float = 80.0
    vmax_kmh: float = 150.0
    vclip_min_kmh: float = 0.0

    max_obs_dist: float = 200.0
    max_deadend_obs: float = 400.0
    adj_obs_unlock_x: float = 200.0

    spawn_min_dist: float = 15.0  # 渋滞対応: 車間距離を詰める
    spawn_lane_cooldown_steps: int = 5
    spawn_attempts_per_step: int = 8

    lane_change_steps: int = 2

    collision_radius: float = 2.5
    collision_dist: float = 2.0
    vehicle_length: Optional[float] = None

    reward_goal: float = 10.0
    reward_crash: float = -100.0

    close_dist_threshold: float = 20.0
    close_penalty_base: float = -40.0
    close_penalty_slope: float = 2.0

    accel_penalty_scale: float = 0.1
    lane_change_penalty: float = -0.1

    deadend_penalty_scale: float = 2.0
    deadend_penalty_warn_m: float = 80.0

    center_lane_reward: float = 0.02
    center_lane_apply_after_x: float = 50.0

    speed_action_deltas_kmh: Tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)

    # 渋滞シチュエーション対応: 高密度状態がたまに発生するよう調整
    density_min: int = 8   # 最低8台（ある程度の混雑を維持）
    density_max: int = 20  # 最大20台（渋滞状態を発生させる）

    @property
    def goal_x(self) -> float:
        return self.pre_merge + self.merge_length + self.post_merge

    @property
    def merge_start(self) -> float:
        return self.pre_merge

    @property
    def merge_end(self) -> float:
        return self.pre_merge + self.merge_length


def kmh_to_mps(v_kmh: float) -> float:
    return float(v_kmh) / 3.6


def mps_to_kmh(v_mps: float) -> float:
    return float(v_mps) * 3.6


class CoopMergeEnv(ParallelEnv):
    metadata = {
        "name": "coop_merge_env",
        "is_parallelizable": True,
        "render_modes": [None],
    }

    def __init__(
        self,
        num_agents: int = 20,  # 渋滞対応: 最大20台
        config: Optional[CoopMergeConfig] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config or CoopMergeConfig()
        self._num_agents = int(num_agents)
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents[:]

        self.obs_dim = 17
        self._obs_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self._act_space = spaces.MultiDiscrete(np.array([len(self.cfg.speed_action_deltas_kmh), 3], dtype=np.int64))

        self.np_random, _ = seeding.np_random(seed)

        self.t = 0
        self.target_active = 0

        self.active = np.zeros((self._num_agents,), dtype=bool)
        self.x = np.zeros((self._num_agents,), dtype=np.float64)
        self.y = np.zeros((self._num_agents,), dtype=np.float64)
        self.lane = np.zeros((self._num_agents,), dtype=np.int64)
        self.v_mps = np.zeros((self._num_agents,), dtype=np.float64)

        self.lc_rem = np.zeros((self._num_agents,), dtype=np.int64)
        self.lc_tot = np.zeros((self._num_agents,), dtype=np.int64)
        self.lc_start_y = np.zeros((self._num_agents,), dtype=np.float64)
        self.lc_end_y = np.zeros((self._num_agents,), dtype=np.float64)
        self.lc_target_lane = np.zeros((self._num_agents,), dtype=np.int64)

        self.spawn_cd = np.zeros((4,), dtype=np.int64)
        self.agent_spawn_cd = np.zeros((self._num_agents,), dtype=np.int64)

        self._last_deactivate_reason = [""] * self._num_agents

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.t = 0
        self.spawn_cd[:] = 0
        self.agent_spawn_cd[:] = 0

        self.active[:] = False
        self.x[:] = 0.0
        self.y[:] = self._lane_center_y(0, 0.0)
        self.lane[:] = 0
        self.v_mps[:] = 0.0

        self.lc_rem[:] = 0
        self.lc_tot[:] = 0
        self.lc_start_y[:] = 0.0
        self.lc_end_y[:] = 0.0
        self.lc_target_lane[:] = 0

        dmin = int(min(self.cfg.density_min, self._num_agents))
        dmax = int(min(self.cfg.density_max, self._num_agents))
        if dmax < dmin:
            dmax = dmin
        self.target_active = int(self.np_random.integers(dmin, dmax + 1))

        self._spawn_to_target()

        obs = {a: self._obs(i) for i, a in enumerate(self.possible_agents)}
        infos = {a: {} for a in self.possible_agents}
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        self.spawn_cd = np.maximum(self.spawn_cd - 1, 0)
        self.agent_spawn_cd = np.maximum(self.agent_spawn_cd - 1, 0)

        applied_abs_dv_kmh = np.zeros((self._num_agents,), dtype=np.float64)
        lane_change_started = np.zeros((self._num_agents,), dtype=bool)
        lane_change_is_merge = np.zeros((self._num_agents,), dtype=bool)

        for i, agent in enumerate(self.possible_agents):
            if not self.active[i]:
                continue

            act = actions.get(agent, np.array([2, 1], dtype=np.int64))
            act = np.asarray(act, dtype=np.int64)

            if self.lc_rem[i] > 0:
                lane_act = 1
            else:
                lane_act = int(act[1])

            speed_act = int(act[0])
            speed_act = int(np.clip(speed_act, 0, len(self.cfg.speed_action_deltas_kmh) - 1))
            dv = float(self.cfg.speed_action_deltas_kmh[speed_act])
            applied_abs_dv_kmh[i] = abs(dv)

            v_kmh = mps_to_kmh(self.v_mps[i]) + dv
            v_kmh = float(np.clip(v_kmh, self.cfg.vclip_min_kmh, self.cfg.vmax_kmh))
            self.v_mps[i] = kmh_to_mps(v_kmh)

            if self.lc_rem[i] == 0:
                dir_ = {0: -1, 1: 0, 2: +1}[lane_act]
                if dir_ != 0:
                    ok, new_lane, is_merge = self._can_start_lane_change(i, dir_)
                    if ok:
                        lane_change_started[i] = True
                        lane_change_is_merge[i] = is_merge
                        self._start_lane_change(i, new_lane)

        for i in range(self._num_agents):
            if not self.active[i]:
                continue

            self.x[i] += self.v_mps[i] * self.cfg.dt

            if self.lc_rem[i] > 0:
                self.lc_rem[i] -= 1
                alpha = 1.0 - (self.lc_rem[i] / max(1, self.lc_tot[i]))
                self.y[i] = (1.0 - alpha) * self.lc_start_y[i] + alpha * self.lc_end_y[i]

                if int(self.lane[i]) == 3 and int(self.lc_target_lane[i]) == 2 and float(self.x[i]) >= self.cfg.merge_end:
                    self.lane[i] = 2
                    self.lc_rem[i] = 0
                    self.lc_tot[i] = 0
                    self.y[i] = self._lane_center_y(2, float(self.x[i]))
                    self.lc_start_y[i] = self.y[i]
                    self.lc_end_y[i] = self.y[i]
                    self.lc_target_lane[i] = 2
                elif self.lc_rem[i] == 0:
                    self.lane[i] = int(self.lc_target_lane[i])
                    self.y[i] = self._lane_center_y(int(self.lane[i]), float(self.x[i]))
            else:
                self.y[i] = self._lane_center_y(int(self.lane[i]), float(self.x[i]))

        deadend_mask = np.zeros((self._num_agents,), dtype=bool)
        for i in range(self._num_agents):
            if not self.active[i]:
                continue
            if int(self.lane[i]) == 3 and float(self.x[i]) >= self.cfg.merge_end:
                deadend_mask[i] = True

        collision_mask = np.zeros((self._num_agents,), dtype=bool)
        collision_dist = float(self.cfg.collision_dist)
        if self.cfg.vehicle_length is not None:
            collision_dist = max(collision_dist, 0.5 * float(self.cfg.vehicle_length))

        if np.any(self.active):
            for L in np.unique(self.lane[self.active].astype(int)):
                idx = np.where(self.active & (self.lane.astype(int) == int(L)))[0]
                if idx.size <= 1:
                    continue
                order = idx[np.argsort(self.x[idx])]
                gaps = np.diff(self.x[order])
                hit = np.where(gaps < collision_dist)[0]
                if hit.size > 0:
                    collision_mask[order[hit]] = True
                    collision_mask[order[hit + 1]] = True

        crash_mask = deadend_mask | collision_mask

        goal_mask = np.zeros((self._num_agents,), dtype=bool)
        for i in range(self._num_agents):
            if not self.active[i]:
                continue
            if float(self.x[i]) >= self.cfg.goal_x:
                goal_mask[i] = True

        active_pre = self.active.copy()
        x_pre = self.x.copy()
        y_pre = self.y.copy()
        lane_pre = self.lane.copy()

        to_pool = crash_mask | goal_mask
        if np.any(to_pool):
            for i in np.where(to_pool)[0]:
                reason = "crash" if crash_mask[i] else "goal"
                self._deactivate_to_pool(int(i), reason=reason)

        self._spawn_to_target()

        self.t += 1
        time_up = self.t >= int(self.cfg.episode_horizon)

        team_reward = 0.0
        n_goal = int(np.sum(goal_mask))
        n_crash = int(np.sum(crash_mask))
        if n_goal > 0:
            team_reward += float(self.cfg.reward_goal) * float(n_goal)
        if n_crash > 0:
            team_reward += float(self.cfg.reward_crash) * float(n_crash)

        per_agent = np.zeros((self._num_agents,), dtype=np.float64)

        per_agent += self._close_front_penalty_each()
        per_agent += self._deadend_progress_penalty_each()
        per_agent += self._center_lane_reward_each()

        per_agent -= self.cfg.accel_penalty_scale * applied_abs_dv_kmh

        non_merge = lane_change_started & (~lane_change_is_merge)
        per_agent[non_merge] += float(self.cfg.lane_change_penalty)

        obs = {a: self._obs(i) for i, a in enumerate(self.possible_agents)}
        rewards = {a: float(team_reward + per_agent[i]) for i, a in enumerate(self.possible_agents)}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: bool(time_up) for a in self.possible_agents}

        infos = {}
        non_merge_lc_count = int(np.sum(non_merge))
        merge_lc_count = int(np.sum(lane_change_started & lane_change_is_merge))
        accel_count = int(np.sum((applied_abs_dv_kmh[self.active] > 0)))

        goal_count = int(n_goal)
        crash_count = int(n_crash)
        deadend_count = int(np.sum(deadend_mask))
        collision_count = int(np.sum(collision_mask))

        for i, a in enumerate(self.possible_agents):
            infos[a] = {
                "active_pre": bool(active_pre[i]),
                "event_goal": bool(goal_mask[i]),
                "event_crash": bool(crash_mask[i]),
                "event_deadend": bool(deadend_mask[i]),
                "event_collision": bool(collision_mask[i]),
                "x": float(self.x[i]),
                "y": float(self.y[i]),
                "lane": int(self.lane[i]),
                "event_x": float(x_pre[i]),
                "event_y": float(y_pre[i]),
                "event_lane": int(lane_pre[i]),
                "team_goal_count": goal_count,
                "team_crash_count": crash_count,
                "team_deadend_count": deadend_count,
                "team_collision_count": collision_count,
                "team_nonmerge_lc_count": non_merge_lc_count,
                "team_merge_lc_count": merge_lc_count,
                "team_accel_count": accel_count,
                "reward_team": float(team_reward),
                "reward_individual": float(per_agent[i]),
                "deactivate_reason": str(self._last_deactivate_reason[i]),
            }

        return obs, rewards, terminations, truncations, infos

    def _lane_center_y(self, lane: int, x: float) -> float:
        if lane in (0, 1, 2):
            return float(lane) * self.cfg.lane_width

        y_merge = 3.0 * self.cfg.lane_width
        y_far = 4.0 * self.cfg.lane_width
        if x <= self.cfg.merge_start:
            if self.cfg.merge_start <= 1e-9:
                return y_merge
            a = float(np.clip(x / self.cfg.merge_start, 0.0, 1.0))
            return (1.0 - a) * y_far + a * y_merge
        return y_merge

    def _can_start_lane_change(self, i: int, dir_: int) -> Tuple[bool, int, bool]:
        cur = int(self.lane[i])
        new_lane = cur + int(dir_)
        if new_lane < 0 or new_lane > 3:
            return False, cur, False
        if cur in (0, 1, 2) and new_lane == 3:
            return False, cur, False
        if cur == 3:
            if new_lane != 2:
                return False, cur, False
            x = float(self.x[i])
            if x < self.cfg.merge_start or x > self.cfg.merge_end:
                return False, cur, False
            return True, new_lane, True
        return True, new_lane, False

    def _start_lane_change(self, i: int, new_lane: int) -> None:
        self.lc_tot[i] = int(self.cfg.lane_change_steps)
        self.lc_rem[i] = int(self.cfg.lane_change_steps)
        self.lc_target_lane[i] = int(new_lane)
        self.lc_start_y[i] = float(self.y[i])
        self.lc_end_y[i] = self._lane_center_y(int(new_lane), float(self.x[i]))

    def _deactivate_to_pool(self, i: int, reason: str = ""):
        self.active[i] = False
        self.agent_spawn_cd[i] = 1

        self.x[i] = 0.0
        self.v_mps[i] = 0.0
        self.lane[i] = 0
        self.y[i] = self._lane_center_y(0, 0.0)

        self.lc_rem[i] = 0
        self.lc_tot[i] = 0
        self.lc_start_y[i] = self.y[i]
        self.lc_end_y[i] = self.y[i]
        self.lc_target_lane[i] = 0

        self._last_deactivate_reason[i] = str(reason)

    def _is_spawn_position_free(self, x: float, y: float) -> bool:
        md = float(self.cfg.spawn_min_dist)
        md2 = md * md
        for j in range(self._num_agents):
            if not self.active[j]:
                continue
            dx = float(self.x[j]) - x
            dy = float(self.y[j]) - y
            if (dx * dx + dy * dy) < md2:
                return False
        return True

    def _spawn_one(self, agent_index: int, lane: int) -> bool:
        if self.spawn_cd[lane] > 0:
            return False
        x0 = 0.0
        y0 = self._lane_center_y(lane, x0)
        if not self._is_spawn_position_free(x0, y0):
            return False
        v0_kmh = float(self.np_random.uniform(self.cfg.vmin_kmh, self.cfg.vmax_kmh))

        self.active[agent_index] = True
        self.x[agent_index] = x0
        self.y[agent_index] = y0
        self.lane[agent_index] = int(lane)
        self.v_mps[agent_index] = kmh_to_mps(v0_kmh)

        self.lc_rem[agent_index] = 0
        self.lc_tot[agent_index] = 0

        self.spawn_cd[lane] = int(self.cfg.spawn_lane_cooldown_steps)
        return True

    def _spawn_to_target(self) -> int:
        max_new = int(max(0, self.target_active - int(np.sum(self.active))))
        if max_new <= 0:
            return 0

        inactive = np.where((~self.active) & (self.agent_spawn_cd == 0))[0]
        if len(inactive) == 0:
            return 0

        spawned = 0
        attempts = int(self.cfg.spawn_attempts_per_step)
        lanes = [0, 1, 2, 3]

        self.np_random.shuffle(inactive)
        for idx in inactive:
            if spawned >= max_new:
                break
            ok = False
            for _ in range(attempts):
                lane = int(self.np_random.choice(lanes))
                if self._spawn_one(int(idx), lane):
                    ok = True
                    spawned += 1
                    break
            if not ok:
                continue
        return spawned

    def _adj_obs_allowed(self, i: int) -> bool:
        li = int(self.lane[i])
        if li in (2, 3):
            return float(self.x[i]) >= self.cfg.adj_obs_unlock_x
        return True

    def _deadend_dist_feature(self, i: int) -> float:
        if int(self.lane[i]) == 3:
            dist = max(0.0, self.cfg.merge_end - float(self.x[i]))
            dist = float(np.clip(dist, 0.0, self.cfg.max_deadend_obs))
            return float(self.cfg.max_deadend_obs - dist)
        return 0.0

    def _nearest_front_back_in_lane(self, i: int, lane_id: int) -> Tuple[float, float, float, float]:
        if not self.active[i]:
            return 0.0, 0.0, 0.0, 0.0

        xi = float(self.x[i])
        best_front = None
        best_back = None
        front_j = -1
        back_j = -1

        for j in range(self._num_agents):
            if j == i or (not self.active[j]):
                continue
            if int(self.lane[j]) != int(lane_id):
                continue
            dx = float(self.x[j]) - xi
            if dx > 0:
                if (best_front is None) or (dx < best_front):
                    best_front = dx
                    front_j = j
            elif dx < 0:
                dd = -dx
                if (best_back is None) or (dd < best_back):
                    best_back = dd
                    back_j = j

        def dist_feat(d: Optional[float]) -> float:
            if d is None:
                return 0.0
            d2 = float(np.clip(d, 0.0, self.cfg.max_obs_dist))
            return float(self.cfg.max_obs_dist - d2)

        if best_front is None:
            fs, fd = 0.0, 0.0
        else:
            fs = float(mps_to_kmh(self.v_mps[front_j]))
            fd = dist_feat(best_front)

        if best_back is None:
            bs, bd = 0.0, 0.0
        else:
            bs = float(mps_to_kmh(self.v_mps[back_j]))
            bd = dist_feat(best_back)

        return fs, fd, bs, bd

    def _norm_speed(self, v_kmh: float) -> float:
        lo = float(self.cfg.vmin_kmh)
        hi = float(self.cfg.vmax_kmh)
        if hi <= lo:
            return 0.0
        return float(np.clip((v_kmh - lo) / (hi - lo), 0.0, 1.0))

    def _norm_dist_feat(self, dfeat: float) -> float:
        m = float(self.cfg.max_obs_dist)
        if m <= 1e-9:
            return 0.0
        return float(np.clip(dfeat / m, 0.0, 1.0))

    def _norm_deadend_feat(self, dfeat: float) -> float:
        m = float(self.cfg.max_deadend_obs)
        if m <= 1e-9:
            return 0.0
        return float(np.clip(dfeat / m, 0.0, 1.0))

    def _obs(self, i: int) -> np.ndarray:
        if not self.active[i]:
            o = np.zeros((self.obs_dim,), dtype=np.float32)
            return o

        li = int(self.lane[i])
        xi = float(self.x[i])
        vi_kmh = float(mps_to_kmh(self.v_mps[i]))

        adj_ok = self._adj_obs_allowed(i)

        if (not adj_ok) and li in (2, 3):
            has_left = 0.0
            has_right = 0.0
        else:
            if li == 3:
                has_left = 1.0
                has_right = 0.0
            else:
                has_left = 1.0 if (li > 0) else 0.0
                if li == 2:
                    has_right = 1.0 if (xi < self.cfg.merge_end) else 0.0
                else:
                    has_right = 0.0

        deadend_feat = self._deadend_dist_feature(i)

        fs, fd, bs, bd = self._nearest_front_back_in_lane(i, li)

        if (not adj_ok) and li in (2, 3):
            lfs = lfd = lbs = lbd = 0.0
            rfs = rfd = rbs = rbd = 0.0
        else:
            if li == 3:
                left_id = 2
                lfs, lfd, lbs, lbd = self._nearest_front_back_in_lane(i, left_id)
            elif li == 0:
                lfs = lfd = lbs = lbd = 0.0
            else:
                left_id = li - 1
                lfs, lfd, lbs, lbd = self._nearest_front_back_in_lane(i, left_id)

            if li == 2 and (xi < self.cfg.merge_end):
                right_id = 3
                rfs, rfd, rbs, rbd = self._nearest_front_back_in_lane(i, right_id)
            elif li in (0, 1):
                right_id = li + 1
                rfs, rfd, rbs, rbd = self._nearest_front_back_in_lane(i, right_id)
            else:
                rfs = rfd = rbs = rbd = 0.0

        o = np.array(
            [
                1.0,
                has_left,
                has_right,
                self._norm_deadend_feat(deadend_feat),
                self._norm_speed(vi_kmh),
                self._norm_speed(fs),
                self._norm_dist_feat(fd),
                self._norm_speed(bs),
                self._norm_dist_feat(bd),
                self._norm_speed(lfs),
                self._norm_dist_feat(lfd),
                self._norm_speed(lbs),
                self._norm_dist_feat(lbd),
                self._norm_speed(rfs),
                self._norm_dist_feat(rfd),
                self._norm_speed(rbs),
                self._norm_dist_feat(rbd),
            ],
            dtype=np.float32,
        )
        return o

    def _close_front_penalty_each(self) -> np.ndarray:
        p = np.zeros((self._num_agents,), dtype=np.float64)
        for i in range(self._num_agents):
            if not self.active[i]:
                continue
            li = int(self.lane[i])
            xi = float(self.x[i])
            best = None
            for j in range(self._num_agents):
                if i == j or (not self.active[j]):
                    continue
                if int(self.lane[j]) != li:
                    continue
                dx = float(self.x[j]) - xi
                if dx <= 0:
                    continue
                if best is None or dx < best:
                    best = dx
            if best is not None and best < float(self.cfg.close_dist_threshold):
                p[i] += float(self.cfg.close_penalty_base) + float(self.cfg.close_penalty_slope) * float(best)
        return p

    def _deadend_progress_penalty_each(self) -> np.ndarray:
        p = np.zeros((self._num_agents,), dtype=np.float64)
        warn = float(self.cfg.deadend_penalty_warn_m)
        scale = float(self.cfg.deadend_penalty_scale)
        if warn <= 1e-9 or scale <= 0:
            return p
        for i in range(self._num_agents):
            if not self.active[i]:
                continue
            if int(self.lane[i]) != 3:
                continue
            dist = float(self.cfg.merge_end) - float(self.x[i])
            if dist < 0:
                continue
            if dist < warn:
                r = float(np.clip(1.0 - (dist / warn), 0.0, 1.0))
                p[i] -= scale * (r * r)
        return p

    def _center_lane_reward_each(self) -> np.ndarray:
        r = np.zeros((self._num_agents,), dtype=np.float64)
        val = float(self.cfg.center_lane_reward)
        x0 = float(self.cfg.center_lane_apply_after_x)
        if val == 0.0:
            return r
        for i in range(self._num_agents):
            if not self.active[i]:
                continue
            if float(self.x[i]) < x0:
                continue
            li = int(self.lane[i])
            if li in (0, 1, 2):
                if li == 1:
                    r[i] += val
                else:
                    r[i] -= val
        return r
