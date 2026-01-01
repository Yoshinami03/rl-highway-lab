# --- 依存（Colab等） ---
# pip install "imageio[ffmpeg]"

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from Env import CoopMergeEnv, CoopMergeConfig

base = CoopMergeEnv(num_agents=12, config=CoopMergeConfig(), seed=0)
venv = ss.pettingzoo_env_to_vec_env_v1(base)
venv = ss.concat_vec_envs_v1(
    venv,
    num_vec_envs=1,
    num_cpus=0,
    base_class="stable_baselines3",
)
venv = VecMonitor(venv)

# 保存済みモデルを選択
# model_path = "./ppo_coopmerge_finetuned_more.zip" # pretrained model
save_path2 = "./ppo_trained2.zip"
model_path = save_path2
model = PPO.load(model_path, env=venv, device="cpu")


# -----------------------------
# 1) camera bounds（あなたの安定版に合わせた簡易版）
# -----------------------------
def compute_fixed_camera_bounds(cfg: CoopMergeConfig, mode="merge", pad=25.0, tail_x=150.0):
    if mode == "full":
        xmin = -pad
        xmax = cfg.goal_x + pad
    else:
        xmin = -pad
        xmax = cfg.merge_end + tail_x + pad

    ymin = -pad
    ymax = 4.0 * cfg.lane_width + pad
    return (xmin, xmax), (ymin, ymax)

# -----------------------------
# 2) ParallelEnv を直接回す（vec化しない）
# -----------------------------
cfg = CoopMergeConfig()
env = CoopMergeEnv(num_agents=12, config=cfg, seed=0)

AGENTS = env.possible_agents  # 固定順序


def obs_dict_to_batch(obs_dict):
    return np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in AGENTS], axis=0)

def act_batch_to_dict(act_batch):
    act_batch = np.asarray(act_batch)
    return {a: act_batch[i].astype(np.int64) for i, a in enumerate(AGENTS)}


# -----------------------------
# 3) warning 判定（描画と同一ロジックを共有）
# -----------------------------
def front_gap_same_lane(env, i: int):
    if not env.active[i]:
        return None
    li = int(env.lane[i])
    xi = float(env.x[i])
    best = None
    for j in range(env._num_agents):
        if j == i or (not env.active[j]):
            continue
        if int(env.lane[j]) != li:
            continue
        dx = float(env.x[j]) - xi
        if dx > 0 and (best is None or dx < best):
            best = dx
    return best

def is_warn_vehicle(env, i: int, close_warn_m: float, deadend_warn_m: float) -> bool:
    if not env.active[i]:
        return False

    close_warn_m = float(close_warn_m)
    deadend_warn_m = float(deadend_warn_m)

    lane = int(env.lane[i])
    x = float(env.x[i])

    too_close = False
    g = front_gap_same_lane(env, i)
    if g is not None and g < close_warn_m:
        too_close = True

    near_deadend = False
    if lane == 3:
        dist_to_end = float(env.cfg.merge_end) - x
        if 0.0 <= dist_to_end < deadend_warn_m:
            near_deadend = True

    return (too_close or near_deadend)

def count_warnings(env, close_warn_m, deadend_warn_m):
    warn_now = 0
    for i in range(env._num_agents):
        if is_warn_vehicle(env, i, close_warn_m, deadend_warn_m):
            warn_now += 1
    return warn_now


# -----------------------------
# 4) 固定カメラ描画（Matplotlib -> RGB）
#   - warning: orange（車そのもの）
#   - collision flash: 「地点」で1秒 red（flash_spotsで制御）
# -----------------------------
def render_fixed_topdown(
    env, xlim, ylim=None, width=960, height=540, dpi=100,
    y_scale=6.0, pad_y=1.0,
    road_lw=10,
    ramp_lw=12,
    car_len_scale=3.2,
    car_width_scale=1.0,
    veh_edge_lw=2.5,
    close_warn_m=None,
    deadend_warn_m=40.0,
    flash_spots=None,          # ★変更：スロットではなく「地点」を赤表示
    flash_color="red",
    warn_color="orange"
):
    cfg = env.cfg

    if close_warn_m is None:
        close_warn_m = float(getattr(cfg, "close_dist_threshold", 20.0))
    close_warn_m = float(close_warn_m)
    deadend_warn_m = float(deadend_warn_m)

    if flash_spots is None:
        flash_spots = []

    if ylim is None:
        y_min = 0.0 - pad_y
        y_max = 4.0 * cfg.lane_width + pad_y
        ylim = (y_min, y_max)

    def Y(y):
        return float(y) * float(y_scale)

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)

    # 上下逆（y反転）
    ax.set_ylim(Y(ylim[1]), Y(ylim[0]))

    ax.set_aspect("auto")
    ax.axis("off")

    # -----------------------------
    # roads (奥)
    # -----------------------------
    xs_full = np.linspace(float(xlim[0]), float(xlim[1]), 250)

    for lane in (0, 1, 2):
        y = lane * cfg.lane_width
        ax.plot(
            xs_full,
            np.full_like(xs_full, Y(y)),
            linewidth=road_lw,
            solid_capstyle="round",
            zorder=1
        )

    # ramp lane(3) は merge_end で打ち切り
    ramp_x1 = min(float(xlim[1]), float(cfg.merge_end))
    ramp_x0 = float(xlim[0])
    xs_ramp = np.linspace(ramp_x0, ramp_x1, 250)
    ys_ramp = np.array([env._lane_center_y(3, float(x)) for x in xs_ramp], dtype=float)
    ax.plot(
        xs_ramp,
        np.array([Y(y) for y in ys_ramp], dtype=float),
        linewidth=ramp_lw,
        solid_capstyle="round",
        zorder=1
    )

    # 区間線（破線）(道路より手前、車より奥)
    ax.axvline(cfg.merge_start, linewidth=1, linestyle=(0, (4, 6)), color="k", alpha=0.6, zorder=2)
    ax.axvline(cfg.merge_end,   linewidth=1, linestyle=(0, (4, 6)), color="k", alpha=0.6, zorder=2)

    # -----------------------------
    # vehicle geometry
    # -----------------------------
    veh_h = float(cfg.lane_width) * float(car_width_scale)   # 幅
    veh_w = float(cfg.lane_width) * float(car_len_scale)     # 長さ
    veh_draw_h = veh_h * float(y_scale)

    # -----------------------------
    # collision flash spots (地点を赤) : 車より「少し奥」に描く
    # -----------------------------
    for s in flash_spots:
        x = float(s["x"])
        y = float(s["y"])
        rect = plt.Rectangle(
            (x - veh_w / 2, Y(y) - veh_draw_h / 2),
            veh_w, veh_draw_h,
            fill=True,
            facecolor=flash_color,
            alpha=0.95,
            linewidth=float(veh_edge_lw) + 1.5,
            edgecolor="k",
            zorder=9,  # 車(zorder=10)より奥
        )
        ax.add_patch(rect)

    # -----------------------------
    # vehicles (最前面)
    # -----------------------------
    for i in range(env._num_agents):
        if not env.active[i]:
            continue

        x = float(env.x[i])
        y = float(env.y[i])
        lane = int(env.lane[i])

        # 枠線
        edge_lw = float(veh_edge_lw) + (1.0 if (lane == 3 or env.lc_rem[i] > 0) else 0.0)

        # warn 判定（orange）
        warn = is_warn_vehicle(env, i, close_warn_m, deadend_warn_m)

        if warn:
            fc, a = warn_color, 0.95
        else:
            fc, a = "white", 0.85

        rect = plt.Rectangle(
            (x - veh_w / 2, Y(y) - veh_draw_h / 2),
            veh_w, veh_draw_h,
            fill=True,
            facecolor=fc,
            alpha=a,
            linewidth=edge_lw,
            edgecolor="k",
            zorder=10,
        )
        ax.add_patch(rect)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


# -----------------------------
# 5) ロールアウトして動画化 + 集計を出力
# -----------------------------
def run_and_record(
    env,
    model,
    out_mp4="demo_coopmerge_fixed.mp4",
    fps=15,
    seconds=12,
    deterministic=True,
    seed=0,
    camera_mode="merge",
    pad=25.0,
    tail_x=180.0,
    width=960,
    height=540,
    # warn/flash params
    close_warn_m=None,
    deadend_warn_m=40.0,
    flash_seconds=1.0,
):
    xlim, ylim = compute_fixed_camera_bounds(env.cfg, mode=camera_mode, pad=pad, tail_x=tail_x)
    print("fixed camera:", "xlim=", xlim, "ylim=", ylim)

    if close_warn_m is None:
        close_warn_m = float(getattr(env.cfg, "close_dist_threshold", 20.0))
    close_warn_m = float(close_warn_m)
    deadend_warn_m = float(deadend_warn_m)

    obs_dict, _ = env.reset(seed=seed)
    frames = []
    n_frames = int(seconds * fps)

    flash_frames = int(round(float(flash_seconds) * fps))
    flash_spots = []  # [{"x":..., "y":..., "ttl":...}]

    # --- counters（出力用）---
    C = dict(
        goal=0, crash=0, collision=0, deadend=0,
        warn_total=0, warn_max=0,
        warn_enter=0,

        # reward
        reward_team_sum=0.0,
        reward_team_min=+1e18,
        reward_team_max=-1e18,
        reward_step_count=0,

        # debug: 衝突地点infoが無い場合のカウント
        collision_missing_xy=0,
    )

    prev_warn = np.zeros(env._num_agents, dtype=bool)

    for _ in range(n_frames):
        # --- policy ---
        obs_batch = obs_dict_to_batch(obs_dict)
        act_batch, _ = model.predict(obs_batch, deterministic=deterministic)
        actions = act_batch_to_dict(act_batch)

        # --- step ---
        obs_dict, _, _, trunc_dict, info_dict = env.step(actions)

        # --- reward（チーム報酬）---
        # 環境は common_reward を全agentに複製して返す設計なので、
        # 代表として先頭agentの値を「チーム報酬」として集計（12倍になるsumはNG）
        head = AGENTS[0]
        team_rew = float(info_dict.get(head, {}).get("reward_team", 0.0))
        C["reward_team_sum"] += team_rew
        C["reward_team_min"] = min(C["reward_team_min"], team_rew)
        C["reward_team_max"] = max(C["reward_team_max"], team_rew)
        C["reward_step_count"] += 1


        # --- infos からイベント集計（active_pre のみ数える）---
        goals = crashes = colls = deads = 0
        for i, a in enumerate(AGENTS):
            inf = info_dict.get(a, {})
            if not inf.get("active_pre", True):
                continue
            goals   += int(inf.get("event_goal", False))
            crashes += int(inf.get("event_crash", False))
            colls   += int(inf.get("event_collision", False))
            deads   += int(inf.get("event_deadend", False))

        C["goal"]      += goals
        C["crash"]     += crashes
        C["collision"] += colls
        C["deadend"]   += deads

        # --- collision を「地点」で 1秒フラッシュ ---
        # 必要: step() info に event_x/event_y を追加（互換性は壊れない）
        for i, a in enumerate(AGENTS):
            inf = info_dict.get(a, {})
            if inf.get("event_collision", False) and inf.get("active_pre", False):
                if ("event_x" in inf) and ("event_y" in inf):
                    flash_spots.append({
                        "x": float(inf["event_x"]),
                        "y": float(inf["event_y"]),
                        "ttl": flash_frames
                    })
                else:
                    # event位置が無い場合は「赤が動く」原因になるので、地点フラッシュは出さない
                    C["collision_missing_xy"] += 1

        # --- warn 台数 + warn_enter ---
        warn_flags = np.zeros(env._num_agents, dtype=bool)
        for i in range(env._num_agents):
            warn_flags[i] = is_warn_vehicle(env, i, close_warn_m, deadend_warn_m)

        warn_now = int(np.sum(warn_flags))
        C["warn_total"] += warn_now
        C["warn_max"] = max(C["warn_max"], warn_now)

        enter = np.logical_and(warn_flags, ~prev_warn)
        C["warn_enter"] += int(np.sum(enter))
        prev_warn = warn_flags

        # --- TTL 減衰（地点）---
        for s in flash_spots:
            s["ttl"] -= 1
        flash_spots = [s for s in flash_spots if s["ttl"] > 0]

        # --- render ---
        frame = render_fixed_topdown(
            env, xlim=xlim, ylim=None,
            width=width, height=height,
            y_scale=4.0, pad_y=10.0,
            road_lw=12, ramp_lw=14,
            car_len_scale=5.0, car_width_scale=0.3,
            veh_edge_lw=3.0,
            close_warn_m=close_warn_m,
            deadend_warn_m=deadend_warn_m,
            flash_spots=flash_spots,
            warn_color="orange",
            flash_color="red",
        )
        frames.append(frame)

        # --- time_up でリセット ---
        if any(trunc_dict.values()):
            obs_dict, _ = env.reset(seed=seed)
            prev_warn[:] = False
            flash_spots = []

    # --- write ---
    out_mp4 = str(out_mp4)
    try:
        imageio.mimwrite(out_mp4, frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    except Exception as e:
        print("WARN: libx264 failed:", repr(e))
        out2 = out_mp4.replace(".mp4", "_fallback.mp4")
        imageio.mimwrite(out2, frames, fps=fps, codec="mpeg4", macro_block_size=1)
        out_mp4 = out2

    # --- summary（動画には入れない）---
    print("=== summary (counts in output window) ===")
    print(f"goal      : {C['goal']}")
    print(f"crash     : {C['crash']}")
    print(f"collision : {C['collision']}")
    print(f"deadend   : {C['deadend']}")
    print(f"warn_total (延べwarn台数/フレーム): {C['warn_total']}")
    print(f"warn_max/frame            : {C['warn_max']}")
    print(f"warn_enter (warn突入回数) : {C['warn_enter']}")

    # reward
    mean_step = (C["reward_team_sum"] / C["reward_step_count"]) if C["reward_step_count"] > 0 else 0.0
    print("--- reward (team, per-step) ---")
    print(f"team_return(sum over steps): {C['reward_team_sum']:.6f}")
    print(f"team_reward_mean_per_step  : {mean_step:.6f}")
    print(f"team_reward_min/max_step   : {C['reward_team_min']:.6f} / {C['reward_team_max']:.6f}")

    # collision XY availability
    if C["collision"] > 0 and C["collision_missing_xy"] > 0:
        print(f"[WARN] collision occurred but event_x/event_y missing in info: {C['collision_missing_xy']} times")
        print("       -> step() の info に event_x/event_y を追加してください（観測/報酬/遷移は不変で互換性OK）")

    return out_mp4


mp4_path = run_and_record(
    env, model,
    out_mp4="demo_coopmerge_fixed.mp4",
    fps=15,
    seconds=12,
    deterministic=True,
    seed=0,
    camera_mode="merge",
    pad=25.0,
    tail_x=180.0,
    width=960,
    height=540,
    close_warn_m=None,     # Noneなら cfg.close_dist_threshold を使う
    deadend_warn_m=40.0,
    flash_seconds=1.0,
)

print("saved:", mp4_path)

from IPython.display import Video, display
display(Video(mp4_path, embed=True))
