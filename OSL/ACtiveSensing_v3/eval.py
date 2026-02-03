import os
import json
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gymnasium.envs.registration import register, registry


# ---- Model (train과 동일) ----
class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def ensure_env_registered(env_id, entry_point, env_kwargs):
    if env_id not in registry:
        register(
            id=env_id,
            entry_point=entry_point,
            kwargs=dict(render_mode=None, **(env_kwargs or {})),
        )

def load_run_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    env_id = cfg.get("env_id", "OdorHold-v2")
    entry_point = cfg.get("entry_point", "odor_env_v2:OdorHoldEnv")
    env_kwargs = cfg.get("env_kwargs", {})
    return env_id, entry_point, env_kwargs


def load_model(ckpt_path, obs_dim, act_dim, device):
    m = QNet(obs_dim, act_dim).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def rollout_trajectories(env_id, env_kwargs, model, device, episodes=20, seed_base=100000):
    env = gym.make(env_id, **(env_kwargs or {}))
    L = float(env.unwrapped.L)
    r_goal = float(env.unwrapped.r_goal)
    sigma_c = float(env.unwrapped.sigma_c)
    bg_c = float(env.unwrapped.bg_c)
    wind_x = float(getattr(env.unwrapped, "wind_x", 0.0))
    wind_y = float(getattr(env.unwrapped, "wind_y", 0.0))

    trajs = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        xs, ys = [float(env.unwrapped.x)], [float(env.unwrapped.y)]
        done = False
        ep_ret = 0.0
        in_goal = 0
        steps = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(torch.argmax(model(obs_t), dim=1).item())  # greedy

            obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)

            xs.append(float(env.unwrapped.x))
            ys.append(float(env.unwrapped.y))

            ep_ret += float(r)
            in_goal += int(info.get("in_goal", 0))
            steps += 1

        trajs.append({
            "seed": seed_base + i,
            "return": float(ep_ret),
            "goal_rate": float(in_goal) / max(1, steps),
            "x": xs,
            "y": ys,
        })

    env.close()
    return trajs, L, r_goal, sigma_c, bg_c, wind_x, wind_y


def save_traj_plot(out_path, trajs, L, r_goal, sigma_c, bg_c, wind_x, wind_y, title):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)

    # world boundary
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # odor concentration heatmap
    res = 220
    xs = np.linspace(-L, L, res, dtype=np.float32)
    ys = np.linspace(-L, L, res, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    wind_mag = float(np.hypot(wind_x, wind_y))
    if wind_mag <= 1e-6:
        r2 = X * X + Y * Y
        c = np.exp(-r2 / (2.0 * sigma_c * sigma_c))
    else:
        wx = wind_x / wind_mag
        wy = wind_y / wind_mag
        t = X * wx + Y * wy
        s = -X * wy + Y * wx
        stretch = 1.0 + min(wind_mag, 2.0)
        sigma_s = sigma_c
        sigma_t = sigma_c * stretch
        sigma_up = sigma_c / stretch
        t_pos = np.maximum(0.0, t)
        t_neg = np.maximum(0.0, -t)
        c = np.exp(-(
            (s * s) / (2.0 * sigma_s * sigma_s) +
            (t_pos * t_pos) / (2.0 * sigma_t * sigma_t) +
            (t_neg * t_neg) / (2.0 * sigma_up * sigma_up)
        ))
    c = bg_c + (1.0 - bg_c) * c
    c = np.clip(c, 0.0, 1.0)
    im = ax.imshow(c, extent=[-L, L, -L, L], origin="lower", cmap="inferno", alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("odor concentration")

    # source at (0,0)
    ax.plot([0], [0], marker="o")
    # goal circle
    th = np.linspace(0, 2*np.pi, 256)
    ax.plot(r_goal*np.cos(th), r_goal*np.sin(th), linewidth=2, alpha=0.8)

    # trajectories
    for t in trajs:
        x = np.asarray(t["x"], dtype=np.float32)
        y = np.asarray(t["y"], dtype=np.float32)
        ax.plot(x, y, alpha=0.6)
        ax.plot([x[0]], [y[0]], marker="x")     # start
        ax.plot([x[-1]], [y[-1]], marker="s")   # end

    # summary
    rets = [t["return"] for t in trajs]
    goals = [t["goal_rate"] for t in trajs]
    ax.set_title(f"{title}\nreturn={np.mean(rets):.2f}±{np.std(rets):.2f} | goal_rate={np.mean(goals):.3f}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed_base", type=int, default=100000)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--save_json", action="store_true")  # trajectory raw 저장 옵션
    args = ap.parse_args()

    env_id, entry_point, env_kwargs = load_run_config(args.run_dir)
    ensure_env_registered(env_id, entry_point, env_kwargs)

    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dim 알아내기
    env = gym.make(env_id, **(env_kwargs or {}))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    media_dir = os.path.join(args.run_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    ckpts = [
        ("init",  os.path.join(ckpt_dir, "init.pt")),
        ("best",   os.path.join(ckpt_dir, "best.pt")),
        ("final", os.path.join(ckpt_dir, "final.pt")),
    ]

    for tag, ckpt_path in ckpts:
        if not os.path.exists(ckpt_path):
            print(f"[WARN] missing: {ckpt_path}")
            continue

        model = load_model(ckpt_path, obs_dim, act_dim, device)
        trajs, L, r_goal, sigma_c, bg_c, wind_x, wind_y = rollout_trajectories(
            env_id, env_kwargs,model, device, episodes=args.episodes, seed_base=args.seed_base
        )

        out_png = os.path.join(media_dir, f"traj_{tag}.png")
        save_traj_plot(out_png, trajs, L, r_goal, sigma_c, bg_c, wind_x, wind_y, title=f"{tag}.pt")

        if args.save_json:
            out_json = os.path.join(media_dir, f"traj_{tag}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(trajs, f, ensure_ascii=False, indent=2)

        rets = [t["return"] for t in trajs]
        goals = [t["goal_rate"] for t in trajs]
        print(f"[EVAL:{tag}] return={np.mean(rets):.3f}±{np.std(rets):.3f} | goal_rate={np.mean(goals):.3f} | saved={out_png}")

if __name__ == "__main__":
    main()
