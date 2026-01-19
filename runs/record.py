# record.py
import os
import glob
import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# -----------------------
# Model (train.py와 동일해야 함)
# -----------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # logits
        self.critic = nn.Linear(hidden, 1)        # V(s)

    def forward(self, x):
        h = self.body(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


def pick_latest_run(runs_dir="runs"):
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"no run subdir under: {runs_dir}")
    return max(subdirs, key=os.path.getmtime)


def load_policy(model_path, env_id="CartPole-v1", force_cpu=True):
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id)  # 모델 차원 확인용
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    model = ActorCritic(obs_dim, act_dim).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, device


def run_and_record(model, device, out_dir, env_id="CartPole-v1", episodes=1, deterministic=True):
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=out_dir,
        name_prefix=os.path.basename(out_dir),
        episode_trigger=lambda ep: True,
        disable_logger=True,
    )

    endured_steps = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            steps += 1
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = model(x)
                dist = Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(dist.probs).item()
                else:
                    action = dist.sample().item()

            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        endured_steps.append(steps)

    env.close()
    return endured_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None, help="e.g., runs/cartpole_ac_YYYYmmdd_HHMMSS")
    p.add_argument("--runs_dir", type=str, default="runs", help="parent runs dir (used if run_dir is None)")
    p.add_argument("--env_id", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--deterministic", action="store_true", help="use argmax action")
    p.add_argument("--cpu", action="store_true", help="force cpu")
    args = p.parse_args()

    run_dir = args.run_dir or pick_latest_run(args.runs_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoints dir not found: {ckpt_dir}")

    # init/mid/final 자동 탐색
    targets = ["init.pt", "mid.pt", "final.pt"]
    found = []
    for t in targets:
        path = os.path.join(ckpt_dir, t)
        if os.path.isfile(path):
            found.append((t.replace(".pt", ""), path))

    if not found:
        # 혹시 다른 이름(best.pt 등)만 있는 경우 대비
        pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        if not pts:
            raise FileNotFoundError(f"no .pt files under {ckpt_dir}")
        found = [(os.path.splitext(os.path.basename(x))[0], x) for x in pts]

    videos_root = os.path.join(run_dir, "videos")
    os.makedirs(videos_root, exist_ok=True)

    print(f"[run_dir] {run_dir}")
    print(f"[ckpts] " + ", ".join([name for name, _ in found]))

    for name, model_path in found:
        out_dir = os.path.join(videos_root, name)
        model, device = load_policy(model_path, env_id=args.env_id, force_cpu=args.cpu)
        steps = run_and_record(
            model, device, out_dir,
            env_id=args.env_id,
            episodes=args.episodes,
            deterministic=args.deterministic,
        )
        print(f"[{name}] endured_steps={steps} -> videos in {out_dir}")


if __name__ == "__main__":
    main()
