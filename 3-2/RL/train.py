import os
import csv
import time
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------
# Model
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


# -----------------------
# Reward shaping
# -----------------------
def shaped_reward(env, obs, terminated, truncated):
    u = env.unwrapped
    x_th = getattr(u, "x_threshold", 2.4)
    th_th = getattr(u, "theta_threshold_radians", 12 * np.pi / 180.0)

    x = float(obs[0])
    th = float(obs[2])

    r = 1.0 - (th / th_th) ** 2 - 0.01 * (x / x_th) ** 2

    if terminated:
        r -= 1.0

    if truncated and (not terminated):
        r += 100.0

    return r


# -----------------------
# Train
# -----------------------
def train(
    env_id="CartPole-v1",
    seed=0,
    total_episodes=2000,
    gamma=0.99,
    lr=3e-4,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    reward_scale=0.01,

    # 저장 관련 추가
    out_dir="runs",
    run_name=None,                 # None이면 timestamp로 자동 생성
    save_mid_episode=None,         # None이면 total_episodes//2
    log_every=50,

    # 느리면 True로(특히 CartPole은 CPU가 더 빠른 경우 많음)
    force_cpu=False,
):
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run dir
    if run_name is None:
        run_name = time.strftime("cartpole_ac_%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # config 저장
    config = dict(
        env_id=env_id, seed=seed, total_episodes=total_episodes, gamma=gamma, lr=lr,
        value_coef=value_coef, entropy_coef=entropy_coef, max_grad_norm=max_grad_norm,
        reward_scale=reward_scale, force_cpu=force_cpu,
        save_mid_episode=save_mid_episode if save_mid_episode is not None else total_episodes // 2,
    )
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # metrics.csv 준비
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode",
            "endured_step",
            "shaped_return",     # reward_scale 적용된 합
            "success",           # truncated & not terminated
            "actor_loss",
            "critic_loss",
        ])

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()

    # 체크포인트: 초기(학습 전) 저장 -> "처음 영상"용
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "init.pt"))

    if save_mid_episode is None:
        save_mid_episode = total_episodes // 2

    ep_returns = []
    ep_steps_hist = []

    for ep in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + ep)

        log_probs = []
        values = []
        rewards = []
        entropies = []

        endured_step = 0
        done = False

        # 에피소드 종료 상태 기록용
        terminated = False
        truncated = False

        while not done:
            endured_step += 1
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            step_out = env.step(int(action.item()))
            if len(step_out) == 5:
                next_obs, _, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_obs, _, done, _ = step_out
                terminated, truncated = done, False

            r = shaped_reward(env, next_obs, terminated, truncated) * reward_scale

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
            entropies.append(entropy)

            obs = next_obs

        # Logging metric
        ep_return = float(torch.stack(rewards).sum().item())
        ep_returns.append(ep_return)
        ep_steps_hist.append(endured_step)
        success = int(truncated and (not terminated))

        # Compute returns G_t (Monte-Carlo)
        returns = []
        G = torch.tensor(0.0, device=device)
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.stack(returns).detach()

        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        adv = (returns - values_t).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_loss = -(log_probs_t * adv).mean()
        critic_loss = huber(values_t, returns)
        entropy_bonus = entropies_t.mean()

        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # metrics.csv append
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep,
                endured_step,
                ep_return,
                success,
                float(actor_loss.item()),
                float(critic_loss.item()),
            ])

        # 체크포인트: 중간 / 최종
        if ep == save_mid_episode:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "mid.pt"))
        if ep == total_episodes:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))

        # Console log (최근 50 에피소드 평균 steps 같이)
        if ep % log_every == 0:
            mean_ret = np.mean(ep_returns[-log_every:])
            mean_steps = np.mean(ep_steps_hist[-log_every:])
            print(f"Episode {ep:4d} | mean_steps({log_every})={mean_steps:6.1f} | mean_return({log_every})={mean_ret: .3f} | "
                  f"actor={actor_loss.item():.4f} critic={critic_loss.item():.4f}")

    env.close()
    return model, run_dir


if __name__ == "__main__":
    train()
