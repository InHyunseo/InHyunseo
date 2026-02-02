import os
import csv
import time
import json
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# --- plotting / gif ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gymnasium.envs.registration import register

# --- 상대경로를 통한 결과 저장 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "runs")


# -----------------------
# Model (DQN)
# -----------------------
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


# -----------------------
# Replay Buffer
# -----------------------
class ReplayBuffer:
    def __init__(self, cap=150000):
        self.cap = int(cap)
        self.buf = []
        self.i = 0

    def push(self, s, a, r, ns, done):
        item = (s, a, r, ns, done)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.i] = item
        self.i = (self.i + 1) % self.cap

    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)


# -----------------------
# Utils: moving average / plots / gif
# -----------------------
def moving_avg(x, window=50):
    if len(x) == 0:
        return np.array([])
    w = max(1, int(window))
    x = np.asarray(x, dtype=np.float32)
    if len(x) < w:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="valid")


def save_training_figure_dqn(run_dir, episodes, success, goal_rate, avg_return, window=50):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ep = np.asarray(episodes, dtype=np.int32)
    success = np.asarray(success, dtype=np.float32)
    goal_rate = np.asarray(goal_rate, dtype=np.float32)
    avg_return = np.asarray(avg_return, dtype=np.float32)

    s_ma = moving_avg(success, window)
    g_ma = moving_avg(goal_rate, window)
    ar_ma = moving_avg(avg_return, window)
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(1, 1, 1)

    if len(s_ma) > 0:
        ax.plot(ep[len(ep) - len(s_ma):], s_ma, label=f"success MovingAverage({window})")
    if len(g_ma) > 0:
        ax.plot(ep[len(ep) - len(g_ma):], g_ma, label=f"goal_rate MovingAverage({window})")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("episode")
    ax.set_ylabel("rate")
    ax.grid(True, alpha=0.3)

    axr = ax.twinx()
    if len(ar_ma) > 0:
        axr.plot(ep[len(ep) - len(ar_ma):], ar_ma, linestyle="--", label=f"avg_return MovingAverage({window})")
    axr.set_ylabel("avg_return")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

    out_path = os.path.join(plots_dir, "training_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
def save_eval_figure_dqn(run_dir, eval_eps, eval_success, eval_goal_rate, eval_avg_return, best_ep=None):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(eval_eps) == 0:
        return None

    ep = np.asarray(eval_eps, dtype=np.int32)
    s = np.asarray(eval_success, dtype=np.float32)
    g = np.asarray(eval_goal_rate, dtype=np.float32)
    ar = np.asarray(eval_avg_return, dtype=np.float32)

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(ep, s, marker="o", label="EVAL success (greedy)")
    ax.plot(ep, g, marker="o", label="EVAL goal_rate (greedy)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("episode")
    ax.set_ylabel("rate")
    ax.grid(True, alpha=0.3)

    axr = ax.twinx()
    axr.plot(ep, ar, marker="o", linestyle="--", label="EVAL avg_return (greedy)")
    axr.set_ylabel("avg_return")

    if best_ep is not None:
        ax.axvline(int(best_ep), linestyle=":", linewidth=2, alpha=0.7, label=f"best_ep={best_ep}")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

    out_path = os.path.join(plots_dir, "eval_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_gif(frames, path, fps=30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration = 1.0 / float(fps)

    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, duration=duration)
        return
    except Exception:
        pass

    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * duration),
        loop=0,
    )

@torch.no_grad()
def eval_greedy(env_id, env_kwargs, model, device, episodes=30, seed_base=100000):
    env = gym.make(env_id, **(env_kwargs or {}))
    model.eval()

    succ = []
    goal_rates = []
    avg_returns = []

    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        steps = 0
        ep_ret = 0.0
        in_goal = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(torch.argmax(model(obs_t), dim=1).item())  # greedy

            obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)

            ep_ret += float(r)
            in_goal += int(info.get("in_goal", 0))
            steps += 1

        s = int(in_goal > 0)
        gr = in_goal / max(1, steps)
        ar = ep_ret / max(1, steps)

        succ.append(s)
        goal_rates.append(gr)
        avg_returns.append(ar)

    env.close()

    # mean, std
    return (
        float(np.mean(succ)), float(np.std(succ)),
        float(np.mean(goal_rates)), float(np.std(goal_rates)),
        float(np.mean(avg_returns)), float(np.std(avg_returns)),
    )
@torch.no_grad()
def rollout_video_gif(env_id, model, device, seed, out_gif_path, fps=30, deterministic=True):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    frames = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        qvals = model(obs_t).squeeze(0)

        if deterministic:
            action = int(torch.argmax(qvals).item())
        else:
            # soft exploration for video (원하면)
            action = int(torch.randint(0, qvals.numel(), (1,)).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs = next_obs

    env.close()
    save_gif(frames, out_gif_path, fps=fps)
    return frames


# -----------------------
# Train (DQN)
# -----------------------
def train(
    env_id="OdorHold-v1",
    seed=0,
    total_episodes=600,

    gamma=0.99,
    lr=3e-4,
    max_grad_norm=10.0,

    # DQN core
    buffer_size=150000,
    batch_size=256,
    learning_starts=5000,      # steps
    target_update_every=1000,  # steps

    # epsilon schedule (step-based)
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=80000,

    # 저장 관련
    out_dir=DEFAULT_OUT_DIR,
    run_name=None,
    log_every=50,
    
    # 평가 관련
    eval_every=50,
    eval_episodes=30,
    eval_seed_base=100000,

    # 시각화 관련
    plot_ma_window=50,
    video_fps=30,
    video_eval_seed=123,

    force_cpu=False,

    env_kwargs = dict(
    sensor_noise=0.01,  
)
):
    # register env (gym.make 사용 유지)
    try:
        gym.make(env_id, **(env_kwargs or {}))
    except Exception:
        register(
            id=env_id,
            entry_point="odor_env_v1:OdorHoldEnv",
            kwargs=dict(render_mode=None, **(env_kwargs or {})),
        )

    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_name is None:
        run_name = time.strftime("odor_dqn_%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    media_dir = os.path.join(run_dir, "media")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    config = dict(
        env_id=env_id, seed=seed, total_episodes=total_episodes,
        gamma=gamma, lr=lr,
        buffer_size=buffer_size, batch_size=batch_size,
        learning_starts=learning_starts, target_update_every=target_update_every,
        eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay_steps,
        max_grad_norm=max_grad_norm, force_cpu=force_cpu,

        log_every=log_every,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        eval_seed_base=eval_seed_base,

        plot_ma_window=plot_ma_window,
        video_fps=video_fps, video_eval_seed=video_eval_seed,
        env_kwargs=env_kwargs,
    )

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "endured_step", "in_goal_steps", "success", "goal_rate", "avg_return"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_id, **(env_kwargs or {}))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q = QNet(obs_dim, act_dim).to(device)
    tq = QNet(obs_dim, act_dim).to(device)
    tq.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()

    rb = ReplayBuffer(buffer_size)

    # init ckpt
    torch.save(q.state_dict(), os.path.join(ckpt_dir, "init.pt"))

    episodes = []
    ep_successes = []     # 0/1
    ep_goal_rates = []    # in_goal_steps / steps
    ep_avg_return = []         # episode_return / steps
    
    best_score = -1e9
    best_ep = None
    best_path = os.path.join(ckpt_dir, "best.pt")

    eval_eps = []
    eval_succ_m = []
    eval_goal_m = []
    eval_avgr_m = []

    eval_metrics_path = os.path.join(run_dir, "metrics_eval.csv")
    with open(eval_metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "success_mean", "goal_rate_mean", "avg_return_mean"])

    global_step = 0

    for ep in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done = False
        endured_step = 0
        ep_ret = 0.0
        in_goal_steps = 0

        while not done:
            endured_step += 1
            global_step += 1

            frac = min(1.0, global_step / float(eps_decay_steps))
            eps = eps_start + frac * (eps_end - eps_start)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    qvals = q(obs_t).squeeze(0)
                    action = int(torch.argmax(qvals).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rb.push(obs, action, float(reward), next_obs, float(done))

            obs = next_obs
            ep_ret += float(reward)
            in_goal_steps += int(info.get("in_goal", 0))

            # learn
            if len(rb) >= learning_starts:
                bs, ba, br, bns, bd = rb.sample(batch_size)

                bs = torch.tensor(bs, dtype=torch.float32, device=device)
                ba = torch.tensor(ba, dtype=torch.int64, device=device).unsqueeze(1)
                br = torch.tensor(br, dtype=torch.float32, device=device).unsqueeze(1)
                bns = torch.tensor(bns, dtype=torch.float32, device=device)
                bd = torch.tensor(bd, dtype=torch.float32, device=device).unsqueeze(1)

                qsa = q(bs).gather(1, ba)

                with torch.no_grad():
                    # Double DQN
                    next_a = torch.argmax(q(bns), dim=1, keepdim=True)
                    next_q = tq(bns).gather(1, next_a)
                    y = br + gamma * (1.0 - bd) * next_q

                loss = huber(qsa, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
                optimizer.step()

                if global_step % target_update_every == 0:
                    tq.load_state_dict(q.state_dict())

        success = int(in_goal_steps > 0)

        episodes.append(ep)
        # ep_returns.append(ep_ret)
        # ep_steps_hist.append(endured_step)
        ep_successes.append(success)
        goal_rate = float(in_goal_steps) / float(endured_step) if endured_step > 0 else 0.0
        avg_return = float(ep_ret) / float(endured_step) if endured_step > 0 else 0.0
        ep_goal_rates.append(goal_rate)
        ep_avg_return.append(avg_return)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ep, endured_step, in_goal_steps, success, goal_rate, avg_return])

        # ---- periodic eval (greedy) + best checkpoint ----
        if (ep % eval_every == 0) or (ep == 1) or (ep == total_episodes):
            sm, ss, gm, gs, am, astd = eval_greedy(
                env_id, env_kwargs, q, device,
                episodes=eval_episodes,
                seed_base=eval_seed_base
            )

            # eval curve 저장용 (mean만)
            eval_eps.append(ep)
            eval_succ_m.append(sm)
            eval_goal_m.append(gm)
            eval_avgr_m.append(am)

            with open(eval_metrics_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([ep, sm, gm, am])

            # best 기준: goal_rate 우선, 동률이면 avg_return
            score = gm * 1000.0 + am
            if score > best_score:
                best_score = score
                best_ep = ep
                torch.save(q.state_dict(), best_path)
                with open(os.path.join(ckpt_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {"best_ep": best_ep, "success": sm, "goal_rate": gm, "avg_return": am},
                        f, ensure_ascii=False, indent=2
                    )

            print(
                f"[EVAL] ep={ep} | success={sm:.3f} | "
                f"goal_rate={gm:.3f} | avg_return={am:.3f} | best_ep={best_ep}"
            )
        

        if ep == total_episodes:
            torch.save(q.state_dict(), os.path.join(ckpt_dir, "final.pt"))

        if ep % log_every == 0:
            k = min(log_every, len(ep_successes))
            mean_succ = float(np.mean(ep_successes[-k:]))
            mean_goal = float(np.mean(ep_goal_rates[-k:]))
            mean_avgr = float(np.mean(ep_avg_return[-k:]))
            print(
                f"[TRAIN] ep={ep:4d} | "
                f"success({k})={mean_succ:.3f} | "
                f"goal_rate({k})={mean_goal:.3f} | "
                f"avg_return({k})={mean_avgr:.3f}"
            )

    env.close()

    save_training_figure_dqn(
        run_dir,
        episodes=episodes,
        success=ep_successes,
        goal_rate=ep_goal_rates,
        avg_return=ep_avg_return,
        window=plot_ma_window,
    )
    
    save_eval_figure_dqn(
        run_dir,
        eval_eps=eval_eps,
        eval_success=eval_succ_m,
        eval_goal_rate=eval_goal_m,
        eval_avg_return=eval_avgr_m,
        best_ep=best_ep,
    )
    def load_ckpt(path):
        m = QNet(obs_dim, act_dim).to(device)
        sd = torch.load(path, map_location=device)
        m.load_state_dict(sd)
        m.eval()
        return m

    init_model = load_ckpt(os.path.join(ckpt_dir, "init.pt"))
    final_model = load_ckpt(os.path.join(ckpt_dir, "final.pt"))

    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, "final.pt")
    best_model = load_ckpt(best_ckpt)

    rollout_video_gif(
        env_id, init_model, device, seed=video_eval_seed,
        out_gif_path=os.path.join(media_dir, "init.gif"),
        fps=video_fps, deterministic=True
    )
    rollout_video_gif(
        env_id, best_model, device, seed=video_eval_seed,
        out_gif_path=os.path.join(media_dir, "best.gif"),
        fps=video_fps, deterministic=True
    )
    rollout_video_gif(
        env_id, final_model, device, seed=video_eval_seed,
        out_gif_path=os.path.join(media_dir, "final.gif"),
        fps=video_fps, deterministic=True
    )

    return q, run_dir


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-episodes", type=int, default=600)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=10.0)

    p.add_argument("--buffer-size", type=int, default=150000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--target-update-every", type=int, default=1000)

    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=80000)

    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-seed-base", type=int, default=100000)

    p.add_argument("--plot-ma-window", type=int, default=50)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-eval-seed", type=int, default=123)

    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--sensor-noise", type=float, default=0.01)

    args = p.parse_args()
    train(seed=args.seed,
        total_episodes=args.total_episodes,
        gamma=args.gamma,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        target_update_every=args.target_update_every,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        out_dir=args.out_dir,
        run_name=args.run_name,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seed_base=args.eval_seed_base,
        plot_ma_window=args.plot_ma_window,
        video_fps=args.video_fps,
        video_eval_seed=args.video_eval_seed,
        force_cpu=args.force_cpu,
        env_kwargs=dict(sensor_noise=args.sensor_noise),
        )