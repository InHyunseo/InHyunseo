# odor_env_2.py (OdorHold-v2) - Option 2: policy learns RUN vs CAST (2 actions)
# - action: 0=RUN, 1=CAST(start)
# - CAST auto-completes L/R/L/R (4 steps) while agent sees no phi-label
# - obs per step: [c, mode], stacked

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OdorHoldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        L=3.0,
        dt=0.1,
        v_fixed=0.25,

        sensor_offset=0.08,
        sigma_c=1.0,
        sigma_r=1.0,
        r_goal=0.35,
        b_hold=0.5,
        b_oob=5.0,

        max_steps=300,
        stack_n=4,
        seed=0,

        bg_c=0.0,
        sensor_noise=0.01,

        scan_penalty=0.01,     # per CAST step cost
        cast_turn=0.6,         # heading change after CAST completes (radians)
        delta_deadband=0.0,    # if |delta| <= deadband, don't turn
    ):
        super().__init__()
        self.render_mode = render_mode

        self.L = float(L)
        self.dt = float(dt)
        self.v = float(v_fixed)
        self.ds = float(sensor_offset)

        self.sigma_c = float(sigma_c)
        self.sigma_r = float(sigma_r)
        self.r_goal = float(r_goal)
        self.b_hold = float(b_hold)
        self.b_oob = float(b_oob)

        self.max_steps = int(max_steps)
        self.stack_n = int(stack_n)

        self.bg_c = float(bg_c)
        self.sensor_noise = float(sensor_noise)

        self.scan_penalty = float(scan_penalty)
        self.cast_turn = float(cast_turn)
        self.delta_deadband = float(delta_deadband)

        # action: 0 RUN, 1 CAST
        self.action_space = spaces.Discrete(2)

        # obs per step: [c, mode] (NO phi label)
        self.obs_step_dim = 2
        obs_dim = self.obs_step_dim * self.stack_n
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._obs_buf = np.zeros((self.stack_n, self.obs_step_dim), dtype=np.float32)
        self.np_random = np.random.default_rng(seed)

        # state
        self._step = 0
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        # CAST internals: L/L/R/R
        self.in_cast = False
        self.cast_phase = 0
        self._scan_dirs = np.array([np.pi / 2, -np.pi / 2], dtype=np.float32)  # L, R
        self._scan_seq = (0, 1, 0, 1)  # L, R, L, R
        self._scan_c = np.zeros(4, dtype=np.float32)
        self._last_scan_delta = 0.0
        self._last_scan_meanL = 0.0
        self._last_scan_meanR = 0.0

        # render cache
        self._img_size = 360
        self._sense_pt = None
        self._render_scan_idx = 0  # 0=F, 1=L, 2=R
        self._render_mode = 0

    def _conc(self, x, y):
        r2 = x * x + y * y
        c = np.exp(-r2 / (2.0 * self.sigma_c * self.sigma_c))
        c = self.bg_c + (1.0 - self.bg_c) * c
        return float(np.clip(c, 0.0, 1.0))

    def _sense(self, phi):
        ang = float(self.th + phi)
        sx = float(self.x + np.cos(ang) * self.ds)
        sy = float(self.y + np.sin(ang) * self.ds)

        c = self._conc(sx, sy)
        if self.sensor_noise > 0:
            c += float(self.np_random.normal(0, self.sensor_noise))
            c = float(np.clip(c, 0.0, 1.0))

        self._sense_pt = (sx, sy)
        return float(c)

    def _push_obs(self, c, mode):
        row = np.array([c, float(mode)], dtype=np.float32)
        self._obs_buf[:-1] = self._obs_buf[1:]
        self._obs_buf[-1] = row

    def _get_obs(self):
        return self._obs_buf.reshape(-1).copy()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # ring spawn
        r_min = max(self.r_goal + 0.25, 0.6)
        r_max = min(0.8 * self.L, self.L - 0.25)
        r0 = self.np_random.uniform(r_min, r_max)
        ang = self.np_random.uniform(-np.pi, np.pi)

        self.x = r0 * np.cos(ang)
        self.y = r0 * np.sin(ang)
        self.th = self.np_random.uniform(-np.pi, np.pi)

        self._step = 0

        self.in_cast = False
        self.cast_phase = 0
        self._scan_c[:] = 0.0
        self._last_scan_delta = 0.0
        self._last_scan_meanL = 0.0
        self._last_scan_meanR = 0.0

        # prefill with forward sensing (RUN-like obs)
        c0 = self._sense(0.0)
        self._render_scan_idx = 0
        self._render_mode = 0

        self._obs_buf[:] = 0.0
        for i in range(self.stack_n):
            self._obs_buf[i] = np.array([c0, 0.0], dtype=np.float32)  # mode=0 (RUN)

        return self._get_obs(), {}

    def _norm_angle(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _cast_step(self):
        i = int(self.cast_phase)
        side = int(self._scan_seq[i])  # 0=L, 1=R
        phi = float(self._scan_dirs[side])

        self._render_scan_idx = 1 if side == 0 else 2
        self._render_mode = 1

        c = self._sense(phi)
        self._scan_c[i] = float(c)
        self._push_obs(c, mode=1)

        self.cast_phase += 1

        finished = (self.cast_phase >= 4)
        if finished:
            meanL = float((self._scan_c[0] + self._scan_c[2]) * 0.5)
            meanR = float((self._scan_c[1] + self._scan_c[3]) * 0.5)
            delta = float(meanL - meanR)

            self._last_scan_meanL = meanL
            self._last_scan_meanR = meanR
            self._last_scan_delta = delta

            # reorient heading toward higher side
            if abs(delta) > self.delta_deadband:
                self.th = self._norm_angle(self.th + (self.cast_turn if delta > 0 else -self.cast_turn))

            # exit cast
            self.in_cast = False
            self.cast_phase = 0
            self._scan_c[:] = 0.0

        return float(c), finished

    def step(self, action):
        self._step += 1
        a = int(action)

        # If currently casting, auto-finish cast regardless of action
        if self.in_cast:
            mode = 1
            c, cast_done = self._cast_step()
            moved = False
        else:
            if a == 1:
                # start CAST and do first scan step immediately
                self.in_cast = True
                self.cast_phase = 0
                self._scan_c[:] = 0.0
                mode = 1
                c, cast_done = self._cast_step()
                moved = False
            else:
                # RUN: move forward (no explicit turning action)
                mode = 0
                self.x += self.v * np.cos(self.th) * self.dt
                self.y += self.v * np.sin(self.th) * self.dt

                self._render_scan_idx = 0
                self._render_mode = 0

                c = self._sense(0.0)
                self._push_obs(c, mode=0)
                moved = True

        oob = (abs(self.x) > self.L) or (abs(self.y) > self.L)
        terminated = bool(oob)
        truncated = bool(self._step >= self.max_steps)

        d = float(np.hypot(self.x, self.y))

        # reward
        r = float(np.exp(-d / self.sigma_r))
        if d < self.r_goal:
            r += self.b_hold
        if mode == 1:
            r -= self.scan_penalty
        if oob:
            r -= self.b_oob

        info = {
            "d": d,
            "c": float(c),
            "mode": int(mode),           # 0 RUN, 1 CAST
            "in_cast": int(self.in_cast),
            "cast_phase": int(self.cast_phase),
            "moved": int(moved),
            "scan_meanL": float(self._last_scan_meanL),
            "scan_meanR": float(self._last_scan_meanR),
            "scan_delta": float(self._last_scan_delta),
            "in_goal": int(d < self.r_goal),
        }
        return self._get_obs(), r, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None

        W = H = self._img_size
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            px = int((x + self.L) / (2 * self.L) * (W - 1))
            py = int((self.L - y) / (2 * self.L) * (H - 1))
            return px, py

        # goal circle + source
        cx, cy = to_px(0.0, 0.0)
        rg = int(self.r_goal / (2 * self.L) * (W - 1))
        draw.ellipse((cx - rg, cy - rg, cx + rg, cy + rg), outline=(180, 180, 180), width=2)
        rs = 4
        draw.ellipse((cx - rs, cy - rs, cx + rs, cy + rs), fill=(0, 0, 0))

        # agent triangle
        ax, ay = to_px(self.x, self.y)
        size = 10
        th = self.th
        p0 = (ax + size * np.cos(th), ay - size * np.sin(th))
        p1 = (ax + size * np.cos(th + 2.5), ay - size * np.sin(th + 2.5))
        p2 = (ax + size * np.cos(th - 2.5), ay - size * np.sin(th - 2.5))
        tri = [tuple(map(int, p0)), tuple(map(int, p1)), tuple(map(int, p2))]
        draw.polygon(tri, fill=(50, 100, 220))

        # sensor ray + label (debug only)
        if self._sense_pt is not None:
            sx, sy = self._sense_pt
            px, py = to_px(sx, sy)
            draw.line((ax, ay, px, py), fill=(220, 60, 60), width=2)
            rr = 3
            draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(220, 60, 60))
            labels = ["F", "L", "R"]
            try:
                draw.text((ax + 8, ay + 8), labels[int(self._render_scan_idx)], fill=(0, 0, 0))
            except Exception:
                pass

        return np.array(img, dtype=np.uint8)

    def close(self):
        pass