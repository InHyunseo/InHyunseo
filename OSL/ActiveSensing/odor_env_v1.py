# odor_env_v1.py (OdorHold-v1) (v0 was 2-sensor-passive-sensing-env) - single sensor + 4-direction scan active sensing
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
        w_list=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
        sensor_offset=0.08,
        sigma_c=1.0,
        sigma_r=1.0,
        r_goal=0.35,
        b_hold=0.5,
        lam_w=0.02,
        b_oob=5.0,
        max_steps=300,
        stack_n=4,
        seed=0,
        bg_c=0.0,
        sensor_noise=0.01,
        scan_every=1,  # direction change period in env steps (1 recommended)
    ):
        super().__init__()
        self.render_mode = render_mode

        self.L = float(L)
        self.dt = float(dt)
        self.v = float(v_fixed)
        self.w_list = np.array(w_list, dtype=np.float32)
        self.ds = float(sensor_offset)

        self.sigma_c = float(sigma_c)
        self.sigma_r = float(sigma_r)
        self.r_goal = float(r_goal)
        self.b_hold = float(b_hold)
        self.lam_w = float(lam_w)
        self.b_oob = float(b_oob)
        self.max_steps = int(max_steps)
        self.stack_n = int(stack_n)
        self.bg_c = float(bg_c)
        self.sensor_noise = float(sensor_noise)

        self.scan_every = int(scan_every)
        self._scan_dirs = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float32)  # F,L,B,R
        self.scan_idx = 0
        self.phi = 0.0
        self.sense_ang = 0.0
        self._sense_pt = None

        self.action_space = spaces.Discrete(len(self.w_list))

        # per-step obs = [c, sin(phi), cos(phi)] => 3
        obs_dim = 3 * self.stack_n
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.np_random = np.random.default_rng(seed)
        self._obs_buf = np.zeros((self.stack_n, 3), dtype=np.float32)
        self._step = 0

        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        # render cache
        self._img_size = 360

    def _conc(self, x, y):
        r2 = x * x + y * y
        c = np.exp(-r2 / (2.0 * self.sigma_c * self.sigma_c))
        c = self.bg_c + (1.0 - self.bg_c) * c
        return float(np.clip(c, 0.0, 1.0))

    def _sense_scan(self, phi):
        ang = float(self.th + phi)
        sx = float(self.x + np.cos(ang) * self.ds)
        sy = float(self.y + np.sin(ang) * self.ds)

        c = self._conc(sx, sy)
        if self.sensor_noise > 0:
            c += float(self.np_random.normal(0, self.sensor_noise))
            c = float(np.clip(c, 0.0, 1.0))

        self._sense_pt = (sx, sy)
        self.phi = float(phi)
        self.sense_ang = ang
        return np.array([c, np.sin(phi), np.cos(phi)], dtype=np.float32)

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
        self.scan_idx = 0

        # prefill stack with F,L,B,R (or repeats if stack_n != 4)
        for i in range(self.stack_n):
            phi = float(self._scan_dirs[i % 4])
            self._obs_buf[i] = self._sense_scan(phi)
        self.scan_idx = (self.stack_n - 1) % 4

        return self._get_obs(), {}

    def step(self, action):
        self._step += 1
        w = float(self.w_list[int(action)])

        # move first (same as original)
        self.x += self.v * np.cos(self.th) * self.dt
        self.y += self.v * np.sin(self.th) * self.dt
        self.th += w * self.dt
        self.th = (self.th + np.pi) % (2 * np.pi) - np.pi

        # update scan direction
        if (self._step % self.scan_every) == 0:
            self.scan_idx = (self.scan_idx + 1) % 4
        phi = float(self._scan_dirs[self.scan_idx])

        s = self._sense_scan(phi)
        self._obs_buf[:-1] = self._obs_buf[1:]
        self._obs_buf[-1] = s

        oob = (abs(self.x) > self.L) or (abs(self.y) > self.L)
        terminated = bool(oob)
        truncated = bool(self._step >= self.max_steps)

        d = float(np.hypot(self.x, self.y))

        r = float(np.exp(-d / self.sigma_r))
        if d < self.r_goal:
            r += self.b_hold
        r -= self.lam_w * abs(w)
        if oob:
            r -= self.b_oob

        info = {
            "d": d,
            "w": w,
            "c": float(s[0]),
            "scan_idx": int(self.scan_idx),
            "phi": float(self.phi),
            "sense_ang": float(self.sense_ang),
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

        # sensor ray + label
        if self._sense_pt is not None:
            sx, sy = self._sense_pt
            px, py = to_px(sx, sy)
            draw.line((ax, ay, px, py), fill=(220, 60, 60), width=2)
            rr = 3
            draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(220, 60, 60))
            labels = ["F", "L", "B", "R"]
            try:
                draw.text((ax + 8, ay + 8), labels[int(self.scan_idx)], fill=(0, 0, 0))
            except Exception:
                pass

        return np.array(img, dtype=np.uint8)

    def close(self):
        pass