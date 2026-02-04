# odor_env_v3.py (OdorHold-v3) - Wind-shaped plume, no wind sensor in obs
# - action: 0=RUN, 1=CAST(start), 2=TURN_L, 3=TURN_R
# - CAST auto-completes L/R/L/R (4 steps) while agent sees no phi-label
# - CAST 완료 직후 need_turn=True가 되고, 그때만 TURN_L/R 1회 선택 가능
# - obs per step: [c, mode], stacked  (mode: 0=RUN, 1=CAST/decision)

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

        wind_x=1.0,
        wind_y=0.0,

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
        turn_penalty=0.01,     # per TURN (or invalid during need_turn) cost
        cast_turn=0.3,         # heading change for TURN_L/R (radians)
    ):
        super().__init__()
        self.render_mode = render_mode

        self.L = float(L)
        self.dt = float(dt)
        self.v = float(v_fixed)
        self.ds = float(sensor_offset)

        self.wind_x = float(wind_x)
        self.wind_y = float(wind_y)
        self._wind_mag = float(np.hypot(self.wind_x, self.wind_y))
        if self._wind_mag > 1e-6:
            self._wind_dir = (self.wind_x / self._wind_mag, self.wind_y / self._wind_mag)
        else:
            self._wind_dir = (1.0, 0.0)

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
        self.turn_penalty = float(turn_penalty)
        self.cast_turn = float(cast_turn)

        # action: 0 RUN, 1 CAST, 2 TURN_L, 3 TURN_R
        self.action_space = spaces.Discrete(4)

        # CAST 완료 직후 1회 TURN 강제
        self.need_turn = False

        # obs per step: [c, mode] (NO phi label)
        self.obs_step_dim = 2
        obs_dim = self.obs_step_dim * self.stack_n
        low = np.zeros((obs_dim,), dtype=np.float32)
        high = np.ones((obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._obs_buf = np.zeros((self.stack_n, self.obs_step_dim), dtype=np.float32)
        self.np_random = np.random.default_rng(seed)

        # state
        self._step = 0
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        # CAST internals: L/R/L/R
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
        self._heatmap_img = None
        self._cbar_img = None

    def _conc(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if self._wind_mag <= 1e-6:
            r2 = x * x + y * y
            c = np.exp(-r2 / (2.0 * self.sigma_c * self.sigma_c))
        else:
            wx, wy = self._wind_dir
            t = x * wx + y * wy      # along-wind
            s = -x * wy + y * wx     # cross-wind

            stretch = 1.0 + min(self._wind_mag, 2.0)
            sigma_s = self.sigma_c
            sigma_t = self.sigma_c * stretch
            sigma_up = self.sigma_c / stretch

            t_pos = np.maximum(0.0, t)
            t_neg = np.maximum(0.0, -t)

            c = np.exp(-(
                (s * s) / (2.0 * sigma_s * sigma_s) +
                (t_pos * t_pos) / (2.0 * sigma_t * sigma_t) +
                (t_neg * t_neg) / (2.0 * sigma_up * sigma_up)
            ))

        c = self.bg_c + (1.0 - self.bg_c) * c
        c = np.clip(c, 0.0, 1.0)
        if np.ndim(c) == 0:
            return float(c)
        return c

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
        self.need_turn = False
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

            # exit cast
            self.in_cast = False
            self.cast_phase = 0
            self._scan_c[:] = 0.0

            # cast 직후 1회 turn 결정 강제
            self.need_turn = True

        return float(c), finished

    def step(self, action):
        self._step += 1
        a = int(action)

        did_scan = False
        did_turn = False
        invalid = False
        moved = False

        # 1) CAST 진행 중이면 action 무시하고 auto-cast
        if self.in_cast:
            mode = 1
            c, _ = self._cast_step()
            did_scan = True

        else:
            # 2) CAST 완료 직후: TURN 1회만 허용
            if self.need_turn:
                if a == 2:
                    self.th = self._norm_angle(self.th + self.cast_turn)
                    self.need_turn = False
                    did_turn = True

                    self._render_scan_idx = 0
                    self._render_mode = 0
                    c = self._sense(0.0)
                    self._push_obs(c, mode=0)
                    mode = 0

                elif a == 3:
                    self.th = self._norm_angle(self.th - self.cast_turn)
                    self.need_turn = False
                    did_turn = True

                    self._render_scan_idx = 0
                    self._render_mode = 0
                    c = self._sense(0.0)
                    self._push_obs(c, mode=0)
                    mode = 0

                else:
                    # RUN/CAST는 무효: obs 유지(캐스트 스택 유지)
                    invalid = True
                    mode = 1
                    c = float(self._obs_buf[-1, 0])

            else:
                # 3) 일반 상태: RUN 또는 CAST 시작만 유효
                if a == 1:
                    self.in_cast = True
                    self.cast_phase = 0
                    self._scan_c[:] = 0.0
                    mode = 1
                    c, _ = self._cast_step()
                    did_scan = True

                elif a == 0:
                    mode = 0
                    self.x += self.v * np.cos(self.th) * self.dt
                    self.y += self.v * np.sin(self.th) * self.dt

                    self._render_scan_idx = 0
                    self._render_mode = 0

                    c = self._sense(0.0)
                    self._push_obs(c, mode=0)
                    moved = True

                else:
                    # TURN은 cast 직후에만 허용
                    invalid = True
                    mode = 0
                    c = float(self._obs_buf[-1, 0])

        oob = (abs(self.x) > self.L) or (abs(self.y) > self.L)
        terminated = bool(oob)
        truncated = bool(self._step >= self.max_steps)

        d = float(np.hypot(self.x, self.y))

        # reward
        r = float(np.exp(-d / self.sigma_r))
        if d < self.r_goal:
            r += self.b_hold
        if did_scan:
            r -= self.scan_penalty
        if did_turn or invalid:
            r -= self.turn_penalty
        if oob:
            r -= self.b_oob

        info = {
            "d": d,
            "c": float(c),
            "mode": int(mode),           # 0 RUN, 1 CAST/decision
            "in_cast": int(self.in_cast),
            "need_turn": int(self.need_turn),
            "cast_phase": int(self.cast_phase),
            "moved": int(moved),
            "did_scan": int(did_scan),
            "did_turn": int(did_turn),
            "invalid": int(invalid),
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
        if self._heatmap_img is None or self._heatmap_img.size != (W, H):
            xs = np.linspace(-self.L, self.L, W, dtype=np.float32)
            ys = np.linspace(self.L, -self.L, H, dtype=np.float32)
            X, Y = np.meshgrid(xs, ys)
            c = self._conc(X, Y)
            try:
                from matplotlib import cm
                rgb = (cm.inferno(c)[..., :3] * 255.0).astype(np.uint8)
            except Exception:
                v = (c * 255.0).astype(np.uint8)
                rgb = np.stack([v, v, v], axis=-1)
            self._heatmap_img = Image.fromarray(rgb, mode="RGB")
            bar_w = max(10, int(W * 0.04))
            bar_h = max(80, int(H * 0.5))
            grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]
            try:
                from matplotlib import cm
                bar_rgb = (cm.inferno(grad)[..., :3] * 255.0).astype(np.uint8)
            except Exception:
                v = (grad * 255.0).astype(np.uint8)
                bar_rgb = np.repeat(v, 3, axis=1)[:, None, :]
            bar_rgb = np.repeat(bar_rgb, bar_w, axis=1)
            self._cbar_img = Image.fromarray(bar_rgb, mode="RGB")

        img = self._heatmap_img.copy()
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            px = int((x + self.L) / (2 * self.L) * (W - 1))
            py = int((self.L - y) / (2 * self.L) * (H - 1))
            return px, py

        cx, cy = to_px(0.0, 0.0)
        rg = int(self.r_goal / (2 * self.L) * (W - 1))
        draw.ellipse((cx - rg, cy - rg, cx + rg, cy + rg), outline=(180, 180, 180), width=2)
        rs = 4
        draw.ellipse((cx - rs, cy - rs, cx + rs, cy + rs), fill=(0, 0, 0))

        ax, ay = to_px(self.x, self.y)
        size = 10
        th = self.th
        p0 = (ax + size * np.cos(th), ay - size * np.sin(th))
        p1 = (ax + size * np.cos(th + 2.5), ay - size * np.sin(th + 2.5))
        p2 = (ax + size * np.cos(th - 2.5), ay - size * np.sin(th - 2.5))
        tri = [tuple(map(int, p0)), tuple(map(int, p1)), tuple(map(int, p2))]
        draw.polygon(tri, fill=(50, 100, 220))

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

        if self._cbar_img is not None:
            pad = 6
            bx = W - self._cbar_img.size[0] - pad
            by = pad
            img.paste(self._cbar_img, (bx, by))
            try:
                draw.text((bx - 2, by - 2), "1.0", fill=(0, 0, 0))
                draw.text((bx - 2, by + self._cbar_img.size[1] - 10), "0.0", fill=(0, 0, 0))
            except Exception:
                pass

        return np.array(img, dtype=np.uint8)

    def close(self):
        pass