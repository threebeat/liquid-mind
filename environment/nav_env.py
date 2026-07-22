"""Gymnasium environment: differential-drive robot navigating to a goal in PyBullet.

Observation (22 floats, all roughly in [-1, 1]):
    [0:16]  ray-cast "lidar" hit fractions (1.0 = nothing within range)
    [16]    goal distance / 10 m
    [17:19] sin/cos of goal bearing relative to robot heading
    [19]    forward speed / 1.2 m/s
    [20]    yaw rate / 3 rad/s
    [21]    dt of the last control step, normalized so nominal 30 Hz -> 1.0

Action (2 floats in [-1, 1]): left/right wheel velocity targets.

Layouts:
    "random" - walled arena with random box/cylinder obstacles (training).
    "u_trap" - a U-shaped wall between robot and goal, opening facing the
               robot; a purely reactive goal-seeker drives in and gets stuck
               (used for the Phase 4 hierarchy demo).

The env supports an irregular control interval (irregular_dt=True): each step
simulates a random number of physics substeps, and the resulting dt is both
returned in the observation and available to time-aware policies.
"""
import math
import os

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from pybullet_utils import bullet_client

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

OBS_DIM = 22
ACT_DIM = 2


class NavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, render_mode: str | None = None,
                 irregular_dt: bool = False, layout: str = "random"):
        cfg = config["env"]
        self.cfg = cfg
        self.render_mode = render_mode
        self.irregular_dt = irregular_dt
        self.layout = layout

        self.half = float(cfg["arena_half_size"])
        self.n_rays = int(cfg["n_rays"])
        self.ray_length = float(cfg["ray_length"])
        self.physics_hz = int(cfg["physics_hz"])
        self.nominal_substeps = int(cfg["control_substeps"])
        self.substeps_min = int(cfg["substeps_min"])
        self.substeps_max = int(cfg["substeps_max"])
        self.max_wheel_speed = float(cfg["max_wheel_speed"])
        self.goal_radius = float(cfg["goal_radius"])
        self.max_episode_steps = int(cfg["max_episode_steps"])
        self.rw = cfg["reward"]

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

        mode = p.GUI if render_mode == "human" else p.DIRECT
        self._pb = bullet_client.BulletClient(connection_mode=mode)
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pb.setGravity(0, 0, -9.8)
        self._pb.setTimeStep(1.0 / self.physics_hz)
        self._pb.loadURDF("plane.urdf")
        self.robot = self._pb.loadURDF(os.path.join(ASSETS, "diffbot.urdf"),
                                       [0, 0, 0.095])
        self._static_ids: list[int] = []   # walls + obstacles
        self._goal_marker = None
        self._build_walls()

        self._ray_angles = np.linspace(0, 2 * math.pi, self.n_rays, endpoint=False)
        self._step_count = 0
        self._prev_goal_dist = 0.0
        self._last_dt = self.nominal_substeps / self.physics_hz
        self.goal = np.zeros(2)
        self._obstacle_ids: list[int] = []

    # ------------------------------------------------------------------ build

    def _add_box(self, half_extents, pos, rgba=(0.6, 0.3, 0.2, 1.0)):
        col = self._pb.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = self._pb.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                         rgbaColor=rgba)
        return self._pb.createMultiBody(0, col, vis, pos)

    def _build_walls(self):
        h, t, z = self.half, 0.1, 0.25
        for cx, cy, ex, ey in [(0, h, h + t, t), (0, -h, h + t, t),
                               (h, 0, t, h + t), (-h, 0, t, h + t)]:
            self._static_ids.append(
                self._add_box([ex, ey, z], [cx, cy, z], (0.4, 0.4, 0.45, 1.0)))

    def _clear_episode_bodies(self):
        for uid in self._obstacle_ids:
            self._pb.removeBody(uid)
        self._obstacle_ids = []
        if self._goal_marker is not None:
            self._pb.removeBody(self._goal_marker)
            self._goal_marker = None

    def _spawn_random_layout(self):
        rng = self.np_random
        m = self.half - 0.8
        start = rng.uniform(-m, m, 2)
        while True:
            goal = rng.uniform(-m, m, 2)
            if np.linalg.norm(goal - start) > 3.0:
                break
        for _ in range(int(self.cfg["n_obstacles"])):
            for _try in range(20):
                pos = rng.uniform(-m, m, 2)
                if (np.linalg.norm(pos - start) > 1.0
                        and np.linalg.norm(pos - goal) > 1.0):
                    break
            if rng.random() < 0.5:
                ext = rng.uniform(0.2, 0.5, 2)
                self._obstacle_ids.append(
                    self._add_box([ext[0], ext[1], 0.25], [pos[0], pos[1], 0.25]))
            else:
                r = rng.uniform(0.2, 0.45)
                col = self._pb.createCollisionShape(p.GEOM_CYLINDER, radius=r,
                                                    height=0.5)
                vis = self._pb.createVisualShape(p.GEOM_CYLINDER, radius=r,
                                                 length=0.5,
                                                 rgbaColor=(0.6, 0.3, 0.2, 1.0))
                self._obstacle_ids.append(
                    self._pb.createMultiBody(0, col, vis, [pos[0], pos[1], 0.25]))
        return start, goal

    def _spawn_u_trap_layout(self):
        rng = self.np_random
        start = np.array([-3.0 + rng.uniform(-0.3, 0.3),
                          rng.uniform(-0.5, 0.5)])
        goal = np.array([2.5, 0.0])
        # U-shaped wall: closed side at x=0.5 (between robot and goal),
        # arms extending back toward the robot so it drives into the pocket.
        z = 0.25
        self._obstacle_ids.append(
            self._add_box([0.1, 1.6, z], [0.5, 0.0, z]))          # back wall
        self._obstacle_ids.append(
            self._add_box([1.0, 0.1, z], [-0.5, 1.5, z]))         # top arm
        self._obstacle_ids.append(
            self._add_box([1.0, 0.1, z], [-0.5, -1.5, z]))        # bottom arm
        return start, goal

    # ------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._clear_episode_bodies()
        if self.layout == "u_trap":
            start, goal = self._spawn_u_trap_layout()
        else:
            start, goal = self._spawn_random_layout()
        self.goal = np.asarray(goal, dtype=np.float64)

        yaw = self.np_random.uniform(-math.pi, math.pi)
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        self._pb.resetBasePositionAndOrientation(
            self.robot, [start[0], start[1], 0.095], quat)
        self._pb.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        for j in (0, 1):
            self._pb.resetJointState(self.robot, j, 0.0, 0.0)

        vis = self._pb.createVisualShape(p.GEOM_CYLINDER, radius=self.goal_radius,
                                         length=0.02, rgbaColor=(0.1, 0.9, 0.1, 0.7))
        self._goal_marker = self._pb.createMultiBody(
            0, -1, vis, [self.goal[0], self.goal[1], 0.02])

        self._step_count = 0
        self._last_dt = self.nominal_substeps / self.physics_hz
        obs = self._observe()
        self._prev_goal_dist = self._goal_dist()
        return obs, {"goal_dist": self._prev_goal_dist}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for j, a in zip((0, 1), action):
            self._pb.setJointMotorControl2(
                self.robot, j, p.VELOCITY_CONTROL,
                targetVelocity=float(a) * self.max_wheel_speed, force=5.0)

        substeps = (int(self.np_random.integers(self.substeps_min,
                                                self.substeps_max + 1))
                    if self.irregular_dt else self.nominal_substeps)
        for _ in range(substeps):
            self._pb.stepSimulation()
        self._last_dt = substeps / self.physics_hz
        self._step_count += 1

        obs = self._observe()
        dist = self._goal_dist()

        reward = (self._prev_goal_dist - dist) * float(self.rw["progress_scale"])
        reward -= float(self.rw["step_penalty"])
        collided = self._in_collision()
        if collided:
            reward -= float(self.rw["collision_penalty"])
        self._prev_goal_dist = dist

        terminated = bool(dist < self.goal_radius)
        if terminated:
            reward += float(self.rw["success_bonus"])
        truncated = self._step_count >= self.max_episode_steps

        info = {"goal_dist": dist, "collided": collided, "dt": self._last_dt,
                "is_success": terminated}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._pb.disconnect()

    # ------------------------------------------------------------ sensing

    def _pose(self):
        pos, quat = self._pb.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(quat)[2]
        return np.array(pos[:2]), yaw

    def _goal_dist(self) -> float:
        pos, _ = self._pose()
        return float(np.linalg.norm(self.goal - pos))

    def _observe(self) -> np.ndarray:
        pos, yaw = self._pose()
        angles = self._ray_angles + yaw
        starts, ends = [], []
        r0 = 0.20   # start rays just outside the chassis
        for a in angles:
            d = np.array([math.cos(a), math.sin(a)])
            starts.append([pos[0] + r0 * d[0], pos[1] + r0 * d[1], 0.15])
            ends.append([pos[0] + (r0 + self.ray_length) * d[0],
                         pos[1] + (r0 + self.ray_length) * d[1], 0.15])
        hits = self._pb.rayTestBatch(starts, ends)
        rays = np.array([h[2] for h in hits], dtype=np.float32)

        vec = self.goal - pos
        dist = np.linalg.norm(vec)
        bearing = math.atan2(vec[1], vec[0]) - yaw
        lin, ang = self._pb.getBaseVelocity(self.robot)
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        fwd_speed = float(np.dot(np.array(lin[:2]), heading))

        obs = np.concatenate([
            rays,
            [dist / 10.0, math.sin(bearing), math.cos(bearing),
             fwd_speed / 1.2, ang[2] / 3.0,
             self._last_dt * (self.physics_hz / self.nominal_substeps)],
        ]).astype(np.float32)
        return obs

    def _in_collision(self) -> bool:
        for uid in self._static_ids + self._obstacle_ids:
            if self._pb.getContactPoints(bodyA=self.robot, bodyB=uid):
                return True
        return False
