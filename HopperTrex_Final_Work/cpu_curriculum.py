"""Position holding and commanded locomotion for the native CPU simulator."""
import numpy as np
from gymnasium import spaces
from cpu_balance import BalanceEnv, POSE, mujoco

class CurriculumEnv(BalanceEnv):
    def __init__(self, stage='stand',wheel_gain=None,physics_timestep=.005):
        self.physics_timestep=physics_timestep
        if wheel_gain is not None:self.wheel_gain=float(wheel_gain)
        self.stage = stage
        self.command = np.zeros(3)
        self.goal_xy = np.zeros(2)
        self.goal_yaw = 0.
        self.fixed_command = None
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (28,), np.float32)

    def _obs(self):
        base = super()._obs()
        rot = self.data.xmat[1].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        pos_error = rot[:2, :2].T @ (self.goal_xy - self.data.qpos[:2])
        if self.stage != 'stand' and np.linalg.norm(self.command) > 1e-6:
            pos_error = np.zeros(2)
            yaw_error = 0.
        else:
            yaw_error = self.goal_yaw - yaw
        return np.concatenate([base, np.clip(pos_error, -2., 2.),
            [np.sin(yaw_error), np.cos(yaw_error)-1.], self.command]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.command = np.zeros(3)
        self.goal_xy = np.zeros(2)
        self.goal_yaw = 0.
        _, info = super().reset(seed=seed, options=options)
        self.goal_xy = self.data.qpos[:2].copy()
        self.max_drift = 0.
        self.max_speed = 0.
        if self.stage == 'drive':
            requested=(options or {}).get('command')
            self.fixed_command=None if requested is None else np.asarray(requested,dtype=float)
            if self.fixed_command is None:
                self._new_command()
            else:
                if self.fixed_command.shape!=(3,) or not np.isfinite(self.fixed_command).all():
                    raise ValueError('command must be three finite values')
                self.command=self.fixed_command.copy()
        return self._obs(), info

    def _new_command(self):
        self.command = np.array([self.np_random.uniform(-.3, .4), 0., self.np_random.uniform(-.5, .5)])
        if self.np_random.random() < .3:
            self.command[:] = 0.
        self.goal_xy = self.data.qpos[:2].copy()
        rot = self.data.xmat[1].reshape(3, 3)
        self.goal_yaw = np.arctan2(rot[1, 0], rot[0, 0])

    def step(self, action):
        previous = self.last_action.copy()
        # Differential-drive feedforward; the policy learns balance corrections.
        vx, _, yaw_rate = self.command
        self.wheel_feedforward = np.array([-(vx-.2384*yaw_rate)/.1,(vx+.2384*yaw_rate)/.1]) if self.stage!='stand' else np.zeros(2)
        _, _, done, timeout, info = super().step(action)
        rot = self.data.xmat[1].reshape(3, 3)
        velocity = rot.T @ self.data.qvel[:3]
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        yaw_error = np.arctan2(np.sin(yaw-self.goal_yaw),np.cos(yaw-self.goal_yaw))
        position_error = np.linalg.norm(self.data.qpos[:2]-self.goal_xy)
        upright = np.exp(-((info['pitch_error']/.15)**2+(info['roll']/.15)**2))
        reward = 1. + 2.*upright
        stationary = self.stage == 'stand' or np.linalg.norm(self.command) < 1e-6
        if stationary:
            reward += np.exp(-(position_error/.15)**2)
            reward -= 8.*min(position_error**2, 4.) + np.sum(velocity[:2]**2) + .5*yaw_error**2 + .5*self.data.qvel[5]**2
            if self.stage=='stand':
                reward-=42.*min(position_error**2,4.)+19.*np.sum(velocity[:2]**2)
                reward-=4.*yaw_error**2+11.*self.data.qvel[4]**2+25.5*self.data.qvel[5]**2
                reward-=80.*np.sum(velocity[:2]**2)
                reward-=.15*np.sum((np.asarray(action)-previous)**2)
            # End escapes early so the value function learns useful local recovery.
            done = done or position_error > .5
        else:
            reward += 2.*np.exp(-np.sum((velocity[:2]-self.command[:2])**2)/.04)
            reward += 3.*np.exp(-(self.data.qvel[5]-self.command[2])**2/.09)
        reward -= .03*self.data.qvel[4]**2 + .01*np.sum((np.asarray(action)-previous)**2)
        if done:
            reward -= 5.
        self.max_drift = max(self.max_drift, info['drift_m'])
        self.max_speed = max(self.max_speed,float(np.linalg.norm(velocity[:2])))
        info.update(position_error_m=float(position_error), speed_m_s=float(np.linalg.norm(velocity[:2])),
            max_drift_m=float(self.max_drift), yaw_error_deg=float(np.rad2deg(yaw_error)),
            velocity_error=float(np.linalg.norm(velocity[:2]-self.command[:2])), yaw_rate_error=float(abs(self.data.qvel[5]-self.command[2])))
        if self.stage == 'drive' and self.fixed_command is None and self.steps % 400 == 0:
            self._new_command()
        return CurriculumEnv._obs(self), float(reward), bool(done), bool(timeout), info
