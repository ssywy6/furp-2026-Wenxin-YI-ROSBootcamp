"""Native MuJoCo / Gymnasium balance environment; no mjlab or CUDA required."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent
LEGS = ['thigh_left_01', 'thigh_right_01', 'knee_left', 'knee_right']
WHEELS = ['wheel_left', 'wheel_right']
# CAD joint zero folds the shins; use a supported pose inside joint limits.
POSE = np.deg2rad([-18., 18., -35., 35.])

class BalanceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        self.pose=np.asarray(getattr(self,'stance_pose',POSE),dtype=float).copy()
        if self.pose.shape!=(4,) or not np.isfinite(self.pose).all():raise ValueError('stance pose must contain four finite joint angles')
        self.wheel_feedforward = np.zeros(2)
        tree = ET.parse(ROOT / 'assets/hoppertrex/robot.xml')
        root = tree.getroot()
        root.find('compiler').set('meshdir', str(ROOT / 'assets/hoppertrex'))
        timestep=float(getattr(self,'physics_timestep',.005))
        if not np.isfinite(timestep) or timestep<=0:raise ValueError('physics timestep must be finite and positive')
        self.physics_substeps=round(.02/timestep)
        if timestep<=0 or self.physics_substeps<1 or abs(self.physics_substeps*timestep-.02)>1e-9:
            raise ValueError('physics timestep must divide the .02 second control interval')
        ET.SubElement(root, 'option', timestep=str(timestep), integrator='implicitfast', iterations='50')
        assets=root.find('asset')
        if assets is None:assets=ET.SubElement(root,'asset')
        ET.SubElement(assets,'texture',name='cpu_ground_grid',type='2d',builtin='checker',rgb1='.22 .28 .32',rgb2='.38 .44 .48',width='512',height='512')
        ET.SubElement(assets,'material',name='cpu_ground_material',texture='cpu_ground_grid',texrepeat='4 4',texuniform='true',reflectance='.05')
        ET.SubElement(root.find('worldbody'), 'geom', name='floor', type='plane', size='0 0 0.1',material='cpu_ground_material', friction='1.6 0.02 0.002')
        self._configure_world(root)
        actuators = ET.SubElement(root, 'actuator')
        for name in LEGS:
            ET.SubElement(actuators, 'position', name=name, joint=name, kp='8000', kv='80', forcelimited='true', forcerange='-97 97')
        for name in WHEELS:
            ET.SubElement(actuators, 'velocity', name=name, joint=name, kv=str(getattr(self,'wheel_gain',200.)), forcelimited='true', forcerange='-5.8 5.8')
        for joint in root.findall('.//joint'):
            joint.set('armature', '0.005' if joint.get('name') in WHEELS else '0.02')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.jids = np.array([self.model.joint(n).id for n in LEGS + WHEELS])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.vids = self.model.jnt_dofadr[self.jids]
        limits=self.model.jnt_range[self.jids[:4]]
        if np.any(self.pose<limits[:,0]+.02) or np.any(self.pose>limits[:,1]-.02):
            raise ValueError('stance pose must remain inside joint limits with a .02rad margin')
        self.wheel_geoms = [self.model.geom(n + '_collision').id for n in WHEELS]
        self.floor_id = self.model.geom('floor').id
        self.action_space = spaces.Box(-1., 1., (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (21,), np.float32)
        self.target_pitch = 0.
        self.reset(seed=0, options={'perturb': False})
        wheel = np.mean(self.data.geom_xpos[self.wheel_geoms], axis=0)
        delta = self.data.subtree_com[1] - wheel
        self.target_pitch = -np.arctan2(delta[0], delta[2])
        self.reset(seed=0)

    def _configure_world(self, root):
        """Optional task geometry hook; each environment owns its model."""

    def _obs(self):
        rot = self.data.xmat[1].reshape(3, 3)
        gravity = rot.T @ np.array([0., 0., -1.])
        return np.concatenate([gravity, rot.T @ self.data.qvel[:3], self.data.qvel[3:6],
            self.data.qpos[self.qids[:4]] - self.pose, self.data.qvel[self.vids] / 12., self.last_action]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        perturb = (options or {}).get('perturb_rad', 0.025)
        pitch = self.target_pitch + (0. if options and not options.get('perturb', True) else self.np_random.uniform(-perturb, perturb))
        self.data.qpos[3:7] = [np.cos(pitch/2), 0., np.sin(pitch/2), 0.]
        self.data.qpos[self.qids[:4]] = self.pose
        self.data.ctrl[:4] = self.pose
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[2] += 0.101 - min(self.data.geom_xpos[g, 2] for g in self.wheel_geoms)
        self.data.qvel[4] = self.np_random.uniform(-0.03, 0.03)
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(2)
        self.steps = 0
        self.start_xy = self.data.qpos[:2].copy()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action), -1., 1.)
        self.data.ctrl[4:] = 12. * action + self.wheel_feedforward
        mujoco.mj_step(self.model,self.data,nstep=self.physics_substeps)
        self.steps += 1
        rot = self.data.xmat[1].reshape(3, 3)
        pitch = np.arctan2(-rot[2, 0], rot[2, 2])
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        error = pitch - self.target_pitch
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        body_contact = any(self.floor_id in (c.geom1, c.geom2) and
            (c.geom2 if c.geom1 == self.floor_id else c.geom1) not in self.wheel_geoms
            for c in self.data.contact)
        fallen = not finite or body_contact or abs(error) > 0.6 or abs(roll) > 0.5 or self.data.qpos[2] < 0.15
        reward = 1. + 2.*np.exp(-((error/0.15)**2 + (roll/0.15)**2))
        reward -= 0.3*np.sum(self.data.qvel[:2]**2) + 0.03*self.data.qvel[4]**2 + 0.5*self.data.qvel[5]**2
        reward -= 0.01*np.sum((action-self.last_action)**2) + 0.002*np.sum(action**2)
        if fallen:
            reward -= 5.
        self.last_action = action.copy()
        return BalanceEnv._obs(self), float(reward), bool(fallen), self.steps >= 3000, {'pitch_error': float(error), 'roll': float(roll), 'drift_m': float(np.linalg.norm(self.data.qpos[:2] - self.start_xy))}
