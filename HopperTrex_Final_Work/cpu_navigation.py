"""Hierarchical navigation: a learned driver at 50Hz, navigation at 10Hz."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from cpu_balance import ROOT
from cpu_curriculum import CurriculumEnv
from cpu_terrain import TerrainEnv
from cpu_route_planner import plan,follow,clearance

class NavigationEnv(gym.Env):
    metadata={'render_modes':[]}
    control_dt=.1
    def __init__(self, controller=None, planner_guard=False):
        self.env=TerrainEnv('avoid')
        self.controller=PPO.load(controller or ROOT/'policy/cpu_drive.zip',device='cpu')
        self.model=self.env.model; self.data=self.env.data
        self.action_space=spaces.Box(-1.,1.,(2,),np.float32)
        self.observation_space=self.env.observation_space
        self.last_command=np.zeros(3)
        self.planner_guard=planner_guard

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs,info=self.env.reset(seed=seed,options=options)
        self.last_command=self.env.command.copy()
        self.path=None;self.path_index=0;self.guard_steps=0;self.meta_steps=0
        self.guarded=False;self.best_distance=self.env.previous_distance;self.last_progress_step=0
        return obs,info

    def step(self, action):
        action=np.clip(np.asarray(action),-1.,1.)
        self.meta_steps+=1
        if self.planner_guard and not self.guarded:
            xy=self.data.qpos[:2].copy();rot=self.data.xmat[1].reshape(3,3)
            heading=np.arctan2(rot[1,0],rot[0,0]);v=.2*(action[0]+1.);w=.5*action[1]
            # Reserve turning space before the wide wheelbase reaches a box.
            hazard=clearance(self.model,xy)<1.05
            for _ in range(20):
                heading+=w*.1;xy+=v*.1*np.array([np.cos(heading),np.sin(heading)])
                hazard=hazard or clearance(self.model,xy)<.5
            if self.env.previous_distance<self.best_distance-.05:
                self.best_distance=self.env.previous_distance;self.last_progress_step=self.meta_steps
            if hazard or self.meta_steps-self.last_progress_step>30:
                self.path=plan(self.model,self.data,self.env.destination)
                self.guarded=self.path is not None
        if self.guarded:
            action,self.path_index=follow(self.data,self.path,self.path_index)
            self.guard_steps+=1
        command=np.array([.2*(action[0]+1.),0.,.5*action[1]])
        if np.linalg.norm(command)<1e-6 and np.linalg.norm(self.last_command)>1e-6:
            self.env.goal_xy=self.data.qpos[:2].copy()
            rot=self.data.xmat[1].reshape(3,3)
            self.env.goal_yaw=np.arctan2(rot[1,0],rot[0,0])
        total_reward=0.
        for _ in range(5):
            self.env.command=command.copy()
            low_obs=CurriculumEnv._obs(self.env)
            wheels=self.controller.predict(low_obs,deterministic=True)[0]
            obs,reward,done,timeout,info=self.env.step(wheels)
            total_reward+=reward
            if done or timeout: break
        self.last_command=command.copy()
        self.env.command=command.copy()
        info['simulated_seconds']=self.env.steps*.02
        info['planner_guard_steps']=self.guard_steps
        info['navigation_steps']=self.meta_steps
        return self.env._obs(),float(total_reward),done,timeout,info

    def close(self): self.env.close()
