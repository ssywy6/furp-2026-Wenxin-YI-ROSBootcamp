"""Simulation-only range sensing, obstacle navigation, and low stair tasks."""
import xml.etree.ElementTree as ET
import numpy as np
from gymnasium import spaces
from cpu_curriculum import CurriculumEnv
from cpu_balance import POSE, mujoco

class TerrainEnv(CurriculumEnv):
    def __init__(self, stage='avoid', stair_height=.02, descending=False, wheel_gain=200., leg_smoothing=.1, leg_speed=.75,stance_pose=None, push_training=False):
        self.wheel_gain=float(wheel_gain)
        self.leg_smoothing=float(leg_smoothing);self.leg_speed=float(leg_speed)
        self.stance_pose=np.asarray(POSE if stance_pose is None else stance_pose,dtype=float).copy()
        self.stair_height = float(stair_height)
        self.descending = descending
        self.destination = np.array([2.4 if stage=='stairs' else 3., 0.])
        self.last_legs = np.zeros(4)
        self.leg_target = self.stance_pose.copy()
        self.previous_distance = 0.
        self.push_training = bool(push_training)
        self.push_next = 0
        super().__init__(stage)
        self.action_space = spaces.Box(-1.,1.,(6 if stage=='stairs' else 2,),np.float32)
        self.observation_space = spaces.Box(-np.inf,np.inf,(46 if stage=='stairs' else 47,),np.float32)

    def _configure_world(self, root):
        world=root.find('worldbody')
        if self.stage=='avoid':
            for i,xy in enumerate([(1.3,0.),(2.2,.7)]):
                ET.SubElement(world,'geom',name=f'obstacle_{i}',type='box',pos=f'{xy[0]} {xy[1]} .35',
                    size='.18 .22 .35',group='4',rgba='.8 .2 .2 1',friction='1.6 .02 .002')
        elif self.stage=='stairs':
            if self.descending:
                ET.SubElement(world,'geom',name='start_platform',type='box',pos=f'.3 0 {1.5*self.stair_height}',
                    size=f'.6 1 {1.5*self.stair_height}',group='4',friction='1.6 .02 .002')
            for i in range(3):
                # The final ascending riser belongs to the end platform.
                # Adding another coplanar box here duplicates wheel contacts.
                if not self.descending and i==2:continue
                height=(2-i if self.descending else i+1)*self.stair_height
                if height <= 0: continue
                ET.SubElement(world,'geom',name=f'step_{i}',type='box',pos=f'{1.05+.3*i} 0 {height/2}',
                    size=f'.15 1 {height/2}',group='4',friction='1.6 .02 .002')
            if not self.descending:
                ET.SubElement(world,'geom',name='end_platform',type='box',pos=f'2.3 0 {1.5*self.stair_height}',
                    size=f'.8 1 {1.5*self.stair_height}',group='4',friction='1.6 .02 .002')

    def _ground_height(self,x):
        if self.stage!='stairs': return 0.
        if self.descending:
            return (3 if x<.9 else 2 if x<1.2 else 1 if x<1.5 else 0)*self.stair_height
        return (0 if x<.9 else 1 if x<1.2 else 2 if x<1.5 else 3)*self.stair_height

    def _navigation_command(self):
        rot=self.data.xmat[1].reshape(3,3)
        local=rot[:2,:2].T @ (self.destination-self.data.qpos[:2])
        bearing=np.arctan2(local[1],local[0])
        self.command=np.array([.25*max(0.,np.cos(bearing)),0.,np.clip(1.5*bearing,-.6,.6)])

    def _obs(self):
        base=super()._obs()
        rot=self.data.xmat[1].reshape(3,3)
        local=rot[:2,:2].T @ (self.destination-self.data.qpos[:2])
        if self.stage=='avoid':
            yaw=np.arctan2(rot[1,0],rot[0,0]); readings=[]
            point=self.data.xpos[1]+rot @ np.array([.055,.126,.153])
            mask=np.array([0,0,0,0,1,0],dtype=np.uint8)
            for angle in np.linspace(-np.pi/2,np.pi/2,17):
                distance=mujoco.mj_ray(self.model,self.data,point,np.array([np.cos(yaw+angle),np.sin(yaw+angle),0.]),mask,True,1,None)
                readings.append(1. if distance<0 else min(distance/3.,1.))
            return np.concatenate([base,np.clip(local/3.,-1.,1.),readings]).astype(np.float32)
        samples=[]
        for ahead in [0.,.3,.6,.9]:
            for lateral in [-.25,0.,.25]:
                xy=self.data.qpos[:2]+rot[:2,:2]@np.array([ahead,lateral])
                samples.append((self._ground_height(xy[0])-self._ground_height(self.data.qpos[0]))/.15)
        leg_state=(self.leg_target-self.pose)/np.array([.35,.35,.5,.5])
        return np.concatenate([base,leg_state,np.clip(local/3.,-1.,1.),samples]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.last_legs=np.zeros(4); self.leg_target=self.stance_pose.copy()
        self.crossed_levels=set()
        self.push_next = int(self.np_random.integers(50, 180))
        if self.stage=='stairs':
            options=dict(options or {})
            # Match the initial disturbance used by deterministic acceptance.
            options.setdefault('perturb_rad',np.deg2rad(5.))
        _,info=super().reset(seed=seed,options=options)
        if self.stage=='avoid':
            self.model.geom_pos[self.model.geom('obstacle_0').id,:2]=[self.np_random.uniform(1.15,1.45),self.np_random.uniform(-.25,.25)]
            self.model.geom_pos[self.model.geom('obstacle_1').id,:2]=[self.np_random.uniform(2.1,2.4),self.np_random.choice([-1.,1.])*self.np_random.uniform(.6,.9)]
            self.destination=np.array([3.,self.np_random.uniform(-.2,.2)])
        elif self.stage=='stairs' and self.descending:
            self.data.qpos[2]+=3*self.stair_height
        mujoco.mj_forward(self.model,self.data)
        self._navigation_command()
        self.previous_distance=float(np.linalg.norm(self.destination-self.data.qpos[:2]))
        return self._obs(),info

    def step(self, action):
        action=np.clip(np.asarray(action),-1.,1.)
        old_legs=self.last_legs.copy()
        if self.stage=='stairs':
            wanted=self.pose+action[:4]*np.array([.35,.35,.5,.5])
            wanted=np.clip(wanted,self.model.jnt_range[self.jids[:4],0]+.02,self.model.jnt_range[self.jids[:4],1]-.02)
            self.leg_target+=np.clip(self.leg_smoothing*(wanted-self.leg_target),-.02*self.leg_speed,.02*self.leg_speed)
            self.data.ctrl[:4]=self.leg_target
            self.last_legs=action[:4].copy()
            wheels=action[4:]
        else:
            wheels=action
        # A short, randomized impulse teaches recovery from sudden impacts.
        # Forces are applied for exactly one control interval and are disabled
        # during evaluation unless --push-training is explicitly requested.
        if self.push_training and self.stage == 'stairs':
            self.push_next -= 1
            if self.push_next <= 0:
                body = self.model.body('chassis_base').id
                angle = self.np_random.uniform(-np.pi, np.pi)
                magnitude = self.np_random.uniform(8.0, 18.0)
                self.data.xfrc_applied[body, :3] = magnitude * np.array([np.cos(angle), np.sin(angle), 0.0])
                self.push_next = int(self.np_random.integers(100, 300))
        _,_,done,timeout,info=super().step(wheels)
        self.data.xfrc_applied[:] = 0.0
        collision=False
        for c in self.data.contact:
            terrain1=self.model.geom_group[c.geom1]==4
            terrain2=self.model.geom_group[c.geom2]==4
            if terrain1==terrain2: continue
            robot_geom=c.geom2 if terrain1 else c.geom1
            if self.stage=='avoid' or robot_geom not in self.wheel_geoms:
                collision=True
        distance=float(np.linalg.norm(self.destination-self.data.qpos[:2]))
        previous_levels=len(self.crossed_levels)
        if self.stage=='stairs':
            axle=self.data.geom_xpos[self.wheel_geoms].mean(axis=0)
            for i,edge in enumerate([.9,1.2,1.5]):
                level_height=(2-i if self.descending else i+1)*self.stair_height
                if axle[0]>edge+.04 and abs(axle[1])<.7 and axle[2]>level_height+.075:
                    self.crossed_levels.add(i)
        progress=self.previous_distance-distance
        self.previous_distance=distance
        # Progress has value; simply waiting upright must not outperform arrival.
        tilt_scale=.35 if self.stage=='stairs' else .15
        upright=np.exp(-((info['pitch_error']/tilt_scale)**2+(info['roll']/.15)**2))
        orientation_weight=.5 if self.stage=='stairs' else 2.
        time_cost=.1 if self.stage=='stairs' else .03
        reward=orientation_weight*(upright-1.)+50.*progress-time_cost-.003*np.sum(action**2)
        if self.stage=='stairs':
            reward+=15.*(len(self.crossed_levels)-previous_levels)
            reward-=.02*np.sum(self.last_legs**2)+.02*np.sum((self.last_legs-old_legs)**2)
        reached=(self.data.qpos[0]>=2.15 and abs(self.data.qpos[1])<.6 and len(self.crossed_levels)==3) if self.stage=='stairs' else distance<.25
        success=reached and not done and not collision
        if success: reward+=100.
        if collision or done: reward-=50. if self.stage=='stairs' else 200.
        done=done or collision or success or abs(self.data.qpos[1])>1.5
        self._navigation_command()
        info.update(success=bool(success),is_success=bool(success),collision=bool(collision),distance_to_goal_m=distance,
            progress_m=float(self.data.qpos[0]),stair_height_m=self.stair_height,descending=self.descending,
            stair_levels_crossed=len(self.crossed_levels))
        return self._obs(),float(reward),bool(done),timeout,info
