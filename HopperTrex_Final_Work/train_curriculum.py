"""Stage training with observation-compatible transfer and measured pass criteria."""
import argparse
import json
from pathlib import Path
import time
import multiprocessing
import hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback,BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from cpu_balance import ROOT,POSE
from cpu_curriculum import CurriculumEnv
from cpu_terrain import TerrainEnv
from cpu_navigation import NavigationEnv

def make_environment(stage, stair_height=.02, descending=False, controller=None, planner_guard=False, leg_smoothing=.1, leg_speed=.75, wheel_gain=200.,stance_pose=None,physics_timestep=.005,push_training=False):
    if multiprocessing.current_process().name != 'MainProcess':
        torch.set_num_threads(1)
    if stage=='avoid': return NavigationEnv(controller,planner_guard)
    return TerrainEnv(stage,stair_height,descending,wheel_gain=wheel_gain,leg_smoothing=leg_smoothing,leg_speed=leg_speed,stance_pose=stance_pose,push_training=push_training) if stage=='stairs' else CurriculumEnv(stage,wheel_gain=wheel_gain,physics_timestep=physics_timestep)

def warm_start(model, checkpoint):
    old = PPO.load(checkpoint, device='cpu')
    source = old.policy.state_dict()
    destination = model.policy.state_dict()
    for key, value in source.items():
        if key not in destination:
            continue
        if value.shape == destination[key].shape:
            destination[key] = value.clone()
        elif key in ('mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight') and value.shape[0]==destination[key].shape[0] and value.shape[1] <= destination[key].shape[1]:
            destination[key].zero_()
            if value.shape[1]==39 and destination[key].shape[1]==46:
                destination[key][:,:28]=value[:,:28]
                destination[key][:,32:34]=value[:,28:30]
            elif value.shape[1]==39 and destination[key].shape[1]==47:
                destination[key][:,:30]=value[:,:30]
                destination[key][:,30+2*torch.arange(9)]=value[:,30:39]
            else:
                destination[key][:, :value.shape[1]] = value
        elif key in ('action_net.weight','action_net.bias','log_std') and value.shape[0]==2 and destination[key].shape[0]==6:
            destination[key].zero_()
            destination[key][4:] = value
            if key=='log_std': destination[key][:4] = np.log(.1)
    model.policy.load_state_dict(destination)

def evaluate(model, stage, episodes=20, seconds=60, perturb=5, stair_height=.02, descending=False, controller=None, planner_guard=False, seed_start=4000,leg_smoothing=None,leg_speed=None,wheel_gain=None,stance_pose=None,physics_timestep=None):
    leg_smoothing=leg_smoothing if leg_smoothing is not None else getattr(model,'stair_leg_smoothing',.1)
    leg_speed=leg_speed if leg_speed is not None else getattr(model,'stair_leg_speed',.75)
    wheel_gain=wheel_gain if wheel_gain is not None else getattr(model,'stand_wheel_gain',getattr(model,'stair_wheel_gain',200.))
    stance_pose=stance_pose if stance_pose is not None else getattr(model,'stair_stance_pose',POSE)
    physics_timestep=physics_timestep if physics_timestep is not None else getattr(model,'stand_physics_timestep',.005)
    env = make_environment(stage,stair_height,descending,controller,planner_guard,leg_smoothing,leg_speed,wheel_gain,stance_pose,physics_timestep)
    dt=getattr(env,'control_dt',.02)
    results=[]
    for seed in range(seed_start,seed_start+episodes):
        obs,_=env.reset(seed=seed,options={'perturb_rad':np.deg2rad(perturb)})
        errors=[]; speeds=[]; vel_errors=[]; yaw_errors=[]; positions=[]; angular=[]
        for step in range(round(seconds/dt)):
            obs,_,done,timeout,info=env.step(model.predict(obs,deterministic=True)[0])
            errors.append(info['pitch_error']); speeds.append(info['speed_m_s'])
            physical=env.env if hasattr(env,'env') else env
            positions.append(physical.data.qpos[:2].copy());angular.append(physical.data.qvel[3:6].copy())
            vel_errors.append(info['velocity_error']); yaw_errors.append(info['yaw_rate_error'])
            if done or timeout: break
        results.append({'seed':seed,'survival_s':info.get('simulated_seconds',(step+1)*dt),'max_drift_m':info['max_drift_m'],
            'final_drift_m':info['drift_m'],'pitch_error_deg':float(np.rad2deg(np.mean(np.abs(errors)))),
            'tail_pitch_std_deg':float(np.rad2deg(np.std(errors[-round(10/dt):]))),
            'tail_angular_rms_rad_s':float(np.sqrt(np.mean(np.square(angular[-round(10/dt):])))),
            'tail_position_range_m':float(np.linalg.norm(np.ptp(positions[-round(10/dt):],axis=0))),
            'mean_speed_m_s':float(np.mean(speeds)), 'tail_speed_m_s':float(np.mean(speeds[-round(10/dt):])),
            'velocity_error_m_s':float(np.mean(vel_errors)), 'yaw_rate_error_rad_s':float(np.mean(yaw_errors)),
            'success':bool(info.get('success',False)), 'collision':bool(info.get('collision',False)),
            'distance_to_goal_m':float(info.get('distance_to_goal_m',0.))})
        if stage=='avoid':results[-1]['planner_guard_fraction']=info['planner_guard_steps']/info['navigation_steps']
        if stage=='stairs':results[-1]['stair_levels_crossed']=info['stair_levels_crossed']
    if stage in ('avoid','stairs'):
        passed=sum(r['success'] and not r['collision'] for r in results)
    else:
        passed=sum(r['survival_s'] >= seconds and (
            (r['max_drift_m']<=.15 and r['tail_speed_m_s']<=.05) if stage=='stand' else
            (r['velocity_error_m_s']<=.12 and r['yaw_rate_error_rad_s']<=.2)) for r in results)
    return {'stage':stage,'episodes':episodes,'seconds':seconds,'perturb_deg':perturb,'passed':passed,
        'stair_height_m':stair_height if stage=='stairs' else None,'descending':descending if stage=='stairs' else None,
        'stand_wheel_gain':wheel_gain if stage=='stand' else None,
        'physics_timestep_s':float(env.model.opt.timestep),'control_interval_s':dt,
        'stair_actuators':{'leg_smoothing':leg_smoothing,'leg_speed_rad_s':leg_speed,'wheel_velocity_gain':wheel_gain,'stance_deg':np.rad2deg(stance_pose).tolist()} if stage=='stairs' else None,
        'criteria': 'survive; max drift <= 0.15m; final 10s mean speed <= 0.05m/s' if stage=='stand' else 'cross all three risers inside stair corridor; finish x>=2.15m and |y|<0.6m without collision or fall' if stage=='stairs' else 'reach goal without collision or fall' if stage=='avoid' else 'survive; mean linear tracking error <= 0.12m/s; mean yaw tracking error <= 0.2rad/s',
        'mean':{k:float(np.mean([r[k] for r in results])) for k in results[0]},'per_episode':results}

def drive_probes(model):
    env=CurriculumEnv('drive'); results=[]
    for i, command in enumerate([[.2,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.2,0,.3],[0,0,0]]):
        obs,_=env.reset(seed=6000+i,options={'command':command,'perturb_rad':np.deg2rad(5)})
        linear=[]; angular=[]
        for step in range(750):
            obs,_,done,timeout,info=env.step(model.predict(obs,deterministic=True)[0])
            linear.append(info['velocity_error']);angular.append(info['yaw_rate_error'])
            if done or timeout: break
        results.append({'command':command,'survival_s':(step+1)*.02,
            'tail_linear_error_m_s':float(np.mean(linear[-250:])),
            'tail_yaw_error_rad_s':float(np.mean(angular[-250:]))})
    return results

class StairAcceptanceCallback(BaseCallback):
    """Keep the best deterministic policy and stop at 20/20 acceptance."""
    def __init__(self,path,height,descending,interval=100000):
        super().__init__()
        self.path=path.with_name(path.stem+'_best.zip')
        self.height=height;self.descending=descending;self.interval=interval
        self.best_score=(-1,-float('inf'));self.last_check=0;self.accepted=False

    def assess(self):
        report=evaluate(self.model,'stairs',stair_height=self.height,descending=self.descending)
        score=(report['passed'],-report['mean']['pitch_error_deg'])
        if score>self.best_score:
            self.best_score=score;self.model.save(self.path)
            report.update(model_file=self.path.name,model_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                provenance='best deterministic acceptance checkpoint during CPU training')
            self.path.with_name(self.path.stem+'_evaluation.json').write_text(json.dumps(report,indent=2))
        print(f'Stair acceptance: {report["passed"]}/{report["episodes"]}; best {self.best_score[0]}/20',flush=True)
        self.accepted=report['passed']==report['episodes']
        return self.accepted

    def _on_training_start(self):
        self.last_check=self.num_timesteps
        self.assess()

    def _on_step(self):
        if self.accepted:return False
        if self.num_timesteps-self.last_check>=self.interval:
            self.last_check=self.num_timesteps
            return not self.assess()
        return True

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['stand','drive','avoid','stairs'],default='stand')
    parser.add_argument('--mode',choices=['train','eval','check','view'],default='train')
    parser.add_argument('--steps',type=int,default=500000)
    parser.add_argument('--envs',type=int,default=4)
    parser.add_argument('--parallel',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--warm-start',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--exploration-std',type=float)
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--gae-lambda',type=float)
    parser.add_argument('--wheel-exploration-std',type=float)
    parser.add_argument('--acceptance-check',action='store_true')
    parser.add_argument('--leg-smoothing',type=float)
    parser.add_argument('--leg-speed',type=float)
    parser.add_argument('--wheel-gain',type=float)
    parser.add_argument('--physics-timestep',type=float)
    parser.add_argument('--stance-deg',type=float,nargs=4)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--eval-seed',type=int,default=4000)
    parser.add_argument('--evaluation-report',type=Path)
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--stair-height',type=float,default=.02)
    parser.add_argument('--descending',action='store_true')
    parser.add_argument('--push-training',action='store_true')
    parser.add_argument('--controller',type=Path,default=ROOT/'policy/cpu_drive.zip')
    parser.add_argument('--planner-guard',action=argparse.BooleanOptionalAction,default=None)
    args=parser.parse_args()
    guarded=args.planner_guard if args.planner_guard is not None else args.mode in ('view','eval')
    if args.steps < 1 or args.envs < 1 or args.episodes < 1 or not .02 <= args.seconds <= 60:
        parser.error('positive steps/envs/episodes and seconds in [.02,60] required')
    if not .005 <= args.stair_height <= .1:
        parser.error('--stair-height must be in [.005,.1] metres')
    if args.resume and args.warm_start:
        parser.error('--resume and --warm-start cannot be combined')
    torch.set_num_threads(2)
    label=f'{args.stair_height*100:g}'.replace('.','p')
    default_name=f'cpu_stairs_{"down" if args.descending else "up"}_{label}cm.zip' if args.stage=='stairs' else f'cpu_{args.stage}.zip'
    path=args.model or ROOT/'policy'/default_name
    cfgpath=args.warm_start if args.mode=='train' and args.warm_start else path if args.mode in ('view','eval') or args.resume else None
    metadata=PPO.load(cfgpath,device='cpu') if cfgpath else None
    leg_smoothing=args.leg_smoothing if args.leg_smoothing is not None else getattr(metadata,'stair_leg_smoothing',.1)
    leg_speed=args.leg_speed if args.leg_speed is not None else getattr(metadata,'stair_leg_speed',.75)
    wheel_gain=args.wheel_gain if args.wheel_gain is not None else getattr(metadata,'stand_wheel_gain',getattr(metadata,'stair_wheel_gain',200.))
    physics_timestep=args.physics_timestep if args.physics_timestep is not None else getattr(metadata,'stand_physics_timestep',.005)
    if args.physics_timestep is not None and args.stage!='stand':parser.error('--physics-timestep currently supports stand only')
    stance_pose=np.deg2rad(args.stance_deg) if args.stance_deg is not None else np.asarray(getattr(metadata,'stair_stance_pose',POSE),dtype=float)
    if not 0<leg_smoothing<=1 or not 0<leg_speed<=10 or not 0<wheel_gain<=500:
        parser.error('leg smoothing in (0,1], leg speed in (0,10], wheel gain in (0,500] required')
    if args.mode=='check':
        check_env(make_environment(args.stage,args.stair_height,args.descending,args.controller,guarded,leg_smoothing,leg_speed,wheel_gain,stance_pose,physics_timestep)); print('Environment check passed'); return
    if args.mode=='train':
        env=make_vec_env(make_environment,n_envs=args.envs,seed=args.seed,
            vec_env_cls=SubprocVecEnv if args.parallel and args.envs>1 else None,
            env_kwargs={'stage':args.stage,'stair_height':args.stair_height,'descending':args.descending,'controller':args.controller,'planner_guard':guarded,'leg_smoothing':leg_smoothing,'leg_speed':leg_speed,'wheel_gain':wheel_gain,'stance_pose':stance_pose,'physics_timestep':physics_timestep,'push_training':args.push_training})
        model=PPO.load(path,env=env,device='cpu') if args.resume else PPO('MlpPolicy',env,device='cpu',seed=args.seed,
            n_steps=512,batch_size=128,n_epochs=5,learning_rate=3e-4,gamma=.999 if args.stage=='stairs' else .995,gae_lambda=.99 if args.stage=='stairs' else .95,
            ent_coef=.001,target_kl=.02,policy_kwargs={'net_arch':[64,64]},verbose=1)
        if args.stage=='avoid' and not args.resume:
            with torch.no_grad(): model.policy.log_std.fill_(np.log(.5))
            model.hierarchical_navigation=True
        if args.warm_start:
            if args.stage=='avoid' and not getattr(PPO.load(args.warm_start,device='cpu'),'hierarchical_navigation',False):
                parser.error('avoid warm-start requires a hierarchical navigation policy; specify driver via --controller')
            warm_start(model,args.warm_start)
            same_stair_space=args.stage=='stairs' and PPO.load(args.warm_start,device='cpu').observation_space.shape==(46,)
            if args.stage in ('avoid','stairs') and not same_stair_space:
                with torch.no_grad():
                    model.policy.value_net.weight.zero_()
                    model.policy.value_net.bias.zero_()
        if args.exploration_std is not None:
            if not 0.01 <= args.exploration_std <= 1.:
                parser.error('--exploration-std must be in [.01,1]')
            with torch.no_grad(): model.policy.log_std.fill_(np.log(args.exploration_std))
        if args.learning_rate is not None:
            if not 0<args.learning_rate<=.01:parser.error('--learning-rate must be in (0,.01]')
            model.learning_rate=args.learning_rate
            model.lr_schedule=lambda _:float(args.learning_rate)
        if args.gae_lambda is not None:
            if not 0<args.gae_lambda<=1:parser.error('--gae-lambda must be in (0,1]')
            model.gae_lambda=args.gae_lambda;model.rollout_buffer.gae_lambda=args.gae_lambda
        if args.wheel_exploration_std is not None:
            if args.stage!='stairs' or not .005<=args.wheel_exploration_std<=1:
                parser.error('--wheel-exploration-std requires stairs and a value in [.005,1]')
            with torch.no_grad():model.policy.log_std[4:].fill_(np.log(args.wheel_exploration_std))
        if args.stage=='stairs':
            model.stair_leg_smoothing=leg_smoothing;model.stair_leg_speed=leg_speed;model.stair_wheel_gain=wheel_gain
            model.stair_stance_pose=stance_pose.tolist()
        if args.stage=='stand':model.stand_wheel_gain=wheel_gain;model.stand_physics_timestep=physics_timestep
        path.parent.mkdir(parents=True,exist_ok=True)
        callbacks=[CheckpointCallback(save_freq=max(1,100000//args.envs),save_path=str(path.parent/f'{path.stem}_checkpoints'))]
        acceptance=StairAcceptanceCallback(path,args.stair_height,args.descending) if args.acceptance_check and args.stage=='stairs' else None
        if acceptance:callbacks.append(acceptance)
        model.learn(args.steps,reset_num_timesteps=not args.resume,callback=callbacks)
        if acceptance:model=PPO.load(acceptance.path,device='cpu')
        model.save(path)
        env.close()
    else:
        model=PPO.load(path,device='cpu')
    if args.mode=='view':
        import mujoco.viewer
        env=make_environment(args.stage,args.stair_height,args.descending,args.controller,guarded,leg_smoothing,leg_speed,wheel_gain,stance_pose,physics_timestep); obs,_=env.reset(seed=4000)
        env.model.vis.headlight.ambient[:]=[.5,.5,.5]
        env.model.vis.headlight.diffuse[:]=[.7,.7,.7]
        env.model.geom_rgba[env.model.geom('floor').id]=[.85,.87,.9,1.]
        playback={'paused':False,'reset':False}
        def on_key(key):
            if key==75:playback['paused']=not playback['paused']
            elif key==82:playback['reset']=True
        print('Viewer: K pause/resume | R reset episode | mouse drag/scroll to adjust camera')
        status_info={}
        with mujoco.viewer.launch_passive(env.model,env.data,key_callback=on_key) as viewer:
            viewer.opt.geomgroup[3]=0
            viewer.opt.geomgroup[0]=1;viewer.opt.geomgroup[4]=1
            viewer.cam.azimuth=135;viewer.cam.elevation=-25
            if args.stage=='stairs':viewer.cam.lookat[:]=[1.2,0,.2];viewer.cam.distance=4.5;viewer.cam.azimuth=45;viewer.cam.elevation=-20
            elif args.stage=='avoid':viewer.cam.lookat[:]=[1.5,0,.2];viewer.cam.distance=5.5
            else:viewer.cam.lookat[:]=[0,0,.25];viewer.cam.distance=1.8
            for geom in range(env.model.ngeom):
                if env.model.geom_group[geom]==4 and args.stage=='stairs':
                    name=env.model.geom(geom).name
                    env.model.geom_rgba[geom]={'step_0':[.95,.65,.25,1],'step_1':[.8,.45,.15,1]}.get(name,[.6,.3,.1,1])
            while viewer.is_running():
                start=time.perf_counter()
                if playback['reset']:
                    obs,_=env.reset(seed=4000);playback['reset']=False
                if not playback['paused']:
                    obs,_,done,timeout,status_info=env.step(model.predict(obs,deterministic=True)[0])
                    if done or timeout: obs,_=env.reset()
                scene_name=args.stage.upper()
                if args.stage=='stairs':scene_name+=f' {"DOWN" if args.descending else "UP"} {args.stair_height*100:g}cm'
                labels='Stage\nSimulation\nState\nPitch error\nSpeed\nKeys'
                values=f'{scene_name}\n{env.data.time:.1f}s\n{"PAUSED" if playback["paused"] else "RUNNING"}\n{np.rad2deg(status_info.get("pitch_error",0)):.2f}deg\n{status_info.get("speed_m_s",0):.3f}m/s\nK:pause  R:reset'
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,labels,values))
                viewer.sync(); time.sleep(max(0,getattr(env,'control_dt',.02)-(time.perf_counter()-start)))
        return
    report=evaluate(model,args.stage,args.episodes,args.seconds,stair_height=args.stair_height,descending=args.descending,controller=args.controller,planner_guard=guarded,seed_start=args.eval_seed,leg_smoothing=leg_smoothing,leg_speed=leg_speed,wheel_gain=wheel_gain,stance_pose=stance_pose,physics_timestep=physics_timestep)
    if args.stage=='drive': report['fixed_command_probes']=drive_probes(model)
    report['model_file']=path.name
    report['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    report['wheel_control']='velocity residual + differential drive feedforward' if args.stage!='stand' else 'velocity target; zero feedforward'
    if args.stage=='avoid':
        report['planner_guard']=guarded
        report['navigation_control']='10Hz speed/yaw commands; frozen 50Hz RL driver'
        report['controller_file']=args.controller.name
        report['controller_sha256']=hashlib.sha256(args.controller.read_bytes()).hexdigest()
    report_path=args.evaluation_report or path.parent/f'{path.stem}_evaluation.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_episode'},indent=2))

if __name__=='__main__': main()
