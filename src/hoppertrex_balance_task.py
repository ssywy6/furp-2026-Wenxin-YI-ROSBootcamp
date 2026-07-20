"""HopperTrex two-wheel balance task for MjLab."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from assets.HopperTrex_CFG import (
    HIP_INITIAL_ANGLE,
    INIT_JOINT_POS,
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WHEEL_VELOCITY_ACTION_SCALE,
    get_hoppertrex_robot_cfg,
)

LEG_INIT_JOINT_POS = {
    name: INIT_JOINT_POS[name]
    for name in LEG_JOINT_NAMES
}


# ============================================================
# 辅助奖励/惩罚函数
# ============================================================
def lin_vel_z_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.square(robot.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=1)

def upright(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    Reward keeping the body vertical.
    1.0 = perfectly upright
    0.0 = horizontal
    """

    robot = env.scene["robot"]

    quat = robot.data.root_link_quat_w

    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]

    # world z direction expressed by quaternion
    up_z = 1.0 - 2.0 * (qx*qx + qy*qy)

    return torch.pow(torch.clamp(up_z, min=0.0), 4)




def knee_ground_penalty(env):

    robot = env.scene["robot"]

    left = robot.site_names.index("left_knee_tip")
    right = robot.site_names.index("right_knee_tip")

    tip_z = robot.data.site_pos_w[:, [left, right], 2]

    violation = torch.relu(0.03 - tip_z)

    return torch.sum(violation**2, dim=1)


def knee_touch_ground(env):

    robot = env.scene["robot"]

    left = robot.site_names.index("left_knee_tip")
    right = robot.site_names.index("right_knee_tip")

    z = robot.data.site_pos_w[:, [left, right], 2]

    return torch.any(z < 0.05, dim=1)

def yaw_vel_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    ang_vel = robot.data.root_link_ang_vel_b
    yaw_vel = ang_vel[:, 2]
    return torch.square(yaw_vel)


def pitch_angle_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """计算基座的俯仰角（Pitch），并返回其平方作为惩罚"""
    robot = env.scene["robot"]
    quat = robot.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    pitch = torch.asin(torch.clamp(-2.0 * (qx * qz - qw * qy), -1.0, 1.0))
    return torch.square(pitch)


def base_lin_vel_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """惩罚水平速度（平方和）"""
    robot = env.scene["robot"]
    vel_xy = robot.data.root_link_lin_vel_b[:, :2]
    return torch.sum(torch.square(vel_xy), dim=1)


def position_xy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """惩罚水平位置偏移（距离原点）"""
    robot = env.scene["robot"]
    pos_xy = robot.data.root_link_pos_w[:, :2]
    return torch.sum(torch.square(pos_xy), dim=1)


# ============================================================
# 环境创建函数
# ============================================================
def create_hoppertrex_balance_env(play: bool = True):
    import torch
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    env_cfg = make_hoppertrex_balance_env_cfg(play=play)
    device = "cpu"
    print(f"使用设备: {device}")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    return env


def make_hoppertrex_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = get_hoppertrex_robot_cfg()
    num_envs = 16 if play else 64

    # ----------------------------------------------------------
    # 观测定义
    # ----------------------------------------------------------
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
                "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
                "velocity_commands": ObservationTermCfg(
                    func=envs_mdp.generated_commands,
                    params={"command_name": "twist"},
                ),
                "joint_pos": ObservationTermCfg(
                    func=envs_mdp.joint_pos_rel,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                    noise=Unoise(n_min=-0.002, n_max=0.002),
                ),
                "joint_vel": ObservationTermCfg(
                    func=envs_mdp.joint_vel_rel,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "actions": ObservationTermCfg(func=envs_mdp.last_action),
            },
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
                "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
                "velocity_commands": ObservationTermCfg(
                    func=envs_mdp.generated_commands,
                    params={"command_name": "twist"},
                ),
                "joint_pos": ObservationTermCfg(
                    func=envs_mdp.joint_pos_rel,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
                "joint_vel": ObservationTermCfg(
                    func=envs_mdp.joint_vel_rel,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
                "actions": ObservationTermCfg(func=envs_mdp.last_action),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ----------------------------------------------------------
    # 动作定义
    # ----------------------------------------------------------
    actions = {

    "leg_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=LEG_JOINT_NAMES,
        scale=0.15,
        offset=LEG_INIT_JOINT_POS,
        use_default_offset=False,
        preserve_order=True,
    ),


    "wheel_vel": JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=WHEEL_JOINT_NAMES,
        scale=12.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
    ),

    }

    # ----------------------------------------------------------
    # 命令（恒零，因为是平衡任务）
    # ----------------------------------------------------------
    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.2,
            rel_heading_envs=0.0,
            rel_forward_envs=0.0,
            heading_command=False,
            debug_vis=play,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )
    }

    # ----------------------------------------------------------
    # 奖励（简洁、干净、平衡导向）
    # ----------------------------------------------------------
    rewards = {


    "alive":
    RewardTermCfg(
        func=envs_mdp.is_alive,
        weight=2.0,
    ),



    "upright":

    RewardTermCfg(
        func=upright,
        weight=5.0,
        
    ),



    "orientation":

    RewardTermCfg(
        func=envs_mdp.flat_orientation_l2,
        weight=-2.0,
    ),



    "angular_motion":

    RewardTermCfg(
        func=ang_vel_xy_l2,
        weight=-0.2,
    ),



    "wheel_smooth":

    RewardTermCfg(
        func=envs_mdp.action_rate_l2,
        weight=-0.002,
    ),

    "knee_contact":

    RewardTermCfg(
        func=knee_ground_penalty,
        weight=-5.0,
    ),

    "height":

    RewardTermCfg(
        func=lin_vel_z_l2,
        weight=-0.01,
    ),

    "wheel_vel":
    RewardTermCfg(
        func=envs_mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg":
            SceneEntityCfg(
            "robot",
            joint_names=WHEEL_JOINT_NAMES
        )
    },
),

    }
    # ----------------------------------------------------------
    # 终止条件
    # ----------------------------------------------------------
    terminations={

    "time_out":

    TerminationTermCfg(
        func=envs_mdp.time_out,
        time_out=True,
    ),


    "bad_orientation":

    TerminationTermCfg(
        func=envs_mdp.bad_orientation,
        params={
            "limit_angle":0.8
        },
    ),

    

    "knee_touch":
    TerminationTermCfg(
        func=knee_touch_ground,
    ),

    "root_low":

    TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        params={
            "minimum_height":0.15
        },
    ),


    "nan":

    TerminationTermCfg(
        func=envs_mdp.nan_detection,
    ),

    }

    
    # ----------------------------------------------------------
    # 场景配置
    # ----------------------------------------------------------
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=2.5,
            terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=2.5),
            entities={"robot": robot_cfg},
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        sim=SimulationCfg(
            nconmax=50,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                integrator="implicitfast",
                cone="elliptic",
                iterations=50,
                ls_iterations=20,
                impratio=10.0,
            ),
        ),
        decimation=4,
        episode_length_s=10.0 if not play else 1.0e9,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="chassis_base",
            distance=2.0,
            elevation=-12.0,
            azimuth=90.0,
        ),
    )

    return cfg