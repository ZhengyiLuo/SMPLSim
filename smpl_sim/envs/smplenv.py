import numpy as np
import mujoco
import torch
from typing import Any, Sequence
from pathlib import Path
from absl import logging

# SMPL imports
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib import smpl_mujoco_new as smplmj
from smpl_sim.smpllib import smpl_xml_addons as smplxadd
from smpl_sim.envs import controllers as ctrlm
import warnings
from collections import OrderedDict

class SMPLHumanoid:

    GENDER2NUM = {
        "female": 2,
        "male": 1,
        "neutral": 0,
    }

    def __init__(
        self,
        motions: Any,
        max_episode_length: int,
        pid_controlled: bool = True,
        data_dir: str = "",
        physics_steps_per_control_step: int = 6,  # 30Hz
        sim_timestep: float = 1.0 / 180.0,
        ### Random initialization
        initial_position: str = "random",
        probabilities_of_random_methods: Sequence[float] = [
            0.5,
            0.5,
            0,
        ],  # only for hybrid mode, [mocap, random, stand]
        seed: int = 0,
    ) -> None:
        assert initial_position in ["mocap", "hybrid", "random", "stand"]
        self._initial_position_str = initial_position
        self._probabilities_of_random_methods = probabilities_of_random_methods
        self._motions = motions
        self._motion_keys = list(self._motions.keys())
        self._max_episode_length = max_episode_length
        self._sim_timestep = sim_timestep
        self._physics_steps_per_control_step = physics_steps_per_control_step
        self._control_frequency = physics_steps_per_control_step * sim_timestep
        if not np.isclose(self._control_frequency, 1.0 / 30.0):
            warnings.warn(
                f"Control frequency is {1. / self._control_frequency:.2f}Hz. Please be sure that data is generated with the same frequency"
            )

        self._seed = seed
        self._np_random = np.random.RandomState(seed=seed)

        self.robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": False,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True,
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False,
            "box_body": False,
            "master_range": 50,
            "model": "smpl",
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
            "gender": "neutral",
        }
        self._smpl_robot = SMPL_Robot(self.robot_cfg, data_dir=data_dir)
        self._pid_controlled = pid_controlled

        # initialize variables
        self.reset()

    def action_spec(self):
        ctrlrange = self.mj_model.actuator_ctrlrange
        return {
            "shape": (self.mj_model.nu,),
            "low": ctrlrange[:, 0],
            "high": ctrlrange[:, 1],
            "name": "\t".join(smplmj.get_actuator_names(self.mj_model)),
        }

    def reset(self) -> None:
        # reset step of the episode
        self.cur_t = 0

        if self._initial_position_str == "mocap":
            self._init_walker_from_mocap(random_timestep=True)
        elif self._initial_position_str == "stand":
            raise NotImplementedError()
        elif self._initial_position_str == "random":
            self._init_walker_random_fall()
        elif self._initial_position_str == "hybrid":
            i_type = self._np_random.choice(
                3, 1, p=self._probabilities_of_random_methods
            ).item()
            if i_type == 0:
                self._init_walker_from_mocap(random_timestep=True)
            elif i_type == 1:
                self._init_walker_random_fall()
            else:
                raise NotImplementedError()
        else:
            raise ValueError(
                f"{self.__class__.__name__}: unknown initial position configuration"
            )

        self.episode = {
            "qpos": [],
            "qvel": [],
            "torque": [],
            "action": [],
        }  # debugging TODO:remove
        return self.get_observation()

    def step(self, action):
        try:
            for _ in range(self._physics_steps_per_control_step):
                self.episode["qpos"].append(
                    self.mj_data.qpos.copy()
                )  # debugging TODO:remove
                self.episode["qvel"].append(
                    self.mj_data.qvel.copy()
                )  # debugging TODO:remove
                self.episode["action"].append(action.copy())  # debugging TODO:remove
                torque = self._controller.control(
                    action, mj_model=self.mj_model, mj_data=self.mj_data
                )
                self.episode["torque"].append(torque.copy())  # debugging TODO:remove
                self.mj_data.ctrl[:] = torque
                mujoco.mj_step(self.mj_model, self.mj_data)

        except Exception as e:
            print("Exception in do_simulation", e, self.cur_t)
            raise e
        self.cur_t += 1
        obs = self.get_observation()
        reward = self.get_reward(obs, action)
        done = self.cur_t == self._max_episode_length
        return obs, reward, done, {}

    def viewer_render(self):
        if not "viewer" in self.__dict__:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        else:
            if self.viewer.is_running():
                self.viewer.sync()

    def render(self) -> np.ndarray:
        self.mj_renderer.update_scene(self.mj_data, camera="back")
        img = self.mj_renderer.render()
        return img

    ###################################################
    ### OBSERVATION AND REWARD
    ###################################################

    def get_observation(self):
        qpos = self.mj_data.qpos.copy()
        qvel = self.mj_data.qvel.copy()

        obs = OrderedDict()
        # morphology
        obs["gender"] = np.array(
            [SMPLHumanoid.GENDER2NUM[str(self._smpl_robot_info["gender"])]]
        )
        obs["beta"] = self._smpl_robot_info["beta"].ravel()
        obs["weight"] = np.array([self._smpl_robot_info["weight"]])
        obs["full_height"] = np.array([self._smpl_robot_info["height"]])

        # positions
        for k, v in self.smpl_qpos_addr.items():
            start, end = v
            obs[f"{k}_pos"] = qpos[start:end]

        # velocities
        for k, v in self.smpl_qpos_addr.items():
            start, end = v
            obs[f"{k}_vel"] = qvel[start:end]

        head_index = self.body_names.index("Head")
        # head pos in global coordinate
        obs["Head_xpos"] = self.mj_data.xpos[head_index].copy()
        # # head quat in global coordinate
        # obs["Head_quat"] = self.mj_data.xquat[head_index].copy()
        # """Returns projection from y-axes of thorax to the z-axes of world."""
        # zy dimension (see https://github.com/google-deepmind/dm_control/blob/f2f0e2333d8bd82c0b6ba83628fe44c2bcc94ef5/dm_control/mujoco/index.py#L103)
        chest_index = self.body_names.index("Chest")
        obs["Chest_upright"] = self.mj_data.xmat[chest_index][-2]
        obs["center_of_mass_velocity"] = self.mj_data.subtree_linvel[chest_index].copy()
        return obs

    def get_reward(self, obs, control) -> float:
        return 1.0

    ###################################################
    ### XML GENERATION AND MJ MODEL CREATION
    ###################################################
    def create_mj_model(self, beta: torch.tensor, gender: str) -> None:
        self._smpl_robot.load_from_skeleton(
            betas=beta, objs_info=None, gender=[SMPLHumanoid.GENDER2NUM[gender]]
        )  # neutral gender

        xml = self._smpl_robot.export_xml_string().decode("utf-8")
        xml = smplxadd.smpl_add_camera(xml)
        xml = smplxadd.smpl_change_world(xml)
        # xml = smplxadd.smpl_add_sensors(xml)
        # Make model and data
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mj_model.opt.timestep = self._sim_timestep

        # setup renderer
        self.mj_renderer = mujoco.Renderer(self.mj_model)
        # needed (see mujoco tutorial)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._smpl_robot_info = {
            "height": self._smpl_robot.height.item(),
            "gender": gender,
            "weight": mujoco.mj_getTotalmass(self.mj_model),
            "beta": beta.numpy().ravel(),
        }
        logging.info(
            f"Created new SMPL Robot: {self._smpl_robot_info['gender']}, {self._smpl_robot_info['height']:.2f}m, {self._smpl_robot_info['weight']:.1f}Kg"
        )

        self.create_controller()

    def create_controller(self):
        """
        crete a torque proportional controller or a PID controller
        """
        self._converter = smplmj.SMPLConverter(
            self.mj_model,
            self.mj_model,
            smpl_model="smpl",
        )
        self.smpl_qpos_addr = smplmj.get_body_qposaddr(self.mj_model)
        self.smpl_qvel_addr = smplmj.get_body_qveladdr(self.mj_model)
        self.smpl_joint_names = list(self.smpl_qpos_addr.keys())
        self.jkd = self._converter.get_new_jkd()
        self.jkp = self._converter.get_new_jkp()
        self.a_scale = self._converter.get_new_a_scale()
        self.torque_lim = self._converter.get_new_torque_limit()
        self.body_names = []
        for i in range(self.mj_model.nbody):
            body_name = self.mj_model.body(i).name
            self.body_names.append(body_name)

        # extract limits of qpos
        jnt_range = smplmj.get_jnt_range(self.mj_model)
        actuators = smplmj.get_actuator_names(self.mj_model)
        limits = zip(*(jnt_range[actuator] for actuator in actuators))
        lower, upper = (np.array(limit) for limit in limits)
        qpos_range = np.concatenate(
            [lower.reshape(-1, 1), upper.reshape(-1, 1)], axis=1
        )
        self.qpos_range = qpos_range
        ctrl_range = self.mj_model.actuator_ctrlrange
        assert np.allclose(ctrl_range[:, 0], -1) and np.allclose(ctrl_range[:, 1], 1)
        self.ctrl_range = ctrl_range

        if self._pid_controlled:
            qvel_lim = (
                np.max(self.mj_model.jnt_dofadr)
                + self.mj_model.jnt_dofadr[-1]
                - self.mj_model.jnt_dofadr[-2]
            )
            qpos_lim = (
                np.max(self.mj_model.jnt_qposadr)
                + self.mj_model.jnt_qposadr[-1]
                - self.mj_model.jnt_qposadr[-2]
            )
            self._controller = ctrlm.UhcPIDController(
                ctrl_range=ctrl_range,
                qpos_range=qpos_range,
                qvel_lim=qvel_lim,
                torque_lim=self.torque_lim,
                jkp=self.jkp,
                jkd=self.jkd,
            )
            # self._controller = ctrlm.SimplePID(
            #     Kp=jkp,
            #     Ki=0.0,
            #     Kd=jkd,
            #     torque_lim=torque_lim,
            #     ctrl_range=ctrl_range,
            #     qpos_range=qpos_range,
            # )
        else:
            self._controller = ctrlm.SimpleTorqueController(
                scale=self.a_scale * 100, torque_lim=self.torque_lim
            )

    ###################################################
    ### INITIALIZATIONS
    ###################################################
    def _init_walker_zero_pos(self):
        # TODO
        pass

    def _init_walker_random_fall(self):
        # sample random beta (see https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf)
        beta_np = 16  # for smpl
        beta = (self._np_random.rand(beta_np) - 0.5) * 2
        tensor_beta = torch.tensor(beta, dtype=torch.float32).reshape(1, -1)

        # sample gender
        gender = self._np_random.choice(["female", "male"])

        # generate the SMPL xml model based on beta and gender
        self.create_mj_model(beta=tensor_beta, gender=gender)

        # set zero pos + z shift
        # TODO: how do we initialize a random position?
        self.mj_data.qpos[:] = np.zeros(self.mj_model.nq)
        self.mj_data.qpos[2] += 1  # set z cordinate
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # take random action
        for _ in range(2):
            # on purpose this is always done in torque space
            action = self._np_random.rand(self.mj_model.nu) * 2.0 - 1.0
            torque = action * self.a_scale * 100
            self.mj_data.ctrl[:] = torque
            for _ in range(self._physics_steps_per_control_step):
                mujoco.mj_step(self.mj_model, self.mj_data)

    def _init_walker_from_mocap(self, key: str = "", random_timestep: bool = False):
        if key:
            self.k_motion = key
        else:
            self.k_motion = self._np_random.choice(self._motion_keys)
        expert_data = self._motions[self.k_motion]

        # convert to appropriate format
        tensor_beta = torch.tensor(expert_data["beta"], dtype=torch.float32).reshape(
            1, -1
        )
        gender = str(expert_data["gender"])
        self.create_mj_model(beta=tensor_beta, gender=gender)

        self.expert_qpos, self.expert_qvel = self._get_qv_from_motion(expert_data)
        time_step = 0
        if random_timestep:
            time_step = self._np_random.choice(self.expert_qpos.shape[0] - 3)
        self.mj_data.qpos[:] = self.expert_qpos[time_step]
        # TODO: get qvel
        # self.mj_data.qvel[:] = self.expert_qvel[0]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    ###################################################
    ### MOCAP UTILS
    ###################################################

    def _get_qv_from_motion(self, expert_data):
        # set qpos from mocap data
        pose_aa = expert_data["pose_aa"]
        trans = expert_data["trans"]
        # convert data into appropriate qpos format
        # TODO: instead of converting the entire sequence we can convert a part of it
        expert_qpos = smplmj.smpl_to_qpose(
            pose=pose_aa,
            mj_model=self.mj_model,
            trans=trans.squeeze(),
            model=self.robot_cfg.get("model", "smpl"),
            count_offset=self.robot_cfg.get("mesh", True),
            euler_order="XYZ",
        )
        expert_qvel = None  # TODO
        return expert_qpos, expert_qvel
