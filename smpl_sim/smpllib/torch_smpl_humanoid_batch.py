import torch
import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import joblib
import mujoco
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES,
)
try:
    # Python < 3.9
    from importlib_resources import files
except ImportError:
    from importlib.resources import files
from omegaconf import DictConfig, OmegaConf
from smpl_sim.envs.humanoid_env import HumanoidEnv
import hydra
from easydict import EasyDict
import scipy.ndimage as ndimage
import smpl_sim.utils.pytorch3d_transforms as tRot
from collections import defaultdict
import smpl_sim.utils.np_transform_utils as npt_utils
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES

class Humanoid_Batch:

    def __init__(self, smpl_model="smpl", data_dir="data/smpl", filter_vel = True):
        self.smpl_model = smpl_model
        if self.smpl_model == "smpl":
                self.smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
                self.smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
                self.smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
                self.bone_mujoco_names = SMPL_MUJOCO_NAMES  
                self._parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22] # Mujoco's order
                self.bone_rodered_names = SMPL_BONE_ORDER_NAMES
                
        elif self.smpl_model == "smplh":
            self.smpl_parser_n = SMPLH_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
            )
            self.smpl_parser_m = SMPLH_Parser(model_path=data_dir, gender="male", use_pca=False, create_transl=False)
            self.smpl_parser_f = SMPLH_Parser(model_path=data_dir, gender="female", use_pca=False, create_transl=False)
            self.bone_mujoco_names = SMPLH_MUJOCO_NAMES 
            self._parents = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11, 12, 11, 14, 15, 16, 17, 18, 19, 17, 21, 22, 17, 24, 25, 17, 27, 28, 17, 30, 31, 11, 33, 34, 35, 36, 37, 38, 36, 40, 41, 36, 43, 44, 36, 46, 47, 36, 49, 50]
            self.bone_rodered_names = SMPLH_BONE_ORDER_NAMES
        elif self.smpl_model == "smplx":
            self.smpl_parser_n = SMPLX_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
                flat_hand_mean = True,
            )
            self.smpl_parser_m = SMPLX_Parser(model_path=data_dir, gender="male", use_pca=False, create_transl=False, flat_hand_mean = True,)
            self.smpl_parser_f = SMPLX_Parser(model_path=data_dir, gender="female", use_pca=False, create_transl=False, flat_hand_mean = True,)
            self.bone_mujoco_names = SMPLH_MUJOCO_NAMES 
            self._parents = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11, 12, 11, 14, 15, 16, 17, 18, 19, 17, 21, 22, 17, 24, 25, 17, 27, 28, 17, 30, 31, 11, 33, 34, 35, 36, 37, 38, 36, 40, 41, 36, 43, 44, 36, 46, 47, 36, 49, 50]
            self.bone_rodered_names = SMPLH_BONE_ORDER_NAMES
            
            
        self.smpl_2_mujoco = [self.bone_rodered_names.index(i) for i in self.bone_mujoco_names] # Apply Mujoco order
        self.mujoco_2_smpl = [self.bone_mujoco_names.index(i) for i in self.bone_rodered_names] # Apply Mujoco order
        self.num_joints = len(self._parents)
        self.dt = 1/30
        self.update_model(torch.zeros((1, 10)), torch.zeros((1))) # default gender 0 and pose 0. 
        self.filter_vel = filter_vel
        

    def update_model(self, betas, gender, dt = 1/30):
        # Betas: Nx10 Gender: N
        betas, gender = betas.cpu().float(), gender.cpu().long()
        B, _ = betas.shape
        betas_f = betas[gender == 2]
        if len(betas_f) > 0:

            _, _, _, _, joint_offsets_f, _, _, _, _, _, _, = self.smpl_parser_f.get_mesh_offsets_batch(betas=betas_f[:, :10])

        betas_n = betas[gender == 0]
        if len(betas_n) > 0:
            _, _, _, _, joint_offsets_n, _, _, _, _, _, _, = self.smpl_parser_n.get_mesh_offsets_batch(betas=betas_n[:, :10])

        betas_m = betas[gender == 1]
        if len(betas_m) > 0:
            _, _, _, _, joint_offsets_m, _, _, _, _, _, _, = self.smpl_parser_m.get_mesh_offsets_batch(betas=betas_m[:, :10])

        joint_offsets_all = dict()
        for n in SMPL_BONE_ORDER_NAMES:
            joint_offsets_all[n] = torch.zeros([B, 3]).float()
            if len(betas_f) > 0:
                joint_offsets_all[n][gender == 2] = joint_offsets_f[n]
            if len(betas_n) > 0:
                joint_offsets_all[n][gender == 0] = joint_offsets_n[n]
            if len(betas_m) > 0:
                joint_offsets_all[n][gender == 1] = joint_offsets_m[n]

        off_sets = []
        for n in self.bone_mujoco_names:
            off_sets.append(joint_offsets_all[n])

        # self._offsets = torch.from_numpy(np.stack(off_sets, axis=1))
        self.dt = dt 
        self._offsets = torch.from_numpy(np.round(np.stack(off_sets, axis=1), decimals=5))
        # self._offsets = joblib.load("curr_offset.pkl")[None, ]

    def fk_batch(self, pose, trans, convert_to_mat=True, count_offset=True, return_full = False):
        # SMPL pose as input, mujoco as output. 
        device, dtype = pose.device, pose.dtype
        assert(len(pose.shape) == 4) # Batch, Seqlen, J, 3
        B, T = pose.shape[:2]
        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose)
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
            
        if len(pose_mat.shape) != 5:
            pose_mat = pose_mat.reshape(B, T, -1, 3, 3)

        if count_offset:
            trans = trans + self._offsets[:, 0:1].to(device)
        
        pose_mat_ordered = pose_mat[:, :, self.smpl_2_mujoco] # Apply Mujoco order

        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat_ordered[:, :, 1:], pose_mat_ordered[:, :, 0:1], trans)
        return_dict = EasyDict()
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        
        if return_full:
            wbody_rot = tRot.matrix_to_quaternion(wbody_mat)
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, self.dt, self.filter_vel)  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, self.dt, self.filter_vel) # Global angular velocity
            return_dict.global_rotation = wbody_rot
            return_dict.local_rotation = pose_quat
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            dof_pos = tRot.matrix_to_euler_angles(pose_mat_ordered, "XYZ")[..., 1:, :] 
            
            return_dict.dof_pos = torch.cat([tRot.fix_continous_dof(dof_pos_t)[None, ] for dof_pos_t in dof_pos], dim = 0) # imperfect fix. 
            # joblib.dump(dof_pos.squeeze(), "dof.pkl")
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1] )/self.dt)
            # while len(dof_vel[dof_vel > np.pi]) > 0: dof_vel[dof_vel > np.pi] -= 2 * np.pi
            # while len(dof_vel[dof_vel < -np.pi]) > 0: dof_vel[dof_vel < -np.pi] += 2 * np.pi
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -1:]], dim = 1)
            return_dict.fps = int(1/self.dt)
            
            return_dict.qpos = torch.cat([trans, pose_quat[..., 0, :], return_dict.dof_pos.view(B, T, -1)], dim = -1)
            
            local_root_angular_velocity = (wbody_mat[:, :, 0, ].transpose(3, 2) @ return_dict.global_root_angular_velocity[..., None])[..., 0]
            return_dict.qvel = torch.cat([return_dict.global_root_velocity, local_root_angular_velocity, return_dict.dof_vels.view(B, T, -1)], dim = -1)

        return return_dict

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """

        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []
        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])

                rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])

                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        assert(len(p.shape) == 4)
        
        velocity = (p[:, 1:, ...] - p[:, :-1, ...])/time_delta
        velocity = torch.cat([velocity, velocity[:, -1:, ...]], dim = 1) # Mujoco
        
        if guassian_filter:
            velocity = torch.from_numpy(ndimage.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(rotations, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        
        diff_quat_data = tRot.quat_identity_like(rotations).to(rotations)
        
        diff_quat_data[..., :-1, :, :] = tRot.quat_mul_norm(rotations[..., 1:, :, :], tRot.quat_inverse(rotations[..., :-1, :, :]))
        diff_angle, diff_axis = tRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        
        if guassian_filter:
            angular_velocity = torch.from_numpy(ndimage.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
        return angular_velocity

    def pose_aa_to_qpos_torch(self, pose_aa, trans):
        assert(len(pose_aa.shape) == 3) # Batch, J, 3
        B, T, _ = dof_pos.shape
        pose_quat = tRot.axis_angle_to_quaternion(pose_aa)
        pose_mat = tRot.quaternion_to_matrix(pose_quat)
        pose_mat_ordered = pose_mat[:, self.smpl_2_mujoco] 
        dof_pos = tRot.matrix_to_euler_angles(pose_mat_ordered, "XYZ")[..., 1:, :] # three d euler
        
        
        import ipdb; ipdb.set_trace()
        qpos = torch.cat([trans, pose_quat[..., 0, :], dof_pos.view(B, T, -1)], dim = -1)
        return qpos
    
    def qpos_to_pose_aa_torch(self, qpos):
        assert(len(qpos.shape) == 2) # Batch, 76
        qpos = torch.from_numpy(qpos)
        root_pos = qpos[:, :3]
        root_rot = qpos[:, 3:7]
        dof_pos = qpos[:, 7:]
        body_pose_mat = tRot.euler_angles_to_matrix(dof_pos.reshape(-1, 23, 3), "XYZ")
        body_pose_aa = tRot.matrix_to_axis_angle(body_pose_mat)
        root_pose_aa = tRot.quaternion_to_axis_angle(root_rot).view(-1, 1, 3)
        pose_aa = torch.cat([root_pose_aa, body_pose_aa], dim = 1)
        pose_aa = pose_aa[:, self.mujoco_2_smpl]
        root_pos = root_pos - self._offsets[0, 0:1].numpy()
        return root_pos, pose_aa
    
    def qpos_to_pose_aa_numpy(self, qpos):
        assert(len(qpos.shape) == 2) # Batch, 76
        root_pos = qpos[:, :3]
        root_rot = qpos[:, 3:7]
        dof_pos = qpos[:, 7:]
        
        body_pose_aa = sRot.from_euler("XYZ", dof_pos.reshape(-1,  3)).as_rotvec().reshape(-1, self.num_joints - 1, 3)
        root_pose_aa = sRot.from_quat(tRot.wxyz_to_xyzw(root_rot)).as_rotvec().reshape(-1, 1, 3)
        pose_aa = np.concatenate([root_pose_aa, body_pose_aa], axis = 1)
        pose_aa = pose_aa[:, self.mujoco_2_smpl]
        root_pos = root_pos - self._offsets[0, 0:1].numpy()
        return root_pos, pose_aa
    

        
        

@hydra.main(version_base=None, config_path=str(files('smpl_sim').joinpath('data/cfg')), config_name="config")
def main(cfg : DictConfig) -> None:
    # motions = joblib.load("sample_data/amass_isaac_standing_upright_slim.pkl")
    cfg.robot.create_vel_sensors = True
    env = HumanoidEnv(cfg)
    
    motions = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_copycat_take6_test.pkl")
    data_key = "0-Transitions_mocap_mazen_c3d_JOOF_walkbackwards_poses"
    pose_aa = motions[data_key]['pose_aa'][:, :66]
    trans = motions[data_key]['trans']
    
    pose_aa_orig = np.concatenate([pose_aa, np.zeros((pose_aa.shape[0], 6))], axis=1).reshape(-1, 24, 3)
    
    humanoid = Humanoid_Batch(filter_vel=False)
    humanoid.update_model(torch.zeros((1, 10)), torch.zeros((1)), dt = env.sim_timestep)
    return_dict = humanoid.fk_batch(torch.from_numpy(pose_aa_orig[None, ]), trans=torch.from_numpy(trans[None, ]), return_full=True)
    
    pose_aa = pose_aa_orig[:, [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]]
    body_pose_aa = pose_aa[:, 1:]
    root_pose_aa = pose_aa[:, 0]
    body_pos = sRot.from_rotvec(body_pose_aa.reshape(-1, 3)).as_euler("XYZ").reshape(-1, 69)
    qpos = np.concatenate([trans + humanoid._offsets[:, 0].numpy(), tRot.xyzw_to_wxyz(sRot.from_rotvec(root_pose_aa).as_quat()), body_pos], axis = -1)
    
    assert(np.abs(return_dict.qpos.numpy()[0] - qpos).sum() < 0.000001) # Verfying pose_aa to qpos conversion
    
    print(f"number of motions: {len(motions)}")
    
    env.reset()
    cur_t, T = 0, 0
    env.recording = True
    state_record = defaultdict(list)
    while True:
        # env.mj_data.qpos[:] = return_dict.qpos[0, cur_t % body_pos.shape[0]]
        # mujoco.mj_forward(env.mj_model, env.mj_data)
        
        # action = return_dict.qpos[0, cur_t % body_pos.shape[0]][7:].numpy()/np.pi

        # cur_t += 1
        # env.step(action=action)
        
        #######  One step Qvel compute. 
        cur_t += 1
        state_record['qpos'].append(env.mj_data.qpos.copy())
        state_record['qvel'].append(env.mj_data.qvel.copy())
        state_record['xpos'].append(env.mj_data.xpos[1:].copy())
        state_record['xquat'].append(env.mj_data.xquat[1:].copy())
        state_record['body_vel'].append(env.mj_data.sensordata[:72].reshape(24, 3).copy())
        state_record['body_angvel'].append(env.mj_data.sensordata[72:].reshape(24, 3).copy())
        
        mujoco.mj_step(env.mj_model, env.mj_data)
        mujoco.mj_forward(env.mj_model, env.mj_data)
        env.render()
        
        
        if cur_t > 10000:
            break
        

    # state_record = { k : np.array(v) for k, v in env.state_record.items()}
    state_record = { k : np.array(v) for k, v in state_record.items()}
    qpos = state_record['qpos']
    qvel = state_record['qvel'] # qvel is multiple frames of simulation result, you can't directly compute it from qpos, but only approximates. 
    xpos = state_record['xpos']
    xquat = state_record['xquat']
    body_vel = state_record['body_vel']
    body_angvel = state_record['body_angvel']
    
    N = qpos.shape[0]
    root_pos_sim, pose_aa_sim = humanoid.qpos_to_pose_aa_torch(qpos)
    return_dict_sim = humanoid.fk_batch(pose_aa_sim[None, ], trans=root_pos_sim[None, ], return_full=True)
    return_dict_sim = EasyDict({k : v.squeeze().numpy() if k != "fps" else v for k, v in return_dict_sim.items()})
    
    
    assert(np.abs(return_dict_sim.qpos[..., :7] - qpos[:, :7]).sum() < 0.00001) ### Root is fine
    # (return_dict_sim.qpos[..., 7:] - qpos[:, 7:]).abs().sum() ### The shoulders are not fine. 
    # assert(np.abs(return_dict_sim.qpos - qpos).sum() < 0.001)
    assert(np.abs(xpos - return_dict_sim.global_translation).max() < 0.001) 
    
    quat_diff = np.abs(npt_utils.quat_unit(npt_utils.quat_mul(xquat,  npt_utils.quat_conjugate(return_dict_sim.global_rotation))))
    assert(np.abs(quat_diff[..., 0] - 1).max() < 0.000001)
    assert(np.abs(quat_diff[..., 1:]).max() < 0.000001) 
    import ipdb; ipdb.set_trace()
    assert(np.abs(return_dict_sim.global_velocity - body_vel).max() < 0.5)# velocities are usually a bit lossy. 
    assert(np.abs(return_dict_sim.global_angular_velocity - body_angvel).max() < 1) # velocities are usually a bit lossy. 
    import ipdb; ipdb.set_trace()
    print('..................')
    # step_diff = (xpos[1:, ] - xpos[:-1, ])/env.sim_timestep
    # big_diff = body_vel[1:] - step_diff
    
    
    
    ### Following is not that good. 
    # diff = (return_dict_sim.qpos[0, ..., 7:].reshape(N, 23, 3)[:, 1] - qpos[:, 7:].reshape(N, 23, 3)[:, 1]).abs()
    # root_pos_sim_np, pose_aa_sim_np = humanoid.qpos_to_pose_aa_numpy(qpos)
    # return_dict_sim_np = humanoid.fk_batch(torch.from_numpy(pose_aa_sim_np[None, ]), trans=torch.from_numpy(root_pos_sim_np[None, ]), return_full=True)
    # assert(np.abs(return_dict_sim_np.qpos - qpos).sum() < 0.001)
    # assert(np.abs(return_dict_sim_np.qvel.squeeze().numpy()[:-1] - qvel[1:]).max() < 0.0001)
    
    # import smpl_sim.utils.math_utils as mRot
    # compute_qvel = [mRot.get_qvel_fd_new(qpos[idx], qpos[idx+1], humanoid.dt) for idx in range(N-1)] # UHC's qvel computation
    # compute_qvel = np.array(compute_qvel)
    # assert(np.abs((compute_qvel - qvel[1:])).max() < 0.01)
    
if __name__ == "__main__":
    main()

 
