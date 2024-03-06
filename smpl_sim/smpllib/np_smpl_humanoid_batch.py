import torch
# import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp
from smpl_sim.utils.torch_ext import dict_to_torch

sys.path.append(os.getcwd())

from smpl_sim.utils.torch_utils import *
from smpl_sim.utils.np_transform_utils import *
from scipy.spatial.transform import Rotation as sRot
import joblib
from mujoco_py import load_model_from_path
from smpl_sim.smpllib.smpl_joint_names import SMPLConverter, smpl_to_qpose, smpl_to_qpose_torch, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPL_EE_NAMES
from smpl_sim.utils.tools import get_expert, get_expert_master
import pytorch3d.transforms as tR
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

import autograd.numpy as np
from autograd import elementwise_grad as egrad


def smpl_op_to_op(pred_joints2d):
    new_2d = np.concatenate([pred_joints2d[..., [1, 4], :].mean(axis = -2, keepdims = True), \
                             pred_joints2d[..., 1:7, :], \
                             pred_joints2d[..., [7, 8, 11], :].mean(axis = -2, keepdims = True), \
                             pred_joints2d[..., 9:11, :], \
                             pred_joints2d[..., 12:, :]], \
                             axis = -2)
    return new_2d


def normalize_screen_coordinates(X, w=1920, h=1080):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to
    #  [-1, 1], while preserving the aspect ratio
    return X / w * 2 - np.array([1, h / w])


def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2))[:, None, None]
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(r.dtype).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.stack([
        z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
        -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick
    ],
                 axis=1).reshape([-1, 3, 3])

    i_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0),
                             [theta.shape[0], 3, 3])
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R


def rodrigues_vec_to_rotation_mat(rot):
    theta = np.linalg.norm(rot, axis=0)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        rot = rot / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([[rot[0] * rot[0], rot[0] * rot[1], rot[0] * rot[2]],
                         [rot[1] * rot[0], rot[1] * rot[1], rot[1] * rot[2]],
                         [rot[2] * rot[0], rot[2] * rot[1], rot[2] * rot[2]]])
        r_cross = np.array([[0, -rot[2], rot[1]], [rot[2], 0, -rot[0]],
                            [-rot[1], rot[0], 0]])
        rotation_mat = np.cos(theta) * I + (
            1 - np.cos(theta)) * r_rT + np.sin(theta) * r_cross

    return rotation_mat


class Humanoid_Batch:
    def __init__(self, smpl_model="smpl", data_dir="data/smpl"):
        self.smpl_model = smpl_model
        if self.smpl_model == "smpl":
            self.smpl_parser_n = SMPL_Parser(model_path=data_dir,
                                             gender="neutral")
            self.smpl_parser_m = SMPL_Parser(model_path=data_dir,
                                             gender="male")
            self.smpl_parser_f = SMPL_Parser(model_path=data_dir,
                                             gender="female")
        elif self.smpl_model == "smplh":
            self.smpl_parser_n = SMPLH_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
            )
            self.smpl_parser_m = SMPLH_Parser(model_path=data_dir,
                                              gender="male",
                                              use_pca=False,
                                              create_transl=False)
            self.smpl_parser_f = SMPLH_Parser(model_path=data_dir,
                                              gender="female",
                                              use_pca=False,
                                              create_transl=False)
        elif self.smpl_model == "smplx":
            self.smpl_parser_n = SMPLX_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
            )
            self.smpl_parser_m = SMPLX_Parser(model_path=data_dir,
                                              gender="male",
                                              use_pca=False,
                                              create_transl=False)
            self.smpl_parser_f = SMPLX_Parser(model_path=data_dir,
                                              gender="female",
                                              use_pca=False,
                                              create_transl=False)

        self.model_names = [
            'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
            'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head',
            'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
            'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
        ]
        self._parents = [
            -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17,
            11, 19, 20, 21, 22
        ]
        self.smpl_index = [
            SMPL_BONE_ORDER_NAMES.index(i) for i in self.model_names
        ]

    def update_model(self, betas, gender):
        betas, gender = betas.cpu().float(), gender.cpu().long()
        B, _ = betas.shape

        betas_f = betas[gender == 2]
        if len(betas_f) > 0:
            _, _, _, _, joint_offsets_f, _, _, _, _, _, _, = self.smpl_parser_f.get_mesh_offsets_batch(
                betas=betas_f[:, :10])

        betas_n = betas[gender == 0]
        if len(betas_n) > 0:
            _, _, _, _, joint_offsets_n, _, _, _, _, _, _, = self.smpl_parser_n.get_mesh_offsets_batch(
                betas=betas_n[:, :10])

        betas_m = betas[gender == 1]
        if len(betas_m) > 0:
            _, _, _, _, joint_offsets_m, _, _, _, _, _, _, = self.smpl_parser_m.get_mesh_offsets_batch(
                betas=betas_m[:, :10])

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
        for n in self.model_names:
            off_sets.append(joint_offsets_all[n])

        # self._offsets = torch.from_numpy(np.stack(off_sets, axis=1))
        self._offsets = np.round(np.stack(off_sets, axis=1), decimals=5)
        self.trans2joint = -self._offsets[:, 0:1]
        self.trans2joint[:, :, 2] = 0
        # self._offsets = joblib.load("curr_offset.pkl")[None, ]

    def update_projection(self, cam_params, smpl2op_map, MUJOCO_2_SMPL):
        self.full_R = cam_params['full_R']
        self.full_t = cam_params['full_t']
        self.K = cam_params['K']
        self.img_w = cam_params['img_w']
        self.img_h = cam_params['img_h']

        self.openpose_subindex = smpl2op_map < 22
        self.smpl2op_map = smpl2op_map
        self.smpl2op_partial = self.smpl2op_map[self.openpose_subindex]
        self.MUJOCO_2_SMPL = MUJOCO_2_SMPL

    def update_tgt_joints(self, tgt_joints, inliers):
        self.gt_2d_joints = tgt_joints
        self.inliers = inliers.astype(bool)

        num_joints = self.gt_2d_joints.shape[-2]
        self.gt_2d_joints_norm = normalize_screen_coordinates(self.gt_2d_joints, self.img_w, self.img_h)


        self.num_frames = self.gt_2d_joints.shape[0]
        self.camera_rays = np.concatenate([self.gt_2d_joints, np.ones([self.num_frames, num_joints, 1])],
            axis=2).dot(np.linalg.inv(self.K).T)
        self.camera_rays /= np.linalg.norm(self.camera_rays, axis=2)[..., None]
        lam = 0.3
        self.weighting = np.exp(lam * -np.arange(self.num_frames)) / np.sum(
            np.exp(lam * -np.arange(self.num_frames)))
        self.weighting = np.tile(self.weighting[:, None, None], [1, num_joints, 2])

        # self.weighting = np.ones(self.num_frames) / self.num_frames

    def proj2d(self, wbpos, return_cam_3d=False):
        # wbpos in mujoco
        pred_joints3d = wbpos.squeeze()[self.MUJOCO_2_SMPL][
            self.smpl2op_partial][None, ]

        pred_joints3d = pred_joints3d @ self.full_R.T + self.full_t
        pred_joints2d = pred_joints3d @ (self.K.T)
        z = pred_joints2d[:, :, 2:]
        pred_joints2d = pred_joints2d[:, :, :2] / z

        pred_joints2d = smpl_op_to_op(pred_joints2d)

        if return_cam_3d:
            return pred_joints2d, pred_joints3d
        else:
            return pred_joints2d

    def proj_2d_line_loss(self, input_vec):
        wbpos = self.fk_batch_grad(input_vec)
        _, pred_joints3d = self.proj2d(wbpos, return_cam_3d=True)
        dist = np.cross(pred_joints3d[0],
                        pred_joints3d[0] - self.camera_rays)**2
        return dist.mean()

    def proj_2d_loss(self, input_vec, ord=2, normalize = True):
        wbpos = self.fk_batch_grad(input_vec)
        pred_joints2d = self.proj2d(wbpos)
        curr_weighting = np.array(self.weighting)
        if normalize:
            pred_joints2d = normalize_screen_coordinates(pred_joints2d, self.img_w, self.img_h)
            gt_2d_joints = self.gt_2d_joints_norm
        else:
            gt_2d_joints = self.gt_2d_joints

        if ord == 1:
            loss = np.abs(
                gt_2d_joints[self.inliers] -
                pred_joints2d.squeeze()[self.inliers]).squeeze().mean()
        else:
            diff = (gt_2d_joints - pred_joints2d.squeeze())**2
            curr_weighting[~self.inliers] = 0
            loss = (diff * curr_weighting).sum(axis=0).mean()
        return loss


    def proj_2d_body_loss(self, input_vec, ord=2,  normalize = False):
        # Has to use the current translation (to roughly put at the same position, and then zero out the translation)
        wbpos = self.fk_batch_grad(input_vec)
        pred_joints2d = self.proj2d(wbpos)

        gt2d_center = self.gt_2d_joints[..., 7:8, :].copy()
        pred_joints2d += (gt2d_center - pred_joints2d[..., 7:8, :])

        curr_weighting = np.array(self.weighting)

        if normalize:
            pred_joints2d = normalize_screen_coordinates(pred_joints2d, self.img_w, self.img_h)
            gt_2d_joints = self.gt_2d_joints_norm
        else:
            gt_2d_joints = self.gt_2d_joints

        if ord == 1:
            loss = np.abs(gt_2d_joints[self.inliers] - pred_joints2d.squeeze()[self.inliers]).squeeze().mean()
        else:
            diff = (gt_2d_joints - pred_joints2d.squeeze())**2
            curr_weighting[~self.inliers] = 0
            loss = (diff * curr_weighting).sum(axis=0).mean()

        return loss

    def proj_2d_root_loss(self, root_pos_rot):
        input_vec = np.concatenate(
            [root_pos_rot.reshape([1, 1, 6]),
             np.zeros([1, 1, 69])], axis=2)
        wbpos = self.fk_batch_grad(input_vec)
        pred_joints2d = self.proj2d(wbpos)
        return np.abs(self.gt_2d_joints[7:8] -
                      pred_joints2d.squeeze()[7:8]).squeeze().mean()

    def fk_batch(self, pose, trans, convert_to_mat=True, count_offset=True):
        pose, trans = pose.cpu().numpy(), trans.cpu().numpy()
        B, seq_len = pose.shape[:2]
        if convert_to_mat:
            pose_mat = rodrigues(pose.reshape(B * seq_len * 24, 1, 3)).reshape(
                B, seq_len, -1, 3, 3)
        else:
            pose_mat = pose
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root

        if count_offset:
            trans = trans + self._offsets[:, 0:1]

        pose_mat_ordered = pose_mat[:, :, self.smpl_index]

        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat_ordered[:, :, 1:], pose_mat_ordered[:, :, 0:1], trans)
        return_dic = {}
        return_dic["wbpos"] = wbody_pos
        return_dic["wbmat"] = wbody_mat

        return return_dic

    def fk_batch_grad(self, input_vec, count_offset=True):
        trans, pose = input_vec[:, :, :3], input_vec[:, :, 3:]
        B, seq_len = pose.shape[:2]

        pose_mat = rodrigues(pose.reshape(-1, 1,
                                          3)).reshape(B, seq_len, -1, 3, 3)
        # pose_mat = [
        #     rodrigues_vec_to_rotation_mat(a) for a in pose.reshape(-1, 3)
        # ]
        # pose_mat = np.stack(pose_mat).reshape(B, seq_len, -1, 3, 3)

        J = pose_mat.shape[2] - 1  # Exclude root

        if count_offset:
            trans = trans + self._offsets[:, 0:1]

        pose_mat_ordered = pose_mat[:, :, self.smpl_index]
        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat_ordered[:, :, 1:], pose_mat_ordered[:, :, 0:1], trans)
        return wbody_pos

    def get_ee_pos(self, body_xpos, root_q, transform):
        ee_name = SMPL_EE_NAMES
        ee_pos = []
        root_pos = body_xpos[:, 0, :]
        for name in ee_name:
            bone_id = self.model._body_name2id[name] - 1
            bone_vec = body_xpos[:, bone_id]

            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec_batch(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)

        return torch.swapaxes(torch.stack(ee_pos, dim=0), 0, 1)

    def forward_kinematics_batch(self, rotations, root_rotations,
                                 root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """

        B, seq_len = rotations.shape[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []
        expanded_offsets = np.repeat(np.repeat(self._offsets, B,
                                               axis=0)[:, None, :],
                                     seq_len,
                                     axis=1)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (
                    np.matmul(rotations_world[self._parents[i]][:, :, 0],
                              expanded_offsets[:, :, i, :, None]).squeeze(-1) +
                    positions_world[self._parents[i]])

                rot_mat = np.matmul(rotations_world[self._parents[i]],
                                    rotations[:, :, (i - 1):i, :])

                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = np.stack(positions_world, axis=2)
        rotations_world = np.concatenate(rotations_world, axis=2)
        return positions_world, rotations_world


if __name__ == "__main__":
    import mujoco_py
    from smpl_sim.smpllib.smpl_robot import Robot
    from smpl_sim.smpllib.torch_smpl_humanoid import Humanoid
    from smpl_sim.utils.config_utils.copycat_config import Config
    from smpl_sim.data_loaders.dataset_amass_single import DatasetAMASSSingle
    from smpl_sim.utils.torch_ext import dict_to_torch
    from smpl_sim.smpllib.smpl_mujoco import smpl_to_qpose_torch, smplh_to_smpl
    torch.manual_seed(0)

    cfg = Config(
        cfg_id="copycat_44",
        create_dirs=False,
    )

    smpl_robot = Robot(
        cfg.robot_cfg,
        data_dir=osp.join(cfg.base_dir, "data/smpl"),
        masterfoot=False,
    )
    dataset = DatasetAMASSSingle(cfg.data_specs, "test")
    humanoid_batch = Humanoid_Batch()

    data_test = dataset.sample_seq()
    data_test = dict_to_torch(data_test)
    pose_aa = data_test['pose_aa']
    trans = data_test['trans']
    beta = data_test['beta']
    gender = data_test['gender']
    count_offset = True

    smpl_robot.load_from_skeleton(beta[0:1, :].cpu().float(),
                                  gender=gender,
                                  objs_info=None)
    model = mujoco_py.load_model_from_xml(
        smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    qpos = smpl_to_qpose_torch(pose_aa,
                               mj_model=model,
                               trans=trans,
                               count_offset=count_offset)
    fk_res = humanoid.qpos_fk(qpos)

    pose_aa_smpl = smplh_to_smpl(pose_aa)
    humanoid_batch.update_model(beta[0:1], gender[0:1])

    pose_aa_np = pose_aa_smpl[None, ].cpu().numpy()
    trans_np = trans[None, ].cpu().numpy()
    input_vec = np.concatenate([trans_np, pose_aa_np], axis=2)
    import time
    t_s = time.time()
    wbpos = humanoid_batch.fk_batch_grad(input_vec)
    dt = time.time() - t_s
    print("1", dt)
    hfk = egrad(humanoid_batch.fk_batch_grad)

    t_s = time.time()
    gradient = hfk(input_vec[:, :, :])
    dt = time.time() - t_s
    print("2", dt)

    # return_dict = humanoid_batch.fk_batch(pose_aa_smpl[None, ], trans[None, ], count_offset=count_offset)
    # diff = return_dict['wbpos'].reshape(-1, 24, 3) - fk_res['wbpos'].reshape(-1, 24, 3)
    diff = wbpos - fk_res['wbpos'].reshape(-1, 24, 3)
    print("diff", np.sum(np.abs(diff)))
    import ipdb
    ipdb.set_trace()