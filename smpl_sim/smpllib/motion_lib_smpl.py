# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

import joblib
import torch
import torch.multiprocessing as mp
import copy
import gc
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot
import random
from smpl_sim.smpllib.motion_lib_base import MotionLibBase, FixHeightMode
from smpl_sim.utils.torch_utils import to_torch
from smpl_sim.smpllib.torch_smpl_humanoid_batch import Humanoid_Batch
from easydict import EasyDict


USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class MotionLibSMPL(MotionLibBase):

    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg=motion_lib_cfg)
        smpl_type = motion_lib_cfg.smpl_type
        data_dir = "data/smpl"
        if smpl_type == "smpl":
            smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
            smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
            smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
        elif smpl_type == "smplx":
            smpl_parser_n = SMPLX_Parser(model_path=data_dir, gender="neutral", use_pca=False, create_transl=False, flat_hand_mean=True)
            smpl_parser_m = SMPLX_Parser(model_path=data_dir, gender="male", use_pca=False, create_transl=False, flat_hand_mean=True)
            smpl_parser_f = SMPLX_Parser(model_path=data_dir, gender="female", use_pca=False, create_transl=False, flat_hand_mean=True)
        self.humanoid = Humanoid_Batch(data_dir = data_dir)
        self.mesh_parsers = EasyDict({"0": smpl_parser_n, "1": smpl_parser_m, "2": smpl_parser_f, "batch": self.humanoid})
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            frame_check = 30
            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[str(gender.int().item())]
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,], trans[:frame_check])
            
            
            if fix_height_mode == FixHeightMode.ankle_fix:
                height_tolorance = -0.025
                assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
                pick = (((assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int() + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int() 
                    + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int() + + (assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()) == 4).nonzero().squeeze()
                diff_fix = (vertices_curr[:, pick][:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            elif fix_height_mode == FixHeightMode.full_fix:
                height_tolorance = 0.0
                diff_fix = (vertices_curr [:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            
            trans[..., -1] -= diff_fix
            return trans, diff_fix

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, shape_params, mesh_parsers, config, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        np.random.seed(np.random.randint(5000)* pid)
        max_len = config.max_length
        res = {}
        assert (len(ids) == len(motion_data_list))
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            fps = curr_file.get("fps", 30)
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = to_torch(shape_params[f])
            
            seq_len = curr_file['pose_aa'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len
            
            trans = to_torch(curr_file['trans'] if "trans" in curr_file else curr_file['trans_orig']).float().clone()[start:end]
            
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).float().clone()
            if pose_aa.shape[1] == 156:
                pose_aa = torch.cat([pose_aa[:, :66], torch.zeros((pose_aa.shape[0], 6))], dim=1).reshape(-1, 24, 3)
            elif pose_aa.shape[1] == 72:
                pose_aa = pose_aa.reshape(-1, 24, 3)
            else:
                import ipdb; ipdb.set_trace()
                print("Error pose_aa")
                
            B, J, N = pose_aa.shape

            ##### ZL: randomize the heading ######
            if config.randomrize_heading:
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_aa[:, 0, :] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, 0, :])).as_rotvec())
                trans = torch.matmul(trans.float(), torch.from_numpy(random_heading_rot.as_matrix().T).float())
            ##### ZL: randomize the heading ######
            trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = config.fix_height)
            
            mesh_parsers.batch.update_model(betas = curr_gender_beta[None, 1:11], gender = curr_gender_beta[:1], dt = 1/fps)
            try:
                curr_motion = mesh_parsers.batch.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, count_offset = True)
            except:
                import ipdb; ipdb.set_trace()
                print("....")
            
            curr_motion = EasyDict({k: v[0] if torch.is_tensor(v) else v for k, v in curr_motion.items() })
            
            curr_motion.gender_beta = curr_gender_beta
            curr_motion.pose_aa = pose_aa
            res[curr_id] = (curr_file, curr_motion)

        if not queue is None:
            queue.put(res)
        else:
            return res


    
    