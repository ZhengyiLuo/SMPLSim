import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import os
import yaml
from tqdm import tqdm

from smpl_sim.utils import torch_utils
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
from enum import Enum
from easydict import EasyDict

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


class MotionlibMode(Enum):
    file = 1
    directory = 2
    
class MotionLibBase():

    def __init__(self, motion_lib_cfg):
        self.mesh_parsers = None
        self.m_cfg = motion_lib_cfg
        self.dtype = np.float32
        self.curr_failed_keys = []
        
        self.load_data(motion_lib_cfg.motion_file,  min_length = motion_lib_cfg.min_length)
        self.setup_constants(fix_height = motion_lib_cfg.fix_height, multi_thread = motion_lib_cfg.multi_thread)

        return
        
    def load_data(self, motion_file,  min_length=-1):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        
        data_list = self._motion_data_load

        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_aa']) >= min_length}
            else:
                data_list = self._motion_data_load

            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = np.zeros(self._num_unique_motions)
        self._success_rate = np.zeros(self._num_unique_motions)
        self._sampling_history = np.zeros(self._num_unique_motions)
        self._sampling_prob = np.ones(self._num_unique_motions) / self._num_unique_motions  # For use in sampling batches
        self._sampling_batch_prob = None  # For use in sampling within batches
        
        
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, shape_params, mesh_parsers, config, queue, pid):
        raise NotImplementedError

    @staticmethod
    def fix_trans_height(pose_aa, trans, shape_params, mesh_parsers, fix_height_mode):
        raise NotImplementedError

    def load_motions(self, m_cfg, shape_params, random_sample=True, start_idx=0, silent= False):
        # load motion load the same number of motions as there are skeletons (humanoids)

        motions = []
        motion_lengths = []
        motion_fps_acc = []
        motion_dt = []
        motion_num_frames = []
        motion_bodies = []
        motion_aa = []

        total_len = 0.0
        
        self.num_joints = len(self.mesh_parsers["0"].joint_names)
        num_motion_to_load = len(shape_params)
        if random_sample:
            sample_idxes = np.random.choice(np.arange(self._num_unique_motions), size = num_motion_to_load, p = self._sampling_prob, replace=True)
        else:
            sample_idxes = np.remainder(np.arange(num_motion_to_load) + start_idx, self._num_unique_motions )

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()


        motion_data_list = self._motion_data_list[sample_idxes]
        mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(min(mp.cpu_count(), 64), num_motion_to_load)
        jobs = motion_data_list
        
        if len(jobs) <= 32 or not self.multi_thread or num_jobs <= 8:
            num_jobs = 1
            
        res_acc = {}  # using dictionary ensures order of the results.
        
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], shape_params[i:i + chunk], self.mesh_parsers, m_cfg) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))
        pbar = tqdm(range(len(jobs) - 1)) if not silent else range(len(jobs) - 1)
        for i in pbar:
            res = queue.get()
            res_acc.update(res)
        pbar = tqdm(range(len(res_acc)))if not silent else range(len(res_acc))
                                                                 
        for f in pbar:
            motion_file_data, curr_motion = res_acc[f]

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            motion_aa.append(curr_motion.pose_aa)
            motion_bodies.append(curr_motion.gender_beta)

            motion_fps_acc.append(motion_fps)
            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            motion_lengths.append(curr_len)
            
            del curr_motion
            
        self._motion_lengths = np.array(motion_lengths).astype(self.dtype)
        self._motion_fps = np.array(motion_fps).astype(self.dtype)
        self._motion_bodies = np.stack(motion_bodies).astype(self.dtype)
        self._motion_aa = np.concatenate(motion_aa).astype(self.dtype)

        self._motion_dt = np.array(motion_dt).astype(self.dtype)
        self._motion_num_frames = np.array(motion_num_frames)
        self._num_motions = len(motions)

        self.gts = np.concatenate([m.global_translation for m in motions], axis=0).astype(self.dtype)
        self.grs = np.concatenate([m.global_rotation for m in motions], axis=0).astype(self.dtype)
        self.lrs = np.concatenate([m.local_rotation for m in motions], axis=0).astype(self.dtype)
        self.grvs = np.concatenate([m.global_root_velocity for m in motions], axis=0).astype(self.dtype)
        self.gravs = np.concatenate([m.global_root_angular_velocity for m in motions], axis=0).astype(self.dtype)
        self.gavs = np.concatenate([m.global_angular_velocity for m in motions], axis=0).astype(self.dtype)
        self.gvs = np.concatenate([m.global_velocity for m in motions], axis=0).astype(self.dtype)
        self.dvs = np.concatenate([m.dof_vels for m in motions], axis=0).astype(self.dtype)
        self.dof_pos = np.concatenate([m.dof_pos for m in motions], axis=0).astype(self.dtype)
        self.qpos = np.concatenate([m.qpos for m in motions], axis=0).astype(self.dtype)
        self.qvel = np.concatenate([m.qvel for m in motions], axis=0).astype(self.dtype)
        
        lengths = self._motion_num_frames
        lengths_shifted = np.roll(lengths, 1, axis = 0) 
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = np.arange(len(motions))
        motion = motions[0]
        self.num_bodies = self.num_joints

        num_motions = self.num_current_motions()
        total_len = self.get_total_length()
        if not silent:
            print(f"###### Sampling {num_motions:d} motions:", sample_idxes[:5], self.curr_motion_keys[:5], f"total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        else:
            print(sample_idxes[:5], end=" ")
        return motions

    def num_current_motions(self):
        return self._num_motions
    
    def num_all_motions(self):
        return self._num_unique_motions

    def get_total_length(self):
        return sum(self._motion_lengths)
    
    def get_termination_history(self):
        return {
            "termination_history": self._termination_history,
            "failed_keys": self.curr_failed_keys,
        }
        
    def set_termination_history(self, termination_history):
        self._termination_history = termination_history["termination_history"]
        self.curr_failed_keys = termination_history["failed_keys"]
        self.update_sampling_prob(self._termination_history)

        
    def update_hard_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only trained on "failed" sequences. Auto PMCP. 
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._sampling_prob[:] = 0
            self._sampling_prob[indexes] = 1/len(indexes)
            print("############################################################ Auto PMCP ############################################################")
            print(f"Training on only {len(failed_keys)} seqs")
            print(failed_keys)
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = np.ones(self._num_unique_motions) / self._num_unique_motions  # For use in sampling batches
            
    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP. 
        if len(failed_keys) > 0:
            self.curr_failed_keys = failed_keys
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)    
            
            
            print("############################################################ Auto PMCP ############################################################")
            print(f"Training mostly on {len(self._sampling_prob.nonzero()[0])} seqs ")
            print(self._motion_data_keys[self._sampling_prob.nonzero()].flatten())
            print(f"###############################################################################################################################")
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = np.ones(self._num_unique_motions) / self._num_unique_motions  # For use in sampling batches

    def update_sampling_prob(self, termination_history):
        print("------------------------------------------------------ Restoring Termination History ------------------------------------------------------")
        if len(self._sampling_prob) == len(self._termination_history):
            self._sampling_prob[:] = termination_history/termination_history.sum()
            self._termination_history = termination_history
            print("Successfully restored termination history")
            return True
        else:
            print("Termination history length does not match")
            return False
         
    def sample_motions(self, n = 1):
        motion_ids =  np.random.choice(np.arange(len(self._sampling_batch_prob)), size = n, p = self._sampling_batch_prob, replace=True)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        
        phase = np.random.random(motion_ids.shape)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = np.random.random(motion_ids.shape)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * 30 / self._motion_fps).astype(int)
        else:
            return (self._motion_num_frames[motion_ids] * 30 / self._motion_fps).astype(int)

    def get_motion_state_intervaled(self, motion_ids, motion_times, offset=None):
        N = len(motion_ids)
        num_bodies = self._get_num_bodies()
        
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        frame_idx = ((1.0 - blend) * frame_idx0 + blend * frame_idx1).astype(int)
        fl = frame_idx + self.length_starts[motion_ids]

        dof_pos = self.dof_pos[fl]
        body_vel = self.gvs[fl]
        body_ang_vel = self.gavs[fl]
        xpos = self.gts[fl, :]
        xquat = self.grs[fl]
        dof_vel = self.dvs[fl]
        qpos = self.qpos[fl]
        qvel = self.qvel[fl]

        vals = [dof_pos, body_vel, body_ang_vel, xpos, dof_vel]

        if not offset is None:
            xpos = xpos + offset[..., None, :]  # ZL: apply offset

        return EasyDict({
            "root_pos": xpos[..., 0, :].copy(),
            "root_rot": xquat[..., 0, :].copy(),
            "dof_pos": dof_pos.copy(),
            "root_vel": body_vel[..., 0, :].copy(),
            "root_ang_vel": body_ang_vel[..., 0, :].copy(),
            "dof_vel": dof_vel.reshape(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[fl],
            "xpos": xpos,
            "xquat": xquat,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "qpos": qpos, 
            "qvel": qvel,
        })
        
        
        
    def get_motion_state(self, motion_ids, motion_times, offset=None):
        N = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        dof_pos0 = self.dof_pos[f0l]
        dof_pos1 = self.dof_pos[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [dof_pos0, dof_pos1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1


        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        
        return EasyDict({
            "root_pos": rg_pos[..., 0, :].copy(),
            "root_rot": rb_rot[..., 0, :].copy(),
            "dof_pos": dof_pos.copy(),
            "root_vel": body_vel[..., 0, :].copy(),
            "root_ang_vel": body_ang_vel[..., 0, :].copy(),
            "dof_vel": dof_vel.reshape(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
        })

    def get_root_pos_smpl(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        vals = [rg_pos0, rg_pos1]

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        return {"root_pos": rg_pos[..., 0, :].copy()}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.copy()
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0
        frame_idx0 = (phase * (num_frames - 1))
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        
        blend = np.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)

    # jp hack
    def _hack_test_vel_consistency(self, motion):
        test_vel = np.loadtxt("output/vel.txt", delimiter=",")
        test_root_vel = test_vel[:, :3]
        test_root_ang_vel = test_vel[:, 3:6]
        test_dof_vel = test_vel[:, 6:]

        dof_vel = motion.dof_vels
        dof_vel_err = test_dof_vel[:-1] - dof_vel[:-1]
        dof_vel_err = np.max(np.abs(dof_vel_err))

        root_vel = motion.global_root_velocity.numpy()
        root_vel_err = test_root_vel[:-1] - root_vel[:-1]
        root_vel_err = np.max(np.abs(root_vel_err))

        root_ang_vel = motion.global_root_angular_velocity.numpy()
        root_ang_vel_err = test_root_ang_vel[:-1] - root_ang_vel[:-1]
        root_ang_vel_err = np.max(np.abs(root_ang_vel_err))

        return