import cv2
import numpy as np
import torch

from .torch_geometry_transforms import *
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as sRot
from scipy.ndimage import gaussian_filter1d

def wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]

def xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def quat_unit(a):
    return normalize(a)

def quat_from_angle_axis(angle, axis):
    theta = (angle / 2)[..., None]
    xyz = normalize(axis) * np.sin(theta)
    w = np.cos(theta)
    return quat_unit(np.concatenate([w, xyz], axis=-1))


def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., None] * 2.0
    c = q_vec * \
        (q_vec.reshape(shape[0], 1, 3) @ v.reshape(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def calc_heading(q):
    ref_dir = np.zeros((q.shape[0], 3))
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


def calc_heading_quat(q):
    heading = calc_heading(q)
    axis = np.zeros((q.shape[0], 3))
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = np.zeros((q.shape[0], 3))
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return np.concatenate((a[:, :1], -a[:, 1:]), axis=-1).reshape(shape)

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([w, x, y, z], axis=-1).reshape(shape)

    return quat