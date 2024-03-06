# Mujoco: wxyz. Isaac and sRot: xyzw.
import numpy as np

def wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]

def xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]

def normalize(v, eps = 1e-9):
    return v / np.linalg.norm(v, ord = 2, axis = -1).clip(min=eps, max=None)[..., None]

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

def quat_to_tan_norm(q):
    ref_tan = np.zeros((q.shape[0], 3))
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = np.zeros((q.shape[0], 3))
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = np.concatenate([tan, norm], axis=len(tan.shape) - 1)
    return norm_tan

def normalize_angle(x):
    return np.arctan2(np.sin(x), np.cos(x))

def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qw, qx, qy, qz = 0, 1, 2, 3

    sin_theta = np.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * np.arccos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta[..., None]
    axis = q[..., 1:4] / sin_theta_expand

    mask = np.abs(sin_theta) > min_theta
    default_axis = np.zeros_like(axis)
    
    default_axis[..., -1] = 1

    angle = np.where(mask, angle, np.zeros_like(angle))
    mask_expand = mask[..., None]
    axis = np.where(mask_expand, axis, default_axis)
    return angle, axis


def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle[..., None]
    exp_map = angle_expand * axis
    return exp_map

def remove_base_rot(quat, humanoid_type = "smpl"):
    # ZL: removing the base rotation for Humanoid model
    if humanoid_type in ["smpl", "smplh", "smplx"]:
        base_rot = quat_conjugate(np.array([[0.5, 0.5, 0.5, 0.5]])) #SMPL

    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))