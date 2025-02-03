import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as sRot


def get_body_qposaddr(model):
    # adapted to mujoco 2.3+
    body_qposaddr = dict()
    for i in range(model.nbody):
        body_name = model.body(i).name
        #body_jntadr: start addr of joints; -1: no joints
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        #body_jntnum: number of joints for this body 
        end_joint = start_joint + model.body_jntnum[i]
        #jnt_qposadr: start addr in 'qpos' for joint's data
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            #nq: number of generalized coordinates = dim(qpos)
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr


def get_body_qveladdr(model):
    # adapted to mujoco 2.3+
    body_qveladdr = dict()
    for i in range(model.nbody):
        body_name = model.body(i).name
        #body_jntadr: start addr of joints; -1: no joints
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        #body_jntnum: number of joints for this body 
        end_joint = start_joint + model.body_jntnum[i]
        start_qveladdr = model.jnt_dofadr[start_joint]
        if end_joint < len(model.jnt_dofadr):
            end_qveladdr = model.jnt_dofadr[end_joint]
        else:
            end_qveladdr = model.nv
        body_qveladdr[body_name] = (start_qveladdr, end_qveladdr)
    return body_qveladdr


def get_jnt_range(model):
    jnt_range = dict()
    for i in range(model.njnt):
        if i == model.njnt - 1:
            end_p = model.name_geomadr[0]
        else:
            end_p = model.name_jntadr[i+1]
        name = model.names[model.name_jntadr[i]:end_p].decode("utf-8").rstrip('\x00')
        jnt_range[name] = model.jnt_range[i]
    return jnt_range


def get_actuator_names(model):
    actuators = []
    for i in range(model.nu):
        if i == model.nu - 1:
            end_p = None
            for el in ["name_sensoradr", "name_numericadr", "name_textadr", "name_tupleadr", "name_keyadr", "name_pluginadr"]:
                v = getattr(model, el)
                if np.any(v):
                    end_p = v[0]
            if end_p is None:
                end_p = model.nnames
        else:
            end_p = model.name_actuatoradr[i+1]
        name = model.names[model.name_actuatoradr[i]:end_p].decode("utf-8").rstrip('\x00')
        actuators.append(name)
    return actuators


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1, point2)

def add_visual_rbox(scene, point1, point2, rgba):
    """Adds one rectangle to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_BOX, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_BOX, 0.01,
                            point1, point2)
