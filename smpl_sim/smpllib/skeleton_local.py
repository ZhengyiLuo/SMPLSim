import glob
import os
import sys
import pdb
import os.path as osp

from smpl_sim.utils.transformation import euler_matrix
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
import math
import re
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot
try:
    # Python < 3.9
    from importlib_resources import files
except ImportError:
    from importlib.resources import files

GEOM_TYPES = {
    'Pelvis': 'sphere',
    'L_Hip': 'capsule',
    'L_Knee': 'capsule',
    'L_Ankle': 'box',
    'L_Toe': 'box',
    'R_Hip': 'capsule',
    'R_Knee': 'capsule',
    'R_Ankle': 'box',
    'R_Toe': 'box',
    'Torso': 'capsule',
    'Spine': 'capsule',
    'Chest': 'capsule',
    'Neck': 'capsule',
    'Head': 'sphere',
    'L_Thorax': 'capsule',
    'L_Shoulder': 'capsule',
    'L_Elbow': 'capsule',
    'L_Wrist': 'capsule',
    'L_Hand': 'sphere',
    # 'L_Hand': 'box',
    'R_Thorax': 'capsule',
    'R_Shoulder': 'capsule',
    'R_Elbow': 'capsule',
    'R_Wrist': 'capsule',
    'R_Hand': 'sphere',
    # 'R_Hand': 'box',
    
    "L_Index1": 'capsule',
    "L_Index2": 'capsule',
    "L_Index3": 'capsule',
    "L_Middle1": 'capsule',
    "L_Middle2": 'capsule',
    "L_Middle3": 'capsule',
    "L_Pinky1": 'capsule',
    "L_Pinky2": 'capsule',
    "L_Pinky3": 'capsule',
    "L_Ring1": 'capsule',
    "L_Ring2": 'capsule',
    "L_Ring3": 'capsule',
    "L_Thumb1": 'capsule',
    "L_Thumb2": 'capsule',
    "L_Thumb3": 'capsule',
    "R_Index1": 'capsule',
    "R_Index2": 'capsule',
    "R_Index3": 'capsule',
    "R_Middle1": 'capsule',
    "R_Middle2": 'capsule',
    "R_Middle3": 'capsule',
    "R_Pinky1": 'capsule',
    "R_Pinky2": 'capsule',
    "R_Pinky3": 'capsule',
    "R_Ring1": 'capsule',
    "R_Ring2": 'capsule',
    "R_Ring3": 'capsule',
    "R_Thumb1": 'capsule',
    "R_Thumb2": 'capsule',
    "R_Thumb3": 'capsule',
}
# KP KD gear max_torque
# GAINS = {
#     "L_Hip":      [800, 80, 30, 500],
#     "L_Knee":     [800, 80, 30, 500],
#     "L_Ankle":    [800, 80, 10, 500],
#     "L_Toe":      [500, 50, 10, 500],
#     "R_Hip":      [800, 80, 30, 500],
#     "R_Knee":     [800, 80, 30, 500],
#     "R_Ankle":    [800, 80, 10, 500],
#     "R_Toe":      [500, 50, 10, 500],
#     "Torso":      [1000, 100, 30, 500],
#     "Spine":      [1000, 100, 30, 500],
#     "Chest":      [1000, 100, 30, 500],
#     "Neck":       [500, 50, 10, 250],
#     "Head":       [500, 50, 5, 250],
#     "L_Thorax":   [500, 50, 30, 500],
#     "L_Shoulder": [500, 50, 30, 500],
#     "L_Elbow":    [500, 50, 10, 150],
#     "L_Wrist":    [300, 30, 10, 150],
#     "L_Hand":     [300, 30, 5, 150],
#     "R_Thorax":   [500, 50, 30, 150],
#     "R_Shoulder": [500, 50, 30, 250],
#     "R_Elbow":    [500, 50, 10, 150],
#     "R_Wrist":    [300, 30, 10, 150],
#     "R_Hand":     [300, 30, 5, 150],
# }

# PHC's gains
GAINS_PHC = {
    "L_Hip": [800, 80, 1, 500],
    "L_Knee": [800, 80, 1, 500],
    "L_Ankle": [800, 80, 1, 500],
    "L_Toe": [500, 50, 1, 500],
    "R_Hip": [800, 80, 1, 500],
    "R_Knee": [800, 80, 1, 500],
    "R_Ankle": [800, 80, 1, 500],
    "R_Toe": [500, 50, 1, 500],
    "Torso": [1000, 100, 1, 500],
    "Spine": [1000, 100, 1, 500],
    "Chest": [1000, 100, 1, 500],
    "Neck": [500, 50, 1, 250],
    "Head": [500, 50, 1, 250],
    "L_Thorax": [500, 50, 1, 500],
    "L_Shoulder": [500, 50, 1, 500],
    "L_Elbow": [500, 50, 1, 150],
    "L_Wrist": [300, 30, 1, 150],
    "L_Hand": [300, 30, 1, 150],
    "R_Thorax": [500, 50, 1, 150],
    "R_Shoulder": [500, 50, 1, 250],
    "R_Elbow": [500, 50, 1, 150],
    "R_Wrist": [300, 30, 1, 150],
    "R_Hand": [300, 30, 1, 150],
    
    "L_Index1": [100, 10, 1, 150],
    "L_Index2": [100, 10, 1, 150],
    "L_Index3": [100, 10, 1, 150],
    "L_Middle1": [100, 10, 1, 150],
    "L_Middle2": [100, 10, 1, 150],
    "L_Middle3": [100, 10, 1, 150],
    "L_Pinky1": [100, 10, 1, 150],
    "L_Pinky2": [100, 10, 1, 150],
    "L_Pinky3": [100, 10, 1, 150],
    "L_Ring1": [100, 10, 1, 150],
    "L_Ring2": [100, 10, 1, 150],
    "L_Ring3": [100, 10, 1, 150],
    "L_Thumb1": [100, 10, 1, 150],
    "L_Thumb2": [100, 10, 1, 150],
    "L_Thumb3": [100, 10, 1, 150],
    "R_Index1": [100, 10, 1, 150],
    "R_Index2": [100, 10, 1, 150],
    "R_Index3": [100, 10, 1, 150],
    "R_Middle1": [100, 10, 1, 150],
    "R_Middle2": [100, 10, 1, 150],
    "R_Middle3": [100, 10, 1, 150],
    "R_Pinky1": [100, 10, 1, 150],
    "R_Pinky2": [100, 10, 1, 150],
    "R_Pinky3": [100, 10, 1, 150],
    "R_Ring1": [100, 10, 1, 150],
    "R_Ring2": [100, 10, 1, 150],
    "R_Ring3": [100, 10, 1, 150],
    "R_Thumb1": [100, 10, 1, 150],
    "R_Thumb2": [100, 10, 1, 150],
    "R_Thumb3": [100, 10, 1, 150],
}

### UHC Phd
# GAINS = {
#     "L_Hip":        [500, 50, 1, 500, 10, 2],
#     "L_Knee":       [500, 50, 1, 500, 10, 2],
#     "L_Ankle":      [500, 50, 1, 500, 10, 2],
#     "L_Toe":        [200, 20, 1, 500, 1, 1],
#     "R_Hip":        [500, 50, 1, 500, 10, 2],
#     "R_Knee":       [500, 50, 1, 500, 10, 2],
#     "R_Ankle":      [500, 50, 1, 500, 10, 2],
#     "R_Toe":        [200, 20, 1, 500, 1, 1],
#     "Torso":        [1000, 100, 1, 500, 10, 2],
#     "Spine":        [1000, 100, 1, 500, 10, 2],
#     "Chest":        [1000, 100, 1, 500, 10, 2],
#     "Neck":         [100, 10, 1, 250, 50, 4],
#     "Head":         [100, 10, 1, 250, 50, 4],
#     "L_Thorax":     [400, 40, 1, 500, 50, 4],
#     "L_Shoulder":   [400, 40, 1, 500, 50, 4],
#     "L_Elbow":      [300, 30, 1, 150, 10, 2],
#     "L_Wrist":      [100, 10, 1, 150, 1, 1],
#     "L_Hand":       [100, 10, 1, 150, 1, 1],
#     "R_Thorax":     [400, 40, 1, 150, 10, 2],
#     "R_Shoulder":   [400, 40, 1, 250, 10, 2],
#     "R_Elbow":      [300, 30, 1, 150, 10, 2],
#     "R_Wrist":      [100, 10, 1, 150, 1, 1],
#     "R_Hand":       [100, 10, 1, 150, 1, 1],
# }

GAINS_MJ = {
    "L_Hip":            [250, 2.5, 1, 500, 10, 2],
    "L_Knee":           [250, 2.5, 1, 500, 10, 2],
    "L_Ankle":          [150, 2.5, 1, 500, 10, 2],
    "L_Toe":            [150, 1, 1, 500, 1, 1],
    "R_Hip":            [250, 2.5, 1, 500, 10, 2],
    "R_Knee":           [250, 2.5, 1, 500, 10, 2],
    "R_Ankle":          [150, 1, 1, 500, 10, 2],
    "R_Toe":            [150, 1, 1, 500, 1, 1],
    "Torso":            [500, 5, 1, 500, 10, 2],
    "Spine":            [500, 5, 1, 500, 10, 2],
    "Chest":            [500, 5, 1, 500, 10, 2],
    "Neck":             [150, 1, 1, 250, 50, 4],
    "Head":             [150, 1, 1, 250, 50, 4],
    "L_Thorax":         [200, 2, 1, 500, 50, 4],
    "L_Shoulder":       [200, 2, 1, 500, 50, 4],
    "L_Elbow":          [150, 1, 1, 150, 10, 2],
    "L_Wrist":          [100, 1, 1, 150, 1, 1],
    "L_Hand":           [50, 1, 1, 150, 1, 1],
    "R_Thorax":         [200, 2, 1, 150, 10, 2],
    "R_Shoulder":       [200, 2, 1, 250, 10, 2],
    "R_Elbow":          [150, 1, 1, 150, 10, 2],
    "R_Wrist":          [100, 1, 1, 150, 1, 1],
    "R_Hand":           [50, 1, 1, 150, 1, 1],
    
    "L_Index1": [100, 10, 1, 150],
    "L_Index2": [100, 10, 1, 150],
    "L_Index3": [100, 10, 1, 150],
    "L_Middle1": [100, 10, 1, 150],
    "L_Middle2": [100, 10, 1, 150],
    "L_Middle3": [100, 10, 1, 150],
    "L_Pinky1": [100, 10, 1, 150],
    "L_Pinky2": [100, 10, 1, 150],
    "L_Pinky3": [100, 10, 1, 150],
    "L_Ring1": [100, 10, 1, 150],
    "L_Ring2": [100, 10, 1, 150],
    "L_Ring3": [100, 10, 1, 150],
    "L_Thumb1": [100, 10, 1, 150],
    "L_Thumb2": [100, 10, 1, 150],
    "L_Thumb3": [100, 10, 1, 150],
    "R_Index1": [100, 10, 1, 150],
    "R_Index2": [100, 10, 1, 150],
    "R_Index3": [100, 10, 1, 150],
    "R_Middle1": [100, 10, 1, 150],
    "R_Middle2": [100, 10, 1, 150],
    "R_Middle3": [100, 10, 1, 150],
    "R_Pinky1": [100, 10, 1, 150],
    "R_Pinky2": [100, 10, 1, 150],
    "R_Pinky3": [100, 10, 1, 150],
    "R_Ring1": [100, 10, 1, 150],
    "R_Ring2": [100, 10, 1, 150],
    "R_Ring3": [100, 10, 1, 150],
    "R_Thumb1": [100, 10, 1, 150],
    "R_Thumb2": [100, 10, 1, 150],
    "R_Thumb3": [100, 10, 1, 150],
}


class Bone:

    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []  # bvh only
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []

        # asf specific
        self.dir = np.zeros(3)
        self.len = 0
        # bvh specific
        self.offset = np.zeros(3)

        # inferred info
        self.pos = np.zeros(3)
        self.end = np.zeros(3)


class Skeleton:

    def __init__(self, smpl_model = "smpl"):
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.dof_name = ["x", "y", "z"]
        self.root = None
        self.exclude_contacts = []
        self.smpl_model = smpl_model

    def forward_bvh(self, bone):
        bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)

    def load_from_offsets(
        self,
        offsets,
        parents,
        scale,
        jrange,
        hull_dict,
        exclude_bones=None,
        channels=None,
        spec_channels=None,
        upright_start=False,
        remove_toe=False,
        freeze_hand= False,
        real_weight_porpotion_capsules=False,
        real_weight_porpotion_boxes = False, 
        real_weight=False,
        big_ankle=False,
        box_body = False, 
        sim='mujoco', 
        ball_joints = False, 
        create_vel_sensors = False, 
        exclude_contacts = []
    ):
        if channels is None:
            channels = ["x", "y", "z"]
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()
        self.exclude_contacts = exclude_contacts
        self.hull_dict = hull_dict
        self.upright_start = upright_start
        self.remove_toe = remove_toe
        self.real_weight_porpotion_capsules = real_weight_porpotion_capsules
        self.real_weight_porpotion_boxes = real_weight_porpotion_boxes
        self.real_weight = real_weight
        self.big_ankle = big_ankle
        self.freeze_hand = freeze_hand
        self.box_body = box_body
        self.ball_joints = ball_joints
        self.sim = sim
        joint_names = list(filter(lambda x: all([t not in x for t in exclude_bones]), offsets.keys()))
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = scale
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = channels
        self.name2bone[self.root.name] = self.root
        self.root.offset = offsets[self.root.name]
        self.bones.append(self.root)
        self.create_vel_sensors = create_vel_sensors
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint

            bone.channels = (spec_channels[joint] if joint in spec_channels.keys() else channels)
            bone.dof_index = [dof_ind[x] for x in bone.channels]
            bone.offset = np.array(offsets[joint]) * self.len_scale
            bone.lb = np.rad2deg(jrange[joint][:, 0])
            bone.ub = np.rad2deg(jrange[joint][:, 1])

            self.bones.append(bone)
            self.name2bone[joint] = bone
        for bone in self.bones[1:]:
            parent_name = parents[bone.name]
            # print(parent_name)
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p
        self.forward_bvh(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.end = bone.pos.copy() + 0.002
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)
                
    def construct_tree(self,
            template_fname=files('smpl_sim').joinpath('data/assets/mjcf/humanoid_template_local.xml'),
            offset=np.array([0, 0, 0]),
            ref_angles=None,
            bump_buffer=False):
        
        if ref_angles is None:
            ref_angles = {}
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")
        self.size_buffer = {}
        self.write_xml_bodynode(self.root, worldbody, offset, ref_angles)

        # create actuators
        actuators = tree.getroot().find("actuator")
        joints = worldbody.findall(".//joint")

        if self.ball_joints:
            for bone, joint in zip(self.bones, joints):
                for ind, channel in enumerate(bone.channels):
                    name = joint.attrib["name"]
                    attr = dict()
                    axis = bone.orient[:, ind]
                    attr["name"] = name + "_" + channel
                    attr["joint"] = name
                    attr["gear"] = "{0:.4f} {1:.4f} {2:.4f}".format(*axis)
                    SubElement(actuators, "motor", attr)
        else:
            for joint in joints[:]:
                name = joint.attrib["name"]
                attr = dict()
                attr["name"] = name
                attr["joint"] = name
                if self.sim in ["mujoco"]:
                    attr["gear"] = str(GAINS_MJ[name[:-2]][2])
                elif self.sim in ["isaacgym"]:
                    attr["gear"] = "500"
                SubElement(actuators, "motor", attr)
            
            
        if bump_buffer:
            SubElement(tree.getroot(), "size", {"njmax": "700", "nconmax": "700"})
            
        c_node = tree.getroot().find("contact")
        for bname1, bname2 in self.exclude_contacts:
            attr = {"body1": bname1, "body2": bname2}
            SubElement(c_node, "exclude", attr)
            
        s_node = tree.getroot().find("sensor")
        if self.create_vel_sensors:
            self.add_vel_sensors(self.root, s_node, "framelinvel")
            self.add_vel_sensors(self.root, s_node, "frameangvel")
            
        return tree
    
    def add_vel_sensors(self, bone, sensor_node, sensor_type = "framelinvel"):
        SubElement(sensor_node, sensor_type, {"name": "sensor_" + bone.name + f"_{sensor_type}", "objtype": 'xbody', "objname": bone.name})
        if bone.child is None:
            pass
        else:
            for bone_c in bone.child:
                self.add_vel_sensors(bone_c, sensor_node, sensor_type)
        
        

    def write_xml(
            self,
            fname,
            template_fname=files('smpl_sim').joinpath('data/assets/mjcf/humanoid_template_local.xml'),
            offset=np.array([0, 0, 0]),
            ref_angles=None,
            bump_buffer=False,
    ):
        tree = self.construct_tree(template_fname, offset, ref_angles, bump_buffer)
        
        tree.write(fname, pretty_print=True)

    def write_str(
            self,
            template_fname=files('smpl_sim').joinpath('data/assets/mjcf/humanoid_template_local.xml'),
            offset=np.array([0, 0, 0]),
            ref_angles=None,
            bump_buffer=False,
    ):
        tree = self.construct_tree(template_fname, offset, ref_angles, bump_buffer)

        return etree.tostring(tree, pretty_print=False)

    def write_xml_bodynode(self, bone, parent_node, offset, ref_angles):
        attr = dict()
        attr["name"] = bone.name
        attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
        node = SubElement(parent_node, "body", attr)

        # SubElement(node, "site", {"name": bone.name, "size": "0.01"}) # Writing site
        
        # write joints
        if bone.parent is None:
            j_attr = dict()
            j_attr["name"] = bone.name
            SubElement(node, "freejoint", j_attr)
        else:
            if self.ball_joints:
                j_attr = dict()
                j_attr["name"] = bone.name 
                j_attr["type"] = "ball"
                j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
                j_attr["user"] = " ".join([ str(s) for s in GAINS_MJ[bone.name]]) # using user to set the max torque
                
                if j_attr["name"] in ref_angles.keys():
                    j_attr["ref"] = f"{ref_angles[j_attr['name']]:.1f}"
                SubElement(node, "joint", j_attr)
            else:
                
                for i in range(len(bone.dof_index)):
                    ind = bone.dof_index[i]
                    axis = bone.orient[:, ind]
                    j_attr = dict()
                    j_attr["name"] = bone.name + "_" + self.dof_name[ind]
                    j_attr["type"] = "hinge"
                    j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
                    j_attr["axis"] = "{0:.4f} {1:.4f} {2:.4f}".format(*axis)
                    
                    if self.sim in ["mujoco"]:
                        j_attr["user"] = " ".join([ str(s) for s in GAINS_MJ[bone.name]]) # using user to set the max torque
                        j_attr["armature"] = "0.01"
                    elif self.sim in ["isaacgym"]:
                        j_attr["stiffness"] = str(GAINS_PHC[bone.name][0])
                        j_attr["damping"] = str(GAINS_PHC[bone.name][1])
                        j_attr["armature"] = "0.02"
                        

                    if i < len(bone.lb):
                        j_attr["range"] = "{0:.4f} {1:.4f}".format(bone.lb[i], bone.ub[i])
                    else:
                        j_attr["range"] = "-180.0 180.0"
                    if j_attr["name"] in ref_angles.keys():
                        j_attr["ref"] = f"{ref_angles[j_attr['name']]:.1f}"

                    SubElement(node, "joint", j_attr)
                
                

        # write geometry
        g_attr = dict()
        
        if not self.freeze_hand:
            GEOM_TYPES['L_Hand'] = 'box'
            GEOM_TYPES['R_Hand'] = 'box'
        
        if self.box_body:
            GEOM_TYPES['Head'] = 'box'
            GEOM_TYPES['Pelvis'] = 'box'
            
        if self.smpl_model == "smplx":
            GEOM_TYPES['L_Wrist'] = 'box' 
            GEOM_TYPES['R_Wrist'] = 'box'
        
        g_attr["type"] = GEOM_TYPES[bone.name]
        g_attr["contype"] = "1"
        g_attr["conaffinity"] = "1"
        if self.real_weight:
            base_density = 1000
        else: 
            base_density = 500
        g_attr["density"] = str(base_density)
        e1 = np.zeros(3)
        e2 = bone.end.copy() + offset
        if bone.name in ["Torso", "Chest", "Spine"]:
            seperation = 0.45
        else:
            seperation = 0.2

        # if bone.name in ["L_Hip"]:
        #     seperation = 0.3

        e1 += e2 * seperation
        e2 -= e2 * seperation
        hull_params = self.hull_dict[bone.name]

        if g_attr["type"] == "capsule":
            g_attr["fromto"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(*np.concatenate([e1, e2]))

            side_len = np.linalg.norm(e2 - e1)
            # radius = 0.067
            # V = np.pi * radius ** 2 * ((4/3) * radius + side_len)

            roots = np.polynomial.polynomial.Polynomial([-hull_params['volume'], 0, side_len * np.pi, 4 / 3 * np.pi]).roots()
            real_valued = roots.real[abs(roots.imag) < 1e-5]
            real_valued = real_valued[real_valued > 0]
            if bone.name in ["Torso", "Spine", "L_Hip", "R_Hip"]:
                real_valued *= 0.7  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:  # If shift is enabled, shift the weight based on teh shrinkage factor
                    g_attr["density"] = str((1 / 0.7**2) * base_density)

            if bone.name in ["Chest"]:
                real_valued *= 0.7  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:
                    g_attr["density"] = str((1 / 0.7**2) * base_density)

            if bone.name in ["L_Knee", 'R_Knee']:
                real_valued *= 0.9  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:
                    g_attr["density"] = str((1 / 0.9**2) * base_density)

            # if bone.name in ["Spine"]:
            # real_valued *= 0.01 # ZL Hack: shrinkage

            # g_attr["size"] = "{0:.4f}".format(*template_attributes["size"])
            g_attr["size"] = "{0:.4f}".format(*real_valued)

        elif g_attr["type"] == "box":
            pos = (e1 + e2) / 2
            min_verts = hull_params['norm_verts'].min(axis=0).values
            size = (hull_params['norm_verts'].max(axis=0).values - min_verts).numpy()
            if self.upright_start:
                if bone.name == "L_Toe" or bone.name == "R_Toe":
                    size[0] = hull_params['volume'] / (size[2] * size[0])
                else:
                    size[2] = hull_params['volume'] / (size[1] * size[0])
            else:
                size[1] = hull_params['volume'] / (size[2] * size[0])
            size /= 2

            if bone.name == "L_Toe" or bone.name == "R_Toe":
                if self.upright_start:
                    pos[2] = -bone.pos[2] / 2 - self.size_buffer[bone.parent.name][2] + size[2]  # To get toe to be at the same height as the parent
                    pos[1] = -bone.pos[1] / 2  # To get toe to be at the same x as the parent
                else:
                    pos[1] = -bone.pos[1] / 2 - self.size_buffer[bone.parent.name][1] + size[1]  # To get toe to be at the same height as the parent
                    pos[0] = -bone.pos[0] / 2  # To get toe to be at the same x as the parent

                if self.remove_toe:
                    size /= 20  # Smaller toes...
                    pos[1] = 0
                    pos[0] = 0
            bone_dir = bone.end / np.linalg.norm(bone.end)
            if not self.remove_toe:
                rot = np.array([1, 0, 0, 0])
            else:
                rot = sRot.from_euler("xyz", [0, 0, np.arctan(bone_dir[1] / bone_dir[0])]).as_quat()[[3, 0, 1, 2]]

            if self.big_ankle:
                # Big ankle override
                g_attr = {}
                hull_params = self.hull_dict[bone.name]
                min_verts, max_verts = hull_params['norm_verts'].min(axis=0).values, hull_params['norm_verts'].max(axis=0).values
                size = max_verts - min_verts

                bone_end = bone.end
                pos = (max_verts + min_verts) / 2
                size /= 2

                if bone.name == "L_Toe" or bone.name == "R_Toe":
                    parent_min, parent_max = self.hull_dict[bone.parent.name]['norm_verts'].min(axis=0).values, self.hull_dict[bone.parent.name]['norm_verts'].max(axis=0).values
                    parent_pos = (parent_max + parent_min) / 2
                    if self.upright_start:
                        pos[2] = parent_min[2] - bone.pos[2] + size[2]  # To get toe to be at the same height as the parent
                        pos[1] = parent_pos[1] - bone.pos[1]  # To get toe to be at the y as the parent
                    else:
                        pos[1] = parent_min[1] - bone.pos[1] + size[1]  # To get toe to be at the same height as the parent
                        pos[0] = parent_pos[0] - bone.pos[0]  # To get toe to be at the y as the parent
                        
                rot = np.array([1, 0, 0, 0])
                
                g_attr["type"] = "box"

            if bone.name == "Pelvis":
                size /= 1.75  # ZL Hack: shrinkage
                
            if bone.name == "Head":
                if self.upright_start:
                    size[0] /= 1.5  # ZL Hack: shrinkage
                    size[1] /= 1.5  # ZL Hack: shrinkage
                else:
                    size[0] /= 1.5  # ZL Hack: shrinkage
                    size[2] /= 1.5  # ZL Hack: shrinkage
            if self.smpl_model == "smplx" and (bone.name == "L_Wrist" or bone.name == "R_Wrist"):
                if self.upright_start:
                    size[0] /= 1.15  # ZL Hack: shrinkage
                    size[1] /= 1.3  # ZL Hack: shrinkage
                    size[2] /= 1.7  # ZL Hack: shrinkage
                else:
                    size[0] /= 1.15  # ZL Hack: shrinkage
                    size[1] /= 1.3  # ZL Hack: shrinkage
                    size[1] /= 1.7  # ZL Hack: shrinkage
                
                
            if self.real_weight_porpotion_boxes:
                g_attr["density"] = str((hull_params['volume'] / (size[0] * size[1] * size[2] * 8).item()) * base_density)
            
            g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*pos)
            g_attr["size"] = "{0:.4f} {1:.4f} {2:.4f}".format(*size)
            g_attr["quat"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(*rot)
            self.size_buffer[bone.name] = size

        elif g_attr["type"] == "sphere":
            pos = np.zeros(3)
            radius = np.cbrt(hull_params['volume'] * 3 / (4 * np.pi))
            if bone.name in ["Pelvis"]:
                radius *= 0.6  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:
                    g_attr["density"] = str((1 / 0.6**3) * base_density)

            g_attr["size"] = "{0:.4f}".format(radius)
            # g_attr["size"] = "{0:.4f}".format(*template_attributes["size"])
            g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*pos)
        g_attr['name'] = bone.name
        SubElement(node, "geom", g_attr)

        # write child bones
        for bone_c in bone.child:
            self.write_xml_bodynode(bone_c, node, offset, ref_angles)
