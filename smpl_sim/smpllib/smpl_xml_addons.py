
from lxml import etree

def smpl_change_world(xml: str):
    tree = etree.XML(xml)
    texplane = tree.xpath("//texture[@name='texplane']")[0]
    newtexplane = etree.fromstring('<texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="100" height="100"></texture>')
    texplane.addnext(newtexplane)
    texplane.getparent().remove(texplane)
    matplane = tree.xpath("//material[@name='MatPlane']")[0]
    matplane.addnext(
    etree.fromstring('<material name="MatPlane" reflectance="0.5" texture="texplane" texrepeat="1 1" texuniform="true"></material>'))
    matplane.getparent().remove(matplane)
    return etree.tostring(tree)

def smpl_add_camera(xml: str):
    # tree = etree.parse("parsed_neutral.xml")
    tree = etree.XML(xml)

    light = tree.xpath("/mujoco/worldbody/light[1]")[0]
    new_light = etree.fromstring('<light name="tracking_light" pos="0 0 7" dir="0 0 -1" directional="true" cutoff="100" exponent="1" diffuse="1 1 1" specular="0.1 0.1 0.1" mode="trackcom"/>') 
    light.addnext(new_light)
    cam_back = etree.fromstring('<camera name="back" pos="0 3 2.4" xyaxes="-1 0 0 0 -1 2" mode="trackcom"/>')
    new_light.addnext(cam_back)
    cam_side = etree.fromstring('<camera name="side" pos="-3 0 2.4" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>')
    cam_back.addnext(cam_side)
    cam_side.addnext(etree.fromstring(
        '<camera name="front_side" pos="-2 -2 .8" xyaxes="0.5 -0.5 0 0.1 0.1 1"  mode="trackcom"/>'
    ))
    light.getparent().remove(light)

    # cam_head = etree.fromstring('<camera name="egocentric" pos="0 0 0" xyaxes="-1 0 0 0 1 0" fovy="80"/>')
    # head = tree.xpath("//joint[@name='Head_x']")[0]
    # head.addprevious(cam_head)
    

    # this is for dumping the file
    # parser = etree.XMLParser(remove_blank_text=True)
    # etree.ElementTree(etree.fromstring(etree.tostring(tree),
    #                                 parser=parser)).write("parsed_TTT.xml",
    #                                                         pretty_print=True)

    # this is to create a new camera for tracking but not working
    # check how to update the position
    # # setup camera
    # self.smpl_body_to_id = {}
    # for i in range(self.mj_model.nbody):
    #     body_name = self.mj_model.body(i).name
    #     self.smpl_body_to_id[body_name] = i
    # self.camera = mujoco.MjvCamera()
    # self.camera.trackbodyid = self.smpl_body_to_id["Pelvis"]
    # self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # mujoco.mjv_defaultCamera(self.camera)
    return etree.tostring(tree)

# def smpl_add_sensors(xml: str):
#     # TODO: verify
#     tree = etree.XML(xml)
#     site_pelvis = etree.fromstring('<site name="Pelvis" size=".01" rgba="0.5 0.5 0.5 0"/>')
#     site_lhand = etree.fromstring('<site name="lhand_touch" size=".012 0.005 0.015" pos="0 0 0" type="ellipsoid" type="sphere" size="0.01" group="4" rgba="1 0 0 .5"/>')
#     site_rhand = etree.fromstring('<site name="rhand_touch" size=".012 0.005 0.015" pos="0 0 0" type="ellipsoid" type="sphere" size="0.01" group="4" rgba="1 0 0 .5"/>')
#     site_rfoot = etree.fromstring('<site name="rfoot_touch" type="capsule" pos="0 0 0" size="0.025 0.01" zaxis="1 0 0" type="sphere" size="0.01" group="4" rgba="1 0 0 .5"/>')
#     site_lfoot = etree.fromstring('<site name="lfoot_touch" type="capsule" pos="0 0 0" size="0.025 0.01" zaxis="1 0 0" type="sphere" size="0.01" group="4" rgba="1 0 0 .5"/>')
#     pelvis = tree.xpath("//joint[@name='Pelvis]")[0]
#     lhand = tree.xpath("//joint[@name='L_Hand_x]")[0]
#     rhand = tree.xpath("//joint[@name='R_Hand_x]")[0]
#     lankle = tree.xpath("//joint[@name='L_Ankle_x]")[0]
#     rankle = tree.xpath("//joint[@name='R_Ankle_x]")[0]
#     pelvis.addnext(site_pelvis)
#     lhand.addnext(site_lhand)
#     rhand.addnext(site_rhand)
#     lankle.addnext(site_lfoot)
#     rankle.addnext(site_rfoot)
#     sensors = etree.XML(
#         """
#         <sensor>
#         <velocimeter name="sensor_Pelvis_veloc" site="Pelvis"/>
#         <gyro name="sensor_Pelvis_gyro" site="Pelvis"/>
#         <accelerometer name="sensor_Pelvis_accel" site="Pelvis"/>
#         <touch name="sensor_touch_lhand" site="lhand_touch"/>
#         <touch name="sensor_touch_rhand" site="rhand_touch"/>
#         <touch name="sensor_touch_rfoot" site="rfoot_touch"/>
#         <touch name="sensor_touch_lfoot" site="lfoot_touch"/>
#         </sensor>
#         """
#     )
#     actuators = tree.xpath("/mujoco/actuator")[0]
#     actuators.addnext(sensors)
#     return etree.tostring(tree)
    