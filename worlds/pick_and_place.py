import time

import yaml
from yarok.comm.components.cam.cam import Cam

from .components.geltip.geltip import GelTip
from .components.tumble_tower.tumble_tower import TumbleTower

from yarok import Platform, Injector, component
from yarok.comm.worlds.empty_world import EmptyWorld
from yarok.comm.components.ur5e.ur5e import UR5e
from yarok.comm.components.robotiq_2f85.robotiq_2f85 import Robotiq2f85

from math import pi
import os
import numpy as np

import cv2


@component(
    extends=EmptyWorld,
    components=[
        GelTip,
        TumbleTower,
        UR5e,
        Robotiq2f85,
        Cam
    ],
    template="""
        <mujoco>
            <asset>
                <texture type="skybox" 
                file="assets/robot_lab.png"
                rgb1="0.6 0.6 0.6" 
                rgb2="0 0 0"/>
                <texture 
                    name="white_wood_texture"
                    type="cube" 
                    file="assets/white_wood.png"
                    width="400" 
                    height="400"/>
                <material name="white_wood" texture="white_wood_texture" rgba="0.6 0.6 0.6 1" specular="0.1"/>
                <material name="gray_wood" texture="white_wood_texture" rgba="0.6 0.4 0.2 1" specular="0.1"/>
                <material name="red_wood" texture="white_wood_texture" rgba="0.8 0 0 1" specular="0.1"/>
                <material name="green_wood" texture="white_wood_texture" rgba="0 0.8 0 1" specular="0.1"/>
                <material name="yellow_wood" texture="white_wood_texture" rgba="0.8 0.8 0 1" specular="0.1"/>
            </asset>
            <worldbody>
                <body euler="1.57 -3.14 0" pos="0.25 -1.3 0.4">
                    <cam name="cam" />
                </body>
                
                <!-- blocks -->
                <body>
                    <freejoint/>
                    <geom 
                        type="box" 
                        size="0.03 0.03 0.03" 
                        pos="0.13 -0.135 0.131" 
                        mass="0.0001"
                        material="red_wood"
                        zaxis="0 1 0"/>
                </body>
                <body>
                    <freejoint/>
                    <geom 
                        type="box" 
                        size="0.03 0.03 0.03" 
                        pos="0.13 -0.135 0.191" 
                        mass="0.0001"
                        material="green_wood"
                        zaxis="0 1 0"/>
                </body>
                
                <body>
                    <freejoint/>
                    <geom 
                        type="box" 
                        size="0.03 0.03 0.03" 
                        pos="0.13 -0.135 0.221" 
                        mass="0.0001"
                        material="yellow_wood"
                        zaxis="0 1 0"/>
                </body>
                
               <body pos="0.3 0.11 0">
                    <geom type="box" pos="-0.45 0.29 0" size="0.1 0.1 0.3" material="gray_wood"/>
                    <geom type="box" pos="0 0 0" size="0.4 0.4 0.1" material="white_wood"/>
               </body>  
                
                <body euler="0 0 1.57" pos="-0.15 0.4 0.3">
                    <ur5e name='arm'>
                       <robotiq-2f85 name="gripper" parent="ee_link"> 
                          <body pos="0.02 -0.017 0.053" xyaxes="0 -1 0 1 0 0" parent="right_tip">
                                <geltip name="left_geltip" parent="left_tip"/>
                            </body>
                           <body pos="-0.02 -0.017 0.053" xyaxes="0 1 0 -1 0 0" parent="left_tip">
                                <geltip name="right_geltip" parent="right_tip"/>
                            </body>
                        </robotiq-2f85> 
                    </ur5e> 
                </body>
            </worldbody>
        </mujoco>
    """
)
class BlocksTowerTestWorld:
    pass


DOWN = [3.11, 1.6e-7, 3.11]

START_POS_UP = [0.3, -0.5, 0.21]
START_POS_DOWN = [0.3, -0.5, 0.11]
END_POS_UP = [0.6, -0.5, 0.21]
END_POS_DOWN = [0.6, -0.5, 0.115]

BLOCK_SIZE = 0.06


def z(pos, delta):
    new_pos = pos.copy()
    new_pos[2] += delta
    return new_pos


class GraspingCylinderBehaviour:

    def __init__(self, injector: Injector):
        self.cam: Cam = injector.get('cam')
        self.arm: UR5e = injector.get('arm')
        self.left_geltip: GelTip = injector.get('left_geltip')
        self.right_geltip: GelTip = injector.get('right_geltip')
        self.gripper: Robotiq2f85 = injector.get('gripper')

        self.arm.set_ws([
            [- pi, pi],  # shoulder pan
            [- pi, -pi / 2],  # shoulder lift,
            [- 2 * pi, 2 * pi],  # elbow
            [-2 * pi, 2 * pi],  # wrist 1
            [0, pi],  # wrist 2
            [- 2 * pi, 2 * pi]  # wrist 3
        ])
        self.arm.set_speed(pi / 24)

        self.pl: Platform = injector.get(Platform)
        self.t = time.time()
        self.i = 0
        self.save_dataset = True
        self.dataset_name = 'pick_and_place'
        self.prepare_dataset_dirs()
        self.ps = None

    def prepare_frame(self, frame):
        frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = frame[:, 40:40 + 240, :]
        frame = cv2.resize(frame, (224, 224))
        return frame

    def prepare_dataset_dirs(self):
        if self.save_dataset:
            os.mkdir(f'data/{self.dataset_name}')
            [os.mkdir(f'data/{self.dataset_name}/{d}') for d in ['c', 'l', 'r']]

    def save_frame(self):
        if self.save_dataset:
            cam_frame = self.prepare_frame(self.cam.read())
            left_touch_frame = self.prepare_frame(self.left_geltip.read())
            right_touch_frame = self.prepare_frame(self.right_geltip.read())

            q_xyz = self.arm.at_xyz()
            p = np.array(q_xyz[0].tolist() + q_xyz[1].tolist() + [self.gripper.at()])

            cv2.imwrite(f'data/{self.dataset_name}/c/frame_{str(self.i).zfill(5)}.jpg', cam_frame)
            cv2.imwrite(f'data/{self.dataset_name}/l/frame_{str(self.i).zfill(5)}.jpg', left_touch_frame)
            cv2.imwrite(f'data/{self.dataset_name}/r/frame_{str(self.i).zfill(5)}.jpg', right_touch_frame)

            self.ps = np.concatenate([self.ps, p.T], axis=0) if self.ps is not None else p.T
            with open(f'data/{self.dataset_name}/p.yaml', 'wb') as f:
                np.save(f, self.ps)
            # yaml.dump(self.ps, open(, 'w'))

            self.i += 1

    def wait(self, arm=None, gripper=None):
        def cb():
            self.save_frame()

            self.pl.wait_seconds(0.1)

            if arm is not None:
                return self.arm.is_at(arm)
            else:
                return self.gripper.is_at(gripper)

        self.pl.wait(cb)

    def on_start(self):
        def move_arm(p):
            q = self.arm.ik_xyz(xyz=p, xyz_angles=DOWN)
            self.arm.move_q(q)
            self.wait(arm=q)

        def move_gripper(q):
            self.gripper.close(q)
            self.wait(gripper=q)

        # set the arm in the initial position
        self.pl.wait(self.arm.move_xyz(xyz=START_POS_UP, xyz_angles=DOWN))

        # do the pick and place.
        for i in range(3):
            # before grasping.
            move_arm(START_POS_UP)

            # grasps block.
            move_arm(z(START_POS_DOWN, -i * BLOCK_SIZE))
            move_gripper(0.26)

            # moves.
            move_arm(START_POS_UP)
            move_arm(END_POS_UP)

            # places.
            move_arm(z(END_POS_DOWN, -(2 - i) * BLOCK_SIZE))
            move_gripper(0)

            # moves back
            move_arm(END_POS_UP)
            move_arm(START_POS_UP)


if __name__ == '__main__':
    Platform.create({
        'world': BlocksTowerTestWorld,
        'behaviour': GraspingCylinderBehaviour,
        'defaults': {
            'plugins': [
            ],
            'components': {
                '/gripper': {
                    'left_tip': False,
                    'right_tip': False
                },
                '/left_geltip': {
                    'cubic_core': True,
                    'label_color': '0 255 0'
                },
                '/right_geltip': {
                    'cubic_core': True,
                    'label_color': '255 0 0'
                }
            }
        },

    }).run()
