import time

from yarok.comm.components.cam.cam import Cam

from .components.geltip.geltip import GelTip
from .components.tumble_tower.tumble_tower import TumbleTower

from yarok import Platform, Injector, component
from yarok.comm.worlds.empty_world import EmptyWorld
from yarok.comm.components.ur5e.ur5e import UR5e
from yarok.comm.components.robotiq_2f85.robotiq_2f85 import Robotiq2f85

from math import pi

import os
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
            </asset>
            <worldbody>
                <body euler="1.57 0 0" pos="1.0 0.8 0.2">
                    <cam name="cam" />
                </body>
                
                <body name="rope" pos="0.08 0.11 0.28">
                    <composite type="grid" count="40 1 1" spacing="0.01" offset="1 0 0">
                        <joint kind="main" damping="0.001" frictionloss='0.0001'/>
                        <tendon kind="main" width="0.007" rgba=".8 .2 .1 1"/>
                        <geom size=".004" rgba=".8 .2 .1 1" mass='0.0001' 
                            friction='0.001 0.005 0.0001'/>
                        <pin coord="39"/> 
                    </composite>
                </body>

                <body euler="0 0 1.57" pos="0.5 -0.37 0">
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


DOWN = [3.11, -1.6e-7, -pi / 2]
# FRONT = [3.11, - pi / 2, pi / 2]
# FRONT = [pi / 2 , - pi / 2 , pi / 2 ]
FRONT = [0.05, 1.5, pi / 2]

START_POS = [0.8, 0.01, 0.2]
END_POS = [0.6, 0.2, 0.2]


class PushingTowerBehaviour:

    def __init__(self, injector: Injector):
        self.cam: Cam = injector.get('cam')
        self.arm: UR5e = injector.get('arm')
        self.gripper: Robotiq2f85 = injector.get('gripper')
        self.left_geltip: GelTip = injector.get('left_geltip')
        self.pl: Platform = injector.get(Platform)

        self.arm.set_ws([
            [- pi, pi],  # shoulder pan
            [- pi, 0],  # shoulder lift,
            [- 2 * pi, 2 * pi],  # elbow
            [-2 * pi, 2 * pi],  # wrist 1
            [- 2 * pi, 2 * pi],  # wrist 2
            [- 2 * pi, 2 * pi]  # wrist 3
        ])

        self.t = time.time()
        self.i = 0
        self.dataset_name = 'sliding_rope'
        self.save_dataset = True
        self.prepare_dataset_dirs()

    def prepare_frame(self, frame):
        frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = frame[:, 40:40 + 240, :]
        frame = cv2.resize(frame, (224, 224))
        return frame

    def prepare_dataset_dirs(self):
        if self.save_dataset:
            os.mkdir(f'data/{self.dataset_name}')
            os.mkdir(f'data/{self.dataset_name}/v')
            os.mkdir(f'data/{self.dataset_name}/lt')

    def save_frame(self):
        if self.save_dataset:
            cam_frame = self.prepare_frame(self.cam.read())
            left_touch_frame = self.prepare_frame(self.left_geltip.read())

            cv2.imwrite(f'data/{self.dataset_name}/v/frame_{str(self.i).zfill(5)}.jpg', cam_frame)
            cv2.imwrite(f'data/{self.dataset_name}/lt/frame_{str(self.i).zfill(5)}.jpg', left_touch_frame)
            self.i += 1

    def on_start(self):
        self.pl.wait(
            self.arm.move_xyz(xyz=[0.75, 0.3, 0.2], xyz_angles=FRONT)
        )
        self.t = time.time()

    def on_update(self):
        self.save_frame()

        delta = time.time() - self.t
        if self.i > 100:
            return False

        self.arm.move_xyz(xyz=[0.75 - 0.01 * delta, 0.3, 0.2], xyz_angles=FRONT)

        self.pl.wait_seconds(0.09)


if __name__ == '__main__':
    Platform.create({
        'world': BlocksTowerTestWorld,
        'behaviour': PushingTowerBehaviour,
        'defaults': {
            'plugins': [
            ]
        },
        'components': {

        }
    }).run()