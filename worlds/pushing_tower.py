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
                <body euler="1.57 -1.57 0" pos="2.1 0.5 0.4">
                    <cam name="cam" />
                </body>
                
                <tumble-tower name="tower"/>

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


FRONT = [3.11, - pi / 2, pi / 2]


class PushingTowerBehaviour:

    def __init__(self, injector: Injector):
        self.cam: Cam = injector.get('cam')
        self.arm: UR5e = injector.get('arm')
        self.arm.set_ws([
            [- pi, pi],  # shoulder pan
            [- pi, 0],  # shoulder lift,
            [- 2 * pi, 2 * pi],  # elbow
            [-2 * pi, 2 * pi],  # wrist 1
            [- 2 * pi, 2 * pi],  # wrist 2
            [- 2 * pi, 2 * pi]  # wrist 3
        ])
        self.gripper: Robotiq2f85 = injector.get('gripper')
        self.left_geltip: GelTip = injector.get('left_geltip')
        self.right_geltip: GelTip = injector.get('right_geltip')
        self.pl: Platform = injector.get(Platform)
        self.t = time.time()
        self.i = 0
        self.save_dataset = True
        self.dataset_name = 'pushing_tower'

    def prepare_frame(self, frame):
        frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = frame[:, 40:40 + 240, :]
        frame = cv2.resize(frame, (224, 224))
        return frame

    def prepare_dataset_dirs(self):
        if self.save_dataset:
            os.mkdir(f'data/{self.dataset_name}')
            os.mkdir(f'data/{self.dataset_name}/c')
            os.mkdir(f'data/{self.dataset_name}/l')
            os.mkdir(f'data/{self.dataset_name}/r')

    def save_frame(self):
        if self.save_dataset:
            cam_frame = self.prepare_frame(self.cam.read())
            left_touch_frame = self.prepare_frame(self.left_geltip.read())
            right_touch_frame = self.prepare_frame(self.right_geltip.read())

            cv2.imwrite(f'data/{self.dataset_name}/c/frame_{str(self.i).zfill(5)}.jpg', cam_frame)
            cv2.imwrite(f'data/{self.dataset_name}/l/frame_{str(self.i).zfill(5)}.jpg', left_touch_frame)
            cv2.imwrite(f'data/{self.dataset_name}/r/frame_{str(self.i).zfill(5)}.jpg', right_touch_frame)
            self.i += 1

    def on_start(self):
        self.prepare_dataset_dirs()
        self.pl.wait(
            self.arm.move_xyz(xyz=[0.1, 0.4, 0.3], xyz_angles=FRONT)
        )

        self.t = time.time()
        self.i = 0

    def on_update(self):
        self.save_frame()

        if self.i > 130:
            return False

        self.arm.move_xyz(xyz=[0.1, 0.4 + 0.0025 * self.i, 0.3], xyz_angles=FRONT)

        self.pl.wait_seconds(0.1)


if __name__ == '__main__':
    Platform.create({
        'world': BlocksTowerTestWorld,
        'behaviour': PushingTowerBehaviour,
        'defaults': {
            'plugins': [
            ]
        }
    }).run()
