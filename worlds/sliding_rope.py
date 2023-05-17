import math
import time

from yarok.comm.components.cam.cam import Cam

from worlds.shared.cross_spawn import parallel_run
from worlds.shared.memory import Memory
from worlds.shared.robotbody import RobotBody
from .components.geltip.geltip import GelTip
from .components.tumble_tower.tumble_tower import TumbleTower

from yarok import Platform, Injector, component, ConfigBlock
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

START_POS = [0.8, 0.01, 0.2]
END_POS = [0.6, 0.2, 0.2]


class PushingTowerBehaviour:

    def __init__(self, injector: Injector, config: ConfigBlock):
        self.body = RobotBody(injector)
        self.body.arm.set_ws([
            [- pi, pi],  # shoulder pan
            [- pi, 0],  # shoulder lift,
            [- 2 * pi, 2 * pi],  # elbow
            [-2 * pi, 2 * pi],  # wrist 1
            [- 2 * pi, 2 * pi],  # wrist 2
            [- 2 * pi, 2 * pi]  # wrist 3
        ])
        self.pl: Platform = injector.get(Platform)
        self.config = config
        self.memory = Memory('sliding_rope', self.body, self.config, skip_right_sensor=True)
        self.a = config['a']
        self.dz = config['dz']
        self.FRONT = [0.05, 1.5, pi / 2 + pi/10]
        self.start_t = None

    def on_start(self):
        self.pl.wait(
            self.body.arm.move_xyz(xyz=[0.75, 0.3, 0.2], xyz_angles=self.FRONT)
        )
        self.start_t = time.time()
        self.memory.prepare()

    def on_update(self):
        self.memory.save()

        delta = time.time() - self.start_t
        if self.memory.i > 100:
            return False

        self.body.arm.move_xyz(xyz=[0.75 - 0.01 * delta,
                                    0.3,
                                    0.2 + 0.001 * self.dz + (1 / 5) * 0.01 * self.a * math.cos(0.01 * delta)],
                               xyz_angles=self.FRONT)

        self.pl.wait_seconds(0.09)


def launch_world(**kwargs):
    Platform.create({
        'world': BlocksTowerTestWorld,
        'behaviour': PushingTowerBehaviour,
        'defaults': {
            'behaviour': kwargs,
            'plugins': [
            ]
        },
        'components': {

        }
    }).run()


if __name__ == '__main__':
    parallel_run(launch_world,
                 {
                     'dz': [-2, 3],
                     'do': [-2, 3],
                     'a': [0, 5],
                 },
                 parallel=8)
