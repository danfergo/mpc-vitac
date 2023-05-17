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
        self.memory = Memory('pushing_tower', self.body, self.config, skip_right_sensor=False)

    def on_start(self):
        self.memory.prepare()
        self.pl.wait(
            self.body.arm.move_xyz(xyz=[0.1, 0.4, 0.3], xyz_angles=FRONT)
        )

    def on_update(self):
        self.memory.save()

        if self.memory.i > 130:
            return False

        self.body.arm.move_xyz(xyz=[0.1 + 0.01 * self.config['ox'],
                                    0.4 + 0.0025 * self.memory.i,
                                    0.3 + 0.01 * self.config['oz'] + 0.0025 * self.memory.i * + 0.001 * self.config[
                                        'dz']
                                    ], xyz_angles=FRONT)

        self.pl.wait_seconds(0.1)


def launch_world(**kwargs):
    Platform.create({
        'world': BlocksTowerTestWorld,
        'behaviour': PushingTowerBehaviour,
        'defaults': {
            'behaviour': kwargs,
            'plugins': [
            ]
        }
    }).run()


if __name__ == '__main__':
    parallel_run(launch_world,
                 {
                     'oz': [0, 6],
                     'dz': [-2, 3],
                     'ox': [-2, 3]
                 },
                 parallel=5)
