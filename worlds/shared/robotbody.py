from yarok import Injector
from yarok.comm.components.cam.cam import Cam
from yarok.comm.components.robotiq_2f85.robotiq_2f85 import Robotiq2f85
from yarok.comm.components.ur5e.ur5e import UR5e

from worlds.components.geltip.geltip import GelTip


class RobotBody:

    def __init__(self, injector: Injector):
        self.cam: Cam = injector.get('cam')
        self.arm: UR5e = injector.get('arm')
        self.gripper: Robotiq2f85 = injector.get('gripper')
        self.left_geltip: GelTip = injector.get('left_geltip')
        self.right_geltip: GelTip = injector.get('right_geltip')
