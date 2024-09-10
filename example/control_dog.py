import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_
from collections import deque
import keyboard
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
LegID = {
    "FR_0": 0,  # Front right hip
    "FR_1": 1,  # Front right thigh
    "FR_2": 2,  # Front right calf
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0

DOG_NAME = 'enxa0cec86c58dc'

def quat_rotate_inverse(q, v):
    shape = q.shape
    #q_w = q[:, -1]
    #q_vec = q[:, :3]
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class ObsHandler(object):
    
    def __init__(self) -> None:
        self.highStateSub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        rsc = RobotStateClient()
        rsc.SetTimeout(3.0)
        rsc.Init()
        self.velocity = deque(maxlen=5)
        self.lowStateQueue = deque(maxlen=5)
        self.highStateSub.Init(self.HighStateHandler, 10)

    def HighStateHandler(self, msg: SportModeState_):
        self.velocity.append(msg.velocity)

    def LowStateHandler(self, msg: LowState_):
        self.lowStateQueue.append(msg)

    def getStateFromLowLevelMsg(self, msg: LowState_):
            ang_vel = msg.imu_state.gyroscope
            quat = msg.imu_state.quaternion
            joint_angle = [msg.motor_state[id]['q'] for id in LegID]
            joint_vel = [msg.motor_state[id]['dq'] for id in LegID]

    def get_state(self, ):
        # base_lin_vel_b
        base_lin_vel = self.velocity[-1]



if __name__ == "__main__":

    
    