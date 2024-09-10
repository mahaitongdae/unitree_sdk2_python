import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

import unitree_legged_const as go2
DOG_NAME = 'enxa0cec86c58dc'

def LowStateHandler(msg: LowState_):
    
    # print front right hip motor states
    print("FR_0 motor state: ", msg.motor_state[go2.LegID["RL_0"]].q)
    print("imu state: ", msg.imu_state)
    # print("IMU state: ", msg.imu_state)
    # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)
    # print(getattr(msg))


if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, "enxa0cec86c58dc")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(LowStateHandler, 10)

    while True:
        time.sleep(10.0)
