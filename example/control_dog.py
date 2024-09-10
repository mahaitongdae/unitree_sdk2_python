import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_
from collections import deque
import keyboard
import onnxruntime as ort
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
import numpy as np
from unitree_sdk2py.utils.crc import CRC
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
onnx_model_path = './model/unitree_go2_flat/policy.onnx'
HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0



DOG_NAME = 'enxa0cec86c58dc'
crc = CRC()

JOINT_LIMIT = np.array([       # Hip, Thigh, Calf
        [-1.047,    -0.663,      -2.9],  # MIN
        [1.047,     2.966,       -0.837]  # MAX
    ])

def quaternion_inverse_rotate (quaternion, vector):
    # quaternion is (w, x, y, z)
    # vector is (vx, vy, vz)
    w, x, y, z = quaternion
    vx, vy, vz = vector

    # Inverse of quaternion
    q_inv = np.array([w, -x, -y, -z])

    # Quaternion multiplication function
    def quaternion_multiply (q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        ])

    # Convert vector to quaternion form (0, vx, vy, vz)
    v_quat = np.array([0, vx, vy, vz])

    # First compute q_inv * v_quat
    temp_result = quaternion_multiply(q_inv, v_quat)

    # Now compute the result of (q_inv * v_quat) * q
    rotated_quat = quaternion_multiply(temp_result, quaternion)

    # The rotated vector is the imaginary part (x, y, z) of the resulting quaternion
    return rotated_quat[1:]  # Return (x, y, z) part

class ObsHandler(object):
    
    def __init__(self) -> None:
        self.highStateSub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        rsc = RobotStateClient()
        rsc.SetTimeout(3.0)
        rsc.Init()
        self.velocity = deque(maxlen=5)
        self.lowStateQueue = deque(maxlen=5)

        self.last_action = [0.0] * 12
        self.offsetIsaac = np.array([0.1, -0.1, 0.1, -0.1,
                                     0.8, 0.8, 1.0, 1.0,
                                     -1.5, -1.5, -1.5, -1.5])
        self.session = ort.InferenceSession(onnx_model_path)
        # Get the model's input name
        self.input_name = self.session.get_inputs()[0].name
        self.offsetGo2 = self.convertJointOrderIsaacToGo2(self.offsetIsaac)
        self.scale = 0.25
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
        return ang_vel, quat, joint_angle, joint_vel

    def convertJointOrderIsaacToGo2(self, IsaacGymJoint):
        '''
        isaac gym: flhip, frhip, rlhip, rrhip,
                   flthigh, frthigh, rlthigh, rrthigh,
                   fl, fr, rl, rr calf
        go2: 0 for hip, 1 for thigh, 2 for calf
        '''
        Go2Joint = np.empty_like(IsaacGymJoint)
        Go2Joint[LegID["FL_0"]] = IsaacGymJoint[0]
        Go2Joint[LegID["FR_0"]] = IsaacGymJoint[1]
        Go2Joint[LegID["RL_0"]] = IsaacGymJoint[2]
        Go2Joint[LegID["RR_0"]] = IsaacGymJoint[3]
        Go2Joint[LegID["FL_1"]] = IsaacGymJoint[4]
        Go2Joint[LegID["FR_1"]] = IsaacGymJoint[5]
        Go2Joint[LegID["RL_1"]] = IsaacGymJoint[6]
        Go2Joint[LegID["RR_1"]] = IsaacGymJoint[7]
        Go2Joint[LegID["FL_2"]] = IsaacGymJoint[8]
        Go2Joint[LegID["FR_2"]] = IsaacGymJoint[9]
        Go2Joint[LegID["RL_2"]] = IsaacGymJoint[10]
        Go2Joint[LegID["RR_2"]] = IsaacGymJoint[11]
        return Go2Joint

    def convertJointOrderGo2ToIsaac(self, Go2Joint):
        '''
        isaac gym: flhip, frhip, rlhip, rrhip,
                   flthigh, frthigh, rlthigh, rrthigh,
                   fl, fr, rl, rr calf
        go2: 0 for hip, 1 for thigh, 2 for calf
        '''
        IsaacGymJoint = np.empty_like(Go2Joint)
        IsaacGymJoint[0] = Go2Joint[LegID["FL_0"]]
        IsaacGymJoint[1] = Go2Joint[LegID["FR_0"]]

        IsaacGymJoint[2] = Go2Joint[LegID["RL_0"]]
        IsaacGymJoint[3] = Go2Joint[LegID["RR_0"]]
        IsaacGymJoint[4] = Go2Joint[LegID["FL_1"]]
        IsaacGymJoint[5] = Go2Joint[LegID["FR_1"]]
        IsaacGymJoint[6] = Go2Joint[LegID["RL_1"]]
        IsaacGymJoint[7] = Go2Joint[LegID["RR_1"]]
        IsaacGymJoint[8] = Go2Joint[LegID["FL_2"]]
        IsaacGymJoint[9] = Go2Joint[LegID["FR_2"]]
        IsaacGymJoint[10] = Go2Joint[LegID["RL_2"]]
        IsaacGymJoint[11] = Go2Joint[LegID["RR_2"]]
        return IsaacGymJoint

    def get_state(self, velo_command = np.array([0.5, 0.0, 0.0])):
        # base_lin_vel_b
        base_lin_vel_w = self.velocity[-1]
        ang_vel_w, quat, joint_angle, joint_vel = self.getStateFromLowLevelMsg(self.lowStateQueue[-1])
        obs = np.empty([1, 48])
        obs[:, :3] = quaternion_inverse_rotate(quat, base_lin_vel_w)
        obs[:, 3:6] = quaternion_inverse_rotate(quat, ang_vel_w)
        obs[:, 6:9] = quaternion_inverse_rotate(quat, np.array([0.0, 0.0, -1.0]))
        obs[:, 9:12] = velo_command
        obs[:, 12:24] = self.convertJointOrderGo2ToIsaac(joint_angle)
        obs[:, 24:36] = self.convertJointOrderGo2ToIsaac(joint_vel)
        obs[:, 36:48] = self.last_action
        return obs

    def convert_action(self, action):
        '''
        action is in IsaacGym, covert Isaac to Go2
        '''
        Go2Action = self.scale * self.convertJointOrderIsaacToGo2(action) + self.offsetGo2

    def onnx_inference(self, input):
        if len(input.shape) == 0:
            input = np.expand_dims(input, axis=0)
        return self.session.run(None, {self.input_name: input})[0][0]

    def get_action(self):
        state = self.get_state()
        output = self.onnx_inference(state)
        action = self.convert_action(output)
        return action

class Controller(object):
    STAND = np.array([
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171
    ])
    def __init__(self):
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.cmd.motor_cmd[i].q = PosStopF
            self.cmd.motor_cmd[i].kp = 0
            self.cmd.motor_cmd[i].dq = VelStopF
            self.cmd.motor_cmd[i].kd = 0
            self.cmd.motor_cmd[i].tau = 0

    def control(self, action):

        while True:

            # Poinstion(rad) control, set RL_0 rad
            for i, joint_pos in enumerate(action):
                self.cmd.motor_cmd[i].q = joint_pos  # Taregt angular(rad)
                self.cmd.motor_cmd[i].kp = 10.0  # Poinstion(rad) control kp gain
                self.cmd.motor_cmd[i].dq = 0.0  # Taregt angular velocity(rad/ss)
                self.cmd.motor_cmd[i].kd = 0.5  # Poinstion(rad) control kd gain
                self.cmd.motor_cmd[i].tau = 0.0  # Feedforward toque 1N.m

            self.cmd.crc = crc.Crc(self.cmd)

            # Publish message
            if self.pub.Write(self.cmd):
                print("Publish success. msg:", self.cmd.crc)
                break
            else:
                print("Waitting for subscriber.")

            time.sleep(0.002)

    def stand(self, action):

        while True:

            # Poinstion(rad) control, set RL_0 rad
            for i, joint_pos in enumerate(action):
                self.cmd.motor_cmd[i].q = joint_pos  # Taregt angular(rad)
                self.cmd.motor_cmd[i].kp = 10.0  # Poinstion(rad) control kp gain
                self.cmd.motor_cmd[i].dq = 0.0  # Taregt angular velocity(rad/ss)
                self.cmd.motor_cmd[i].kd = 0.5  # Poinstion(rad) control kd gain
                self.cmd.motor_cmd[i].tau = 0.0  # Feedforward toque 1N.m

            self.cmd.crc = crc.Crc(self.cmd)

            # Publish message
            if self.pub.Write(self.cmd):
                print("Publish success. msg:", self.cmd.crc)
                break
            else:
                print("Waitting for subscriber.")

            time.sleep(0.002)

    def soft_emergency_stop(self):

        while True:

            # Poinstion(rad) control, set RL_0 rad
            for i, joint_pos in enumerate(LegID.keys()):
                self.cmd.motor_cmd[i].q = 0.0  # Taregt angular(rad)
                self.cmd.motor_cmd[i].kp = 0.0  # Poinstion(rad) control kp gain
                self.cmd.motor_cmd[i].dq = 0.0  # Taregt angular velocity(rad/ss)
                self.cmd.motor_cmd[i].kd = 1.0  # Poinstion(rad) control kd gain
                self.cmd.motor_cmd[i].tau = 0.0  # Feedforward toque 1N.m

            self.cmd.crc = crc.Crc(self.cmd)

            # Publish message
            if self.pub.Write(self.cmd):
                print("Emergency stop success. msg:", self.cmd.crc)
                break
            else:
                print("Waitting for subscriber.")

            time.sleep(0.002)

if __name__ == "__main__":
    ChannelFactoryInitialize(0, DOG_NAME)
    obs_handle = ObsHandler()
    controller = Controller()
    while True:
        if keyboard.is_pressed('enter'):
            controller.soft_emergency_stop()
        else:
            action = obs_handle.get_action()
            controller.control(controller.STAND)
        time.sleep(0.005)  # Adding a small delay to avoid high CPU usage


    
    