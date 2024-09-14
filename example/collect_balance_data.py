import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_
from collections import deque
# import keyboard
import onnxruntime as ort
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
import numpy as np
from unitree_sdk2py.utils.crc import CRC
import select
import sys
import termios
import tty

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
onnx_model_path = './model/onnx/policy091213.onnx'
HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0

STAND = np.array([
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5
    ])

FREQ = 100
DT = 1./FREQ



DOG_NAME = 'eth0'
crc = CRC()

JOINT_LIMIT = np.array([       # Hip, Thigh, Calf
        [-1.047,    -0.663,      -2.9],  # MIN
        [1.047,     2.966,       -0.837]  # MAX
    ])

def emergency_stop(key): 
    if key == ' ':
        code = controller.rsc.ServiceSwitch("sport_mode", False)
        if code != 0:
            print("service stop sport_mode error. code:", code)
        else:
            print("service stop sport_mode success. code:", code)
        for _ in range(20):
            controller.soft_emergency_stop()
        print('exit')
        sys.exit(0)
    elif key == '1':
        sys.exit(0)

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

def convertJointOrderIsaacToGo2(IsaacGymJoint):
    '''
    isaac gym: flhip, frhip, rlhip, rrhip,
                flthigh, frthigh, rlthigh, rrthigh,
                fl, fr, rl, rr calf
    go2: 0 for hip, 1 for thigh, 2 for calf
    '''
    Go2Joint = np.empty_like(IsaacGymJoint)
    Go2Joint[LegID["FL_0"]] = IsaacGymJoint[0]
    Go2Joint[LegID["FL_1"]] = IsaacGymJoint[1]
    Go2Joint[LegID["FL_2"]] = IsaacGymJoint[2]
    Go2Joint[LegID["FR_0"]] = IsaacGymJoint[3]
    Go2Joint[LegID["FR_1"]] = IsaacGymJoint[4]
    Go2Joint[LegID["FR_2"]] = IsaacGymJoint[5]
    Go2Joint[LegID["RL_0"]] = IsaacGymJoint[6]
    Go2Joint[LegID["RL_1"]] = IsaacGymJoint[7]
    Go2Joint[LegID["RL_2"]] = IsaacGymJoint[8]
    Go2Joint[LegID["RR_0"]] = IsaacGymJoint[9]
    Go2Joint[LegID["RR_1"]] = IsaacGymJoint[10]
    Go2Joint[LegID["RR_2"]] = IsaacGymJoint[11]
    return Go2Joint

def convertJointOrderGo2ToIsaac(Go2Joint):
    '''
    isaac gym: FLhip, flthigh, flcalf;
               FR;
               RL;
               RR
    go2: 0 for hip, 1 for thigh, 2 for calf
    '''
    IsaacGymJoint = np.empty_like(Go2Joint)
    IsaacGymJoint[0] = Go2Joint[LegID["FL_0"]]
    IsaacGymJoint[1] = Go2Joint[LegID["FL_1"]]
    IsaacGymJoint[2] = Go2Joint[LegID["FL_2"]]
    IsaacGymJoint[3] = Go2Joint[LegID["FR_0"]]
    IsaacGymJoint[4] = Go2Joint[LegID["FR_1"]]
    IsaacGymJoint[5] = Go2Joint[LegID["FR_2"]]
    IsaacGymJoint[6] = Go2Joint[LegID["RL_0"]]
    IsaacGymJoint[7] = Go2Joint[LegID["RL_1"]]
    IsaacGymJoint[8] = Go2Joint[LegID["RL_2"]]
    IsaacGymJoint[9] = Go2Joint[LegID["RR_0"]]
    IsaacGymJoint[10] = Go2Joint[LegID["RR_1"]]
    IsaacGymJoint[11] = Go2Joint[LegID["RR_2"]]
    return IsaacGymJoint

class ObsHandler(object):
    
    def __init__(self) -> None:
        self.highStateSub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        # rsc = RobotStateClient()
        # rsc.SetTimeout(3.0)
        # rsc.Init()
        self.velocity = deque(maxlen=5)
        self.lowStateQueue = deque(maxlen=5)

        self.last_action = [0.0] * 12
        
        # self.session = ort.InferenceSession(onnx_model_path)
        # Get the model's input name
        # self.input_name = self.session.get_inputs()[0].name
        # self.offsetGo2 = convertJointOrderIsaacToGo2(self.offsetIsaac)
        # print(self.offsetGo2)
        self.scale = 0.25
        self.last_state = None
        self.last_joint_angle = None
        self.lowStateSub = ChannelSubscriber("rt/lowstate", LowState_)
        self.highStateSub.Init(self.HighStateHandler, 10)
        self.lowStateSub.Init(self.LowStateHandler, 10)

    def HighStateHandler(self, msg: SportModeState_):
        
        self.velocity.append(msg.velocity)

    def LowStateHandler(self, msg: LowState_):
        # print("get low state")
        self.lowStateQueue.append(msg)

    def getStateFromLowLevelMsg(self, msg: LowState_):
        ang_vel = msg.imu_state.gyroscope
        quat = msg.imu_state.quaternion
        joint_angle = [msg.motor_state[id].q for id in range(12)]
        joint_vel = [msg.motor_state[id].dq for id in range(12)]
        return ang_vel, quat, joint_angle, joint_vel

    def get_state(self):
        # base_lin_vel_b
        # if self.last_state is None:
        #     print("pause one time step to init last state")
        base_lin_vel_w = self.velocity[-1]
        ang_vel_w, quat, joint_angle, joint_vel = self.getStateFromLowLevelMsg(self.lowStateQueue[-1])
        obs = np.empty([1, 48])
        obs[:, :3] = 2.0 * np.array(base_lin_vel_w) # quaternion_inverse_rotate(quat, )
        obs[:, 3:6] = 0.25 * np.array(ang_vel_w) # quaternion_inverse_rotate(quat, )
        obs[:, 6:9] = quaternion_inverse_rotate(quat, np.array([0.0, 0.0, -1.0]))
        obs[:, 9:12] = np.multiply(np.array([2.0, 2.0, 0.25]), velo_command)
        obs[:, 12:24] = convertJointOrderGo2ToIsaac(joint_angle) - ISAAC_OFFSET
        obs[:, 24:36] = convertJointOrderGo2ToIsaac(joint_vel) * 0.05
        obs[:, 36:48] = self.last_action
        return obs
    
    def get_joint_diff(self, joint_pos):
        diff = []
        for i in range(len(joint)):
            diff.append(joint[i] - self.last_joint_angle[i])
        return diff
    
    def init_last(self):
        if self.last_state is None:
            joint_pos = self.get_joint_pos()
            self.last_joint_angle = joint_pos
            time.sleep(DT)
            '''
            need to initilalized twice since we have others in the observation.
            '''
            self.last_state, self.last_joint_angle = self.get_state()
            time.sleep(DT)
    
    def get_transition(self):
        print('Start data collection!')
        states = []
        actions = []
        next_states = []
        self.init_last()
        start = time.time()
        for i in range(1000):
            state = self.last_state
            next_state, joint_angle = self.get_state()
            action = self.get_joint_diff(joint_angle)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            self.last_state = next_state
            self.last_joint_angle = joint_angle
            time.sleep(DT)
        print(f"Collection finished! time cost: {time.time() - start}")
        states = np.concatenate(states, axis=0)
        actions = np.vstack(actions)
        next_states = np.concatenate(next_states, axis=0)
        print(states.shape, actions.shape, next_states.shape)
        print("Collected 1000 transitions!")
        filename = input("Please enter the filename (without extension): ")
        np.savez_compressed(f'{filename}.npz', expert_state=states, 
                            expert_actions=actions, expert_next_states=next_states)
        print(f"Arrays saved to {filename}.npz")
           
    
    def get_joint_pos(self):
        ang_vel_w, quat, joint_angle, joint_vel = self.getStateFromLowLevelMsg(self.lowStateQueue[-1])
        return joint_angle

    # def onnx_inference(self, input):
    #     if len(input.shape) == 0:
    #         input = np.expand_dims(input, axis=0)
    #     return self.session.run(None, {self.input_name: input})[0][0]

    # def get_action(self, velo_command = np.array([0.3, 0.3, 0.3])):
    #     state = self.get_state(velo_command)
    #     output = self.onnx_inference(state.astype(np.float32))
    #     # action = self.convert_action(output)
    #     return output

    # def update_action(self, raw_action):
    #     self.last_action = raw_action

class Controller(object):
    STAND = np.array([
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5
    ])
    SIT = np.array([
        -0.0, 1.2, -2.5,
        -0.0, 1.2, -2.5,
        -0.0, 1.2, -2.5,
        -0.0, 1.2, -2.5,
    ])
    def __init__(self):
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        self.rsc = RobotStateClient()
        self.rsc.SetTimeout(3.0)
        self.rsc.Init()
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.cmd.motor_cmd[i].q = PosStopF
            self.cmd.motor_cmd[i].kp = 0
            self.cmd.motor_cmd[i].dq = VelStopF
            self.cmd.motor_cmd[i].kd = 0
            self.cmd.motor_cmd[i].tau = 0
        self.sportsModeDisabled = False
        self.offsetIsaac = np.array([0.1, -0.1, 0.1, -0.1,
                                     0.8, 0.8, 1.0, 1.0,
                                     -1.5, -1.5, -1.5, -1.5])
    
    def verifySportsMode(self):
        if not self.sportsModeDisabled:
            code = controller.rsc.ServiceSwitch("sport_mode", False)
            if code != 0:
                print("service stop sport_mode error. code:", code)
            else:
                print("service stop sport_mode success. code:", code)
            self.sportsModeDisabled = True
    
    def controlIsaacAction(self, action):
        go2Action = self.convertIsaacAction2Go2Action(action)
        self.controlGo2Action(go2Action)
    
    def controlGo2Action(self, action):

        # Poinstion(rad) control, set RL_0 rad
        for i, joint_pos in enumerate(action):
            self.cmd.motor_cmd[i].q = joint_pos  # Taregt angular(rad)
            self.cmd.motor_cmd[i].kp = 30.0  # Poinstion(rad) control kp gain
            self.cmd.motor_cmd[i].dq = 0.0  # Taregt angular velocity(rad/ss)
            self.cmd.motor_cmd[i].kd = 0.5  # Poinstion(rad) control kd gain
            self.cmd.motor_cmd[i].tau = 0.0  # Feedforward toque 1N.m

        self.cmd.crc = crc.Crc(self.cmd)

        # Publish message
        if self.pub.Write(self.cmd):
            print("Publish success. msg:", self.cmd.crc)
            # break
        else:
            print("Waitting for subscriber.")
    
    def convertIsaacAction2Go2Action(self, isaacAction):
        scaled_action = 0.25 * isaacAction
        abs_action = scaled_action + ISAAC_OFFSET
        absActionGo2 = convertJointOrderIsaacToGo2(abs_action)
        return absActionGo2

    def stop(self):

        while True:

            # Poinstion(rad) control, set RL_0 rad
            for i, joint_pos in enumerate(self.STAND):
                self.cmd.motor_cmd[i].q = joint_pos  # Taregt angular(rad)
                if i <= 5 :
                    self.cmd.motor_cmd[i].kp = 20.0  # Poinstion(rad) control kp gain
                else: 
                    self.cmd.motor_cmd[i].kp = 30.0  # Poinstion(rad) control kp gai
                self.cmd.motor_cmd[i].dq = 0.0  # Taregt angular velocity(rad/ss)
                self.cmd.motor_cmd[i].kd = 0.3  # Poinstion(rad) control kd gain
                # if i % 3 == 2:
                self.cmd.motor_cmd[i].tau = 0.0  # Feedforward toque 1N.m
                # else:
                #     self.cmd.motor_cmd[i].tau = 0.0  # Feedforward toque 1N.m

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
                self.cmd.motor_cmd[i].kd = 2.0  # Poinstion(rad) control kd gain
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
    def get_key(settings):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = None

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    key_settings = termios.tcgetattr(sys.stdin)
    ChannelFactoryInitialize(0, DOG_NAME)
    obs_handle = ObsHandler()
    time.sleep(2)
    controller = Controller()
    first_run = True
    
    while True:
        key = get_key(key_settings)
        emergency_stop(key)
        if first_run:
            obs_handle.init_last()
            first_run = False
            print("initialize")
        if key is not None:
            # pass
            break
        else:
            obs_handle.init_last()
            # print("Current State:", obs_handle.get_state()[0])
            _, joint = obs_handle.get_state()
            print("Current joint:", joint)
            print("Joint diff:", obs_handle.get_joint_diff(joint))
            print("Press Space for emergency stop, 1 to exit, other to proceed.")
        time.sleep(DT)  # Adding a small delay to avoid high CPU usage
    print('Start to walk')
    first_run = True
    while True:
        key = get_key(key_settings)
        emergency_stop(key)
        if first_run:
            obs_handle.init_last()
            first_run = False
            print("initialize")
            print("Press Space for emergency stop, 1 to exit, r to start collection")
        if key == 'r':
            start = time.time()
            obs_handle.get_transition()
            print(f"Collection finished! time cost: {time.time() - start}")
        else:
            pass
        # isaacAction = obs_handle.get_action(velo_command=[0.0, 0.3, 0.0])
        # go2Action = controller.convertIsaacAction2Go2Action(isaacAction)
        # jointAngles = obs_handle.get_joint_pos()
        # # controller.controlIsaacAction(isaacAction)
        # print('Diff GO2:', jointAngles - go2Action)
        # print('Action isaac:',isaacAction)
        # # print(obs_handle.get_state())
        # obs_handle.update_action(isaacAction)
        time.sleep(0.02)



    
    