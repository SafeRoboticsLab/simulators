import pybullet as p
import os
import math
import numpy as np
from simulators.dynamics.resources.utils import *

class GVR:
    def __init__(self, client, height, orientation, envtype = None, **kwargs):
        self.client = client
        self.height = height

        ox = 0
        oy = 0

        self.urdf = "gvr_bot/gvrbot_updated.urdf"

        if envtype != None:
            raise NotImplementedError
        
        f_name = os.path.join(os.path.dirname(__file__), self.urdf)

        self.id = p.loadURDF(fileName = f_name, basePosition=[ox, oy, self.height], baseOrientation = orientation, physicsClientId = client)

        '''
        0 b'base_to_L1_wheel'
        1 b'base_to_L2_wheel'
        2 b'base_to_L3_wheel'
        3 b'base_to_L4_wheel'
        4 b'base_to_L5_wheel'
        5 b'base_to_L6_wheel'
        6 b'base_to_L7_wheel'
        7 b'base_to_L8_wheel'
        8 b'base_to_L9_wheel'
        9 b'base_to_L10_wheel'
        10 b'base_to_L11_wheel'
        11 b'base_to_R1_wheel'
        12 b'base_to_R2_wheel'
        13 b'base_to_R3_wheel'
        14 b'base_to_R4_wheel'
        15 b'base_to_R5_wheel'
        16 b'base_to_R6_wheel'
        17 b'base_to_R7_wheel'
        18 b'base_to_R8_wheel'
        19 b'base_to_R9_wheel'
        20 b'base_to_R10_wheel'
        21 b'base_to_R11_wheel'
        22 b'base_to_LF_flipper'
        23 b'base_to_RF_flipper'
        '''

        self.flipper_joint_index = [22, 23]
        self.left_wheel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.right_wheel_index = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    def get_ids(self):
        return self.id, self.client
    
    def reset(self, position):
        for i in range(len(self.flipper_joint_index)):
            p.resetJointState(self.id, self.flipper_joint_index[i], position[i], physicsClientId = self.client)
    
    def apply_action(self, action):
        # first 2 actions are the increment of the left and right flipper angular position, next 2 actions are the target velocity of the left and right wheel, respectively
        new_flipper_angle = np.array(self.get_flipper_joint_position()) + np.array(action)
        for joint in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, joint)
            if "LF_flipper" in str(info[1]):
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = new_flipper_angle[0], force=info[10], maxVelocity=info[11])
            elif "RF_flipper" in str(info[1]):
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = new_flipper_angle[1], force=info[10], maxVelocity=info[11])
            else:
                if "base_to_L" in str(info[1]):
                    p.setJointMotorControl2(self.id, joint, p.VELOCITY_CONTROL, targetVelocity = action[2])
                elif "base_to_R" in str(info[1]):
                    p.setJointMotorControl2(self.id, joint, p.VELOCITY_CONTROL, targetVelocity = action[3])
    
    def apply_position(self, action):
        # only apply for flippers
        for joint in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, joint)
            lower_limit = info[8]
            upper_limit = info[9]

            if "LF_flipper" in str(info[1]):
                pos = min(max(lower_limit, action[0]), upper_limit)
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = pos, force=info[10], maxVelocity=info[11])
            elif "RF_flipper" in str(info[1]):
                pos = min(max(lower_limit, action[1]), upper_limit)
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = action[1], force=info[10], maxVelocity=info[11])
    
    def get_obs(self):
        # similar to spirit.py
        pos, ang = p.getBasePositionAndOrientation(self.id, physicsClientId = self.client)
        ang = p.getEulerFromQuaternion(ang, physicsClientId = self.client)
        vel = p.getBaseVelocity(self.id, physicsClientId = self.client)[0][:]
        observation = (pos + ang + vel)
        return observation
    
    def safety_margin(self, state):
        return {
            "roll": abs(state[3]) - math.pi * 0.2,
            "pitch": abs(state[4]) - math.pi * 0.2
        }
    
    def target_margin(self, state):
        # for now, let's just use target_margin smaller than safety_margin, as we are running avoidonly anyway (not using target margin)
        return {
            "roll": abs(state[3]) - math.pi * 0.1,
            "pitch": abs(state[4]) - math.pi * 0.1
        }
    
    def get_flipper_joint_position(self):
        joint_state = p.getJointStates(self.id, jointIndices = self.flipper_joint_index, physicsClientId = self.client)
        position = [state[0] for state in joint_state]
        return position
    
    def get_wheel_velocity(self):
        left_wheel_joint_state = p.getJointStates(self.id, jointIndices = self.left_wheel_index, physicsClientId = self.client)
        right_wheel_joint_state = p.getJointStates(self.id, jointIndices = self.right_wheel_index, physicsClientId = self.client)
        #! NEED CHECK: assume that all wheels of each side is the same, so only take the first wheel vel and return
        return [left_wheel_joint_state[0], right_wheel_joint_state[0]]

