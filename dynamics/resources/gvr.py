import pybullet as p
import os
import math
import numpy as np
from simulators.dynamics.resources.utils import *

class GVR:
    def __init__(self, client, height, orientation, envtype="normal", payload=0.0 , payload_max=10.0, **kwargs):
        self.client = client
        self.height = height

        ox = 0
        oy = 0

        self.urdf = "gvr_bot/gvrbot_updated.urdf"
        self.urdf_path = os.path.join(os.path.dirname(__file__), self.urdf)

        self.payload = payload
        self.payload_max = payload_max
        self.envtype = envtype

        if envtype != "normal":
            raise NotImplementedError

        self.id = p.loadURDF(fileName = self.urdf_path, basePosition=[ox, oy, self.height], baseOrientation = orientation, physicsClientId = client)

        # self.left_wheel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # self.right_wheel_index = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        # self.flipper_joint_index = [22, 23]

        self.left_wheel_index = []
        self.right_wheel_index = []
        self.flipper_joint_index = []  
        num_joints = p.getNumJoints(self.id)
        for joint in range(num_joints):
            info = p.getJointInfo(self.id, joint)
            if 'L' in str(info[1]) and 'wheel' in str(info[1]):
                self.left_wheel_index.append(info[0])
            elif 'R' in str(info[1]) and 'wheel' in str(info[1]):
                self.right_wheel_index.append(info[0])
            elif 'flipper' in str(info[1]):
                self.flipper_joint_index.append(info[0])
        
        num_valid_indices = len(self.left_wheel_index) + len(self.right_wheel_index) + len(self.flipper_joint_index)
        self.body_indices = list(np.arange(num_valid_indices))

        self.max_linear_vel = 2 #from AndrosBot Guide
        self.max_angular_vel = 3 # approximation so far, need to refine
        self.max_wheel_vel = 20 #rad/s from max linear velocity
        self.max_flipper_vel = 0.5 # rad/s slow deployment approximate
        self.Rw = 0.0862 #m wheel radius
        self.W = 0.35 #m wheelbase

    def get_ids(self):
        return self.id, self.client
    
    def reset(self, position: float):
        """_summary_

        Args:
            position (float): Reset the flippers to a angular position
        """

        for i in range(len(self.flipper_joint_index)):
            p.resetJointState(self.id, self.flipper_joint_index[i], position, physicsClientId = self.client)
    
    def apply_action(self, user_action):
        """_summary_

        Args:
            action (np.ndarray): 3-D action [linear_x, angular_z, flip_pos_increment]
        """
        action = np.zeros(len(self.body_indices))
        left_vel = (user_action[0] - user_action[1]*self.W/2)/self.Rw
        right_vel = (user_action[0] + user_action[1]*self.W/2)/self.Rw

        left_vel = np.clip(left_vel, -self.max_wheel_vel, self.max_wheel_vel)
        right_vel = np.clip(right_vel, -self.max_wheel_vel, self.max_wheel_vel)

        action[self.left_wheel_index] = left_vel
        action[self.right_wheel_index] = right_vel

        max_wheel_torque = 1.8*np.ones_like(action)
        kP_p = 0.8*np.ones_like(action)
        kP_v = 2*np.ones_like(action)

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.body_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=list(action),
            forces=max_wheel_torque)
            
        for joint in self.flipper_joint_index:
            p.setJointMotorControl2(
                self.id, 
                joint, 
                p.POSITION_CONTROL,
                targetPosition = user_action[2], 
                maxVelocity=self.max_flipper_vel
            )
    
    def apply_position(self, action:float):
        # only apply for flippers
        for joint in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, joint)
            if "LF_flipper" in str(info[1]):
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = action, force=info[10], maxVelocity=info[11])
            elif "RF_flipper" in str(info[1]):
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = action, force=info[10], maxVelocity=info[11])
    
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
        left_vel = [state[1] for state in left_wheel_joint_state]
        right_vel = [state[1] for state in right_wheel_joint_state]
        return [left_vel[0], right_vel[0]]
    
    def get_link_id(self, name):
        _link_name_to_index = {p.getBodyInfo(self.id)[0].decode('UTF-8'):-1,}
        
        for _id in range(p.getNumJoints(self.id)):
            _name = p.getJointInfo(self.id, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id
        
        return _link_name_to_index[name]

