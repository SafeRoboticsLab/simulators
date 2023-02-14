import pybullet as p
import os
import math
import numpy as np
from scipy.spatial.transform import Rotation
from simulators.dynamics.resources.utils import *

class GVR:
    def __init__(self, client, height, orientation, envtype="normal", **kwargs):
        self.client = client
        self.height = height

        ox = 0
        oy = 0

        self.urdf = "gvr_bot/gvrbot_updated.urdf"
        self.urdf_path = os.path.join(os.path.dirname(__file__), self.urdf)

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
        self.max_wheel_vel = 25 #rad/s from max linear velocity
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
            action (np.ndarray): 3-D action [linear_x, angular_z, flip_pos]
        """
        # scale the user's input with the actual real-life reaction
        user_action[1] = user_action[1] * 5.5
        action = np.zeros(len(self.body_indices))
        left_vel = (user_action[0] - user_action[1]*self.W/2)/self.Rw
        right_vel = (user_action[0] + user_action[1]*self.W/2)/self.Rw

        if abs(left_vel) >= (right_vel):
            if abs(left_vel) > self.max_wheel_vel:
                ratio = right_vel/left_vel
                left_vel = np.clip(left_vel, -self.max_wheel_vel, self.max_wheel_vel)
                right_vel = left_vel*ratio
        else:
            if abs(right_vel) > self.max_wheel_vel:
                ratio = left_vel/right_vel
                right_vel = np.clip(right_vel, -self.max_wheel_vel, self.max_wheel_vel)
                left_vel = right_vel*ratio

        action[self.left_wheel_index] = left_vel
        action[self.right_wheel_index] = right_vel

        max_wheel_torque = 3.0*np.ones_like(action)
        kP_v_drive = 1.5*np.ones_like(action)

        max_flipper_torque = 30*np.ones_like(self.flipper_joint_index)
        kP_v_flipper = 1*np.ones_like(self.flipper_joint_index)

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.body_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=list(action),
            forces=max_wheel_torque,
            velocityGains = kP_v_drive)
            
        p.setJointMotorControl2(
            bodyUniqueId=self.id,
            jointIndex=self.flipper_joint_index[0],
            controlMode=p.POSITION_CONTROL,
            targetPosition=user_action[2],
            force=max_flipper_torque[0]
        )
    
    def apply_position(self, action:float):
        for joint in self.flipper_joint_index:
            p.setJointMotorControl2(
                self.id, 
                joint, 
                p.POSITION_CONTROL,
                targetPosition = action, 
                maxVelocity=self.max_flipper_vel
            )
    
    def get_obs(self):
        """Get observation 13-D:
            x_dot, y_dot, z_dot,
            roll, pitch, yaw
            w_x, w_y, w_z,
            flipper_pos, flipper_angular_vel,
            v_left, v_right

        Returns:
            observation (Tuple): 13-D observation
        """
        pos, ang = p.getBasePositionAndOrientation(self.id, physicsClientId = self.client)
        rotmat = Rotation.from_quat(ang).as_matrix()
        ang = p.getEulerFromQuaternion(ang, physicsClientId = self.client)
        linear_vel, angular_vel = p.getBaseVelocity(self.id, physicsClientId = self.client)
        robot_body_linear_vel = (np.linalg.inv(rotmat)@np.array(linear_vel).T)   
        robot_body_angular_vel = (np.linalg.inv(rotmat)@np.array(angular_vel).T)
        observation = (tuple(robot_body_linear_vel) + ang + tuple(robot_body_angular_vel) + self.get_flipper_state() +  self.get_track_velocity())
        return observation
    
    def safety_margin(self, state):
        return {
            "roll": abs(state[3]) - math.pi * 1./9.,
            "pitch": abs(state[4]) - math.pi * 1./6.,
            "body_ang_x": abs(state[6]) - math.pi * 0.5,
            "body_ang_y": abs(state[7]) - math.pi * 0.5,
            "linear_x": abs(state[0]) - 0.7
        }
    
    def target_margin(self, state):
        # for now, let's just use target_margin smaller than safety_margin, as we are running avoidonly anyway (not using target margin)
        return {
            "roll": abs(state[3]) - math.pi * 0.1,
            "pitch": abs(state[4]) - math.pi * 0.1
        }
    
    def get_flipper_state(self):
        flip_state = p.getJointState(self.id, self.flipper_joint_index[0], physicsClientId = self.client)
        flip_pos, flip_vel, flip_force, flip_torque = flip_state
        return flip_pos, flip_vel
    
    def get_track_velocity(self):
        lt_ang_vel = 0
        for i in range(len(self.left_wheel_index)):
            left_track_state = p.getJointState(self.id, self.left_wheel_index[i], physicsClientId = self.client)
            lt_rad, lt_vel, lt_force, lt_torque = left_track_state
            lt_ang_vel += lt_vel/len(self.left_wheel_index)

        rt_ang_vel = 0
        for i in range(len(self.right_wheel_index)):
            right_track_state = p.getJointState(self.id, self.right_wheel_index[i], physicsClientId = self.client)
            rt_rad, rt_vel, rt_force, rt_torque = right_track_state
            rt_ang_vel += rt_vel/len(self.right_wheel_index)
        
        return lt_ang_vel, rt_ang_vel
    
    def get_link_id(self, name):
        _link_name_to_index = {p.getBodyInfo(self.id)[0].decode('UTF-8'):-1,}
        
        for _id in range(p.getNumJoints(self.id)):
            _name = p.getJointInfo(self.id, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id
        
        return _link_name_to_index[name]
    
    def linc_get_pos(self):
        """
        Return the position list of 3 floats and orientation as list of
        4 floats in [x,y,z,w] order. Use pb.getEulerFromQuaternion to convert
        the quaternion to Euler if needed.
        """
        return p.getBasePositionAndOrientation(self.id, physicsClientId = self.client)

