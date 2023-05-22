import pybullet as p
import os
import math
import numpy as np
from scipy.spatial.transform import Rotation
from simulators.dynamics.resources.utils import *
from simulators.dynamics.utils import change_dynamics

class GVR:
    def __init__(self, client, height, orientation, envtype="normal", **kwargs):
        self.client = client
        self.height = height
        self.orientation = orientation

        # fine tuned car env params
        self.car_lat_friction = 0.4597457696689967
        self.car_ani_friction = [16.034153317830327, 38.63070609519674, 2.7524624880209765]
        self.car_roll_friction = 0
        self.car_spin_friction = 0
        self.car_mass = 16.4
        self.car_inertia = 0.668
        self.wheel_mass = 0.12455
        self.wheel_inertia = 0.0002693
        self.damping = 15853.49778551601
        self.stiffness = 51886.04540801367

        self.envtype = envtype

        self.change_dynamics = change_dynamics

        if envtype != "normal":
            raise NotImplementedError

        self.load_robot()
        p.enableJointForceTorqueSensor(self.id, 0)
    
    def load_robot(self):
        self.urdf = "gvr_bot/gvrbot_updated.urdf"
        self.urdf_path = os.path.join(os.path.dirname(__file__), self.urdf)
        self.id = p.loadURDF(fileName=self.urdf_path, basePosition=np.array([0, 0, self.height]), 
                             baseOrientation=self.orientation, physicsClientId=self.client)

        self.left_wheel_index = []
        self.right_wheel_index = []
        self.flipper_joint_index = []
        num_joints = p.getNumJoints(self.id)
        # change base mass and inertia values
        current_inertia = np.array(p.getDynamicsInfo(self.id, -1)[2])
        current_inertia[2] = self.car_inertia
        self.change_dynamics(self.id, link_id=-1, mass=self.car_mass, local_inertia_diagonal=current_inertia)
        current_inertia = np.array(p.getDynamicsInfo(self.id, -1)[2])

        for joint in range(num_joints):
            info = p.getJointInfo(self.id, joint)
            change = 0
            if 'L' in str(info[1]) and 'wheel' in str(info[1]):
                self.left_wheel_index.append(info[0])
                p.enableJointForceTorqueSensor(self.id, info[0], 1)
                change = 1
            elif 'R' in str(info[1]) and 'wheel' in str(info[1]):
                self.right_wheel_index.append(info[0])
                p.enableJointForceTorqueSensor(self.id, info[0], 1)
                change = 1
            elif 'base_to_flippers' in str(info[1]):
                self.flipper_joint_index.append(info[0])
                p.enableJointForceTorqueSensor(self.id, info[0], 1)
                change = 2

            if change == 1:
                wheel_current_inertia = np.array(p.getDynamicsInfo(self.id, joint)[2])
                wheel_current_inertia[:2] = self.wheel_inertia
                self.change_dynamics(self.id, link_id=joint, mass=self.wheel_mass, local_inertia_diagonal=wheel_current_inertia, contact_damping=self.damping, contact_stiffness=self.stiffness,
                                    lateral_friction=self.car_lat_friction, anisotropic_friction=self.car_ani_friction, rolling_friction=self.car_roll_friction, spinning_friction=self.car_spin_friction)
            else:
                self.change_dynamics(self.id, link_id=joint, lateral_friction=0.42, anisotropic_friction=[4, 4, 4], rolling_friction=0, spinning_friction=0)

        num_valid_indices = len(self.left_wheel_index) + len(self.right_wheel_index) + len(self.flipper_joint_index)
        self.body_indices = list(np.arange(num_valid_indices))

        self.max_linear_vel = 2.0  # from AndrosBot Guide
        self.max_angular_vel = 3.0  # approximation so far, need to refine
        self.max_wheel_vel = 25.0  # rad/s from max linear velocity
        self.max_flipper_vel = 0.5  # rad/s slow deployment approximate
        self.Rw = 0.0862  # m wheel radius
        self.W = 0.35  # m wheelbase

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
            "body_ang_z": abs(state[8]) - math.pi * 0.5,
            "vel_left": abs(state[11]) - 0.7,
            "vel_right": abs(state[12]) - 0.7
        }
    
    def target_margin(self, state):
        # for now, let's just use target_margin smaller than safety_margin, as we are running avoidonly anyway (not using target margin)
        return {
            "roll": abs(state[3]) - math.pi * 0.1,
            "pitch": abs(state[4]) - math.pi * 0.1,
            "body_ang_x": abs(state[6]) - math.pi * 0.02,
            "body_ang_y": abs(state[7]) - math.pi * 0.02,
            "body_ang_z": abs(state[8]) - math.pi * 0.02,
            "vel_left": abs(state[11]) - 0.2,
            "vel_right": abs(state[12]) - 0.2
        }

        # return {
        #     "roll": abs(state[3]) - math.pi * 1./9.,
        #     "pitch": abs(state[4]) - math.pi * 1./6.,
        #     "body_ang_x": abs(state[6]) - math.pi * 0.5,
        #     "body_ang_y": abs(state[7]) - math.pi * 0.5,
        #     "linear_x": abs(state[0]) - 0.7
        # }
    
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

