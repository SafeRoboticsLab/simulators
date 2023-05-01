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
        self.orientation = orientation

        self.lateral_friction = 0.8
        self.anisotropic_friction = [1, 1, 1]
        self.rolling_friction = 0
        self.spinning_friction = 0
        self.restitution = 0.98

        self.payload = "none" # sail, sloshy
        self.payload_mass = 10
        self.payload_blocks = 5
        self.theta = 20
        self.terrain = "plane"
        self.terrain_coeff = 1.0
        self.terrain_height = 0.2

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

        if envtype != "normal":
            raise NotImplementedError

        self.load_robot()
        p.enableJointForceTorqueSensor(self.id, 0)
    
    def load_robot(self):
        self.urdf = "gvr_bot/gvrbot_updated.urdf"
        self.urdf_path = os.path.join(os.path.dirname(__file__), self.urdf)
        self.id = p.loadURDF(fileName=self.urdf_path, basePosition=np.array([0, 0, self.height]), baseOrientation=self.orientation, physicsClientId=self.client)

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

        self.max_linear_vel = 2  # from AndrosBot Guide
        self.max_angular_vel = 3  # approximation so far, need to refine
        self.max_wheel_vel = 25  # rad/s from max linear velocity
        self.max_flipper_vel = 0.5  # rad/s slow deployment approximate
        self.Rw = 0.0862  # m wheel radius
        self.W = 0.35  # m wheelbase
    
    def change_dynamics(self, body_id, link_id=-1, mass=None, lateral_friction=None, spinning_friction=None,
                        rolling_friction=None, anisotropic_friction=None, restitution=None, linear_damping=None, angular_damping=None,
                        contact_stiffness=None, contact_damping=None, friction_anchor=None,
                        local_inertia_diagonal=None, inertia_position=None, inertia_orientation=None,
                        joint_damping=None, joint_friction=None, joint_force=None):

        kwargs = {}
        if mass is not None:
            kwargs['mass'] = mass
        if lateral_friction is not None:
            kwargs['lateralFriction'] = lateral_friction
        if spinning_friction is not None:
            kwargs['spinningFriction'] = spinning_friction
        if rolling_friction is not None:
            kwargs['rollingFriction'] = rolling_friction
        if anisotropic_friction is not None:
            kwargs['anisotropicFriction'] = anisotropic_friction
        if restitution is not None:
            kwargs['restitution'] = restitution
        if linear_damping is not None:
            kwargs['linearDamping'] = linear_damping
        if angular_damping is not None:
            kwargs['angularDamping'] = angular_damping
        if contact_stiffness is not None:
            kwargs['contactStiffness'] = contact_stiffness
        if contact_damping is not None:
            kwargs['contactDamping'] = contact_damping
        if friction_anchor is not None:
            kwargs['frictionAnchor'] = friction_anchor
        if local_inertia_diagonal is not None:
            kwargs['localInertiaDiagonal'] = local_inertia_diagonal
        if joint_damping is not None:
            kwargs['jointDamping'] = joint_damping
        if joint_force is not None:
            kwargs['jointLimitForce'] = joint_force

        p.changeDynamics(body_id, link_id, **kwargs)

    def change_COM(self, mass, world_location, body_location):
        if self.box is not None:
            p.removeBody(self.box)
        cuid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01])
        # i.e. this should be started close to the robot
        self.box = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=cuid, basePosition=world_location)
        self.box_location = np.array(body_location)
        self.box_mass = mass
        inertia = mass * 0.2 ** 2 / 6
        p.changeDynamics(self.box, -1, localInertiaDiagonal=[inertia, inertia, inertia])
        p.createConstraint(self.box, -1, self.id, 0, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=body_location)

    def get_COM(self):
        masses = [p.getDynamicsInfo(self.id, k)[0] for k in range(-1, p.getNumJoints(self.id))]
        total_mass = sum(masses)
        pos, _ = p.getBasePositionAndOrientation(self.id, self.client)
        link_states = p.getLinkStates(self.id, range(p.getNumJoints(self.id)))
        link_positions = [pos] + [state[0] for state in link_states]
        center_of_mass = np.sum(np.array(masses)[:, np.newaxis] * (np.array(link_positions) - np.array(pos)), axis=0)
        center_of_mass += self.box_location * self.box_mass  # add to the sum in the body frame
        total_mass += self.box_mass  # total mass of the system (including the box)
        center_of_mass /= total_mass
        return center_of_mass

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
            "body_ang_x": abs(state[6]) - math.pi * 0.25,
            "body_ang_y": abs(state[7]) - math.pi * 0.25,
            "linear_x": abs(state[0]) - 1.0
        }
    
    def target_margin(self, state):
        # for now, let's just use target_margin smaller than safety_margin, as we are running avoidonly anyway (not using target margin)
        return {
            "roll": abs(state[3]) - math.pi * 0.1,
            "pitch": abs(state[4]) - math.pi * 0.1,
            "body_ang_x": abs(state[6]) - math.pi * 0.02,
            "body_ang_y": abs(state[7]) - math.pi * 0.02,
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

