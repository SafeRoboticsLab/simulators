import pybullet as p
import os
import math
import numpy as np
from simulators.dynamics.resources.utils import *

class Spirit:
    def __init__(self, client, height, orientation, 
        envtype = None, payload = 0, payload_max = 0, **kwargs):
        self.client = client
        self.urdf = "spirit40.urdf"
        self.height = height
        
        ox = 0
        oy = 0
        
        for key in kwargs.keys():
            if key == "ox":
                ox = kwargs["ox"]
            if key == "oy":
                oy = kwargs["oy"]
        
        if envtype != None:
            # TODO: create different env here
            # self._gen_urdf(envtype, payload, payload_max)
            pass
            
        f_name = os.path.join(os.path.dirname(__file__), self.urdf)

        self.id = p.loadURDF(fileName = f_name, basePosition=[ox, oy, self.height], baseOrientation = orientation, physicsClientId = client)
        
        # self.joint_index = range(p.getNumJoints(self.id))
        self.joint_index = self.make_joint_list()
        self.torque_gain = 10.0

    def get_ids(self):
        return self.id, self.client

    def reset(self, position):
        for i in range(len(self.joint_index)):
            p.resetJointState(self.id, self.joint_index[i], position[i], physicsClientId = self.client)
    
    def apply_action(self, action):
        """
        Action is the angular increase for each of the joint wrt to the current position

        Args:
            action (_type_): angular positional increase
        """
        new_angle = np.array(self.get_joint_position()) + np.array(action)

        for i in range(len(self.joint_index)):
            info = p.getJointInfo(self.id, self.joint_index[i], physicsClientId = self.client)
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]
            pos = min(max(lower_limit, new_angle[i]), upper_limit)

            p.setJointMotorControl2(
                self.id, 
                self.joint_index[i],
                p.POSITION_CONTROL, 
                targetPosition = pos, 
                positionGain=1./12.,
                velocityGain=0.4,
                force=max_force,
                maxVelocity=max_velocity, 
                physicsClientId = self.client
            )

    def apply_position(self, action):
        for i in range(len(self.joint_index)):
            info = p.getJointInfo(self.id, self.joint_index[i], physicsClientId = self.client)
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]
            pos = min(max(lower_limit, action[i]), upper_limit)

            p.setJointMotorControl2(
                self.id, 
                self.joint_index[i],
                p.POSITION_CONTROL, 
                targetPosition = pos, 
                positionGain=1./12.,
                velocityGain=0.4,
                force=max_force,
                maxVelocity=max_velocity, 
                physicsClientId = self.client
            )

    def get_obs(self):
        # Get the position and orientation of robot in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ang = p.getEulerFromQuaternion(ang, physicsClientId = self.client)
        
        # ori = (math.cos(ang[0]), math.sin(ang[0]), math.cos(ang[1]), math.sin(ang[1]), math.cos(ang[2]), math.sin(ang[2]))
        
        # Get the velocity of robot
        vel = p.getBaseVelocity(self.id, self.client)[0][:]

        # self.previous_position = pos[0:2]

        # Concatenate position, orientation, velocity
        # VALUE:
        # 0 - x value of body wrt to env
        # 1 - y value of body wrt to env
        # 2 - z value of body wrt to env
        # 3 - roll value, rad
        # 4 - pitch value, rad
        # 5 - yaw value, rad
        # 6 - x velocity of body wrt to env
        # 7 - y velocity of body wrt to env
        # 8 - z velocity of body wrt to env

        observation = (pos + ang + vel) # 3 + 3 + 3

        return observation # return observation size of 12

    def get_joint_position(self):
        joint_state = p.getJointStates(self.id, jointIndices = self.joint_index, physicsClientId = self.client)
        position = [state[0] for state in joint_state]
        return position

    def get_joint_torque(self):
        joint_state = p.getJointStates(self.id, jointIndices = self.joint_index, physicsClientId = self.client)
        torque = [state[3] for state in joint_state]
        return torque

    def safety_margin(self, state):
        """
        Safety margin of the robot. 
        If the robot gets too close to the ground, or if any of the knee touches the ground (within an error margin)
        """
        # height, roll, pitch
        # rotate_margin = np.array([0.16, 0.16, 0.16]) * np.pi
        # dt = 0.008
        # new_obs = state[:9]
        # old_obs = state[9:18]
        # accel = (new_obs - old_obs)/dt
        # rotate_accel = accel[3:6]
        # rotate_error = abs(np.array(rotate_accel))  - np.array(rotate_margin)

        # return {
        #     "height_lower": 0.1 - state[2],
        #     "rotate_error": max(rotate_error)
        # }

        # NEW SAFETY MARGIN
        # Only consider if the robot flips

        """
        test stance
            0.0, 0.2, 1.0,
            0.0, 0.2, 1.0,
            0.0, 0.2, 1.0,
            0.0, 0.2, 1.0
        """

        corners = self.get_body_corners()
        corner_height = corners[2, :]

        elbows = self.get_elbows()
        elbow_height = elbows[2, :]

        return {
            "corner_height": 0.1 - min(corner_height),
            "elbow_height": 0.05 - min(elbow_height)
        }

    def target_margin(self, state):
        corners = self.get_body_corners()
        corner_height = corners[2, :]

        elbows = self.get_elbows()
        elbow_height = elbows[2, :]
        
        return {
            # "corner_height": 0.16 - min(corner_height),
            # "elbow_height": 0.12 - min(elbow_height),
            "roll": abs(state[3]) - math.pi * 0.2,
            "pitch": abs(state[4]) - math.pi * 0.2
        }

    def make_joint_list(self):
        damaged_legs = []
        joint_names = [
            b'hip0', b'upper0', b'lower0',
            b'hip1', b'upper1', b'lower1',
            b'hip2', b'upper2', b'lower2',
            b'hip3', b'upper3', b'lower3'
        ]
        joint_list = []
        for n in joint_names:
            joint_found = False
            for joint in range(p.getNumJoints(self.id, physicsClientId = self.client)):
                name = p.getJointInfo(self.id, joint, physicsClientId = self.client)[12]
                if name == n and name not in damaged_legs:
                    joint_list += [joint]
                    joint_found = True
                elif name == n and name in damaged_legs:
                    p.changeVisualShape(
                        1, joint, rgbaColor=[0.5, 0.5, 0.5, 0.5], physicsClientId = self.client)
            if joint_found is False:
                # if the joint is not here (aka broken leg case) put 1000
                # joint_list += [1000]
                continue
        return joint_list

    def linc_get_joints_positions(self):
        """Return the actual position in the physics engine"""
        pos = np.zeros(len(self.joint_index))
        i = 0
        # be careful that the joint_list is not necessarily in the same order as
        # in bullet (see make_joint_list)
        for joint in self.joint_index:
            if joint != 1000:
                pos[i] = p.getJointState(self.id, joint, physicsClientId = self.client)[0]
                i += 1
        return pos

    def linc_get_pos(self):
        """
        Return the position list of 3 floats and orientation as list of
        4 floats in [x,y,z,w] order. Use pb.getEulerFromQuaternion to convert
        the quaternion to Euler if needed.
        """
        return p.getBasePositionAndOrientation(self.id, physicsClientId = self.client)
    
    def linc_get_ground_contacts(self):
        leg_link_ids = [17, 14, 2, 5, 8, 11]
        descriptor = {17: [], 14: [], 2: [], 5: [], 8: [], 11: []}
        ground_contacts = np.zeros_like(leg_link_ids)

        # Get contact points between robot and world plane
        contact_points = p.getContactPoints(self.id, physicsClientId = self.client)
        link_ids = []  # list of links in contact with the ground plane
        if len(contact_points) > 0:
            for cn in contact_points:
                linkid = cn[3]  # robot link id in contact with world plane
                if linkid not in link_ids:
                    link_ids.append(linkid)
        for l in leg_link_ids:
            cns = descriptor[l]
            if l in link_ids:
                cns.append(1)
            else:
                cns.append(0)
            descriptor[l] = cns

        for i, ll in enumerate(leg_link_ids):
            if ll in link_ids:
                ground_contacts[i] = 1
        
        return ground_contacts

    def linc_get_state(self, t):
        """Combine the elements of the state vector."""
        state = np.concatenate([
            [t],
            list(sum(self.linc_get_pos(), ())),
            self.linc_get_joints_positions(),
            self.linc_get_ground_contacts()])
        return state

    def get_joint_position_wrt_body(self, alpha, beta):
        # get the joint position wrt to body (which joint is closer to the body), from this, the further a joint away from the body, the closer the joint to ground
        # leg length l1 = l2 = 0.206
        # alpha is the angle between upper link and body (upper joint)
        # beta is the angle between lower link and upper link (lower joint)
        # ------ O -- BODY ---------> HEAD
        #      |  \\         | h2
        #      |   \\       B 
        #     h1    \\    //
        #      |      A //
        #----------- GROUND --------
        # for all legs, upper joint moving forward to the head will be to 3.14 (180 degree)
        l1 = 0.206
        l2 = 0.206
        h1 = math.sin(math.pi - alpha) * l1
        theta = math.pi * 1.5 - (math.pi - alpha) - beta
        OB = math.sqrt(l1*l1 + l2*l2 - 2*l1*l2*math.cos(beta))
        if OB == 0:
            return h1, 0
        theta_1 = math.acos((l1 ** 2 + OB ** 2 - l2 ** 2) / (2 * l1 * OB))
        theta_2 = theta - theta_1
        h2 = math.cos(theta_2) * OB
        return h1, h2

    def calculate_ground_footing(self):
        joints = self.get_joint_position()
        leg0h1, leg0h2 = self.get_joint_position_wrt_body(joints[1], joints[2])
        leg1h1, leg1h2 = self.get_joint_position_wrt_body(joints[4], joints[5])
        leg2h1, leg2h2 = self.get_joint_position_wrt_body(joints[7], joints[8])
        leg3h1, leg3h2 = self.get_joint_position_wrt_body(joints[10], joints[11])

        return leg0h1, leg0h2, leg1h1, leg1h2, leg2h1, leg2h2, leg3h1, leg3h2
    
    def get_elbows(self):
        obs = self.get_obs()
        
        initial_pos = np.array([0, 0, obs[2]]).reshape((3, 1))

        # rotate_z
        yaw = obs[5]
        # rotate_y
        pitch = obs[4]
        # rotate_x
        roll = obs[3]

        body_l = 0.335
        body_w = 0.24
        body_h = 0.104
        upper_link_l = 0.206
        hip_from_body_l = 0.2263
        hip_from_body_w = 0.07
        upper_from_hip_w = 0.1
        hip_w = 0.11
        hip_l = 0.08

        current_joint = self.get_joint_position()

        hip_FL = current_joint[0]
        hip_FR = current_joint[6]
        hip_BL = current_joint[3]
        hip_BR = current_joint[9]
        upper_FL = current_joint[1]
        upper_FR = current_joint[7]
        upper_BL = current_joint[4]
        upper_BR = current_joint[10]

        FL_elbow = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(hip_from_body_l) + rotate_x(-np.pi * 0.5) @ (translate_z(hip_from_body_w) + rotate_x(hip_FL) @ (translate_x(hip_l*0.5) + translate_z(upper_from_hip_w) + rotate_z(np.pi - upper_FL) @ (translate_x(upper_link_l) + rotate_z(-np.pi+upper_FL) @ rotate_x(-hip_FL) @ rotate_x(np.pi*0.5) @ initial_pos))))

        FR_elbow = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(hip_from_body_l) + rotate_x(np.pi * 0.5) @ (translate_z(hip_from_body_w) + rotate_x(-hip_FR) @ (translate_x(hip_l*0.5) + translate_z(upper_from_hip_w) + rotate_z(np.pi + upper_FR) @ (translate_x(upper_link_l) + rotate_z(-np.pi - upper_FR) @ rotate_x(hip_FR) @ rotate_x(-np.pi*0.5) @ initial_pos))))

        BL_elbow = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(-hip_from_body_l) + rotate_x(-np.pi * 0.5) @ (translate_z(hip_from_body_w) + rotate_x(hip_BL) @ (translate_x(hip_l*0.5) + translate_z(upper_from_hip_w) + rotate_z(np.pi - upper_BL) @ (translate_x(upper_link_l) + rotate_z(-np.pi+upper_BL) @ rotate_x(-hip_BL) @ rotate_x(np.pi*0.5) @ initial_pos))))

        BR_elbow = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(-hip_from_body_l) + rotate_x(np.pi * 0.5) @ (translate_z(hip_from_body_w) + rotate_x(-hip_BR) @ (translate_x(hip_l*0.5) + translate_z(upper_from_hip_w) + rotate_z(np.pi + upper_BR) @ (translate_x(upper_link_l) + rotate_z(-np.pi-upper_BR) @ rotate_x(hip_BR) @ rotate_x(-np.pi*0.5) @ initial_pos))))

        return np.concatenate([FL_elbow, FR_elbow, BL_elbow, BR_elbow], axis=1)
    
    def get_body_corners(self):
        obs = self.get_obs()
        
        initial_pos = np.array([0, 0, obs[2]]).reshape((3, 1))

        # rotate_z
        yaw = obs[5]
        # rotate_y
        pitch = obs[4]
        # rotate_x
        roll = obs[3]

        # 0.335 0.24 0.104

        L = 0.36
        W = 0.35
        H = 0.104

        FL_top = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(L*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(H*0.5) + initial_pos)))
        FL_bot = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(L*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(-H*0.5) + initial_pos)))

        FR_top = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(L*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(H*0.5) + initial_pos)))
        FR_bot = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(L*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(-H*0.5) + initial_pos)))

        BL_top = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(-L*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(H*0.5) + initial_pos)))
        BL_bot = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(-L*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(-H*0.5) + initial_pos)))

        BR_top = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(-L*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(H*0.5) + initial_pos)))
        BR_bot = rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ (translate_x(-L*0.5) + rotate_x(np.pi * 0.5) @ (translate_z(W*0.5) + rotate_x(-np.pi * 0.5) @ (translate_z(-H*0.5) + initial_pos)))

        return np.concatenate([FL_top, FL_bot, FR_top, FR_bot, BL_top, BL_bot, BR_top, BR_bot], axis=1)