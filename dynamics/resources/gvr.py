import pybullet as p
import os
import math
import numpy as np
from simulators.dynamics.resources.utils import *

class GVR:
    def __init__(self, client, height, orientation, envtype="normal", payload=0.0 , payload_max=10.0, has_flipper=False, **kwargs):
        self.client = client
        self.height = height

        ox = 0
        oy = 0

        self.urdf = "gvr_bot/gvrbot_updated.urdf"
        self.urdf_path = os.path.join(os.path.dirname(__file__), self.urdf)

        self.payload = payload
        self.payload_max = payload_max
        self.envtype = envtype

        self.has_flipper = has_flipper

        if envtype != "normal":
            self._gen_urdf()

        self.id = p.loadURDF(fileName = self.urdf_path, basePosition=[ox, oy, self.height], baseOrientation = orientation, physicsClientId = client)

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

        self.left_wheel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.right_wheel_index = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.flipper_joint_index = [22, 23]

    def get_ids(self):
        return self.id, self.client
    
    def reset(self, position):
        for i in range(len(self.flipper_joint_index)):
            p.resetJointState(self.id, self.flipper_joint_index[i], position[i], physicsClientId = self.client)
    
    def apply_action(self, action):
        # first 2 actions are the increment of the left and right flipper angular position, next 2 actions are the target velocity of the left and right wheel, respectively
        # if there is no flipper, action will only be for the wheels
        if self.has_flipper:
            new_flipper_angle = np.array(self.get_flipper_joint_position()) + np.array(action[0:2])
        for joint in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, joint)
            if self.has_flipper:
                if "LF_flipper" in str(info[1]):
                    p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = new_flipper_angle[0], force=info[10], maxVelocity=info[11])
                elif "RF_flipper" in str(info[1]):
                    p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition = new_flipper_angle[1], force=info[10], maxVelocity=info[11])
                else:
                    if "base_to_L" in str(info[1]):
                        p.setJointMotorControl2(self.id, joint, p.VELOCITY_CONTROL, targetVelocity = action[2])
                    elif "base_to_R" in str(info[1]):
                        p.setJointMotorControl2(self.id, joint, p.VELOCITY_CONTROL, targetVelocity = action[3])
            else:
                if "base_to_L" in str(info[1]):
                        p.setJointMotorControl2(self.id, joint, p.VELOCITY_CONTROL, targetVelocity = action[0])
                elif "base_to_R" in str(info[1]):
                    p.setJointMotorControl2(self.id, joint, p.VELOCITY_CONTROL, targetVelocity = action[1])
    
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
        left_vel = [state[1] for state in left_wheel_joint_state]
        right_vel = [state[1] for state in right_wheel_joint_state]
        return [left_vel[0], right_vel[0]]

    def _gen_urdf(self):
        fin = open(self.urdf_path)
        urdf_content = fin.read()
        fin.close()

        flipper_definition = """
            <!-- FRONT LEFT FLIPPER-->

            <joint name="base_to_LF_flipper" type="revolute">
                <parent link="base_link"/>
                <child link="LF_flipper_main"/>
                <limit effort="30.0" lower="-1.57079632679" upper="1.57079632679" velocity="7.0"/>
                <!--Add the wheel thickness and the half width of the flipper and 7 mm to get the y offset-->
                <origin xyz="0.2605 -0.2418 0.0125"/>
                <axis xyz="0 1 0"/>
                <dynamics damping="0.0"/>
            </joint>

            <link name="LF_flipper_main">
                <visual>
                <geometry>
                    <mesh filename="gvrbot_step_files/flipper.stl" scale="0.001 0.001 0.001"/>
                </geometry>
                <origin rpy="0 0 1.57075" xyz="0 0 0"/>
                <material name="Blue" />
                </visual>
                <collision>
                <geometry>
                    <mesh filename="gvrbot_step_files/flipper.stl" scale="0.001 0.001 0.001"/>
                </geometry>
                <origin rpy="0 0 1.57075" xyz="0 0 0"/>
                </collision>
                <inertial>
                    <mass value="1.026" />
                    <origin rpy="0 0 0" xyz="0.03 0 0 " />
                    <inertia ixx="0.000926" ixy="0" ixz="0" iyy="0.001459" iyz="0" izz="0.00825" />
                </inertial>
            </link>
            
            <!-- FRONT RIGHT FLIPPER-->

            <joint name="base_to_RF_flipper" type="revolute">
                <parent link="base_link"/>
                <child link="RF_flipper_main"/>
                <limit effort="30.0" lower="-1.57079632679" upper="1.57079632679" velocity="7.0"/>
                <origin xyz="0.2605 0.2418 0.0125"/>
                <axis xyz="0 1 0"/>
                <dynamics damping="0.0"/>
            </joint>

            <link name="RF_flipper_main">
                <visual>
                <geometry>
                    <mesh filename="gvrbot_step_files/flipper.stl" scale="0.001 0.001 0.001"/>
                </geometry>
                <origin rpy="0 0 1.57075" xyz="0 0 0"/>
                <material name="Blue" />
                </visual>
                <collision>
                <geometry>
                    <mesh filename="gvrbot_step_files/flipper.stl" scale="0.001 0.001 0.001"/>
                </geometry>
                <origin rpy="0 0 1.57075" xyz="0 0 0"/>
                </collision>
                <inertial>
                    <mass value="1.026" />
                    <origin rpy="0 0 0" xyz="0.03 0 0" />
                    <inertia ixx="0.000926" ixy="0" ixz="0" iyy="0.001459" iyz="0" izz="0.00825" />
                </inertial>
            </link>
        """

        payload_definition = ""
        
        # Update payload scale
        if self.envtype == 'payload':
            payload_definition = """<!-- ADDING PAYLOAD -->
                <joint name="body_payload" type="fixed">
                    <parent link="base_link"/>
                    <child link="payload"/>
                </joint>
                <link name="payload">
                    <visual>
                    <origin rpy="0 0 0" xyz="0.0 0.0 @ZLOCATION"/>  <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 @HEIGHT"/>
                    </geometry>
                    <material name="Yellow"/>
                    </visual>
                    <collision>
                    <origin rpy="0 0 0" xyz="0.00 0.00 @ZLOCATION"/> <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 @HEIGHT"/>
                    </geometry>
                    </collision>
                    <inertial>
                    <!-- CENTER OF MASS -->
                    <origin rpy="0 0 0" xyz="0.0 0.0 @ZCOM"/>
                    <mass value="@MASS"/>
                    <!-- box inertia: 1/12*m(y^2+z^2), ... -->
                    <inertia ixx="@IXX" ixy="0" ixz="0" iyy="@IYY" iyz="0" izz="@IZZ"/>
                    </inertial>
                </link>
                <!-- END PAYLOAD --> 
            """

            payload_mass = self.payload * self.payload_max
            # payload_height = self.payload * 0.45
            payload_height = self.payload * 1.5
            payload_definition = payload_definition \
                .replace('@MASS', str(payload_mass) ) \
                .replace('@HEIGHT', str(payload_height) ) \
                .replace('@ZLOCATION', str(payload_height/2+0.02) ) \
                .replace('@IXX', str(1/12 * payload_mass * (0.15*0.15 + payload_height * payload_height))) \
                .replace('@IYY', str(1/12 * payload_mass * (0.15*0.15 + payload_height * payload_height))) \
                .replace('@IZZ', str(1/12 * payload_mass * (2 * 0.15*0.15)))
        
        elif self.envtype == 'spring':
            payload_mass = self.payload * self.payload_max
            payload_blocks = 6
            payload_active = int(self.payload * payload_blocks)
            block_mass = self.payload_max / float(payload_blocks)

            remaining = 0
            if not payload_active * block_mass == payload_mass:
                remaining = payload_mass - payload_active * block_mass
                
            fixed_joint_definition = """
                <joint name="@JOINTNAME" type="fixed">
                    <parent link="@PARENTLINK"/>
                    <child link="@CHILDLINK"/>
                </joint>
            """

            revolute_joint_definition = """
                <joint name="@JOINTNAME" type="revolute">
                    <parent link="@PARENTLINK"/>
                    <child link="@CHILDLINK"/>
                    <limit effort="10.0" lower="-0.5" upper="0.5" velocity="20.0"/>
                    <origin rpy="0 0 0.0" xyz="0.0 0.0 @ZLOCATION"/>
                    <axis xyz="@AXIS"/>
                    <dynamics damping="0.02"/>
                </joint>
            """

            block_definition = """
                <link name="@LINKNAME">
                    <visual>
                    <origin rpy="0 0 0" xyz="0.0 0.0 @ZLOCATION"/>  <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 0.2"/>
                    </geometry>
                    <material name="@COLOR"/>
                    </visual>
                    <collision>
                    <origin rpy="0 0 0" xyz="0.00 0.00 @ZLOCATION"/> <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 0.2"/>
                    </geometry>
                    </collision>
                    <inertial>
                    <!-- CENTER OF MASS -->
                    <origin rpy="0 0 0" xyz="0.0 0.0 0.1"/>
                    <mass value="@MASS"/>
                    <!-- box inertia: 1/12*m(y^2+z^2), ... -->
                    <inertia ixx="@IXX" ixy="0" ixz="0" iyy="@IYY" iyz="0" izz="@IZZ"/>
                    </inertial>
                </link>
            """

            payload_definition = "<!-- ADDING PAYLOAD -->\n"
            if remaining > 0.01:
                total_range = payload_active +1
            else:
                total_range = payload_active
            for se_ in range(total_range):
                if not se_:
                    #the fixed joint should always be the first element of the definition
                    payload_definition += fixed_joint_definition\
                        .replace("@JOINTNAME","body_payload")\
                        .replace("@PARENTLINK","base_link")\
                        .replace("@CHILDLINK",f"payload_block_{se_}")
                else:
                    payload_definition += revolute_joint_definition\
                        .replace("@JOINTNAME",f"payload{se_-1}_payload{se_}")\
                        .replace("@PARENTLINK",f"payload_block_{se_-1}")\
                        .replace("@CHILDLINK",f"payload_block_{se_}")\
                        .replace("@ZLOCATION","0.2")\
                        .replace("@AXIS","0 1 0" if se_ % 2 else "1 0 0")
                current_block_mass = block_mass if se_ != payload_active else remaining
                payload_definition += block_definition\
                    .replace("@LINKNAME",f"payload_block_{se_}")\
                    .replace("@COLOR", "Yellow" if se_ % 2 else "Green" )\
                    .replace("@ZLOCATION","0.2")\
                    .replace("@MASS",str(block_mass))\
                    .replace('@IXX', str(1/12 * block_mass * (0.15*0.15 + 0.2 * 0.2))) \
                    .replace('@IYY', str(1/12 * block_mass * (0.15*0.15 + 0.2 * 0.2))) \
                    .replace('@IZZ', str(1/12 * block_mass * (2 * 0.15*0.15)))
            payload_definition += " \n<!-- END PAYLOAD -->"
        
        elif self.envtype == "sail":
            payload_definition = """
                <!-- ADDING PAYLOAD -->
                <joint name="body_payload" type="fixed">
                    <parent link="base_link"/>
                    <child link="sail_base"/>
                </joint>

                <link name="sail_base">
                    <visual>
                        <origin rpy="0 0 0" xyz="-0.194 0.0 0.48"/>
                        <geometry>
                            <box size="0.03 0.05 0.90"/>
                        </geometry>
                        <material name="Black"/>
                    </visual>
                    <collision>
                        <origin rpy="0 0 0" xyz="-0.194 0 0.48"/>
                        <geometry>
                            <box size="0.03 0.05 0.90"/>
                        </geometry>
                    </collision>
                    <inertial>
                        <origin rpy="0 0 0" xyz="0.015 0.025 0.45"/>
                        <mass value="1.0"/>
                        <inertia ixx="0.017" ixy="0.0" ixz="0.0" iyy="0.017" iyz="0.0" izz="0.00007"/>
                    </inertial>
                </link>

                <joint name="sail_connector" type="fixed">
                    <parent link="sail_base"/>
                    <child link="sail_panel"/>
                </joint>

                <link name="sail_panel">
                    <visual>
                        <origin rpy="0 0 0" xyz="0.271 0.0 0.53"/>
                        <geometry>
                            <box size="0.9 0.02 0.8"/>
                        </geometry>
                        <material name="Black"/>
                    </visual>
                    <collision>
                        <origin rpy="0 0 0" xyz="0.271 0 0.53"/>
                        <geometry>
                            <box size="0.9 0.02 0.8"/>
                        </geometry>
                    </collision>
                    <inertial>
                        <origin rpy="0 0 0" xyz="0.45 0.01 0.4"/>
                        <mass value="1.0"/>
                        <inertia ixx="0.053" ixy="0.0" ixz="0.0" iyy="0.12" iyz="0.0" izz="0.067"/>
                    </inertial>
                </link>
                <!-- END PAYLOAD -->
            """
        
        self.urdf_path = self.urdf_path[:-5]+"_tmp.urdf"
        fout = open(self.urdf_path, "w")
        
        new_content = urdf_content.replace("<!-- PAYLOAD_PLACEHOLDER -->", payload_definition)
        if self.has_flipper:
            new_content = new_content.replace("<!-- FLIPPERS_PLACEHOLDER -->", flipper_definition)
        
        fout.write(new_content)
        fout.close()
