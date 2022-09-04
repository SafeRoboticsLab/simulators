import numpy as np 
import pybullet as p
from .base_pybullet_dynamics import BasePybulletDynamics
from typing import Optional, Tuple, Any
from .resources.spirit import Spirit
import time
import matplotlib.pyplot as plt

class SpiritDynamicsPybullet(BasePybulletDynamics):
    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        super().__init__(config, action_space)

        self.dim_x = 42
        self.dim_u = 12 # Spirit angular joint position

        # action range
        # The 12 joints are categorized into abduction, hip and knee
        # Joints of similar category share similar max, min
        # NOTE: This is not the joint range, this is the increment range constraint
        self.abduction_increment_min = action_space[0, 0]
        self.abduction_increment_max = action_space[0, 1]
        self.hip_increment_min = action_space[1, 0]
        self.hip_increment_max = action_space[1, 1]
        self.knee_increment_min = action_space[2, 0]
        self.knee_increment_max = action_space[2, 1]

        # TODO: Read this value from URDF, or pass from config
        #! This is hardware constraints, different from action range
        self.abduction_min = -0.5
        self.abduction_max = 0.5
        self.hip_min = 0.5
        self.hip_max = 2.64
        self.knee_min = 0.5
        self.knee_max = 2.64

        self.initial_height = None
        self.initial_rotation = None
        self.initial_joint_value = None

        self.rendered_img = None

        self.reset()
    
    def reset(self, **kwargs):
        # rejection sampling until outside target set and safe set
        while True:
            super().reset(**kwargs)

            if "initial_height" in kwargs.keys():
                height = kwargs["initial_height"]
            else:
                height = None
            
            if "initial_rotation" in kwargs.keys():
                rotate = kwargs["initial_rotation"]
            else:
                rotate = None
            
            if "initial_joint_value" in kwargs.keys():
                random_joint_value = kwargs["initial_joint_value"]
            else:
                random_joint_value = None

            if "is_rollout_shielding_reset" in kwargs.keys():
                is_rollout_shielding_reset = kwargs["is_rollout_shielding_reset"]
            else:
                is_rollout_shielding_reset = False
            
            if height is None:
                if self.height_reset:  # Drops from the air.
                    height = 0.4 + np.random.rand()*0.2
                else:
                    height = 0.6
            self.initial_height = height

            if rotate is None:
                if self.rotate_reset:  # Resets the yaw, pitch, roll.
                    rotate = p.getQuaternionFromEuler((np.random.rand(3)-0.5) * np.pi * 0.125)
                else:
                    rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            self.initial_rotation = rotate
            
            self.robot = Spirit(self.client, height, rotate, **kwargs)

            if not is_rollout_shielding_reset:
                if random_joint_value is None:
                    random_joint_value = self.get_random_joint_value()
                self.initial_joint_value = random_joint_value
                
                self.robot.reset(random_joint_value)
                self.robot.apply_position(random_joint_value)

                for t in range(0, 100):
                    p.stepSimulation(physicsClientId = self.client)

            spirit_initial_obs = self.robot.get_obs()
            self.state = np.concatenate((np.array(spirit_initial_obs, dtype=np.float32), np.array(spirit_initial_obs, dtype=np.float32), random_joint_value, random_joint_value), axis = 0)

            # print(
            #     max(self.robot.safety_margin(self.state).values()), 
            #     max(self.robot.target_margin(self.state).values())
            # )

            # input()

            # if max(self.robot.target_margin(self.state).values()) > 0 and max(self.robot.safety_margin(self.state).values()) <= 0:
            
            if max(self.robot.safety_margin(self.state).values()) <= 0 or is_rollout_shielding_reset:
                break
    
    def get_constraints(self):
        return self.robot.safety_margin(self.state)

    def get_target_margin(self):
        return self.robot.target_margin(self.state)

    def get_random_joint_value(self):
        # return (
        #     np.random.uniform(self.abduction_min, self.abduction_max),
        #     np.random.uniform(self.hip_min, self.hip_max),
        #     np.random.uniform(self.knee_min, self.knee_max),
        #     np.random.uniform(self.abduction_min, self.abduction_max),
        #     np.random.uniform(self.hip_min, self.hip_max),
        #     np.random.uniform(self.knee_min, self.knee_max),
        #     np.random.uniform(self.abduction_min, self.abduction_max),
        #     np.random.uniform(self.hip_min, self.hip_max),
        #     np.random.uniform(self.knee_min, self.knee_max),
        #     np.random.uniform(self.abduction_min, self.abduction_max),
        #     np.random.uniform(self.hip_min, self.hip_max),
        #     np.random.uniform(self.knee_min, self.knee_max)
        # )

        return (
            np.random.uniform(self.abduction_min, self.abduction_max),
            0.6 + np.random.uniform(-0.5, 0.5),
            1.45 + np.random.uniform(-0.5, 0.5),
            np.random.uniform(self.abduction_min, self.abduction_max),
            0.6 + np.random.uniform(-0.5, 0.5),
            1.45 + np.random.uniform(-0.5, 0.5),
            np.random.uniform(self.abduction_min, self.abduction_max),
            0.6 + np.random.uniform(-0.5, 0.5),
            1.45 + np.random.uniform(-0.5, 0.5),
            np.random.uniform(self.abduction_min, self.abduction_max),
            0.6 + np.random.uniform(-0.5, 0.5),
            1.45 + np.random.uniform(-0.5, 0.5)
        )
    
    def get_random_joint_increment_from_current(self):
        increment = (
            np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
            np.random.uniform(self.hip_increment_min, self.hip_increment_max),
            np.random.uniform(self.knee_increment_min, self.knee_increment_max),
            np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
            np.random.uniform(self.hip_increment_min, self.hip_increment_max),
            np.random.uniform(self.knee_increment_min, self.knee_increment_max),
            np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
            np.random.uniform(self.hip_increment_min, self.hip_increment_max),
            np.random.uniform(self.knee_increment_min, self.knee_increment_max),
            np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
            np.random.uniform(self.hip_increment_min, self.hip_increment_max),
            np.random.uniform(self.knee_increment_min, self.knee_increment_max)
        )

        return np.array(self.robot.get_joint_position()) + np.array(increment)

    def integrate_forward(self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the Pybullet physics simulation engine to do 1-step integrate_forward.

        Args:
            state (np.ndarray): Dummy data, the system state will be maintained by Pybullet instead of passing in.
            control (np.ndarray): _description_
            num_segment (Optional[int], optional): _description_. Defaults to 1.
            noise (Optional[np.ndarray], optional): _description_. Defaults to None.
            noise_type (Optional[str], optional): _description_. Defaults to 'unif'.
            adversary (Optional[np.ndarray], optional): The adversarial action, this is the force vector, force applied position, and terrain information. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        if adversary is not None:
            has_adversarial = True
        else:
            has_adversarial = False

        # get the current state of the robot
        spirit_old_obs = self.robot.get_obs()
        spirit_old_joint_pos = np.array(self.robot.get_joint_position(), dtype = np.float32)

        # check clipped control
        clipped_control = []
        for i, j in enumerate(control):
            if i % 3 == 0:
                increment = np.clip(j, self.abduction_increment_min, self.abduction_increment_max)
                if self.abduction_min <= spirit_old_joint_pos[i] + increment <= self.abduction_max:
                    clipped_control.append(increment)
                else:
                    clipped_control.append(
                        np.clip(spirit_old_joint_pos[i] + increment, self.abduction_min, self.abduction_max) - spirit_old_joint_pos[i]
                    )
            elif i % 3 == 1:
                increment = np.clip(j, self.hip_increment_min, self.hip_increment_max)
                if self.hip_min <= spirit_old_joint_pos[i] + increment <= self.hip_max:
                    clipped_control.append(increment)
                else:
                    clipped_control.append(
                        np.clip(spirit_old_joint_pos[i] + increment, self.hip_min, self.hip_max) - spirit_old_joint_pos[i]
                    )
            elif i % 3 == 2:
                increment = np.clip(j, self.knee_increment_min, self.knee_increment_max)
                if self.knee_min <= spirit_old_joint_pos[i] + increment <= self.knee_max:
                    clipped_control.append(increment)
                else:
                    clipped_control.append(
                        np.clip(spirit_old_joint_pos[i] + increment, self.knee_min, self.knee_max) - spirit_old_joint_pos[i]
                    )
        
        # TODO: check clipped adversarial control
        
        self.robot.apply_action(clipped_control)
        if has_adversarial:
            force_vector = adversary[0:3]
            position_vector = adversary[3:]
            self._apply_adversarial_force(force_vector=force_vector, position_vector=position_vector)
        else:
            self._apply_force()

        p.stepSimulation(physicsClientId = self.client)

        if self.gui:
            if has_adversarial:
                p.addUserDebugLine(position_vector, position_vector + force_vector * self.force, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.1, physicsClientId=self.client, parentObjectUniqueId=self.robot.id)
            time.sleep(self.dt)
            
            if self.video_output_file is not None:
                self._save_frames()
            
            self.debugger.cam_and_robotstates(self.robot.id)
        elif self.gui_imaginary:
            self.render()

        spirit_new_obs = np.array(self.robot.get_obs(), dtype = np.float32)
        spirit_new_joint_pos = np.array(self.robot.get_joint_position(), dtype = np.float32)

        self.state = np.concatenate((spirit_new_obs, spirit_old_obs, spirit_new_joint_pos, spirit_old_joint_pos), axis=0)

        self.cnt += 1

        return self.state, clipped_control
    
    def render(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((200, 200, 4)))

        # Base information
        robot_id, client_id = self.robot.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100, physicsClientId = self.client)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(robot_id, client_id)]

        pos[0] += 1.0
        pos[1] -= 1.0
        pos[2] += 0.7
        ori = p.getQuaternionFromEuler([0, 0.2, np.pi * 0.8])

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(200, 200, view_matrix, proj_matrix, physicsClientId = self.client)[2]
        frame = np.reshape(frame, (200, 200, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.axis('off')
        plt.title("Rollout imagine env")
        plt.pause(.00001)
