import numpy as np 
import pybullet as p
from .base_pybullet_dynamics import BasePybulletDynamics
from typing import Optional, Tuple, Any
from .resources.spirit import Spirit
import time

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

        self.reset()        
    
    def reset(self):
        super().reset()
        # add the robot to the Pybullet engine
        if self.height_reset:  # Drops from the air.
            height = 0.4 + np.random.rand()*0.2
        else:
            height = 0.6

        if self.rotate_reset:  # Resets the yaw, pitch, roll.
            rotate = p.getQuaternionFromEuler((np.random.rand(3)-0.5) * np.pi * 0.125)
        else:
            rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        
        self.robot = Spirit(self.client, height, rotate)

        random_joint_value = self.get_random_joint_value(target_set=True)
        
        self.robot.reset(random_joint_value)
        self.robot.apply_position(random_joint_value)

        for t in range(0, 100):
            p.stepSimulation()

        spirit_initial_obs = self.robot.get_obs()
        self.state = np.concatenate((np.array(spirit_initial_obs, dtype=np.float32), np.array(spirit_initial_obs, dtype=np.float32), random_joint_value, random_joint_value), axis = 0)
    
    def get_constraints(self):
        return self.robot.safety_margin()

    def get_target_margin(self):
        return self.robot.target_margin()

    def get_random_joint_value(self, target_set = False):
        if target_set:
            return (
                0.0 + np.random.uniform(-0.3, 0.3),
                0.6 + np.random.uniform(-0.3, 0.3),
                1.45 + np.random.uniform(-0.25, 0.25),
                0.0 + np.random.uniform(-0.3, 0.3),
                0.6 + np.random.uniform(-0.3, 0.3),
                1.45 + np.random.uniform(-0.25, 0.25),
                0.0 + np.random.uniform(-0.3, 0.3),
                0.6 + np.random.uniform(-0.3, 0.3),
                1.45 + np.random.uniform(-0.25, 0.25),
                0.0 + np.random.uniform(-0.3, 0.3),
                0.6 + np.random.uniform(-0.3, 0.3),
                1.45 + np.random.uniform(-0.25, 0.25)
            )
        else:
            return (
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.5, 2.64),
                np.random.uniform(0.5, 2.64),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.5, 2.64),
                np.random.uniform(0.5, 2.64),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.5, 2.64),
                np.random.uniform(0.5, 2.64),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.5, 2.64),
                np.random.uniform(0.5, 2.64)
            )
    
    def get_random_joint_increment_from_current(self):
        increment = (
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05),
            np.random.uniform(-np.pi * 0.05, np.pi * 0.05)
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
            adversary (Optional[np.ndarray], optional): _description_. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        # get the current state of the robot
        spirit_old_obs = self.robot.get_obs()
        spirit_old_joint_pos = np.array(self.robot.get_joint_position(), dtype = np.float32)

        # check clipped control
        clipped_control = []
        for i, j in enumerate(control):
            if i % 3 == 0:
                if self.abduction_min <= spirit_old_joint_pos[i] + j <= self.abduction_max:
                    clipped_control.append(np.clip(j, self.abduction_increment_min, self.abduction_increment_max))
                else:
                    clipped_control.append(
                        np.clip(spirit_old_joint_pos[i] + j, self.abduction_min, self.abduction_max) - spirit_old_joint_pos[i]
                    )
            elif i % 3 == 1:
                if self.hip_min <= spirit_old_joint_pos[i] + j <= self.hip_max:
                    clipped_control.append(np.clip(j, self.hip_increment_min, self.hip_increment_max))
                else:
                    clipped_control.append(
                        np.clip(spirit_old_joint_pos[i] + j, self.hip_min, self.hip_max) - spirit_old_joint_pos[i]
                    )
            elif i % 3 == 2:
                if self.knee_min <= spirit_old_joint_pos[i] + j <= self.knee_max:
                    clipped_control.append(np.clip(j, self.knee_increment_min, self.knee_increment_max))
                else:
                    clipped_control.append(
                        np.clip(spirit_old_joint_pos[i] + j, self.knee_min, self.knee_max) - spirit_old_joint_pos[i]
                    )

        self.robot.apply_action(clipped_control)
        self._apply_force()

        p.stepSimulation(physicsClientId = self.client)

        if self.gui:
            time.sleep(self.dt)

        spirit_new_obs = np.array(self.robot.get_obs(), dtype = np.float32)
        spirit_new_joint_pos = np.array(self.robot.get_joint_position(), dtype = np.float32)

        self.state = np.concatenate((spirit_new_obs, spirit_old_obs, spirit_new_joint_pos, spirit_old_joint_pos), axis=0)
        
        self.debugger.cam_and_robotstates(self.robot.id)

        return self.state, clipped_control