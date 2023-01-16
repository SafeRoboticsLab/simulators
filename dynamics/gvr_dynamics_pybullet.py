import numpy as np 
import pybullet as p
from .base_pybullet_dynamics import BasePybulletDynamics
from typing import Optional, Tuple, Any
from .resources.gvr import GVR
import time
import matplotlib.pyplot as plt
from jaxlib.xla_extension import DeviceArray

class GVRDynamicsPybullet(BasePybulletDynamics):
    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        #! TODO: FIX THIS, SO THAT THERE WILL BE A SEPARATE DYNAMICS WHEN WE USE ISAACS (BaseDstbDynamics instead of BaseDynamics)
        if isinstance(action_space, dict):
            super().__init__(config, action_space["ctrl"])
            self.flipper_increment_min = action_space["ctrl"][0, 0]
            self.flipper_increment_max = action_space["ctrl"][0, 1]
            self.wheel_velocity_min = action_space["ctrl"][2, 0]
            self.wheel_velocity_max = action_space["ctrl"][2, 1]
        else:
            super().__init__(config, action_space)
            self.flipper_increment_min = action_space[0, 0]
            self.flipper_increment_max = action_space[0, 1]
            self.wheel_velocity_min = action_space[2, 0]
            self.wheel_velocity_max = action_space[2, 1]
        
        self.dim_x = 26 # 9 + 9 + 4 + 4
        self.dim_u = 4

        self.flipper_min = -1.57
        self.flipper_max = 1.57

        self.initial_height = None
        self.initial_rotation = None
        self.initial_joint_value = None

        self.rendered_img = None
        self.adv_debug_line_id = None
        self.shielding_status_debug_text_id = None

        self.envtype = config.ENVTYPE
        self.payload = config.PAYLOAD
        self.payload_max = config.PAYLOAD_MAX

        self.reset()
    
    def reset(self, **kwargs):
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
                    height = 2.4 + np.random.rand()*0.2
                else:
                    height = 2.8
            self.initial_height = height

            if rotate is None:
                if self.rotate_reset:  # Resets the yaw, pitch, roll.
                    rotate = p.getQuaternionFromEuler((np.random.rand(3)-0.5) * np.pi * 0.125)
                else:
                    rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            self.initial_rotation = rotate
            
            self.robot = GVR(self.client, height, rotate, 
                envtype=self.envtype, payload=self.payload, payload_max=self.payload_max, **kwargs)

            if not is_rollout_shielding_reset:
                if random_joint_value is None:
                    random_joint_value = self.get_random_joint_value()
                self.initial_joint_value = random_joint_value
                
                self.robot.reset(random_joint_value)
                self.robot.apply_position(random_joint_value)

                for t in range(0, 100):
                    p.stepSimulation(physicsClientId = self.client)

            spirit_initial_obs = self.robot.get_obs()
            self.state = np.concatenate((
                np.array(spirit_initial_obs, dtype=np.float32), 
                np.array(spirit_initial_obs, dtype=np.float32), 
                np.concatenate((random_joint_value, np.array([0.0, 0.0])), axis=0), 
                np.concatenate((random_joint_value, np.array([0.0, 0.0])), axis=0),
            ), axis = 0)

            if max(self.robot.safety_margin(self.state).values()) <= 0 or is_rollout_shielding_reset:
                break
    
    def get_constraints(self):
        return self.robot.safety_margin(self.state)

    def get_target_margin(self):
        return self.robot.target_margin(self.state)

    def get_random_joint_value(self):
        return (
            np.random.uniform(self.flipper_min, self.flipper_max),
            np.random.uniform(self.flipper_min, self.flipper_max)
        )
    
    def integrate_forward(self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if adversary is not None:
            has_adversarial = True
        else:
            has_adversarial = False

        gvr_old_obs = self.robot.get_obs()
        gvr_old_joint_pos = np.array(self.robot.get_flipper_joint_position(), dtype = np.float32)
        gvr_old_wheel_vel = np.array(self.robot.get_wheel_velocity(), dtype = np.float32)

        clipped_control = []

        for i, j in enumerate(control[0:2]):
            increment = np.clip(j, self.flipper_increment_min, self.flipper_increment_max)
            if self.flipper_min <= gvr_old_joint_pos[i] + increment <= self.flipper_max:
                clipped_control.append(increment)
            else:
                clipped_control.append(
                    np.clip(gvr_old_joint_pos[i] + increment, self.flipper_min, self.flipper_max) - gvr_old_joint_pos[i]
                )
        
        for i, j in enumerate(control[2:4]):
            clipped_control.append(np.clip(j, self.wheel_velocity_min, self.wheel_velocity_max))
        
        self.robot.apply_action(clipped_control)
        if has_adversarial:
            force_vector = adversary[0:3]
            position_vector = adversary[3:]
            self._apply_adversarial_force(force_vector=force_vector, position_vector=position_vector)
        else:
            self._apply_force()
        
        # weird hack from env_hexapod so that the spring payload will work well
        p.setGravity(0, 0, self.gravity, physicsClientId = self.client)
        p.stepSimulation(physicsClientId = self.client)

        if self.gui:
            if has_adversarial:
                if self.adv_debug_line_id is not None:
                    p.removeUserDebugItem(self.adv_debug_line_id)
                self.adv_debug_line_id = p.addUserDebugLine(position_vector, position_vector + force_vector * self.force, lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id)
            time.sleep(self.dt)
            
            if self.video_output_file is not None:
                self._save_frames()
            
            self.debugger.cam_and_robotstates(self.robot.id)
        elif self.gui_imaginary:
            self.render()
        
        gvr_new_obs = np.array(self.robot.get_obs(), dtype = np.float32)
        gvr_new_joint_pos = np.array(self.robot.get_flipper_joint_position(), dtype = np.float32)
        gvr_new_wheel_vel = np.array(self.robot.get_wheel_velocity(), dtype = np.float32)

        self.state = np.concatenate((
            gvr_new_obs, 
            gvr_old_obs, 
            np.concatenate((gvr_new_joint_pos, gvr_new_wheel_vel), axis=0), 
            np.concatenate((gvr_old_joint_pos, gvr_old_wheel_vel), axis=0)), axis=0)
        
        self.cnt += 1

        if has_adversarial:
            return self.state, clipped_control, adversary
        else:
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

    def integrate_forward_jax(self, state: DeviceArray, control: DeviceArray) -> Tuple[DeviceArray, DeviceArray]:
        return super().integrate_forward_jax(state, control)
    
    def _integrate_forward(self, state: DeviceArray, control: DeviceArray) -> DeviceArray:
        return super()._integrate_forward(state, control)