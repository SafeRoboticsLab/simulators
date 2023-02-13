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
            ctrl_action_space = action_space["ctrl"]
        else:
            super().__init__(config, action_space)
            ctrl_action_space = action_space

        self.linear_vel_min = ctrl_action_space[0, 0]
        self.linear_vel_max = ctrl_action_space[0, 1]
        self.angular_vel_min = ctrl_action_space[1, 0]
        self.angular_vel_max = ctrl_action_space[1, 1]
        self.flipper_min = ctrl_action_space[2, 0]
        self.flipper_max = ctrl_action_space[2, 1]

        self.flipper_increment_min = self.dt/10*2.6
        self.flipper_increment_max = self.dt/10*2.6

        self.dim_u = 3 # user's input linear_x, angular_z, flip_pos
        self.dim_x = 13

        self.initial_height = None
        self.initial_rotation = None
        self.initial_joint_value = None

        self.rendered_img = None
        self.adv_debug_line_id = None
        self.shielding_status_debug_text_id = None

        self.envtype = config.ENVTYPE

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
                if self.height_reset:
                    height = 0.4 + np.random.rand()*0.2
                else:
                    height = 0.6
            self.initial_height = height

            if rotate is None:
                if self.rotate_reset:  # Resets the x, y, z.
                    rotate = p.getQuaternionFromEuler(np.concatenate((
                            (np.random.rand(2)-0.5) * np.pi * 0.125,
                            np.array([np.random.uniform(0.0, 2*np.pi)])
                        ), axis=0))
                else:
                    rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            self.initial_rotation = rotate
            
            self.robot = GVR(self.client, height, rotate, 
                envtype=self.envtype, **kwargs)

            if not is_rollout_shielding_reset:
                if random_joint_value is None:
                    random_joint_value = self.get_random_joint_value()
                self.initial_joint_value = random_joint_value
                
                self.robot.reset(random_joint_value)
                self.robot.apply_position(random_joint_value)

                p.setGravity(0, 0, self.gravity*0.1, physicsClientId = self.client)
                for t in range(0, 100):
                    p.stepSimulation(physicsClientId = self.client)
                p.setGravity(0, 0, self.gravity, physicsClientId = self.client)

            self.state = np.array(self.robot.get_obs(), dtype = np.float32)

            if max(self.robot.safety_margin(self.state).values()) <= 0 or is_rollout_shielding_reset:
                break
    
    def get_constraints(self):
        return self.robot.safety_margin(self.state)

    def get_target_margin(self):
        return self.robot.target_margin(self.state)

    def get_random_joint_value(self):
        return np.random.uniform(self.flipper_min, self.flipper_max)
    
    def integrate_forward(self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: 13-D
            x_dot, y_dot, z_dot,
            roll, pitch, yaw
            w_x, w_y, w_z,
            flipper_pos, flipper_angular_vel,
            v_left, v_right
        Action: 3-D
        """
        # the control is user's control: 
        #! TODO: CHECK IF THE CLIPPED CONTROL IS CORRECT HERE, CURRENTLY NOT CLIPPING

        if adversary is not None:
            has_adversarial = True
        else:
            has_adversarial = False
        
        # clip the increment of flippers
        flipper_cur_pos, _ = self.robot.get_flipper_state()
        control[2] = flipper_cur_pos + np.clip(control[2], self.flipper_increment_min, self.flipper_increment_max)
        
        self.robot.apply_action(control)
        
        if not self._apply_dstb_from_adversarial_sequence():
            if has_adversarial:
                force_vector = adversary[0:3]
                position_vector = adversary[3:]
                self._apply_adversarial_force(force_vector=force_vector, position_vector=position_vector)
            else:
                self._apply_force()
        
        p.stepSimulation(physicsClientId = self.client)

        if self.gui:
            if has_adversarial:
                if self.adv_debug_line_id is not None:
                    p.removeUserDebugItem(self.adv_debug_line_id)
                if self.link_name is not None:
                    self.adv_debug_line_id = p.addUserDebugLine(position_vector, position_vector + force_vector * self.force, lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id, parentLinkIndex=self.robot.get_link_id(self.link_name))
                else:
                    self.adv_debug_line_id = p.addUserDebugLine(position_vector, position_vector + force_vector * self.force, lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id)
            time.sleep(self.dt)
            
            if self.video_output_file is not None:
                self._save_frames()
            
            self.debugger.cam_and_robotstates(self.robot.id)
        elif self.gui_imaginary:
            self.render()
        
        self.state = np.array(self.robot.get_obs(), dtype = np.float32)
        
        self.cnt += 1

        if has_adversarial:
            return self.state, control, adversary
        else:
            return self.state, control
    
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