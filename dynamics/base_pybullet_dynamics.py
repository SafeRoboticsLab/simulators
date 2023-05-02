from turtle import pos
from typing import Optional, Tuple, Any
import numpy as np
from simulators.dynamics.resources.ramp import Ramp
from simulators.dynamics.utils import change_dynamics
from simulators.pybullet_debugger import pybulletDebug
from .resources.plane import Plane
from .base_dynamics import BaseDynamics
import pybullet as p
import subprocess
import os
import math

class BasePybulletDynamics(BaseDynamics):
    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        super().__init__(config, action_space)
        """
        Initialize a Pybullet physics simulator to keep track of robot dynamics. This will only work for single agent

        Args:
            config (Any): an object specifies configuration. This will correspond to config_agent of the yaml config file
            action_space (np.ndarray): action space.
        """

        self.verbose = config.VERBOSE
        self.gui = config.GUI
        self.gui_imaginary = config.GUI_IMAGINARY
        self.dt = config.DT
        self.gravity = -9.81

        # configure force in the dynamics
        if config.APPLY_FORCE:
            self.force = float(config.FORCE) * float(config.FORCE_SCALE)
        else:
            self.force = 0.0
        self.force_info = config.FORCE_INFO
        self.elapsed_force_applied = 0
        self.force_applied_reset = config.FORCE_RESET_TIME

        self.rotate_reset = config.ROTATE_RESET
        self.height_reset = config.HEIGHT_RESET

        self.force_applied_force_vector = None
        self.force_applied_position_vector = None
        self.force_random = config.FORCE_RANDOM
        self.link_name = config.LINK_NAME

        # configure terrain in the dynamics (legacy)
        self.terrain_type = config.TERRAIN
        self.terrain_height = config.TERRAIN_HEIGHT
        self.terrain_gridsize = config.TERRAIN_GRIDSIZE
        self.terrain_friction = config.TERRAIN_FRICTION

        self.terrain_data = None

        self.change_dynamics = change_dynamics

        # extra stuff from LINC
        self.env_kwargs = {
            "terrain": config.TERRAIN,
            "lateral_friction": config.LATERAL_FRICTION,
            "anisotropic_friction": config.ANISOTROPIC_FRICTION,
            "rolling_friction": config.ROLLING_FRICTION,
            "spinning_friction": config.SPINNING_FRICTION,
            "restitution": config.RESTITUTION,
            "theta": config.THETA,
            "terrain_coeff": config.TERRAIN_COEFF,
            "terrain_height": config.TERRAIN_HEIGHT,
            "payload": config.PAYLOAD,
            "payload_mass": config.PAYLOAD_MASS,
            "payload_blocks": config.PAYLOAD_BLOCKS
        }

        self.world_objects = []

        # initialize a pybullet client (GUI/DIRECT)
        if self.gui:
            # Setup the GUI (disable the useless windows)
            self.camera_info = {
                'camera': {'distance': 12, 'yaw': -0, 'pitch': -89},
                'lookat': [0, 0, 0]}
            self._render_width = 640
            self._render_height = 480
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1,
                cameraYaw=20,
                cameraPitch=-20,
                cameraTargetPosition=[1, -0.5, 0.8])
            self.debugger = pybulletDebug()
        else:
            self.client = p.connect(p.DIRECT)

        self.video_output_file = None
        self.ffmpeg_pipe = None
        self.cnt = None

    def _init_frames(self):
        """
        Initialize the pipe for streaming frames to the video file.
        Warning: video slows down the simulation!
        """
        if self.ffmpeg_pipe is not None:
            try:
                if self.video_output_file is not None:
                    self.ffmpeg_pipe.stdin.close()
                    self.ffmpeg_pipe.stderr.close()
                    ret = self.ffmpeg_pipe.wait()
            except Exception as e:
                print(
                    "VideoRecorder encoder exited with status {}".format(ret))

        if self.video_output_file is not None:
            camera = p.getDebugVisualizerCamera()
            command = [
                'ffmpeg',
                '-y',
                '-r', str(24),
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', '{}x{}'.format(camera[0], camera[1]),
                '-pix_fmt', 'rgba',
                '-i', '-',
                '-an',
                '-vcodec', 'mpeg4',
                '-vb', '20M',
                self.video_output_file]
            self.ffmpeg_pipe = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _save_frames(self):
        """
        Write frame at each step.
        24 FPS dt = 1/240 : every 10 frames
        """
        # if self.video_output_file is not None and self.cnt % (int(1. / (self.dt * 24))) == 0:
        if self.video_output_file is not None:
            camera = p.getDebugVisualizerCamera(physicsClientId=self.client)
            img = p.getCameraImage(
                camera[0], camera[1],
                renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.client)
            self.ffmpeg_pipe.stdin.write(img[2].tobytes())
    
    def destroy(self):
        """Properly close the simulation."""
        try:
            p.disconnect()
        except p.error as e:
            print("Warning (destructor of simulator):", e)
        
        self.close_video_stream()
    
    def close_video_stream(self):        
        try:
            if self.gui and self.video_output_file is not None:
                self.ffmpeg_pipe.stdin.close()
                self.ffmpeg_pipe.stderr.close()
                ret = self.ffmpeg_pipe.wait()
        except Exception as e:
            print(
                "VideoRecorder encoder exited with status {}".format(ret))

    def reset(self, **kwargs):
        if "video_output_file" in kwargs.keys():
            self.video_output_file = kwargs["video_output_file"]
        
        if "adversarial_sequence" in kwargs.keys():
            self.adversarial_sequence = kwargs["adversarial_sequence"]
        else:
            # clean adversarial sequence, so that any adversarial sequence passed during a reset will only be used once
            self.adversarial_sequence = None
            
        self.cnt = 0
        p.resetSimulation(physicsClientId = self.client)
        p.setGravity(0, 0, self.gravity, physicsClientId = self.client)
        p.setTimeStep(self.dt, physicsClientId = self.client)
        p.setPhysicsEngineParameter(fixedTimeStep = self.dt, physicsClientId = self.client)
        p.setRealTimeSimulation(0, physicsClientId = self.client)

        if "terrain_data" in kwargs.keys():
            terrain_data = kwargs["terrain_data"]
        else:
            terrain_data = None
        
        if self.terrain_type == "normal": # legacy simple plane terrain
            self.base_plane = Plane(self.client).id
        elif self.terrain_type == "rough": # legacy rough terrain
            self.base_plane = Plane(self.client).id
            if terrain_data is None:
                self._gen_terrain(mesh_scale=[self.terrain_gridsize, self.terrain_gridsize, 2.0])
            else:
                self._set_terrain(terrain_data, mesh_scale=[self.terrain_gridsize, self.terrain_gridsize, 2.0])
        else: # updated LINC rough terrain [sawtooth, pyramids, etc,.]
            self.load_terrain()
        
        self._gen_force()

        if self.gui and self.video_output_file is not None:
            self._init_frames()

    def integrate_forward(self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return super().integrate_forward(state, control, num_segment, noise, noise_type, adversary, **kwargs)
    
    def get_jacobian(self, nominal_states: np.ndarray, nominal_controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super().get_jacobian(nominal_states, nominal_controls)

    def _gen_force(self):
        """
        Create a random force to be applied onto the robot
        The force will be applied when integrate_forward is called
        """
        # create a random force applied on the robot
        self.elapsed_force_applied = 0
        if self.force_info != None:
            self.force_applied_force_vector = np.array([
                np.random.uniform(self.force_info["vector"][0][0], self.force_info["vector"][0][1]),
                np.random.uniform(self.force_info["vector"][1][0], self.force_info["vector"][1][1]),
                np.random.uniform(self.force_info["vector"][2][0], self.force_info["vector"][2][1])
            ]) * self.force
            self.force_applied_position_vector = np.array([
                np.random.uniform(self.force_info["position"][0][0], self.force_info["position"][0][1]),
                np.random.uniform(self.force_info["position"][1][0], self.force_info["position"][1][1]),
                np.random.uniform(self.force_info["position"][2][0], self.force_info["position"][2][1])
            ])
        elif self.force_random:
            self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
            self.force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
        else:
            self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
            self.force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
    
    def _apply_adversarial_force(self, force_vector, position_vector):
        # clip adversarial force
        self.force_applied_force_vector = np.clip(
            force_vector * self.force,
            self.force_applied_force_vector - np.ones(3)*self.force*0.1,
            self.force_applied_force_vector + np.ones(3)*self.force*0.1
        )

        self.force_applied_position_vector = np.clip(
            position_vector,
            self.force_applied_position_vector - np.array([0.4, 0.01, 0.01])*0.1,
            self.force_applied_position_vector + np.array([0.4, 0.01, 0.01])*0.1
        )

        if self.link_name is not None:
            p.applyExternalForce(self.robot.id, self.robot.get_link_id(self.link_name), self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.client)
        else:
            p.applyExternalForce(self.robot.id, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.client)

    def _apply_force(self):
        if self.elapsed_force_applied > self.force_applied_reset:
            self._gen_force()
        else:
            self.elapsed_force_applied += 1

        if self.adv_debug_line_id is not None:
            p.removeUserDebugItem(self.adv_debug_line_id)
        
        if self.link_name is not None:
            p.applyExternalForce(self.robot.id, self.robot.get_link_id(self.link_name), self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.client)
            self.adv_debug_line_id = p.addUserDebugLine(self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector, lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id, parentLinkIndex=self.robot.get_link_id(self.link_name))
        else:
            p.applyExternalForce(self.robot.id, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.client)
            self.adv_debug_line_id = p.addUserDebugLine(self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector, lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id)
    
    def _apply_dstb_from_adversarial_sequence(self):
        if self.adversarial_sequence is None:
            return False
        try:
            self.force = self.adversarial_sequence[self.cnt]["force"]
            self.force_applied_force_vector = self.adversarial_sequence[self.cnt]["force_applied_force_vector"]
            self.force_applied_position_vector = self.adversarial_sequence[self.cnt]["force_applied_position_vector"]
        except IndexError:
            return False
        
        if self.adv_debug_line_id is not None:
            p.removeUserDebugItem(self.adv_debug_line_id)

        if self.link_name is not None:
            p.applyExternalForce(self.robot.id, self.robot.get_link_id(self.link_name), self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.client)
            self.adv_debug_line_id = p.addUserDebugLine(self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector, lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id, parentLinkIndex=self.robot.get_link_id(self.link_name))
        else:
            p.applyExternalForce(self.robot.id, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.client)
            self.adv_debug_line_id = p.addUserDebugLine(self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector, lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id)
        
        return True
    
    def _gen_terrain(self, terrain_height: Optional[int]=None, mesh_scale: Optional[np.ndarray]=None):
        """
        Create a randomized terrain to be applied into the dynamics
        The terrain will be applied from the beginning
        """
        if terrain_height is not None:
            self.terrain_height = terrain_height
        
        if mesh_scale is None:
            mesh_scale = [0.08, 0.08, 1.0] # [x, y, z]
            
        heightPerturbationRange = self.terrain_height
        numHeightfieldRows = 256
        numHeightfieldColumns = 256

        terrainShape = 0
        heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

        heightPerturbationRange = heightPerturbationRange
        for j in range(int(numHeightfieldColumns / 2)):
            for i in range(int(numHeightfieldRows / 2)):
                height = np.random.uniform(0, heightPerturbationRange)
                heightfieldData[2 * i +
                                        2 * j * numHeightfieldRows] = height
                heightfieldData[2 * i + 1 +
                                        2 * j * numHeightfieldRows] = height
                heightfieldData[2 * i + (2 * j + 1) *
                                        numHeightfieldRows] = height
                heightfieldData[2 * i + 1 + (2 * j + 1) *
                                        numHeightfieldRows] = height

        terrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            # meshScale=[0.08, 0.08, 1.0], # [x, y, z]
            meshScale=mesh_scale,
            heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
            heightfieldData = heightfieldData,
            numHeightfieldRows=numHeightfieldRows,
            numHeightfieldColumns=numHeightfieldColumns,
            physicsClientId = self.client)
        
        terrain = p.createMultiBody(0, terrainShape, physicsClientId = self.client)

        p.resetBasePositionAndOrientation(
            terrain, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId = self.client)
        
        p.changeDynamics(terrain, -1, lateralFriction=self.terrain_friction, physicsClientId = self.client)
        p.changeVisualShape(terrain, -1, rgbaColor=[0.2, 0.8, 0.8, 1], physicsClientId = self.client)

        self.terrain_data = heightfieldData

    def _set_terrain(self, terrain_data, terrain_height: Optional[int]=None, mesh_scale: Optional[np.ndarray]=None):
        """
        Create a randomized terrain to be applied into the dynamics
        The terrain will be applied from the beginning
        """
        if terrain_height is not None:
            self.terrain_height = terrain_height
        
        if mesh_scale is None:
            mesh_scale = [0.08, 0.08, 1.0] # [x, y, z]
            
        numHeightfieldRows = 256
        numHeightfieldColumns = 256

        terrainShape = 0
        heightfieldData = terrain_data
        
        terrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
            heightfieldData = heightfieldData,
            numHeightfieldRows=numHeightfieldRows,
            numHeightfieldColumns=numHeightfieldColumns,
            physicsClientId = self.client)
        
        terrain = p.createMultiBody(0, terrainShape, physicsClientId = self.client)

        p.resetBasePositionAndOrientation(
            terrain, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId = self.client)
        
        p.changeDynamics(terrain, -1, lateralFriction=self.terrain_friction, physicsClientId = self.client)
        p.changeVisualShape(terrain, -1, rgbaColor=[0.2, 0.8, 0.8, 1], physicsClientId = self.client)

        self.terrain_data = heightfieldData
    
    def load_terrain(self):
        full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/simple_plane.urdf')
        self.base_plane = p.loadURDF(fileName=full_file, basePosition=[0, 0, 0], physicsClientId=self.client)
        self.change_dynamics(
            self.base_plane, link_id=0, 
            lateral_friction=self.env_kwargs['lateral_friction'], anisotropic_friction=self.env_kwargs['anisotropic_friction'], 
            rolling_friction=self.env_kwargs['rolling_friction'], spinning_friction=self.env_kwargs['spinning_friction'], 
            restitution=self.env_kwargs['restitution']
        )
        self.add_terrain()
    
    def add_terrain(self):
        """
        Function from LINC extension, to add in more terrain objects into the env
        Function uses self.env_kwargs initialized in the __init__ from yaml config files
        Objects created will be added into self.world_objects
        """
        if self.terrain_type == 'ramp':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/ramp.urdf')
            old_urdf = open(full_file, 'r')
            old_urdf_list = old_urdf.readlines()
            old_urdf_lines = ""
            for line in old_urdf_list:
                old_urdf_lines += line

            theta = math.pi / 2 - self.env_kwargs['theta'] * math.pi / 180
            new_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/ramp_tmp.urdf')
            fout = open(new_file, "w")
            fout.write(old_urdf_lines.replace("<origin rpy='0 1.2 0' xyz='0 0 0'/>", "<origin rpy='0 " + str(theta) + " 0' xyz='0 0 0'/>"))
            fout.close()
            self.terrain = p.loadURDF(fileName=new_file, basePosition=[-0.5, 0, 0], physicsClientId=self.client)
            self.world_objects.append(self.terrain)

        elif self.terrain_type == 'pyramids_sawtooth':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Block/Terrain_Test_Block.urdf')
            width = 0.6105
            length = 0.6105
            n = 4
            # base_pos = np.array([n / 2, 0, 0])
            base_pos = np.array([-0.5, 0, 0])
            for col in range(n):
                for row in range(n):
                    pos = np.array([-(length * n / 2) + col * (length), -(width * n / 2) + row * (width), 0.0])
                    if (col + row) % 2 == 0:
                        ang = p.getQuaternionFromEuler([0, 0.0, 0.0])
                    else:
                        ang = p.getQuaternionFromEuler([0, 0.0, math.pi])
                        pos += np.array([length, 0, 0])
                    self.terrain = p.loadURDF(fileName=full_file, basePosition=pos + base_pos, baseOrientation=ang, physicsClientId=self.client)
                    self.world_objects.append(self.terrain)

        elif self.terrain_type == 'pyramids_wave':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Block/Terrain_Test_Block.urdf')
            width = 0.6105
            length = 0.6105
            n = 4
            # base_pos = np.array([n / 2, 0, 0])
            base_pos = np.array([-0.5, 0, 0])
            for col in range(n):
                for row in range(n):
                    pos = np.array([-(length * n / 2) + col * (length), -(width * n / 2) + row * (width), 0.0])
                    if col % 2 == 0:
                        ang = p.getQuaternionFromEuler([0, 0.0, 0.0])
                    else:
                        ang = p.getQuaternionFromEuler([0, 0.0, math.pi])
                        pos += np.array([length, 0, 0])
                    self.terrain = p.loadURDF(fileName=full_file, basePosition=pos + base_pos, baseOrientation=ang, physicsClientId=self.client)
                    self.world_objects.append(self.terrain)

        elif self.terrain_type == 'pyramids_randomized':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Block/Terrain_Test_Block.urdf')
            width = 0.6105
            length = 0.6105
            n = 4
            base_pos = np.array([n / 2, 0, 0])
            for col in range(n):
                for row in range(n):
                    pos = np.array([-(length * n / 2) + col * (length), -(width * n / 2) + row * (width), 0.0])
                    if np.random.rand() < .5:
                        ang = p.getQuaternionFromEuler([0, 0.0, 0.0])
                    else:
                        ang = p.getQuaternionFromEuler([0, 0.0, math.pi])
                        pos += np.array([length, 0, 0])
                    self.terrain = p.loadURDF(fileName=full_file, basePosition=pos + base_pos, baseOrientation=ang, physicsClientId=self.client)
                    self.world_objects.append(self.terrain)

        elif self.terrain_type == 'slope':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Block/Terrain_Test_Block_10x.urdf')
            width = 0.6105 * 10
            length = 0.6105 * 10
            n = 2
            base_pos = np.array([n * 10 / 2, 0, -0.1])
            for col in range(n):
                for row in range(n):
                    pos = np.array([-(length * n / 2) + col * (length), -(width * n / 2) + row * (width), 0.0])
                    if col % 2 == 0:
                        ang = p.getQuaternionFromEuler([0, 0.0, 0.0])
                    else:
                        ang = p.getQuaternionFromEuler([0, 0.0, math.pi])
                        pos += np.array([length, 0, 0])
                    self.terrain = p.loadURDF(fileName=full_file, basePosition=pos + base_pos, baseOrientation=ang, physicsClientId=self.client)
                    self.world_objects.append(self.terrain)

        elif self.terrain_type == 'chicane':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Chicane/chicane.urdf')
            self.terrain = p.loadURDF(fileName=full_file, basePosition=[5, 0, 0], physicsClientId=self.client)
            self.world_objects.append(self.terrain)

        elif self.terrain_type == 'stairs':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Stairs/stairs.urdf')
            self.terrain = p.loadURDF(fileName=full_file, basePosition=[5, 0, 0], physicsClientId=self.client)
            self.world_objects.append(self.terrain)

        elif self.terrain_type == 'cliff':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Cliff/cliff.urdf')
            self.terrain = p.loadURDF(fileName=full_file, basePosition=[5, 0, 0], physicsClientId=self.client)
            self.world_objects.append(self.terrain)

        elif self.terrain_type == 'potholes':
            full_file = os.path.join(os.path.dirname(__file__), 'resources/gvr_bot/test_track/Potholes/potholes.urdf')
            self.terrain = p.loadURDF(fileName=full_file, basePosition=[5, 0, 0], physicsClientId=self.client)
            self.world_objects.append(self.terrain)

        elif self.terrain_type == 'rugged':
            heightfield = np.random.uniform(low=0.0, high=1.0, size=(50 * 50,))
            size = int(math.sqrt(heightfield.shape[0]))
            mesh_scale = [1.0 / self.env_kwargs['terrain_coeff'], 1.0 / self.env_kwargs['terrain_coeff'], self.env_kwargs['terrain_height']]
            terrain_texture_file = "resources/gvr_bot/grid_rugged.png"
            texUid = p.loadTexture(os.path.join(os.path.dirname(os.path.abspath(__file__)), terrain_texture_file))

            terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=mesh_scale,
                numHeightfieldRows=size,
                numHeightfieldColumns=size,
                # replaceHeightfieldIndex=self.terrain_shape,
                heightfieldData=heightfield,
                heightfieldTextureScaling=128,
            )
            self.terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape, basePosition=[0, 0, 0.01])
            self.world_objects.append(self.terrain)
            p.changeVisualShape(self.terrain, 0, textureUniqueId=texUid)

        for item in self.world_objects:
            self.change_dynamics(
                item, link_id=0, 
                lateral_friction=self.env_kwargs['lateral_friction'], anisotropic_friction=self.env_kwargs['anisotropic_friction'], 
                rolling_friction=self.env_kwargs['rolling_friction'], spinning_friction=self.env_kwargs['spinning_friction'], 
                restitution=self.env_kwargs['restitution'])