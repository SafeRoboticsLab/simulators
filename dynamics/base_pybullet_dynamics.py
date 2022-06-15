from typing import Optional, Tuple, Any
import numpy as np
from simulators.pybullet_debugger import pybulletDebug
from .resources.plane import Plane
from .base_dynamics import BaseDynamics
import pybullet as p

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
        self.dt = config.DT
        self.gravity = -9.81

        # configure force in the dynamics
        self.force = float(config.FORCE)
        self.elapsed_force_applied = 0
        self.force_applied_reset = config.FORCE_RESET_TIME

        self.rotate_reset = config.ROTATE_RESET
        self.height_reset = config.HEIGHT_RESET

        self.force_applied_force_vector = None
        self.force_applied_position_vector = None
        self.force_random = config.FORCE_RANDOM

        # configure terrain in the dynamics
        self.terrain = config.TERRAIN
        self.terrain_height = config.TERRAIN_HEIGHT
        self.terrain_friction = config.TERRAIN_FRICTION

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
        else:
            self.client = p.connect(p.DIRECT)
        
        self.debugger = pybulletDebug()

        self.reset()

    def reset(self):
        p.resetSimulation(physicsClientId = self.client)
        p.setGravity(0, 0, self.gravity, physicsClientId = self.client)
        p.setTimeStep(self.dt, physicsClientId = self.client)
        p.setPhysicsEngineParameter(fixedTimeStep = self.dt, physicsClientId = self.client)
        p.setRealTimeSimulation(0)
        Plane(self.client)

        if self.terrain == "rough":
            self._gen_terrain()
        self._gen_force()

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
        if self.force_random:
            self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        else:
            self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
        self.force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
    
    def _apply_force(self):
        if self.elapsed_force_applied > self.force_applied_reset:
            self._gen_force()
        else:
            self.elapsed_force_applied += 1

        p.applyExternalForce(self.robot.id, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME)
    
    def _gen_terrain(self):
        """
        Create a randomized terrain to be applied into the dynamics
        The terrain will be applied from the beginning
        """
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
            meshScale=[0.08, 0.08, 1.0], # [x, y, z]
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