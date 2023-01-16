import pybullet as p
import os
import pybullet_data

class Ramp:
    def __init__(self, client):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(
            os.path.join(os.path.dirname(__file__), 'gvr_bot/test_track/Block/Terrain_Test_Block.urdf'), 
            [-5.0, 0, -0.1], baseOrientation=p.getQuaternionFromEuler([1.57, -0.2, 0.0]),
            useFixedBase=True, globalScaling=10.0
        )