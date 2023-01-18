import pybullet as p
import os
import pybullet_data

class Ramp:
    def __init__(self, client):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(os.path.join(os.path.dirname(__file__), 'gvr_bot/test_track/CustomBlockTerrain/ramp_mesh.urdf'), [0.0, 0.0, 0.0], baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]), useFixedBase=True)