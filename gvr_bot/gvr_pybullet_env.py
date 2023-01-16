from abc import ABC
from typing import Any

class GVRPybulletEnv(ABC):
    """
    Dummy environment when wanting to use GVR with Pybullet
    As the GVR dynamics is stored in Pybullet and can be accessed through dynamics/gvr_dynamics_pybullet.py, this is just dummy env to pass data along

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, config_env: Any, config_agent: Any) -> None:
        pass