from abc import ABC
from typing import Any, Tuple, Optional, Callable, List, Dict, Union
import numpy as np
from gym import Space, spaces
import torch
from simulators.pybullet_debugger import pybulletDebug


class SpiritPybulletEnv(ABC):
  """
    Dummy environment when wanting to use Spirit with Pybullet
    As the Spirit dynamics is stored in Pybullet and can be accessed through dynamics/spirit_dynamics_pybullet.py, this is just dummy env to pass data along

    Args:
        ABC (_type_): _description_
    """

  def __init__(self, cfg_env: Any, cfg_agent: Any) -> None:
    pass
