from typing import Callable, Dict, List, Tuple, Type, Optional, Union, Set, Any
import numpy as np

from car_ma_env import CarMultiAgentEnv


class CarZeroSumEnv(CarMultiAgentEnv):
    def __init__(self, config) -> None:
        assert config.NUM_AGENTS == 2, (
            "Zero-Sum Game currently only supports two agents!")
        assert config.CTDE is False, (
            "Zero-Sum Game has only one physical agent!")
        super().__init__(config)
        self.action_dim_ctrl = self.action_dim[0]
        self.action_dim_dstb = self.action_dim[1]
        self.action_space_ctrl = self.action_space[0]
        self.action_space_dstb = self.action_space[1]

    def step(
        self, action: np.ndarray
    ) -> Tuple[List[np.ndarray], float, bool, Dict[str, Any]]:
        pass
