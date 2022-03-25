from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import random
import numpy as np
import gym
import torch


class BaseEnv(gym.Env, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[List[np.ndarray], float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    def seed(self, seed: int = 0) -> None:
        self.seed_val = seed
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
        random.seed(self.seed_val)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
