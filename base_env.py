"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import random
import numpy as np
import gym
import torch

from .utils import GenericAction, GenericState


class BaseEnv(gym.Env, ABC):

  def __init__(self, config_env) -> None:
    super().__init__()
    self.cnt = 0
    self.timeout = config_env.TIMEOUT
    self.end_criterion = config_env.END_CRITERION

  @abstractmethod
  def step(self,
           action: GenericAction) -> Tuple[GenericState, float, bool, Dict]:
    raise NotImplementedError

  def reset(self) -> GenericState:
    self.cnt = 0

  @abstractmethod
  def render(self):
    raise NotImplementedError

  def seed(self, seed: int = 0) -> None:
    self.seed_val = seed
    self.rng = np.random.default_rng(seed)
    random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    self.action_space.seed(seed)
    self.observation_space.seed(seed)
