# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for (control) policy.

This file implements a parent class for (control) policy. A child class should
implement `get_action()`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import numpy as np


class BasePolicy(ABC):
  obsrv_dict: Optional[Dict]

  @property
  @abstractmethod
  def is_stochastic(self) -> bool:
    raise NotImplementedError

  def __init__(self, id: str, obsrv_dict: Optional[Dict] = None, **kwargs) -> None:
    super().__init__()
    self.id = id
    self.obsrv_dict = obsrv_dict

  @abstractmethod
  def get_action(self, obsrv: np.ndarray, agents_action: Optional[Dict[str, np.ndarray]] = None,
                 **kwargs) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        obsrv (np.ndarray): current observation.
        agents_action (Optional[Dict]): other agents' actions that are
            observable to the ego agent.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    raise NotImplementedError

  def report(self):
    print(self.id)
    if self.obsrv_dict is not None:
      print("  - The policy can observe:", end=' ')
      for i, k in enumerate(self.obsrv_dict.keys()):
        print(k, end='')
        if i == len(self.obsrv_dict) - 1:
          print('.')
        else:
          print(', ', end='')
    else:
      print("  - The policy can only access observation.")
