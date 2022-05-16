"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from abc import ABC, abstractmethod
import copy
from typing import Tuple
import numpy as np


class BasePolicy(ABC):

  def __init__(self, env, config) -> None:
    super().__init__()
    self.env = copy.deepcopy(env)
    self.has_safety = getattr(config, "HAS_SAFETY", False)
    if self.has_safety:
      pass  #TODO: load safety

  def get_action(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    action, solver_info = self._get_action(state, **kwargs)
    if self.has_safety:
      pass  #TODO

    return action, solver_info

  @abstractmethod
  def _get_action(self, state: np.ndarray,
                  **kwargs) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    raise NotImplementedError
