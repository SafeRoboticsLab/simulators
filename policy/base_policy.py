"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BasePolicy(ABC):

  def __init__(self) -> None:
    super().__init__()

  @abstractmethod
  def get_action(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    raise NotImplementedError
