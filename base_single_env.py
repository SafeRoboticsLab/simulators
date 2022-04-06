"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from abc import abstractmethod
from typing import Any, Tuple, Dict, Optional
import numpy as np
from gym import spaces

from .agent import Agent
from .base_env import BaseEnv


class BaseSingleEnv(BaseEnv):
  """Implements an environment of a single agent.
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    super().__init__(config_env)

    # Action Space.
    action_space = np.array(config_agent.ACTION_RANGE)
    self.action_dim = action_space.shape[0]
    self.agent = Agent(config_agent, action_space)
    self.action_space = spaces.Box(
        low=action_space[:, 0], high=action_space[:, 1]
    )

    self.integrate_kwargs = getattr(config_agent, "INTEGRATE_KWARGS", {})
    if "noise" in self.integrate_kwargs:
      self.integrate_kwargs['noise'] = np.array(self.integrate_kwargs['noise'])

  def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
    """Implements the step function in the environment.

    Args:
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].

    Returns:
        np.ndarray: next state.
        float: the reward that ctrl wants to maximize and dstb wants to
            minimize.
        bool: True if the episode ends.
        Dict[str, Any]]: additional information of the step, such as target
            margin and safety margin used in reachability analysis.
    """
    self.cnt += 1
    state_nxt, _ = self.agent.integrate_forward(
        state=self.state, control=action, **self.integrate_kwargs
    )
    constraints = self.get_constraints(self.state, action, state_nxt)
    cost = self.get_cost(self.state, action, state_nxt, constraints)
    targets = self.get_target_margin(self.state, action, state_nxt)
    done, info = self.get_done_and_info(constraints, targets)

    self.state = np.copy(state_nxt.reshape(-1, 1))

    return np.copy(state_nxt.reshape(-1, 1)), -cost, done, info

  @abstractmethod
  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Optional[dict] = None
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        float: the cost to minimize.
    """
    raise NotImplementedError

  @abstractmethod
  def get_constraints(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_target_margin(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all target margin functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).

    Returns:
        Dict: each (key, value) pair is the name and value of a target margin
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_and_info(self, constraints: Dict,
                        targets: Dict) -> Tuple[bool, Dict]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets.

    Args:
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.
        targets (Dict): each (key, value) pair is the name and value of a
            target margin function.

    Returns:
        bool: True if the episode ends.
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    raise NotImplementedError
