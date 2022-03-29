from abc import abstractmethod
from typing import Dict, Tuple
import numpy as np
from gym import spaces

from base_env import BaseEnv
from ..utils import get_agent
from .track import Track


class RaceCarSingleEnv(BaseEnv):
  """
  Implements an environment of a single Princeton Race Car.
  """

  def __init__(self, config) -> None:
    assert config.NUM_AGENTS == 1, "This environment only has one agent!"
    super().__init__()

    # Action Space.
    action_space = np.array(config.ACTION_RANGE)
    self.action_dim = action_space.shape[0]
    self.agent = get_agent(  # Currently only supports 'Bicycle' model.
        dyn='Bicycle', config=config, action_space=action_space
    )
    self.action_space = spaces.Box(
        low=action_space[:, 0], high=action_space[:, 1]
    )
    self.integrate_kwargs = config.INTEGRATE_KWARGS

    # Observation Space.
    self.observation_space = spaces.Box(
        low=config.OBS_LOW, high=config.OBS_HIGH, shape=config.OBS_DIM
    )
    self.observation_dim = self.observation_space.low.shape
    self.state = self.observation_space.sample()  # Overriden by reset later.

    # Environment
    self.track = Track(
        center_line=config.CENTER_LINE, width_left=config.WIDTH_LEFT,
        width_right=config.WIDTH_RIGHT, loop=getattr(config, 'LOOP', True)
    )

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
    state_nxt, _ = self.agent.integrate_forward(
        state=self.state, control=action, **self.integrate_kwargs
    )
    constraints = self.get_constraints(self.state, action, state_nxt)
    cost = self.get_cost(self.state, action, state_nxt, constraints)
    done = self.get_done_flag(self.state, action, state_nxt, constraints)
    info = self.get_info(self.state, action, state_nxt, cost, constraints)

    return np.copy(state_nxt), -cost, done, info

  @abstractmethod
  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: dict
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current state.
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        float: the cost that ctrl wants to minimize and dstb wants to maximize.
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
        state (np.ndarray): current state.
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_flag(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Dict
  ) -> bool:
    """
    Gets the done flag given current state, current action, next state, and
    constraints.

    Args:
        state (np.ndarray): current state.
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        bool: True if the episode ends.
    """
    raise NotImplementedError

  @abstractmethod
  def get_info(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      cost: float, constraints: Dict
  ) -> Dict:
    """
    Gets a dictionary to provide additional information of the step function
    given current state, current action, next state, cost, and constraints.

    Args:
        state (np.ndarray): current state.
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.
        cost (float): the cost that ctrl wants to minimize and dstb wants to
            maximize.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    raise NotImplementedError
