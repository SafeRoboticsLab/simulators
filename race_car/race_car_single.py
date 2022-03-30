from abc import abstractmethod
from typing import Dict, Tuple, List, Any
import numpy as np
from gym import spaces
import csv

from ..base_env import BaseEnv
from ..ell_reach.ellipse import Ellipse
from .track import Track
from .constraints import Constraints
from ..agent import Agent


class RaceCarSingleEnv(BaseEnv):
  """
  Implements an environment of a single Princeton Race Car.
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    assert config_env.NUM_AGENTS == 1, "This environment only has one agent!"
    super().__init__()

    # Environment
    x = []
    y = []
    filepath = config_env.TRACK_FILE
    with open(filepath) as f:
      spamreader = csv.reader(f, delimiter=',')
      for i, row in enumerate(spamreader):
        if i > 0:
          x.append(float(row[0]))
          y.append(float(row[1]))

    center_line = np.array([x, y])
    self.track = Track(
        center_line=center_line, width_left=config_env.TRACK_WIDTH_LEFT,
        width_right=config_env.TRACK_WIDTH_RIGHT,
        loop=getattr(config_env, 'LOOP', True)
    )
    self.constraints = Constraints(
        config_env=config_env, config_agent=config_agent
    )

    # Action Space.
    action_space = np.array(config_agent.ACTION_RANGE)
    self.action_dim = action_space.shape[0]
    self.agent = Agent(config_agent, action_space)
    self.action_space = spaces.Box(
        low=action_space[:, 0], high=action_space[:, 1]
    )

    self.integrate_kwargs = {}
    if hasattr(config_agent, 'INTEGRATE_KWARGS'):
      self.integrate_kwargs = config_agent.INTEGRATE_KWARGS
      self.integrate_kwargs['noise'] = np.array(self.integrate_kwargs['noise'])

    # Observation Space.
    low = np.zeros((4,))
    low[:2] = np.min(center_line, axis=1)
    high = np.zeros((4,))
    high[:2] = np.max(center_line, axis=1)
    high[2] = config_agent.V_MAX
    high[3] = 2 * np.pi
    self.observation_space = spaces.Box(low=low, high=high)
    self.observation_dim = self.observation_space.low.shape
    self.state = self.observation_space.sample()  # Overriden by reset later.

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

  def reset(self) -> np.ndarray:
    return super().reset()

  def render(self):
    pass

  def update_obs(self, obs_list: List[List[Ellipse]]):
    self.constraints.update_obs(obs_list)

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
    pass

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
    pass

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
    pass

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
    pass
