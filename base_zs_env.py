from abc import abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
from gym import spaces

from base_env import BaseEnv
# TODO: add other simulators.
from race_car.bicycle_dynamics import BicycleDynamics
from utils import ActionZS


class BaseZeroSumEnv(BaseEnv):
  """
  Implements an environment of two-player discrete-time zero-sum dynamic game.
  The agent minimizing the cost is called 'ctrl', while the agent maximizing
  the cost is called 'dstb'.
  """

  def __init__(self, config) -> None:
    assert config.NUM_AGENTS == 2, (
        "Zero-Sum Game currently only supports two agents!"
    )
    assert config.CTDE is False, ("Zero-Sum Game has only one physical agent!")
    super().__init__(config)
    self.observation_space = spaces.Box(
        low=config.OBS_LOW, high=config.OBS_HIGH, shape=config.OBS_DIM
    )

    ctrl_space = np.array(config.ACTION_RANGE['CTRL'])
    self.action_space_ctrl = spaces.Box(
        low=ctrl_space[:, 0], high=ctrl_space[:, 1]
    )
    self.action_dim_ctrl = ctrl_space.shape[0]

    dstb_space = np.array(config.ACTION_RANGE['DSTB'])
    self.action_space_dstb = spaces.Box(
        low=dstb_space[:, 0], high=dstb_space[:, 1]
    )
    self.action_dim_dstb = dstb_space.shape[0]

    self.action_space = spaces.Dict(
        dict(ctrl=self.action_space_ctrl, dstb=self.action_space_dstb)
    )

    # TODO: add other simulators.
    if config.DYNAMICS == "Bicycle":
      self.agent = BicycleDynamics(config, self.action_space_ctrl)

    # Other keyword arguments for integrate_forward, such as step, noise,
    # noise_type.
    self.integrate_kwargs = config.INTEGRATE_KWARGS

    self.state = self.observation_space.sample()  # Overriden by reset later.

  def step(self,
           action: ActionZS) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """Implements the step function in the environment.

    Args:
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].

    Returns:
        np.ndarray: next state.
        float: the cost that ctrl wants to minimize and dstb wants to maximize.
        bool: True if the episode ends.
        Dict[str, Any]]: additional information of the step, such as target
            margin and safety margin used in reachability analysis.
    """
    state_nxt, _ = self.agent.integrate_forward(
        state=self.state, control=action['ctrl'], adversary=action['dstb'],
        **self.integrate_kwargs
    )
    cost = self.get_cost(self.state, action, state_nxt)
    constraints = self.get_constraints(self.state, action, state_nxt)
    done = self.get_done_flag(self.state, action, state_nxt, constraints)
    info = self.get_info(self.state, action, state_nxt, cost, constraints)

    return np.copy(state_nxt), cost, done, info

  @abstractmethod
  def get_cost(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current state.
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.

    Returns:
        float: the cost that ctrl wants to minimize and dstb wants to maximize.
    """
    raise NotImplementedError

  @abstractmethod
  def get_constraints(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current state.
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_flag(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray,
      constraints: Dict
  ) -> bool:
    """
    Gets the done flag given current state, current action, next state, and
    constraints.

    Args:
        state (np.ndarray): current state.
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
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
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray,
      cost: float, constraints: Dict
  ) -> Dict:
    """
    Gets a dictionary to provide additional information of the step function
    given current state, current action, next state, cost, and constraints.

    Args:
        state (np.ndarray): current state.
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
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
