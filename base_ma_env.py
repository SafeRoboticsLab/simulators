from abc import abstractmethod
from typing import Dict, Tuple, Any, List
import numpy as np
from gym import spaces

from base_env import BaseEnv
from .utils import get_agent


class BaseMultiAgentEnv(BaseEnv):
  """
  Implements an environment of an N-person discrete-time general-sum infinite
  dynamic game. The planners should take care of the following training options
  in their implementation:
      (1) Centralized Training for Decentralized Execution (CTDE).
      (2) Global Observation Broadcast.
  """

  def __init__(self, config: Any) -> None:
    super().__init__()
    self.num_agents = config.NUM_AGENTS

    # placeholder
    _obs_space = {}
    _action_space = {}
    self.action_dim = np.empty(shape=(self.num_agents,))
    self.observation_dim = [None for _ in range(self.num_agents)]
    self.agent = [None for _ in range(self.num_agents)]
    self.integrate_kwargs = [None for _ in range(self.num_agents)]

    for i in range(self.num_agents):
      agent_name = 'agent_' + str(i)
      # Action Space.
      tmp_action_space = np.array(config.ACTION_RANGE[agent_name])
      _action_space[agent_name] = spaces.Box(
          low=tmp_action_space[:, 0], high=tmp_action_space[:, 1]
      )
      self.action_dim[i] = _action_space[agent_name].low.shape
      self.agent[i] = get_agent(
          dyn=config.DYNAMICS[agent_name], config=config.PHYSICS[agent_name],
          action_space=tmp_action_space
      )
      self.integrate_kwargs[i] = config.INTEGRATE_KWARGS[agent_name]

      # Observation space.
      tmp_observation_space = np.array(config.OBS_RANGE[agent_name])
      if tmp_observation_space.ndim == 2:  # e.g., state.
        _obs_space[agent_name] = spaces.Box(
            low=tmp_observation_space[:, 0], high=tmp_observation_space[:, 1]
        )
      elif tmp_observation_space.ndim == 4:  # e.g., RGB-D.
        _obs_space[agent_name] = spaces.Box(
            low=tmp_observation_space[:, :, :, 0],
            high=tmp_observation_space[:, :, :, 1]
        )
      else:  # Each dimension shares the same min and max.
        assert tmp_observation_space.ndim == 1, "Unsupported obs space dim!"
        _obs_space[agent_name] = spaces.Box(
            low=tmp_observation_space[0], high=tmp_observation_space[1],
            shape=(config.OBS_DIM[agent_name])
        )
      self.observation_dim[i] = _obs_space[agent_name].low.shape

    # Required attributes for gym env.
    self.action_space = spaces.Dict(_action_space)
    self.observation_space = spaces.Dict(_obs_space)
    self.state = self.observation_space.sample()  # Overriden by reset later.

  def step(
      self, action: List[np.ndarray]
  ) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
    """
    Implements the step function in the environment. We assume each agent's
    action only influences its own state.

    Args:
        action (List[np.ndarray]): a list consisting of the action of each
            agent.

    Returns:
        List[np.ndarray]: a list consisting of the next state of each agent.
        List[float]: a list consisting of the reward (cost*-1) of each agent.
        List[bool]: a list consisting of the done flag of each agent. True if
            the episode of that agent ends.
        List[Dict]]: a list consisting of additional information of each agent
            after the step, such as target margin and safety margin used in
            reachability analysis.
    """
    state_nxt = [None for _ in range(self.num_agents)]

    for i, (state_i, action_i) in enumerate(zip(self.state, action)):
      state_nxt_i, _ = self.agent[i].integrate_forward(
          state=state_i, control=action_i, **self.integrate_kwargs[i]
      )
      state_nxt[i] = np.copy(state_nxt_i)

    cost = self.get_cost(self.state, action, state_nxt)
    constraints = self.get_constraints(self.state, action, state_nxt)
    done = self.get_done_flag(self.state, action, state_nxt, constraints)
    info = self.get_info(self.state, action, state_nxt, cost, constraints)

    return state_nxt, -cost, done, info

  @abstractmethod
  def get_cost(
      self, state: List[np.ndarray], action: List[np.ndarray],
      state_nxt: List[np.ndarray]
  ) -> List[float]:
    """
    Gets the cost given current state, current action, and next state of each
    agent.

    Args:
        state (List[np.ndarray]): a list consisting of the current state of
            each agent.
        action (List[np.ndarray]): a list consisting of the action of each
            agent.
        state_nxt (List[np.ndarray]): a list consisting of the next state of
            each agent.

    Returns:
        List[float]: a list consisting of the cost of each agent.
    """
    raise NotImplementedError

  @abstractmethod
  def get_constraints(
      self, state: List[np.ndarray], action: List[np.ndarray],
      state_nxt: List[np.ndarray]
  ) -> List[Dict]:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state of each agent.

    Args:
        state (List[np.ndarray]): a list consisting of the current state of
            each agent.
        action (List[np.ndarray]): a list consisting of the action of each
            agent.
        state_nxt (List[np.ndarray]): a list consisting of the next state of
            each agent.

    Returns:
        List[Dict]: a list consisting of the constraints dictionary of
            each agent. Each (key, value) pair in the dictionary is the
            name and value of a constraint function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_flag(
      self, state: List[np.ndarray], action: List[np.ndarray],
      state_nxt: List[np.ndarray], constraints: List[Dict]
  ) -> bool:
    """
    Gets the done flag given current state, current action, next state, and
    constraints of each agent.

    Args:
        state (List[np.ndarray]): a list consisting of the current state of
            each agent.
        action (List[np.ndarray]): a list consisting of the action of each
            agent.
        state_nxt (List[np.ndarray]): a list consisting of the next state of
            each agent.
        constraints (List[Dict]): a list consisting of the constraints
            dictionary of each agent. Each (key, value) pair in the dictionary
            is the name and value of a constraint function.

    Returns:
        List[bool]: a list consisting of the done flag of each agent. True if
            the episode of that agent ends.
    """
    raise NotImplementedError

  @abstractmethod
  def get_info(
      self, state: List[np.ndarray], action: List[np.ndarray],
      state_nxt: List[np.ndarray], cost: List[float], constraints: List[Dict]
  ) -> List[Dict]:
    """
    Gets a dictionary to provide additional information of the step function
    given current state, current action, next state, cost, and constraints of
    each agent.

    Args:
        state (List[np.ndarray]): a list consisting of the current state of
            each agent.
        action (List[np.ndarray]): a list consisting of the action of each
            agent.
        state_nxt (List[np.ndarray]): a list consisting of the next state of
            each agent.
        cost (List[float]): a list consisting of the cost of each agent.
        constraints (List[Dict]): a list consisting of the constraints
            dictionary of each agent. Each (key, value) pair in the dictionary
            is the name and value of a constraint function.

    Returns:
        List[Dict]: a list consisting of additional information of each agent
            after the step, such as target margin and safety margin used in
            reachability analysis.
    """
    raise NotImplementedError
