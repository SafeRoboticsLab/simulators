"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from abc import abstractmethod
import copy
from typing import Dict, Tuple, Any, List, Union, Optional
import numpy as np
import torch
from gym import spaces

from .agent import Agent
from .base_env import BaseEnv
from .utils import build_space, cast_numpy


class BaseMultiAgentEnv(BaseEnv):
  """
  Implements an environment of an N-person discrete-time general-sum infinite
  dynamic game. The planners should take care of the following training options
  in their implementation:
      (1) Centralized Training for Decentralized Execution (CTDE).
      (2) Global Observation Broadcast (GLOBAL).
  """
  agents: Dict[str, Agent]
  state: Dict[str, np.ndarray]

  def __init__(
      self, config_env: Any, config_agent: List[Any],
      agent_name_list: Optional[List[str]] = None
  ) -> None:
    super().__init__(config_env)
    self.env_type = "multi-agent"
    self.num_agents: int = config_env.NUM_AGENTS
    if agent_name_list is None:
      agent_name_list = [f'agent_{i}' for i in range(self.num_agents)]
    else:
      len(agent_name_list) == self.num_agents
    self.agent_name_list: List[str] = agent_name_list

    _action_space = {}
    self.action_dim = {}
    self.agents = {}
    self.integrate_kwargs = {}
    for i, a_name in enumerate(agent_name_list):
      tmp_action_space = np.array(config_agent[i].ACTION_RANGE)
      _action_space[a_name] = spaces.Box(
          low=tmp_action_space[:, 0], high=tmp_action_space[:, 1]
      )
      self.action_dim[a_name] = _action_space[a_name].low.shape
      self.agents[a_name] = Agent(
          config_agent[i], action_space=tmp_action_space
      )
      self.integrate_kwargs[a_name] = config_agent[i].AGENT_INTEGRATE_KWARGS
    self.action_space = spaces.Dict(_action_space)

    self.build_rst_sapce(config_env, config_agent)
    self.build_obs_space(config_env, config_agent)
    self.seed(config_env.SEED)
    self.state = self.reset_sample_space.sample()  # Overriden by reset later.

  def build_rst_sapce(self, config_env: Any, config_agent: List[Any]):
    _rst_space = {}
    self.state_dim = {}
    for i, a_name in enumerate(self.agent_name_list):
      self.state_dim[a_name] = self.agents[a_name].dyn.dim_x
      rst_spec = np.array(config_agent[i].RST_RANGE)
      _rst_space[a_name] = build_space(
          space_spec=rst_spec, space_dim=self.state_dim[a_name]
      )
    self.reset_sample_space = spaces.Dict(_rst_space)

  def build_obs_space(self, config_env: Any, config_agent: List[Any]):
    _obs_space = {}
    self.observation_dim = {}
    for i, a_name in enumerate(self.agent_name_list):
      obs_spec = np.array(config_agent[i].OBS_RANGE)
      _obs_space[a_name] = build_space(
          space_spec=obs_spec, space_dim=config_agent[i].OBS_DIM
      )
      self.observation_dim[a_name] = _obs_space[a_name].low.shape
    self.observation_space = spaces.Dict(_obs_space)

  def step(
      self, action: Dict[str, Union[np.ndarray, torch.Tensor]],
      cast_torch: bool = False
  ) -> Tuple[Union[Union[np.ndarray, torch.Tensor], Dict[str, Union[
      np.ndarray, torch.Tensor]]], Dict[str, float], bool, Dict[str, bool],
             Dict[str, Dict]]:  # noqa
    """
    Implements the step function in the environment. We assume each agent's
    action only influences its own state.

    Args:
        action (Dict[str, np.ndarray]): a dict consisting of the action of each
            agent.
        cast_torch (bool): cast state to torch if True.

    Returns:
        Dict | np.ndarray | torch.Tensor]: a dict consisting of observations
            of each agent or an observation of the whole system.
        Dict[str, float]: a dict consisting of the reward (cost*-1) of each
            agent.
        Dict[str, bool]: a dict consisting of the done flag of each agent. True
            if the episode of that agent ends.
        Dict[str, Dict]]: a dict consisting of additional information of each
            agent after the step, such as target margin and safety margin used
            in the reachability analysis.
    """
    state_nxt = {}

    for a_name, state_i in self.state.items():
      action_i = cast_numpy(action[a_name])
      state_nxt_i, _ = self.agents[a_name].integrate_forward(
          state=state_i, control=action_i, **self.integrate_kwargs[a_name]
      )
      state_nxt[a_name] = np.copy(state_nxt_i)

    constraints = self.get_constraints(self.state, action, state_nxt)
    cost = self.get_cost(self.state, action, state_nxt, constraints)
    targets = self.get_target_margin(self.state, action, state_nxt)
    done, info = self.get_done_and_info(constraints, targets)

    self.state = copy.deepcopy(state_nxt)
    obs = self.get_obs(state_nxt)
    if cast_torch:
      obs = torch.FloatTensor(obs)

    reward = {}
    for k, v in cost.items():
      reward[k] = -v

    return obs, reward, done, info

  @abstractmethod
  def get_cost(
      self, state: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
      state_nxt: Dict[str, np.ndarray], constraints: Dict[str, Dict]
  ) -> Dict[str, float]:
    """
    Gets the cost given current state, current action, and next state of each
    agent.

    Args:
        state (Dict[str, np.ndarray]): a dict consisting of the current state
            of each agent.
        action (Dict[str, np.ndarray]): a dict consisting of the action of each
            agent.
        state_nxt (Dict[str, np.ndarray]): a dict consisting of the next state
            of each agent.
        constraints (Dict[str, Dict]): the keys of the 1st level keys specify
            an agent's constraint dictionary and the (key, value) pair of the
            2nd level is the name and value of a constraint function.

    Returns:
        Dict[str, float]: a dict consisting of the cost of each agent.
    """
    raise NotImplementedError

  @abstractmethod
  def get_constraints(
      self, state: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
      state_nxt: Dict[str, np.ndarray]
  ) -> Dict[str, Dict]:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state of each agent.

    Args:
        state (Dict[str, np.ndarray]): a dict consisting of the current state
            of each agent.
        action (Dict[str, np.ndarray]): a dict consisting of the action of each
            agent.
        state_nxt (Dict[str, np.ndarray]): a dict consisting of the next state
            of each agent.

    Returns:
        Dict[str, dict]: the keys of the 1st level keys specify an agent's
            constraint dictionary and the (key, value) pair of the 2nd level is
            the name and value of a constraint function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_target_margin(
      self, state: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
      state_nxt: Dict[str, np.ndarray]
  ) -> Dict[str, Dict]:
    """
    Gets the values of all target margin functions given current state, current
    action, and next state of each agent.

    Args:
        state (Dict[str, np.ndarray]): a dict consisting of the current state
            of each agent.
        action (Dict[str, np.ndarray]): a dict consisting of the action of each
            agent.
        state_nxt (Dict[str, np.ndarray]): a dict consisting of the next state
            of each agent.

    Returns:
        Dict[str, Dict]: the keys of the 1st level keys specify
            an agent's target dictionary and the (key, value) pair of the
            2nd level is the name and value of a target margin function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_and_info(
      self, constraints: Dict[str, Dict], targets: Dict[str, Dict]
  ) -> Tuple[Dict[str, bool], Dict[str, Dict]]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets of each agent.

    Args:
        constraints (Dict[str, Dict]): the keys of the 1st level keys specify
            an agent's constraint dictionary and the (key, value) pair of the
            2nd level is the name and value of a constraint function.
        targets Dict[str, Dict]: the keys of the 1st level keys specify
            an agent's target dictionary and the (key, value) pair of the
            2nd level is the name and value of a target margin function.

    Returns:
        Dict[str, bool]: a dict consisting of the done flag of each agent. True
            if the episode of that agent ends.
        Dict[str, Dict]: a dict consisting of additional information of each
            agent after the step, such as target margin and safety margin used
            in the reachability analysis.
    """
    raise NotImplementedError
