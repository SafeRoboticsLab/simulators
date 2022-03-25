from typing import Callable, Dict, List, Tuple, Type, Optional, Union, Set, Any
import numpy as np
from gym import spaces

from base_env import BaseEnv


class CarMultiAgentEnv(BaseEnv):
  """
    Implements a multi-agent car simulator.
        (1) Centralized Training for Decentralized Execution (CTDE).
        (2) Global Observation Broadcast.
    """

  def __init__(self, config) -> None:
    super().__init__()
    self.num_agents = config.NUM_AGENTS
    self.CTDE = config.CTDE

    # placeholder
    if self.CTDE:
      _observation_space = {}
    _action_space = {}
    self.action_dim = np.empty(shape=(self.num_agents,))

    for i in range(self.num_agents):
      tmp_action_sapce = np.array(config.ACTION_RANGE['AGENT_' + str(i)])
      tmp_observation_sapce = np.array(config.OBS_RANGE['AGENT_' + str(i)])

      _action_space['agent_' + str(i)] = spaces.Box(
          low=tmp_action_sapce[:, 0], high=tmp_action_sapce[:, 1],
          seed=config.SEED
      )
      self.action_dim[i] = tmp_action_sapce.shape[0]
      if self.CTDE:
        _observation_space['agent_' + str(i)] = spaces.Box(
            low=tmp_observation_sapce[:, 0], high=tmp_observation_sapce[:, 1],
            seed=config.SEED
        )

    # required attributes for gym env.
    self.action_space = spaces.Dict(_action_space)
    if self.CTDE:
      self.observation_space = spaces.Dict(_observation_space)
    else:
      self.observation_space = spaces.Box(
          low=config.OBS_LOW, high=config.OBS_HIGH, shape=(config.OBS_DIM,),
          seed=config.SEED
      )

  def step(
      self, action: np.ndarray
  ) -> Tuple[List[np.ndarray], float, bool, Dict[str, Any]]:
    pass

  def reset(self) -> List[np.ndarray]:
    pass

  def render(self):
    pass
