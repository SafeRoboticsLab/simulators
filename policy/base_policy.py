"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Callable, Tuple, Union, Dict, List, Optional
import numpy as np
import torch


class BasePolicy(ABC):
  _critic: Optional[Union[torch.nn.Module, Callable[
      [np.ndarray, np.ndarray, Optional[np.ndarray]], float]]]
  critic_agents_order: Optional[List]
  policy_observable: Optional[List]

  def __init__(self, id: str, cfg) -> None:
    super().__init__()
    self.id = id
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.policy_observable = None
    self._critic = None
    self.critic_agents_order = None

  @abstractmethod
  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        obs (np.ndarray): current observation.
        agents_action (Optional[Dict]): other agents' actions that are
            observable to the ego agent.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    raise NotImplementedError

  def critic(
      self, obs: np.ndarray, action: Union[np.ndarray, Dict[str, np.ndarray]],
      append: Optional[np.ndarray] = None
  ) -> float:
    assert self._critic is not None
    assert self.critic_agents_order is not None

    if isinstance(action, dict):
      flat_action = np.concatenate([
          action[k].copy() for k in self.critic_agents_order
      ], axis=0)
    else:
      flat_action = action.copy()

    if isinstance(self._critic, torch.nn.Module):
      q_pi_1, q_pi_2 = self._critic(obs, flat_action, append=append)
      return (q_pi_1+q_pi_2) / 2
    else:
      return self._critic(obs, flat_action, append)

  def report(self):
    print(self.id)
    if self.policy_observable is not None:
      print("  - The policy can observe:", end=' ')
      for i, k in enumerate(self.policy_observable):
        print(k, end='')
        if i == len(self.policy_observable) - 1:
          print('.')
        else:
          print(', ', end='')
    else:
      print("  - The policy can only access observation.")

  def to(self, device):
    self.device = device
    if self._critic is not None:
      if isinstance(self._critic, torch.nn.Module):
        self._critic.to(self.device)
