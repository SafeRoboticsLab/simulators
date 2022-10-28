"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
from typing import Tuple, Optional, Any, Dict
import copy
import time
import numpy as np
import torch

from .base_policy import BasePolicy


class NeuralNetworkControlSystem(BasePolicy):

  def __init__(self, id: str, actor: torch.nn.Module, config: Any):
    super().__init__(id, config)
    self.policy_type = "NNCS"

    # Constructs NNs.
    self.actor = copy.deepcopy(actor)
    self.actor.to(self.device)

    # Loads weights if specified
    if hasattr(config, 'ACTOR_PATH'):
      actor_path = config.ACTOR_PATH
      self.actor.load_state_dict(
          torch.load(actor_path, map_location=self.device)
      )
      print("--> Load actor wights from {}".format(actor_path))

  def update_policy(self, actor: torch.nn.Module):
    self.actor.load_state_dict(actor.state_dict())

  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    if self.policy_observable is not None:
      flat_action = np.concatenate([
          agents_action[k].copy() for k in self.policy_observable
      ], axis=0)
      obs = np.concatenate((obs, flat_action), axis=0)
    append = kwargs.get("append", None)
    latent = kwargs.get("latent", None)

    time0 = time.time()
    action = self.actor(obs, append=append, latent=latent)
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)

  def to(self, device):
    super().to(device)
    self.actor.to(self.device)
