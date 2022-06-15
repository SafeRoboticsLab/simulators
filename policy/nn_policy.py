"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
from typing import Tuple, Optional, Any
import copy
import time
import numpy as np
import torch

from .base_policy import BasePolicy


class NeuralNetworkControlSystem(BasePolicy):

  def __init__(
      self, env, critic: torch.nn.Module, actor: torch.nn.Module, config: Any
  ) -> None:
    super().__init__(env, config)
    self.policy_type = "NNCS"
    self.device = config.DEVICE
    self.critic_has_act_ind = config.CRITIC_HAS_ACT_IND
    if self.critic_has_act_ind:
      assert hasattr(config, "ACT_IND"), "Needs action indicator!"
    if hasattr(config, "ACT_IND"):
      self.act_ind = torch.FloatTensor(config.ACT_IND).to(self.device)
      self.act_ind_dim = self.act_ind.shape[0]

    # Constructs NNs.
    self.critic = copy.deepcopy(critic)
    self.actor = copy.deepcopy(actor)

    # Loads weights if specified
    if hasattr(config, 'CRITIC_PATH'):
      critic_path = config.CRITIC_PATH
      self.critic.load_state_dict(
          torch.load(critic_path, map_location=self.device)
      )
      print("--> Load critic wights from {}".format(critic_path))

    if hasattr(config, 'ACTOR_PATH'):
      actor_path = config.ACTOR_PATH
      self.actor.load_state_dict(
          torch.load(actor_path, map_location=self.device)
      )
      print("--> Load actor wights from {}".format(actor_path))

  def update_policy(
      self, critic: Optional[torch.nn.Module] = None,
      actor: Optional[torch.nn.Module] = None
  ):
    if critic is not None:
      self.critic.load_state_dict(critic.state_dict())

    if actor is not None:
      self.actor.load_state_dict(actor.state_dict())

  def _get_action(self, state: np.ndarray,
                  **kwargs) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    append = kwargs.get("append", None)
    latent = kwargs.get("latent", None)

    state_tensor = torch.FloatTensor(state).to(self.device)
    time0 = time.time()
    with torch.no_grad():
      action = self.actor(state_tensor, append=append, latent=latent)
    t_process = time.time() - time0
    action = action.cpu().numpy()
    status = 1
    return action, dict(t_process=t_process, status=status)

  def get_value(self, state, append=None):
    action = self.actor(state, append=append)
    action = torch.FloatTensor(action).to(self.device)
    if self.critic_has_act_ind:
      act_ind_rep = self.act_ind.repeat(action.shape[0], 1)
      action = torch.cat((action, act_ind_rep), dim=-1)

    q_pi_1, q_pi_2 = self.critic(state, action, append=append)
    value = (q_pi_1+q_pi_2) / 2
    return value
