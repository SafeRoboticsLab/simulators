"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List, Union, Callable
import random
import numpy as np
import gym
import torch

from .utils import GenericAction, GenericState


class BaseEnv(gym.Env, ABC):

  def __init__(self, config_env) -> None:
    gym.Env.__init__(self)
    self.cnt = 0
    self.timeout = config_env.TIMEOUT
    self.end_criterion = config_env.END_CRITERION

  @abstractmethod
  def step(self,
           action: GenericAction) -> Tuple[GenericState, float, bool, Dict]:
    raise NotImplementedError

  @abstractmethod
  def get_obs(self, state: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def reset(
      self, state: Optional[GenericState] = None, cast_torch: bool = False,
      **kwargs
  ) -> GenericState:
    """
    Resets the environment and returns the new state.

    Args:
        state (Optional[GenericState], optional): reset to this state if
            provided. Defaults to None.
        cast_torch (bool): cast state to torch if True.

    Returns:
        np.ndarray: the new state of the shape (4, ).
    """
    self.cnt = 0

  @abstractmethod
  def render(self):
    raise NotImplementedError

  def seed(self, seed: int = 0) -> None:
    self.seed_val = seed
    self.rng = np.random.default_rng(seed)
    random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    self.action_space.seed(seed)
    self.observation_space.seed(seed)

  @abstractmethod
  def report(self):
    raise NotImplementedError

  @abstractmethod
  def simulate_one_trajectory(
      self, T_rollout: int, end_criterion: str,
      reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Tuple[np.ndarray, int, Dict]:
    """
    Rolls out the trajectory given the horizon, termination criterion, reset
    keyword arguments, callback afeter every step, and callout after the
    rollout.

    Args:
        T_rollout (int): rollout horizon.
        end_criterion (str): termination criterion.
        reset_kwargs (Dict): keyword argument dictionary for reset function.
        action_kwargs (Dict): keyword argument dictionary for get_action
            function.
        rollout_step_callback (Callable): function to call after every step.
        rollout_episode_callback (Callable): function to call after rollout.

    Returns:
        np.ndarray: state trajectory.
        int: result (0: unfinished, 1: success, -1: failure).
        Dict: auxiliarry information -
            "action_hist": action sequence.
            "plan_hist": planning info for every step.
            "reward_hist": rewards for every step.
            "step_hist": information for every step.
    """
    raise NotImplementedError

  def simulate_trajectories(
      self,
      num_trajectories: int,
      T_rollout: int,
      end_criterion: str,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
  ):
    """
    Rolls out multiple trajectories given the horizon, termination criterion,
    reset keyword arguments, callback afeter every step, and callback after the
    rollout. Need to call env.reset() after this function to revert back to the
    training mode.
    """

    if isinstance(reset_kwargs_list, list):
      assert num_trajectories == len(reset_kwargs_list), (
          "The length of reset_kwargs_list does not match with",
          "the number of rollout trajectories"
      )
    if isinstance(action_kwargs_list, list):
      assert num_trajectories == len(action_kwargs_list), (
          "The length of action_kwargs_list does not match with",
          "the number of rollout trajectories"
      )

    results = np.empty(shape=(num_trajectories,), dtype=int)
    length = np.empty(shape=(num_trajectories,), dtype=int)

    trajectories = []
    for trial in range(num_trajectories):
      if isinstance(reset_kwargs_list, list):
        reset_kwargs = reset_kwargs_list[trial]
      else:
        reset_kwargs = reset_kwargs_list
      if isinstance(action_kwargs_list, list):
        action_kwargs = action_kwargs_list[trial]
      else:
        action_kwargs = action_kwargs_list

      state_hist, result, _ = self.simulate_one_trajectory(
          T_rollout=T_rollout, end_criterion=end_criterion,
          reset_kwargs=reset_kwargs, action_kwargs=action_kwargs,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback
      )
      trajectories.append(state_hist)
      results[trial] = result
      length[trial] = len(state_hist)

    return trajectories, results, length
