"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from abc import abstractmethod
from typing import Any, Tuple, Optional, Callable, List, Dict, Union
import numpy as np
import torch
from gym import spaces

from .agent import Agent
from .base_env import BaseEnv


class BaseSingleEnv(BaseEnv):
  """Implements an environment of a single agent.
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    super().__init__(config_env)

    # Action Space.
    action_space = np.array(config_agent.ACTION_LIMIT, dtype=np.float32)
    self.action_dim = action_space.shape[0]
    self.agent = Agent(config_agent, action_space)
    self.action_space = spaces.Box(
        low=action_space[:, 0], high=action_space[:, 1]
    )
    self.state_dim = self.agent.dyn.dim_x

    self.integrate_kwargs = getattr(config_env, "INTEGRATE_KWARGS", {})
    if "noise" in self.integrate_kwargs:
      if self.integrate_kwargs['noise'] is not None:
        self.integrate_kwargs['noise'] = np.array(
            self.integrate_kwargs['noise']
        )

  def step(
      self, action: np.ndarray, cast_torch: bool = False
  ) -> Tuple[Union[np.ndarray, torch.FloatTensor], float, bool, Dict]:
    """Implements the step function in the environment.

    Args:
        action (np.ndarray).
        cast_torch (bool): cast state to torch if True.

    Returns:
        np.ndarray: next state.
        float: the reward that ctrl wants to maximize and dstb wants to
            minimize.
        bool: True if the episode ends.
        Dict[str, Any]]: additional information of the step, such as target
            margin and safety margin used in reachability analysis.
    """

    self.cnt += 1
    state_nxt, _ = self.agent.integrate_forward(
        state=self.state, control=action, **self.integrate_kwargs
    )
    constraints = self.get_constraints(self.state, action, state_nxt)
    cost = self.get_cost(self.state, action, state_nxt, constraints)
    targets = self.get_target_margin(self.state, action, state_nxt)
    done, info = self.get_done_and_info(constraints, targets)

    self.state = np.copy(state_nxt)

    obs = self.get_obs(state_nxt)
    if cast_torch:
      obs = torch.FloatTensor(obs)

    return obs, -cost, done, info

  @abstractmethod
  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Optional[Dict] = None
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current states.
        action (np.ndarray): current actions.
        state_nxt (np.ndarray): next state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        float: the cost to minimize.
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
        state (np.ndarray): current states.
        action (np.ndarray): current actions.
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_target_margin(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all target margin functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states.
        action (np.ndarray): current actions.
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a target margin
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_and_info(
      self, constraints: Dict, targets: Dict, final_only: bool = True,
      end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets.

    Args:
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.
        targets (Dict): each (key, value) pair is the name and value of a
            target margin function.

    Returns:
        bool: True if the episode ends.
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    raise NotImplementedError

  def simulate_one_trajectory(
      self,
      T_rollout: int,
      end_criterion: str,
      reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
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
    # Stores the environment attributes and sets to rollout settings.
    timeout_backup = self.timeout
    end_criterion_backup = self.end_criterion
    self.timeout = T_rollout
    self.end_criterion = end_criterion
    if reset_kwargs is None:
      reset_kwargs = {}
    if action_kwargs is None:
      action_kwargs = {}

    state_hist = []
    action_hist = []
    reward_hist = []
    plan_hist = []
    step_hist = []

    # Initializes robot.
    if self.agent.policy.policy_type == "iLQR":
      init_control = np.zeros((self.action_dim, self.agent.policy.N - 1))
    result = 0
    obs = self.reset(**reset_kwargs)
    state_hist.append(self.state)

    for t in range(T_rollout):
      # Gets action.
      if self.agent.policy.policy_type == "iLQR":
        action, solver_info = self.agent.policy.get_action(
            state=self.state, controls=init_control, **action_kwargs
        )
      elif self.agent.policy.policy_type == "NNCS":
        with torch.no_grad():
          obs_tensor = torch.FloatTensor(obs).to(self.agent.policy.device)
          action, solver_info = self.agent.policy.get_action(
              state=obs_tensor, **action_kwargs
          )

      # Applies action: `done` and `info` are evaluated at the next state.
      obs, reward, done, step_info = self.step(action)

      # Executes step callback and stores history.
      state_hist.append(self.state)
      action_hist.append(action)
      plan_hist.append(solver_info)
      reward_hist.append(reward)
      step_hist.append(step_info)
      if rollout_step_callback is not None:
        rollout_step_callback(
            self, state_hist, action_hist, plan_hist, step_hist
        )

      # Checks termination criterion.
      if done:
        if step_info["done_type"] == "success":
          result = 1
        elif step_info["done_type"] == "failure":
          result = -1
        break

      if self.agent.policy.policy_type == "iLQR":  # Better warmup.
        init_control[:, :-1] = solver_info['controls'][:, 1:]
        init_control[:, -1] = 0.

    if rollout_episode_callback is not None:
      rollout_episode_callback(
          self, state_hist, action_hist, plan_hist, step_hist
      )
    # Reverts to training setting.
    self.timeout = timeout_backup
    self.end_criterion = end_criterion_backup
    return np.array(state_hist), result, dict(
        action_hist=np.array(action_hist), plan_hist=plan_hist,
        reward_hist=np.array(reward_hist), step_hist=step_hist
    )

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
