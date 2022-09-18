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
from .utils import ActionZS, build_obs_space

import copy


class BaseZeroSumEnv(BaseEnv):
  """
  Implements an environment of a two-player discrete-time zero-sum dynamic
  game. The agent minimizing the cost is called 'ctrl', while the agent
  maximizing the cost is called 'dstb'.
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    assert config_env.NUM_AGENTS == 2, (
        "Zero-Sum Game currently only supports two agents!"
    )
    super().__init__(config_env)

    # Action Space.
    ctrl_space = np.array(config_agent.ACTION_RANGE['CTRL'])
    self.action_space_ctrl = spaces.Box(
        low=ctrl_space[:, 0], high=ctrl_space[:, 1]
    )
    self.action_dim_ctrl = ctrl_space.shape[0]
    self.agent = Agent(config_agent, ctrl_space)
    self.state_dim = self.agent.dyn.dim_x

    self.integrate_kwargs = getattr(config_env, "INTEGRATE_KWARGS", {})
    if "noise" in self.integrate_kwargs:
      if self.integrate_kwargs['noise'] is not None:
        self.integrate_kwargs['noise'] = np.array(
            self.integrate_kwargs['noise']
        )

    dstb_space = np.array(config_agent.ACTION_RANGE['DSTB'])
    self.action_space_dstb = spaces.Box(
        low=dstb_space[:, 0], high=dstb_space[:, 1]
    )
    self.action_dim_dstb = dstb_space.shape[0]
    self.action_space = spaces.Dict(
        dict(ctrl=self.action_space_ctrl, dstb=self.action_space_dstb)
    )

  def step(
      self, action: ActionZS, cast_torch: bool = False
  ) -> Tuple[Union[np.ndarray, torch.Tensor], float, bool, Dict]:
    """Implements the step function in the environment.

    Args:
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
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
        state=self.state, control=action['ctrl'], adversary=action['dstb'],
        **self.integrate_kwargs
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
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray,
      constraints: Optional[Dict] = None
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current state.
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

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
  def get_target_margin(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all target margin functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states.
        action (ActionZS): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
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
      self, T_rollout: int, end_criterion: str,
      adversary: Callable[[np.ndarray, np.ndarray, Any],
                          np.ndarray], reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None, **kwargs
  ) -> Tuple[np.ndarray, int, Dict]:
    """
    Rolls out the trajectory given the horizon, termination criterion, reset
    keyword arguments, callback afeter every step, and callout after the
    rollout.

    Args:
        T_rollout (int): rollout horizon.
        end_criterion (str): termination criterion.
        adversary (callable): a mapping from current state and ctrl to
            adversarial ctrl (dstb).
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

    controller = None
    if "controller" in kwargs.keys():
      controller = copy.deepcopy(kwargs["controller"])

    state_hist = []
    obs_hist = []
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
      if controller is None:
        # Gets action.
        if self.agent.policy.policy_type == "iLQR":
          ctrl, solver_info = self.agent.policy.get_action(
              state=self.state, controls=init_control, **action_kwargs
          )
        elif self.agent.policy.policy_type == "NNCS":
          with torch.no_grad():
            ctrl, solver_info = self.agent.policy.get_action(
                state=obs, **action_kwargs
            )
      else:
        new_joint_pos = controller.get_action()
        ctrl = new_joint_pos - np.array(
            self.agent.dyn.robot.get_joint_position()
        )
        solver_info = None

      # Applies action: `done` and `info` are evaluated at the next state.
      action = {'ctrl': ctrl, 'dstb': adversary(obs, ctrl)}
      obs, reward, done, step_info = self.step(action)

      # Executes step callback and stores history.
      state_hist.append(self.state)
      obs_hist.append(obs)
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
        obs_hist=obs_hist, action_hist=action_hist, plan_hist=plan_hist,
        reward_hist=np.array(reward_hist), step_hist=step_hist
    )

  def simulate_trajectories(
      self, num_trajectories: int, T_rollout: int, end_criterion: str,
      adversary: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None, **kwargs
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
          adversary=adversary, reset_kwargs=reset_kwargs,
          action_kwargs=action_kwargs,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback, **kwargs
      )
      trajectories.append(state_hist)
      results[trial] = result
      length[trial] = len(state_hist)

    return trajectories, results, length
