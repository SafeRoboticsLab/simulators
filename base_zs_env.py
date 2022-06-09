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
from .utils import ActionZS, build_obs_space, cast_numpy


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
    # Other keyword arguments for integrate_forward, such as step, noise,
    # noise_type.
    self.integrate_kwargs = config_agent.INTEGRATE_KWARGS

    dstb_space = np.array(config_agent.ACTION_RANGE['DSTB'])
    self.action_space_dstb = spaces.Box(
        low=dstb_space[:, 0], high=dstb_space[:, 1]
    )
    self.action_dim_dstb = dstb_space.shape[0]
    self.action_space = spaces.Dict(
        dict(ctrl=self.action_space_ctrl, dstb=self.action_space_dstb)
    )

    # Observation Space.
    obs_spec = np.array(config_agent.OBS_RANGE)
    self.observation_space = build_obs_space(
        obs_spec=obs_spec, obs_dim=config_agent.OBS_DIM
    )
    self.observation_dim = self.observation_space.low.shape
    self.state = self.observation_space.sample()  # Overriden by reset later.

  def step(self, action: ActionZS,
           cast_torch: bool = False) -> Tuple[np.ndarray, float, bool, Dict]:
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
    ctrl = action['ctrl']
    dstb = action['dstb']
    ctrl = cast_numpy(ctrl)
    dstb = cast_numpy(dstb)

    self.cnt += 1
    state_nxt, _ = self.agent.integrate_forward(
        state=self.state, control=ctrl, adversary=dstb, **self.integrate_kwargs
    )
    constraints = self.get_constraints(self.state, ctrl, state_nxt)
    cost = self.get_cost(self.state, ctrl, state_nxt, constraints)
    targets = self.get_target_margin(self.state, ctrl, state_nxt)
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
      self,
      T_rollout: int,
      end_criterion: str,
      adversary: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
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
        ctrl, solver_info = self.agent.policy.get_action(
            state=self.state, controls=init_control, **action_kwargs
        )
      elif self.agent.policy.policy_type == "NNCS":
        with torch.no_grad():
          obs_tensor = torch.FloatTensor(obs).to(self.agent.policy.device)
          ctrl, solver_info = self.agent.policy.get_action(
              state=obs_tensor, **action_kwargs
          )

      # Applies action: `done` and `info` are evaluated at the next state.
      action = {'ctrl': ctrl, 'dstb': adversary(self.state, ctrl)}
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
        action_hist=action_hist, plan_hist=plan_hist,
        reward_hist=np.array(reward_hist), step_hist=step_hist
    )
