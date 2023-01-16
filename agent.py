"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from multiprocessing.sharedctypes import Value
from typing import Optional, Tuple, Any, Union, Dict, List, Callable
import copy
import numpy as np
import torch

# Dynamics.
# from .dynamics.bicycle4D import Bicycle4D
# from .dynamics.bicycle5D import Bicycle5D
# from .dynamics.bicycle5D_dstb import BicycleDstb5D

from .cost.base_cost import BaseCost

# Footprint.
# from .ell_reach.ellipse import Ellipse
# from .footprint.box import BoxFootprint

# Policy.
from .policy.base_policy import BasePolicy
# from .policy.ilqr_policy import iLQR
# from .policy.ilqr_spline_policy import iLQRSpline
# from .policy.ilqr_reachability_spline_policy import iLQRReachabilitySpline
from .policy.nn_policy import NeuralNetworkControlSystem


class Agent:
  """A basic unit in our environments.

  Attributes:
      dyn (object): agent's dynamics.
      footprint (object): agent's shape.
      policy (object): agent's policy.
  """
  policy: Optional[BasePolicy]
  safety_policy: Optional[BasePolicy]
  ego_observable: Optional[List]
  agents_policy: Dict[str, BasePolicy]
  agents_order: Optional[List]

  def __init__(self, config, action_space: np.ndarray, env=None) -> None:
    if config.DYN == "Bicycle4D":
      self.dyn = Bicycle4D(config, action_space)
    elif config.DYN == "Bicycle5D":
      self.dyn = Bicycle5D(config, action_space)
    elif config.DYN == "BicycleDstb5D":
      self.dyn = BicycleDstb5D(config, action_space)
    elif config.DYN == "SpiritPybullet":
      # Prevents from opening a pybullet simulator when we don't need to.
      from .dynamics.spirit_dynamics_pybullet import SpiritDynamicsPybullet
      self.dyn = SpiritDynamicsPybullet(config, action_space)
    elif config.DYN == "GVRPybullet":
      from .dynamics.gvr_dynamics_pybullet import GVRDynamicsPybullet
      self.dyn = GVRDynamicsPybullet(config, action_space)
    else:
      raise ValueError("Dynamics type not supported!")

    try:
      self.env = copy.deepcopy(env)  # imaginary environment
    except Exception as e:
      print("WARNING: Cannot copy env - {}".format(e))

    if config.FOOTPRINT == "Ellipse":
      ego_a = config.LENGTH / 2.0
      ego_b = config.WIDTH / 2.0
      ego_q = np.array([config.CENTER, 0])[:, np.newaxis]
      ego_Q = np.diag([ego_a**2, ego_b**2])
      self.footprint = Ellipse(q=ego_q, Q=ego_Q)
    elif config.FOOTPRINT == "Box":  # TODO
      self.footprint = BoxFootprint(box_limit=config.BOX_LIMIT)

    # Policy should be initialized by `init_policy()`.
    self.policy = None
    self.safety_policy = None
    self.id: str = config.AGENT_ID
    self.ego_observable = None
    self.agents_policy = {}
    self.agents_order = None

  def integrate_forward(
      self, state: np.ndarray, control: Optional[Union[np.ndarray,
                                                       torch.Tensor]] = None,
      num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif',
      adversary: Optional[Union[np.ndarray, torch.Tensor]] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state and
    control input.

    Args:
        state (np.ndarray): (dyn.dim_x, ) array.
        control (np.ndarray): (dyn.dim_u, ) array.
        num_segment (int, optional): The number of segements to forward the
            dynamics. Defaults to 1.
        noise (np.ndarray, optional): the ball radius or standard
            deviation of the Gaussian noise. The magnitude should be in the
            sense of self.dt. Defaults to None.
        noise_type(str, optional): Uniform or Gaussian. Defaults to 'unif'.
        adversary (np.ndarray, optional): adversarial control (disturbance).
            Defaults to None.

    Returns:
        np.ndarray: next state.
        np.ndarray: clipped control.
    """
    assert control is not None or self.policy is not None, (
        "You need to either pass in a control or construct a policy!"
    )
    if control is None:
      obs: np.ndarray = kwargs.get('obs')
      kwargs['state'] = state.copy()
      control = self.get_action(obs=obs.copy(), **kwargs)[0]
    elif not isinstance(control, np.ndarray):
      control = control.cpu().numpy()
    if noise is not None:
      assert isinstance(noise, np.ndarray)
    if adversary is not None and not isinstance(adversary, np.ndarray):
      adversary = adversary.cpu().numpy()

    return self.dyn.integrate_forward(
        state=state, control=control, num_segment=num_segment, noise=noise,
        noise_type=noise_type, adversary=adversary, **kwargs
    )

  def get_dyn_jacobian(
      self, nominal_states: np.ndarray, nominal_controls: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the linearized 'A' and 'B' matrix of the ego vehicle around
    nominal states and controls.

    Args:
        nominal_states (np.ndarray): states along the nominal trajectory.
        nominal_controls (np.ndarray): controls along the trajectory.

    Returns:
        np.ndarray: the Jacobian of next state w.r.t. the current state.
        np.ndarray: the Jacobian of next state w.r.t. the current control.
    """
    A, B = self.dyn.get_jacobian(nominal_states, nominal_controls)
    return np.asarray(A), np.asarray(B)

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
    if self.ego_observable is not None:
      for agent_id in self.ego_observable:
        assert agent_id in agents_action

    if agents_action is not None:
      _action_dict = copy.deepcopy(agents_action)
    else:
      _action_dict = {}
    _action, _solver_info = self.policy.get_action(  # Proposed action.
        obs=obs, agents_action=agents_action, **kwargs
    )
    _action_dict[self.id] = _action

    agents_kwargs = kwargs.get("agents_kwargs", None)
    if self.agents_order is not None:
      for agent_id in self.agents_order:
        # print(f"get action for {agent_id}")
        if agent_id not in _action_dict:
          agent_policy: BasePolicy = self.agents_policy[agent_id]
          agent_kwarg = {"state": kwargs.get("state", None)}
          if agents_kwargs is not None:
            if agent_id in agents_kwargs:
              agent_kwarg.update(agents_kwargs[agent_id])
          _action_dict[agent_policy.id] = agent_policy.get_action(
              obs=obs, agents_action=_action_dict, **agent_kwarg
          )[0]

    policy_type = kwargs.get('policy_type', 'task')
    # TODO: different action after computing others' imaginary actions for
    # TODO: task and safety.
    if policy_type == 'task':
      pass
    elif policy_type == 'safety':
      assert self.safety_policy is not None
      _action, _solver_info = self.safety_policy.get_action(
          obs=obs, agents_action=_action_dict, **kwargs
      )
    elif policy_type == 'shield':
      shield_kwargs: Dict = kwargs.get("shield_kwargs")
      use_safe_action = False
      shield_type = shield_kwargs['type']
      if shield_type == "value":
        shield_thr = shield_kwargs['thr']
        critic_action_dict = {
            k: _action_dict[k] for k in self.safety_policy.critic_agents_order
        }
        safety_value = self.safety_policy.critic(
            obs=obs, action=critic_action_dict
        )
        if safety_value > shield_thr:
          use_safe_action = True
      elif shield_type == "rollout" or shield_type == "mixed":
        #! bugs!!!!
        imag_rollout_steps = shield_kwargs['imag_rollout_steps']
        imag_end_criterion = shield_kwargs['imag_end_criterion']

        if self.env.env_type == "zero-sum":
          adversary = self.agents_policy['dstb']

          def adversary_fn(
              obs: np.ndarray, ctrl: np.ndarray, **kwargs
          ) -> np.ndarray:
            return adversary.get_action(
                obs=obs, agents_action={self.id: ctrl}, **kwargs
            )[0]

          tmp_action = {
              'ctrl':
                  _action,
              'dstb':
                  adversary_fn(
                      obs=obs, ctrl=_action.copy(), state=kwargs.get('state')
                  )
          }
          self.env.reset(state=kwargs.get('state'))
          self.env.end_criterion = imag_end_criterion
          _, _, done, step_info = self.env.step(tmp_action)
          if done:
            result = 0
            if step_info["done_type"] == "success":
              result = 1
            elif step_info["done_type"] == "failure":
              result = -1
          else:
            traj, result, info = self.env.simulate_one_trajectory(
                T_rollout=imag_rollout_steps, end_criterion=imag_end_criterion,
                adversary=adversary_fn,
                reset_kwargs=dict(state=self.env.state.copy())
            )
            if shield_type == "mixed" and result == 0:
              # skips if it enters in the target or failure set.
              state_ter = traj[-1]
              obs_ter = info['obs_hist'][-1]

              # Collects all other agents' actions.
              action_dict_ter = {}
              for agent_id in self.agents_order:
                # print(f"get action for {agent_id}")
                if agent_id == 'ego':
                  agent_policy: BasePolicy = self.safety_policy
                else:
                  agent_policy: BasePolicy = self.agents_policy[agent_id]
                agent_kwarg = {"state": state_ter}
                if agents_kwargs is not None:
                  if agent_id in agents_kwargs:
                    agent_kwarg.update(agents_kwargs[agent_id])
                action_dict_ter[agent_policy.id] = agent_policy.get_action(
                    obs=obs_ter, agents_action=action_dict_ter, **agent_kwarg
                )[0]

              critic_action_dict_ter = {
                  k: action_dict_ter[k]
                  for k in self.safety_policy.critic_agents_order
              }
              safety_value_ter = self.safety_policy.critic(
                  obs=obs_ter, action=critic_action_dict_ter
              )
              if safety_value_ter > shield_kwargs['thr']:
                result = -1

          if imag_end_criterion == "reach-avoid" and result != 1:
            use_safe_action = True
          elif imag_end_criterion == "failure" and result == -1:
            use_safe_action = True
        else:
          pass  # TODO
      else:
        raise ValueError(f"Not supported shielding type ({shield_type})")
      if use_safe_action:
        _solver_info['task_action'] = _action.copy()
        safety_policy_kwargs = kwargs.get('safety_policy_kwargs', {})
        _action, _ = self.safety_policy.get_action(
            obs=obs, agents_action=agents_action, **safety_policy_kwargs
        )
        _solver_info['shield'] = True
      else:
        _action = _action.copy()
        _solver_info['shield'] = False
    else:
      raise ValueError(f"Not supported policy type ({policy_type})!")
    return _action, _solver_info

  def init_policy(
      self, policy_type: str, config, cost: Optional[BaseCost] = None, **kwargs
  ):
    if policy_type == "iLQR":
      self.policy = iLQR(self.id, config, self.dyn, cost, **kwargs)
    elif policy_type == "iLQRSpline":
      self.policy = iLQRSpline(self.id, config, self.dyn, cost, **kwargs)
    elif policy_type == "iLQRReachabilitySpline":
      self.policy = iLQRReachabilitySpline(
          self.id, config, self.dyn, cost, **kwargs
      )
    # elif policy_type == "MPC":
    elif policy_type == "NNCS":
      self.policy = NeuralNetworkControlSystem(
          id=self.id, config=config, **kwargs
      )
    else:
      raise ValueError(
          "The policy type ({}) is not supported!".format(policy_type)
      )

  def update_agents_policy(
      self, agents_policy: Dict[str, BasePolicy], agents_order: List,
      ego_observable: Optional[List] = None
  ):
    """
    The policy can have imaginary gameplays with other agents. We use a
    dictionary to keep other agents' policy (actor) and a list of agents whose
    actions are observable to that specific agent.

    Args:
        agents_policy (Union[BasePolicy, Dict[str, BasePolicy]])
        agents_order (List)
        ego_observable (Optional[List])
    """
    for key in agents_policy.keys():
      self.agents_policy[key] = copy.deepcopy(agents_policy[key])

    self.agents_order = copy.deepcopy(agents_order)

    if ego_observable is not None:
      idx_ego = agents_order.index(self.id)
      for agent_id in ego_observable:
        assert agent_id in self.agents_policy
        idx_agent = agents_order.index(agent_id)
        assert idx_agent < idx_ego
      self.ego_observable = copy.deepcopy(ego_observable)

  def update_safety_module(
      self,
      policy: BasePolicy,
      critic: Optional[Union[torch.nn.Module, Callable[
          [np.ndarray, np.ndarray, Optional[np.ndarray]], float]]] = None,
      critic_agents_order: Optional[List] = None,
  ):

    self.safety_policy = copy.deepcopy(policy)
    if critic is not None:
      if isinstance(critic, torch.nn.Module):
        self.safety_policy._critic = copy.deepcopy(critic)
      else:
        self.safety_policy._critic = critic
      self.safety_policy.critic_agents_order = copy.deepcopy(
          critic_agents_order
      )

  def report(self):
    print(self.id)
    if self.ego_observable is not None:
      print("  - The agent can observe:", end=' ')
      for i, k in enumerate(self.ego_observable):
        print(k, end='')
        if i == len(self.ego_observable) - 1:
          print('.')
        else:
          print(', ', end='')
    else:
      print("  - The agent can only access observation.")

    if self.agents_order is not None:
      print("  - The agent keeps agents order:", end=' ')
      for i, k in enumerate(self.agents_order):
        print(k, end='')
        if i == len(self.agents_order) - 1:
          print('.')
        else:
          print(' -> ', end='')
