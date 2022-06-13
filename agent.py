"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Optional, Tuple, Any
import numpy as np

from .dynamics.spirit_dynamics_pybullet import SpiritDynamicsPybullet

# Dynamics.
from .dynamics.bicycle_dynamics_v1 import BicycleDynamicsV1
from .dynamics.bicycle_dynamics_v2 import BicycleDynamicsV2

# Footprint.
# from .ell_reach.ellipse import Ellipse

# Policy.
from .policy.ilqr_policy import iLQR
from .policy.nn_policy import NeuralNetworkControlSystem


class Agent:
  """A basic unit in our environments.

  Attributes:
      dyn (object): agent's dynamics.
      footprint (object): agent's shape.
      policy (object): agent's policy.
  """

  def __init__(self, config, action_space: np.ndarray) -> None:
    if config.DYN == "BicycleV1":
      self.dyn = BicycleDynamicsV1(config, action_space)
    elif config.DYN == "BicycleV2":
      self.dyn = BicycleDynamicsV2(config, action_space)
    elif config.DYN == "SpiritPybullet":
      self.dyn = SpiritDynamicsPybullet(config, action_space)
    else:
      raise ValueError("Dynamics type not supported!")

    if config.FOOTPRINT == "Ellipse":
      ego_a = config.LENGTH / 2.0
      ego_b = config.WIDTH / 2.0
      ego_q = np.array([config.CENTER, 0])[:, np.newaxis]
      ego_Q = np.diag([ego_a**2, ego_b**2])
      self.footprint = Ellipse(q=ego_q, Q=ego_Q)

    # Policy should be initialized by `init_policy()`.
    self.policy = None

  def integrate_forward(
      self, state: np.ndarray, control: Optional[np.ndarray] = None,
      num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif',
      adversary: Optional[np.ndarray] = None, **kwargs
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
    assert control is not None or self._policy is not None, (
        "You need to either pass in a control or construct a policy!"
    )
    if control is None:
      control = self.policy.get_action(state, **kwargs)[0]
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
    return self.dyn.get_jacobian(nominal_states, nominal_controls)

  def init_policy(
      self, policy_type: str, config: Any, env: Optional[Any] = None, **kwargs
  ):
    if policy_type == "iLQR":
      self.policy = iLQR(env, config)
    # elif policy_type == "MPC":
    elif policy_type == "NNCS":
      self.policy = NeuralNetworkControlSystem(
          env, kwargs['critic'], kwargs['actor'], config
      )
    else:
      raise ValueError(
          "The policy type ({}) is not supported!".format(policy_type)
      )
