"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Optional, Tuple, Callable, Any
import numpy as np

from .dynamics.bicycle_dynamics import BicycleDynamics
from .ell_reach.ellipse import Ellipse


class Agent:
  """A basic unit in our environments.

  Attributes:
      dyn (object): agent's dynamics.
      footprint (object): agent's shape.
      nominal_states (np.ndarray): nominal states of the planned trajectory.
      nominal_controls (np.ndarray): nominal controls of the planned
          trajectory.
  """

  def __init__(self, config, action_space: np.ndarray) -> None:
    if config.DYN == "Bicycle":
      self.dyn = BicycleDynamics(config, action_space)

    if config.FOOTPRINT == "Ellipse":
      ego_a = config.LENGTH / 2.0
      ego_b = config.WIDTH / 2.0
      ego_q = np.array([config.CENTER, 0])[:, np.newaxis]
      ego_Q = np.diag([ego_a**2, ego_b**2])
      self.footprint = Ellipse(q=ego_q, Q=ego_Q)

    self._policy = None
    self._nominal_trajectory = {}

  @property
  def nominal_states(self):
    assert self._nominal_trajectory, "The nominal trajectory is empty!"
    return self._nominal_trajectory["nominal_states"].copy()

  @property
  def nominal_controls(self):
    assert self._nominal_trajectory, "The nominal trajectory is empty!"
    return self._nominal_trajectory["nominal_controls"].copy()

  def update_policy(self, policy: Callable[[np.ndarray, Any], np.ndarray]):
    self._policy = policy

  def update_nominal_trajectory(
      self, nominal_states: np.ndarray, nominal_controls: np.ndarray
  ):
    self._nominal_trajectory["nominal_states"] = nominal_states
    self._nominal_trajectory["nominal_controls"] = nominal_controls

  def integrate_forward(
      self, state: np.ndarray, control: Optional[np.ndarray] = None,
      step: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif',
      adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state and
    control input.

    Args:
        state (np.ndarray): (4, ) array [X, Y, V, psi].
        control (np.ndarray): (2, ) array [a, delta].
        step (int, optional): The number of segements to forward the
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
        "Tou need to either pass in a control or construct a policy!"
    )
    if control is None:
      control = self.policy(state, **kwargs)
    return self.dyn.integrate_forward(
        state, control, step, noise, noise_type, adversary, **kwargs
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

  def policy(self, state: np.ndarray, **kwargs) -> np.ndarray:
    assert self._policy is not None, "You need to first pass in a policy"
    return self._policy(state, **kwargs)
