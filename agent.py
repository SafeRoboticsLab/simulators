from typing import Optional, Tuple
import numpy as np

from .race_car.bicycle_dynamics import BicycleDynamics
from .ell_reach.ellipse import Ellipse


class Agent:

  def __init__(self, config, action_space: np.ndarray) -> None:
    if config.DYN == "Bicycle":
      self.dyn = BicycleDynamics(config, action_space)

    if config.FOOTPRINT == "Ellipse":
      ego_a = config.LENGTH / 2.0
      ego_b = config.WIDTH / 2.0
      ego_q = np.array([config.CENTER, 0])[:, np.newaxis]
      ego_Q = np.diag([ego_a**2, ego_b**2])
      self.footprint = Ellipse(q=ego_q, Q=ego_Q)

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray, step: Optional[int] = 1,
      noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif',
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
    return self.dyn.get_dyn_jacobian(nominal_states, nominal_controls)
