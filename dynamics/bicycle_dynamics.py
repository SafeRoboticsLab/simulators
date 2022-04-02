from typing import Optional, Tuple, Any
import numpy as np

from ..base_dynamics import BaseDynamics


class BicycleDynamics(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    self.dim_x = 4
    self.dim_u = 2

    # load parameters
    self.dt = config.DT  # time step for each planning step
    self.wheelbase = config.WHEELBASE  # vehicle chassis length
    self.a_min = action_space[0, 0]  # min longitudial accel
    self.a_max = action_space[0, 1]  # max longitudial accel
    self.delta_min = action_space[1, 0]  # min steering angle rad
    self.delta_max = action_space[1, 1]  # max steering angle rad

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
    """

    # Clips the controller values between min and max accel and steer values.
    accel = np.clip(control[0], self.a_min, self.a_max)
    delta = np.clip(control[1], self.delta_min, self.delta_max)
    control_clip = np.array([accel, delta])
    if adversary is not None:
      accel = accel + adversary[0]
      delta = delta + adversary[1]

    # Euler method
    state_nxt = np.copy(state)
    dt_step = self.dt / step  # step size of Euler method
    for _ in range(step):
      # State: [x, y, v, psi]
      d_x = ((state_nxt[2] * dt_step + 0.5 * accel * dt_step**2)
             * np.cos(state_nxt[3]))
      d_y = ((state_nxt[2] * dt_step + 0.5 * accel * dt_step**2)
             * np.sin(state_nxt[3]))
      d_v = accel * dt_step
      d_psi = ((state_nxt[2] * dt_step + 0.5 * accel * dt_step**2)
               * np.tan(delta) / self.wheelbase)
      state_nxt = state_nxt + np.array([d_x, d_y, d_v, d_psi])

      # Adds noises.
      if noise is not None:
        transform_mtx = np.array([[
            np.cos(state_nxt[-1]),
            np.sin(state_nxt[-1]), 0, 0
        ], [-np.sin(state_nxt[-1]),
            np.cos(state_nxt[-1]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if noise_type == 'unif':
          rv = (np.random.rand(4) - 0.5) * 2  # Maps to [-1, 1]
        else:
          rv = np.random.normal(size=(4))
        state_nxt = state_nxt + (transform_mtx@noise) * rv / step

    return state_nxt, control_clip

  def get_jacobian(
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
    self.N = nominal_states.shape[1]  # number of planning steps
    self.zeros = np.zeros((self.N))
    self.ones = np.ones((self.N))

    v = nominal_states[2, :]
    psi = nominal_states[3, :]
    accel = nominal_controls[0, :]
    delta = nominal_controls[1, :]

    A = np.empty((4, 4, self.N), dtype=float)
    A[0, :, :] = [
        self.ones, self.zeros,
        np.cos(psi) * self.dt,
        -(v * self.dt + 0.5 * accel * self.dt**2) * np.sin(psi)
    ]
    A[1, :, :] = [
        self.zeros, self.ones,
        np.sin(psi) * self.dt,
        (v * self.dt + 0.5 * accel * self.dt**2) * np.cos(psi)
    ]
    A[2, :, :] = [self.zeros, self.zeros, self.ones, self.zeros]
    A[3, :, :] = [
        self.zeros, self.zeros,
        np.tan(delta) * self.dt / self.wheelbase, self.ones
    ]

    B = np.empty((4, 2, self.N), dtype=float)
    B[0, :, :] = [self.dt**2 * np.cos(psi) / 2, self.zeros]
    B[1, :, :] = [self.dt**2 * np.sin(psi) / 2, self.zeros]
    B[2, :, :] = [self.dt * self.ones, self.zeros]
    B[3, :, :] = [
        np.tan(delta) * self.dt**2 / (2 * self.wheelbase),
        (v * self.dt + 0.5 * accel * self.dt**2) /
        (self.wheelbase * np.cos(delta)**2)
    ]

    return A, B
