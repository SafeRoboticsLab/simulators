"""
Please contact the author(s) of this library if you have any questions.
Authors:  Zixu Zhang ( zixuz@princeton.edu )
          Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Optional, Tuple, Any
import numpy as np

from .base_dynamics import BaseDynamics


class BicycleDynamicsV1(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    self.dim_x = 4  # [x, y, v, psi].
    self.dim_u = 2  # [a, delta].

    # load parameters
    self.dt = config.DT  # time step for each planning step
    self.wheelbase = config.WHEELBASE  # vehicle chassis length
    self.a_min = action_space[0, 0]  # min longitudial accel
    self.a_max = action_space[0, 1]  # max longitudial accel
    self.delta_min = action_space[1, 0]  # min steering angle rad
    self.delta_max = action_space[1, 1]  # max steering angle rad

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray,
      num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif',
      adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state and
    control input.

    Args:
        state (np.ndarray): (4, ) array [x, y, v, psi].
        control (np.ndarray): (2, ) array [a, delta].
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
    dt_step = self.dt / num_segment  # step size of Euler method
    for _ in range(num_segment):
      # State: [x, y, v, psi]
      d_x = ((state_nxt[2] * dt_step + 0.5 * accel * dt_step**2)
             * np.cos(state_nxt[3]))
      d_y = ((state_nxt[2] * dt_step + 0.5 * accel * dt_step**2)
             * np.sin(state_nxt[3]))
      d_v = accel * dt_step
      d_psi = ((state_nxt[2] * dt_step + 0.5 * accel * dt_step**2)
               * np.tan(delta) / self.wheelbase)
      state_nxt += np.array([d_x, d_y, d_v, d_psi])

      # Adds noises.
      if noise is not None:
        assert noise.shape[0] == self.dim_x, ("Noise dim. is incorrect!")
        cos = np.cos(state_nxt[-1])
        sin = np.sin(state_nxt[-1])
        transform_mtx = np.array([[cos, sin, 0, 0], [-sin, cos, 0, 0],
                                  [0, 0, 1, 0], [0, 0, 0, 1]])
        if noise_type == 'unif':
          rv = (np.random.rand(self.dim_x) - 0.5) * 2  # Maps to [-1, 1]
        else:
          rv = np.random.normal(size=(self.dim_x))
        state_nxt = state_nxt + (transform_mtx@noise) * rv / num_segment

    state_nxt[3] = np.mod(state_nxt[3], 2 * np.pi)
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
    zeros = np.zeros((self.N))
    ones = np.ones((self.N))

    v = nominal_states[2, :]
    psi = nominal_states[3, :]
    accel = nominal_controls[0, :]
    delta = nominal_controls[1, :]

    A = np.empty((self.dim_x, self.dim_x, self.N), dtype=float)
    A[0, :, :] = [
        ones, zeros,
        np.cos(psi) * self.dt,
        -(v * self.dt + 0.5 * accel * self.dt**2) * np.sin(psi)
    ]
    A[1, :, :] = [
        zeros, ones,
        np.sin(psi) * self.dt,
        (v * self.dt + 0.5 * accel * self.dt**2) * np.cos(psi)
    ]
    A[2, :, :] = [zeros, zeros, ones, zeros]
    A[3, :, :] = [zeros, zeros, np.tan(delta) * self.dt / self.wheelbase, ones]

    B = np.empty((self.dim_x, self.dim_u, self.N), dtype=float)
    B[0, :, :] = [self.dt**2 * np.cos(psi) / 2, zeros]
    B[1, :, :] = [self.dt**2 * np.sin(psi) / 2, zeros]
    B[2, :, :] = [self.dt * ones, zeros]
    B[3, :, :] = [
        np.tan(delta) * self.dt**2 / (2 * self.wheelbase),
        (v * self.dt + 0.5 * accel * self.dt**2) /
        (self.wheelbase * np.cos(delta)**2)
    ]

    return A, B
