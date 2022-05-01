"""
Please contact the author(s) of this library if you have any questions.
Authors:  Zixu Zhang ( zixuz@princeton.edu )
          Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Optional, Tuple, Any
import numpy as np

from .base_dynamics import BaseDynamics


class BicycleCosSinDynamics(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    self.dim_x = 5  # [X, Y, V, cos psi, sin psi].
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
        state (np.ndarray): (5, ) array [X, Y, V, psi].
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
      # State: [x, y, v, cos psi, sin psi]
      phi = state_nxt[2] * dt_step + 0.5 * accel * dt_step**2
      tmp = phi * np.tan(delta) / self.wheelbase
      d_x = phi * state_nxt[3]
      d_y = phi * state_nxt[4]
      d_v = accel * dt_step
      new_alpha = state_nxt[3] * np.cos(tmp) - state_nxt[4] * np.sin(tmp)
      new_beta = state_nxt[3] * np.sin(tmp) + state_nxt[4] * np.cos(tmp)

      state_nxt = np.array([
          state_nxt[0] + d_x, state_nxt[1] + d_y, state_nxt[2] + d_v,
          new_alpha, new_beta
      ])

      # Adds noises.
      if noise is not None:
        transform_mtx = np.array([[state_nxt[3], state_nxt[4], 0, 0, 0],
                                  [-state_nxt[4], state_nxt[3], 0, 0, 0],
                                  [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]])
        if noise_type == 'unif':
          rv = (np.random.rand(self.dim_x) - 0.5) * 2  # Maps to [-1, 1]
        else:
          rv = np.random.normal(size=(self.dim_x))
        state_nxt = state_nxt + (transform_mtx@noise) * rv / num_segment

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
    alpha = nominal_states[3, :]
    beta = nominal_states[4, :]
    accel = nominal_controls[0, :]
    delta = nominal_controls[1, :]

    phi = v * self.dt + 0.5 * accel * self.dt**2
    tmp = phi * np.tan(delta) / self.wheelbase
    cos_tmp = np.cos(tmp)
    sin_tmp = np.sin(tmp)

    tmp1 = -alpha * sin_tmp - beta*cos_tmp
    tmp2 = alpha*cos_tmp - beta*sin_tmp

    coeff1 = np.tan(delta) / self.wheelbase * self.dt
    coeff2 = 0.5 * np.tan(delta) / self.wheelbase * self.dt**2
    coeff3 = phi / (self.wheelbase * np.cos(delta)**2)

    A = np.empty((self.dim_x, self.dim_x, self.N), dtype=float)
    A[0, :, :] = [ones, zeros, alpha * self.dt, phi, zeros]
    A[1, :, :] = [zeros, ones, beta * self.dt, zeros, phi]
    A[2, :, :] = [zeros, zeros, ones, zeros, zeros]
    A[3, :, :] = [zeros, zeros, coeff1 * tmp1, cos_tmp, -sin_tmp]
    A[4, :, :] = [zeros, zeros, coeff1 * tmp2, sin_tmp, cos_tmp]

    B = np.empty((self.dim_x, self.dim_u, self.N), dtype=float)
    B[0, :, :] = [self.dt**2 * alpha / 2, zeros]
    B[1, :, :] = [self.dt**2 * beta / 2, zeros]
    B[2, :, :] = [self.dt * ones, zeros]
    B[3, :, :] = [coeff2 * tmp1, coeff3 * tmp1]
    B[4, :, :] = [coeff2 * tmp2, coeff3 * tmp2]

    return A, B