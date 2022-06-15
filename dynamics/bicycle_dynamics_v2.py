"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Optional, Tuple, Any
import numpy as np

from .base_dynamics import BaseDynamics


class BicycleDynamicsV2(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    self.dim_x = 5  # [x, y, v, psi, delta].
    self.dim_u = 2  # [a, w].

    # load parameters
    self.dt = config.DT  # time step for each planning step
    self.wheelbase = config.WHEELBASE  # vehicle chassis length
    self.a_min = action_space[0, 0]  # min longitudial accel
    self.a_max = action_space[0, 1]  # max longitudial accel
    self.w_min = action_space[1, 0]  # min steering vel rad/s
    self.w_max = action_space[1, 1]  # max steering vel rad/s

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray,
      num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif',
      adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state and
    control input. The discrete-time dynamics is as below
        x_k+1 = x_k + v_k cos(psi_k) dt
        y_k+1 = y_k + v_k sin(psi_k) dt
        v_k+1 = v_k + u1_k dt + d1_k dt
        psi_k+1 = psi_k + v_k tan(delta_k) / L dt + d2_k dt
        delta_k+1 = delta_k + u2_k dt

    Args:
        state (np.ndarray): (5, ) array [x, y, v, psi, delta].
        control (np.ndarray): (2, ) array [v_dot, delta_dot].
        num_segment (int, optional): The number of segements to forward the
            dynamics. Defaults to 1.
        noise (np.ndarray, optional): the ball radius or standard
            deviation of the Gaussian noise. The magnitude should be in the
            sense of self.dt. Defaults to None.
        noise_type(str, optional): Uniform or Gaussian. Defaults to 'unif'.
        adversary (np.ndarray, optional): adversarial control (disturbance)
            with array [d_v, d_psi]. Defaults to None.

    Returns:
        np.ndarray: next state.
    """

    # Clips the controller values between min and max accel and steer values.
    accel = np.clip(control[0], self.a_min, self.a_max)
    omega = np.clip(control[1], self.w_min, self.w_max)
    control_clip = np.array([accel, omega])

    # Euler method
    state_nxt = np.copy(state)
    dt_step = self.dt / num_segment  # step size of Euler method
    for _ in range(num_segment):
      # State: [x, y, v, psi, delta]
      d_x = state_nxt[2] * np.cos(state_nxt[3]) * dt_step
      d_y = state_nxt[2] * np.sin(state_nxt[3]) * dt_step
      d_v = accel * dt_step
      d_psi = state_nxt[2] * np.tan(state_nxt[4]) / self.wheelbase * dt_step
      d_delta = omega * dt_step
      state_nxt += np.array([d_x, d_y, d_v, d_psi, d_delta])

      if adversary is not None:
        d_v_dstb = adversary[0] * dt_step
        d_psi_dstb = adversary[1] * dt_step
        state_nxt += np.array([0, 0, d_v_dstb, d_psi_dstb, 0])

      # Adds noises.
      if noise is not None:
        assert noise.shape[0] == self.dim_x, ("Noise dim. is incorrect!")
        cos = np.cos(state_nxt[-1])
        sin = np.sin(state_nxt[-1])
        transform_mtx = np.array([[cos, sin, 0, 0, 0], [-sin, cos, 0, 0, 0],
                                  [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]])
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
    delta = nominal_states[4, :]

    A = np.empty((self.dim_x, self.dim_x, self.N), dtype=float)
    cos = np.cos(psi) * self.dt
    sin = np.sin(psi) * self.dt
    constant = self.dt / self.wheelbase
    A[0, :, :] = [ones, zeros, cos, -v * sin, zeros]
    A[1, :, :] = [zeros, ones, sin, v * cos, zeros]
    A[2, :, :] = [zeros, zeros, ones, zeros, zeros]
    A[3, :, :] = [
        zeros, zeros,
        np.tan(delta) * constant, ones, v * constant / np.cos(delta)**2
    ]
    A[4, :, :] = [zeros, zeros, zeros, zeros, ones]

    B = np.empty((self.dim_x, self.dim_u, self.N), dtype=float)
    B[0, :, :] = [zeros, zeros]
    B[1, :, :] = [zeros, zeros]
    B[2, :, :] = [self.dt * ones, zeros]
    B[3, :, :] = [zeros, zeros]
    B[4, :, :] = [zeros, self.dt * ones]

    return A, B
