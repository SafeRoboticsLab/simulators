"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
from typing import Tuple, Optional
import copy
import time
import numpy as np

from .base_policy import BasePolicy


class iLQR(BasePolicy):

  def __init__(self, env, config) -> None:
    super().__init__()
    self.policy_type = "iLQR"

    self.N = config.N
    self.max_iter = config.MAX_ITER

    self.env = copy.deepcopy(env)

    self.tol = 1e-3  # ILQR update tolerance.
    self.eps = 10  # Numerical stability for Q inverse.
    self.eps_max = 100
    self.eps_min = 1e-3

    self.alphas = 1.1**(-np.arange(10)**2)  # Stepsize scheduler.

  def get_action(
      self, state: np.ndarray, controls: Optional[np.ndarray] = None, **kwargs
  ) -> np.ndarray:
    status = 0
    self.eps = 10

    if controls is None:
      controls = np.zeros((self.env.action_dim, self.N - 1))

    # Rolls out.
    states = np.zeros((self.env.observation_dim, self.N - 1))
    states[:, 0] = state
    for i in range(self.N - 1):
      state_nxt, _ = self.env.agent.integrate_forward(
          states[:, i], controls[:, i], **self.env.integrate_kwargs
      )
      if i == self.N - 2:
        state_final = state_nxt.copy()
      else:
        states[:, i + 1] = state_nxt.copy()

    # Initial Cost.
    J = self.env.get_cost(state=states, action=controls, state_nxt=state_final)

    converged = False
    time0 = time.time()
    for i in range(self.max_iter):
      K_closed_loop, k_open_loop = self.backward_pass(
          nominal_states=states, nominal_controls=controls,
          nominal_state_final=state_final
      )
      updated = False
      for alpha in self.alphas:
        X_new, U_new, x_final_new, J_new = (
            self.forward_pass(
                states, controls, K_closed_loop, k_open_loop, alpha
            )
        )
        if J_new <= J:  # Improved!
          if np.abs((J-J_new) / J) < self.tol:  # Small improvement.
            converged = True

          # Updates nominal trajectory and best cost.
          J = J_new
          states = X_new
          controls = U_new
          state_final = x_final_new
          updated = True
          break
      if updated:
        self.eps *= 0.7
      else:
        status = 2
        break
      # self.eps = min(max(self.eps_min, self.eps), self.eps_max)
      self.eps = max(self.eps_min, self.eps)

      if converged:
        status = 1
        break
    t_process = time.time() - time0

    solver_info = dict(
        states=states, controls=controls, state_final=state_final,
        t_process=t_process, status=status
    )
    return controls[:, 0], solver_info

  def forward_pass(
      self, nominal_states: np.ndarray, nominal_controls: np.ndarray,
      K_closed_loop: np.ndarray, k_open_loop: np.ndarray, alpha: float
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    self._check_shape(
        dict(
            nominal_states=nominal_states, nominal_controls=nominal_controls,
            K_closed_loop=K_closed_loop, k_open_loop=k_open_loop
        )
    )
    X = np.zeros_like(nominal_states)  # (observation_dim, N-1)
    U = np.zeros_like(nominal_controls)  # (action_dim, N-1)

    X[:, 0] = nominal_states[:, 0]
    for i in range(self.N - 1):
      K = K_closed_loop[:, :, i]
      k = k_open_loop[:, i]
      u = (
          nominal_controls[:, i] + alpha*k
          + K @ (X[:, i] - nominal_states[:, i])
      )
      state_nxt, control_clip = self.env.agent.integrate_forward(
          state=X[:, i], control=u, **self.env.integrate_kwargs
      )
      U[:, i] = control_clip
      if i == self.N - 2:
        state_final = state_nxt.copy()
      else:
        X[:, i + 1] = state_nxt.copy()

    J = self.env.get_cost(state=X, action=U, state_nxt=state_final)

    return X, U, state_final, J

  def backward_pass(
      self, nominal_states: np.ndarray, nominal_controls: np.ndarray,
      nominal_state_final: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    self._check_shape(
        dict(nominal_states=nominal_states, nominal_controls=nominal_controls)
    )

    # Gets quadratized cost and linearized system.
    c_x, c_xx, c_u, c_uu, c_ux = self.env.get_derivatives(
        nominal_states, nominal_controls, nominal_state_final
    )
    fx, fu = self.env.agent.get_dyn_jacobian(nominal_states, nominal_controls)

    # Placeholders.
    k_open_loop = np.zeros((self.env.action_dim, self.N - 1))
    K_closed_loop = np.zeros(
        (self.env.action_dim, self.env.observation_dim, self.N - 1)
    )
    Q_u_hist = np.zeros([self.env.action_dim, self.N - 1])
    Q_uu_hist = np.zeros([
        self.env.action_dim, self.env.action_dim, self.N - 1
    ])

    # derivative of value function at final step
    V_x = c_x[:, -1]
    V_xx = c_xx[:, :, -1]
    reg_mat = self.eps * np.eye(self.env.action_dim)  # Numeric stability.

    for i in range(self.N - 2, -1, -1):
      Q_x = c_x[:, i] + fx[:, :, i].T @ V_x
      Q_u = c_u[:, i] + fu[:, :, i].T @ V_x
      Q_xx = c_xx[:, :, i] + fx[:, :, i].T @ V_xx @ fx[:, :, i]
      Q_ux = fu[:, :, i].T @ V_xx @ fx[:, :, i] + c_ux[:, :, i]
      Q_uu = c_uu[:, :, i] + fu[:, :, i].T @ V_xx @ fu[:, :, i]

      Q_uu_inv = np.linalg.inv(Q_uu + reg_mat)
      k_open_loop[:, i] = -Q_uu_inv @ Q_u
      K_closed_loop[:, :, i] = -Q_uu_inv @ Q_ux

      # Update value function derivative for the previous time step
      V_x = Q_x - K_closed_loop[:, :, i].T @ Q_uu @ k_open_loop[:, i]
      V_xx = Q_xx - K_closed_loop[:, :, i].T @ Q_uu @ K_closed_loop[:, :, i]

      Q_u_hist[:, i] = Q_u
      Q_uu_hist[:, :, i] = Q_uu

    return K_closed_loop, k_open_loop

  def _check_shape(self, array_dict: dict):
    for key, value in array_dict.items():
      assert value.shape[-1] == self.N - 1, (
          "The length of {} should be {} but get {}.".format(
              key, self.N - 1, value.shape[-1]
          )
      )
