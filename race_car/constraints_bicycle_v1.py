"""
Please contact the author(s) of this library if you have any questions.
Authors:  Zixu Zhang ( zixuz@princeton.edu )
          Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from __future__ import annotations
from typing import Any, Tuple
import numpy as np

from .base_constraints import BaseConstraints


class ConstraintsBicycleV1(BaseConstraints):
  """
  A constraint class that computes the constraint function values after every
  step. The obstacles and footprint are assumed to be ellipses (2D).
  """

  def __init__(self, config_agent: Any, config_env: Any):
    super().__init__(config_agent, config_env)

  @property
  def dim_x(self) -> int:
    return 4

  def _lat_accel_cons(self, states: np.ndarray,
                      controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the lateral acceleration constraint.

    Args:
        states (np.ndarray): of the shape (4, N).
        controls (np.ndarray): of the shape (2, N).

    Returns:
        np.ndarray: constarint fucntion value of the lateral accel minimum.
        np.ndarray: constarint fucntion value of the lateral accel maximum.
    """
    accel = states[2:3, :]**2 * np.tan(controls[1, :]) / self.wheelbase
    cons_a_lat_max = accel - self.alat_max
    cons_a_lat_min = self.alat_min - accel
    return cons_a_lat_min, cons_a_lat_max

  def _lat_accel_bound_derivative(
      self, states: np.ndarray, controls: np.ndarray,
      cons_a_lat_min: np.ndarray, cons_a_lat_max: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of Lateral Acceleration soft constraint
        cost.

    Args:
        states (np.ndarray): of the shape (4, N).
        controls (np.ndarray): of the shape (2, N).
        cons_a_lat_min (np.ndarray): constarint value of the minimum lateral
            acceleration.
        cons_a_lat_max (np.ndarray): constarint value of the maximum lateral
            acceleration.

    Returns:
        np.ndarray: c_x of the shape (4, N).
        np.ndarray: c_xx of the shape (4, 4, N).
        np.ndarray: c_u of the shape (2, N).
        np.ndarray: c_uu of the shape (2, 2, N).
        np.ndarray: c_ux of the shape (2, 4, N).
    """
    cost_a_lat_min = self.q1_lat * (
        np.exp(np.clip(self.q2_lat * cons_a_lat_min, None, self.barrier_thr))
    )
    cost_a_lat_max = self.q1_lat * (
        np.exp(np.clip(self.q2_lat * cons_a_lat_max, None, self.barrier_thr))
    )
    num_steps = states.shape[1]
    c_x = np.zeros((self.dim_x, num_steps))
    c_xx = np.zeros((self.dim_x, self.dim_x, num_steps))
    c_u = np.zeros((2, num_steps))
    c_uu = np.zeros((2, 2, num_steps))
    c_ux = np.zeros((2, self.dim_x, num_steps))

    da_dx = 2 * states[2, :] * np.tan(controls[1, :]) / self.wheelbase
    da_dxx = 2 * np.tan(controls[1, :]) / self.wheelbase

    da_du = states[2, :]**2 / (np.cos(controls[1, :])**2 * self.wheelbase)
    da_duu = (
        states[2, :]**2 * np.sin(controls[1, :]) /
        (np.cos(controls[1, :])**3 * self.wheelbase)
    )

    da_dux = 2 * states[2, :] / (np.cos(controls[1, :])**2 * self.wheelbase)

    c_x[2, :] = self.q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_dx
    c_u[1, :] = self.q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_du

    c_xx[2, 2, :] = self.q2_lat**2 * (
        cost_a_lat_max+cost_a_lat_min
    ) * da_dx**2 + self.q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_dxx
    c_uu[1, 1, :] = self.q2_lat**2 * (
        cost_a_lat_max+cost_a_lat_min
    ) * da_du**2 + self.q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_duu

    c_ux[1, 2, :] = (
        self.q2_lat**2 *
        (cost_a_lat_max+cost_a_lat_min) * da_dx * da_du + self.q2_lat *
        (cost_a_lat_max-cost_a_lat_min) * da_dux
    )
    return c_x, c_xx, c_u, c_uu, c_ux
