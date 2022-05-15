"""
Please contact the author(s) of this library if you have any questions.
Authors:  Zixu Zhang ( zixuz@princeton.edu )
          Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from __future__ import annotations
from typing import Any, Tuple, Optional
import numpy as np

from .base_constraints import BaseConstraints
from ..ell_reach.ellipse import Ellipse
from ..utils import barrier_function


class ConstraintsBicycleV2(BaseConstraints):
  """
  A constraint class that computes the constraint function values after every
  step. The obstacles and footprint are assumed to be ellipses (2D).
  """

  def __init__(self, config_agent: Any, config_env: Any):
    super().__init__(config_agent, config_env)
    # State Constraint.
    self.delta_min = config_agent.DELTA_MIN  # min steering
    self.delta_max = config_agent.DELTA_MAX  # max steering

    # Parameter for barrier functions
    self.q1_delta = config_env.Q1_DELTA
    self.q2_delta = config_env.Q2_DELTA

  @property
  def dim_x(self) -> int:
    return 5

  def get_constraint(
      self, footprint: Ellipse, states: np.ndarray, controls: np.ndarray,
      close_pts: np.ndarray, slopes: np.ndarray,
      get_obs_circ_index: Optional[bool] = False
  ) -> Tuple[dict, np.ndarray] | dict:
    """
    Gets the constraint function values given the interested states, the
    closest points on the centerline, the slope of their tangent lines, and the
    interested controls. This augements the parent class with additional
    steering constraints.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (dim_x, N).
        controls (np.ndarray): of the shape (2, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array is of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector is of the shape (1, N).
        get_obs_circ_index (Optional[bool], optional): returns the index of the
            closest circles if True. Defaults to False.

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
        np.ndarray: the index of the closest circles of ego and obstacle. For
            example, x[:, i, j] = (idx_ego_circ, idx_obs_circ). Only returns if
            get_obs_circ_index is True.
    """
    if get_obs_circ_index:
      cons_dict, obs_circ_idx = super().get_constraint(
          footprint, states, controls, close_pts, slopes, get_obs_circ_index
      )
    else:
      cons_dict = super().get_constraint(
          footprint, states, controls, close_pts, slopes, get_obs_circ_index
      )

    # Steering constraint
    cons_delta_min, cons_delta_max = self._delta_cons(states)
    cons_dict["cons_delta_min"] = cons_delta_min
    cons_dict["cons_delta_max"] = cons_delta_max

    if get_obs_circ_index:
      return cons_dict, obs_circ_idx
    return cons_dict

  def get_soft_cons_cost(
      self, footprint: Ellipse, states: np.ndarray, controls: np.ndarray,
      close_pts: Optional[np.ndarray] = None,
      slopes: Optional[np.ndarray] = None, cons_dict: Optional[dict] = None,
      return_cons_dict: Optional[bool] = False
  ) -> np.ndarray:
    """
    Gets the barrier cost of constraint function values given the interested
    states, the closest points on the centerline, the slope of their tangent
    lines, and the interested controls. This augements the parent class with
    additional steering constraints.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (5, N).
        controls (np.ndarray): of the shape (2, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array is of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector is of the shape (1, N).
        cons_dict (Optional[dict], optional): if provided, directly using the
            constraint values within. Defaults to None.

    Returns:
        np.ndarray: the soft constraint cost at each time step, of the shape
            (1, N).
    """
    soft_cons_cost, cons_dict = super().get_soft_cons_cost(
        footprint, states, controls, close_pts, slopes, cons_dict, True
    )

    c_delta_min, c_delta_max = self._delta_cost(
        cons_dict['cons_delta_min'], cons_dict['cons_delta_max']
    )
    if return_cons_dict:
      return soft_cons_cost + c_delta_min + c_delta_max, cons_dict
    return soft_cons_cost + c_delta_min + c_delta_max

  def get_derivatives(
      self, footprint: Ellipse, states: np.ndarray, controls: np.ndarray,
      close_pts: np.ndarray, slopes: np.ndarray,
      return_cons_dict: Optional[bool] = False
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of soft constraint cost given the
    interested states, the closest points on the centerline, the slope of their
    tangent lines, and the interested controls. This augements the parent class
    with additional steering constraints.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (dim_x, N).
        controls (np.ndarray): of the shape (2, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array is of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector is of the shape (1, N).

    Returns:
        np.ndarray: c_x of the shape (dim_x, N).
        np.ndarray: c_xx of the shape (dim_x, dim_x, N).
        np.ndarray: c_u of the shape (2, N).
        np.ndarray: c_uu of the shape (2, 2, N).
        np.ndarray: c_ux of the shape (2, dim_x, N).
        dict: constraint dictionary.
    """
    c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons, cons_dict = super(
    ).get_derivatives(footprint, states, controls, close_pts, slopes, True)

    c_x_delta, c_xx_delta = self._delta_bound_derivative(
        cons_dict['cons_delta_min'], cons_dict['cons_delta_max']
    )
    c_x_cons += c_x_delta
    c_xx_cons += c_xx_delta

    if return_cons_dict:
      return c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons, cons_dict
    return c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons

  def _lat_accel_cons(self, states: np.ndarray,
                      controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the lateral acceleration constraint.

    Args:
        states (np.ndarray): of the shape (5, N).
        controls (np.ndarray): of the shape (2, N). Not used in bicycle V2.

    Returns:
        np.ndarray: constarint fucntion value of the lateral accel minimum.
        np.ndarray: constarint fucntion value of the lateral accel maximum.
    """
    accel = states[2:3, :]**2 * np.tan(states[4:5, :]) / self.wheelbase
    cons_a_lat_max = accel - self.alat_max
    cons_a_lat_min = self.alat_min - accel
    return cons_a_lat_min, cons_a_lat_max

  def _delta_cons(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the road boundary constraint.

    Args:
        states (np.ndarray): of the shape (5, N).

    Returns:
        np.ndarray: constarint fucntion value of the steering minimum.
        np.ndarray: constarint fucntion value of the steering maximum.
    """
    cons_delta_max = states[4:5, :] - self.delta_max
    cons_delta_min = self.delta_min - states[4:5, :]
    return cons_delta_min, cons_delta_max

  def _delta_cost(
      self, cons_delta_min: np.ndarray, cons_delta_max: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the constraint values of the steering limit to the corresponding
    barrier costs.

    Args:
        cons_delta_min (np.ndarray): constarint value of the minimum steering.
        cons_delta_max (np.ndarray): constarint value of the maximum steering.

    Returns:
        np.ndarray: barrier cost of the minimum steering.
        np.ndarray: barrier cost of the maximum steering.
    """
    barrier_delta_min = self.q1_delta * (
        np.exp(
            np.clip(self.q2_delta * cons_delta_min, None, self.barrier_thr)
        )
    )
    barrier_delta_max = self.q1_delta * (
        np.exp(
            np.clip(self.q2_delta * cons_delta_max, None, self.barrier_thr)
        )
    )
    return barrier_delta_min, barrier_delta_max

  def _lat_accel_bound_derivative(
      self, states: np.ndarray, controls: np.ndarray,
      cons_a_lat_min: np.ndarray, cons_a_lat_max: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of Lateral Acceleration soft constraint
        cost.

    Args:
        states (np.ndarray): of the shape (5, N).
        controls (np.ndarray): of the shape (2, N). Not used in bicycle V2.
        cons_a_lat_min (np.ndarray): constarint value of the minimum lateral
            acceleration.
        cons_a_lat_max (np.ndarray): constarint value of the maximum lateral
            acceleration.

    Returns:
        np.ndarray: c_x of the shape (5, N).
        np.ndarray: c_xx of the shape (5, 5, N).
        np.ndarray: c_u of the shape (2, N).
        np.ndarray: c_uu of the shape (2, 2, N).
        np.ndarray: c_ux of the shape (2, 5, N).
    """
    num_steps = states.shape[1]
    c_x = np.zeros((self.dim_x, num_steps))
    c_xx = np.zeros((self.dim_x, self.dim_x, num_steps))
    c_u = np.zeros((2, num_steps))
    c_uu = np.zeros((2, 2, num_steps))
    c_ux = np.zeros((2, self.dim_x, num_steps))

    zeros = np.zeros((num_steps))
    da_dv = 2 * states[2, :] * np.tan(states[4, :]) / self.wheelbase
    da_ddelta = states[2, :]**2 / (self.wheelbase * np.cos(states[4, :])**2)
    transform = np.array([zeros, zeros, da_dv, zeros, da_ddelta])

    _c_x_min, _c_xx_min = barrier_function(
        q1=self.q1_lat, q2=self.q2_lat, cons=cons_a_lat_min,
        cons_dot=-transform, cons_max=self.barrier_thr
    )
    _c_x_max, _c_xx_max = barrier_function(
        q1=self.q1_lat, q2=self.q2_lat, cons=cons_a_lat_max,
        cons_dot=transform, cons_max=self.barrier_thr
    )
    c_x = _c_x_min + _c_x_max
    c_xx = _c_xx_min + _c_xx_max
    return c_x, c_xx, c_u, c_uu, c_ux

  def _delta_bound_derivative(
      self, cons_delta_min: np.ndarray, cons_delta_max: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the Jacobian and Hessian of steering soft constraint cost.

    Args:
        cons_delta_min (np.ndarray): constarint value of the minimum steering.
        cons_delta_max (np.ndarray): constarint value of the maximum steering.

    Returns:
        np.ndarray: c_x of the shape (5, N).
        np.ndarray: c_xx of the shape (5, 5, N).
    """
    N = cons_delta_min.shape[1]
    transform = np.ones((1, N))
    c_x = np.zeros((self.dim_x, N))
    c_xx = np.zeros((self.dim_x, self.dim_x, N))

    _c_x_min, _c_xx_min = barrier_function(
        self.q1_delta, self.q2_delta, cons_delta_min, -transform,
        cons_max=self.barrier_thr
    )
    _c_x_max, _c_xx_max = barrier_function(
        self.q1_delta, self.q2_delta, cons_delta_max, transform,
        cons_max=self.barrier_thr
    )
    c_x[4, :] = _c_x_min + _c_x_max
    c_xx[4, 4, :] = _c_xx_min + _c_xx_max

    return c_x, c_xx
