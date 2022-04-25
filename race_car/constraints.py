"""
Please contact the author(s) of this library if you have any questions.
Authors:  Zixu Zhang ( zixuz@princeton.edu )
          Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from __future__ import annotations
import time
from typing import List, Any, Optional, Tuple
import numpy as np

from ..ell_reach.ellipse import Ellipse
from ..utils import barrier_function

# TODO: We currently only support Ellipse obstacles.


class Constraints:
  """
  A constraint class that computes the constraint function values after every
  step. The obstacles and footprint are assumed to be ellipses (2D).
  """

  def __init__(self, config_agent: Any, config_env: Any):
    self.wheelbase = config_agent.WHEELBASE  # vehicle chassis length

    # State Constraint.
    self.v_min = config_agent.V_MIN  # min velocity
    self.v_max = config_agent.V_MAX  # max velocity
    self.track_width_L = config_env.TRACK_WIDTH_LEFT
    self.track_width_R = config_env.TRACK_WIDTH_RIGHT

    # Dynamics Constraint
    self.alat_max = config_agent.ALAT_MAX  # max lateral accel
    self.alat_min = -config_agent.ALAT_MAX  # min lateral accel

    # Obstacles
    self.buffer = getattr(config_env, "BUFFER", 0.)
    self.obs_list = None

    # Parameter for barrier functions
    self.barrier_thr = 20
    self.q1_v = config_env.Q1_V
    self.q2_v = config_env.Q2_V
    self.q1_road = config_env.Q1_ROAD
    self.q2_road = config_env.Q2_ROAD
    self.road_thr = -0.025

    self.q1_lat = config_env.Q1_LAT
    self.q2_lat = config_env.Q2_LAT

    self.q1_obs = config_env.Q1_OBS
    self.q2_obs = config_env.Q2_OBS
    self.gamma = getattr(config_env, "OBS_COST_GAMMA", 1.)

  def update_obs(self, obs_list: List[List[Ellipse]]):
    """Updates the obstacles.

    Args:
        obs_list (List[List[Ellipse]]): a list of ellipse lists. Each Ellipse
        in the list is an obstacle at each time step.
    """
    dim = np.array([len(x) for x in obs_list])
    assert np.all(dim == dim[0]), ("The length of each list does not match!")
    self.obs_list = obs_list

  def get_constraint(
      self, footprint: Ellipse, states: np.ndarray, controls: np.ndarray,
      close_pts: np.ndarray, slopes: np.ndarray,
      get_obs_circ_index: Optional[bool] = False
  ) -> Tuple[dict, np.ndarray] | dict:
    """
    Gets the constraint function values given the interested states, the
    closest points on the centerline, the slope of their tangent lines, and the
    interested controls.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (4, N).
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
    self._check_input(states, controls, close_pts, slopes)

    num_steps = states.shape[1]

    # Road boundary constarint.
    cons_road_l, cons_road_r = self._road_boundary_cons(
        footprint, states, close_pts, slopes
    )

    # Velocity constraint.
    cons_v_min = self.v_min - states[2:3, :]
    cons_v_max = states[2:3, :] - self.v_max

    # Lateral acceleration constraint
    accel = states[2:3, :]**2 * np.tan(controls[1, :]) / self.wheelbase
    cons_a_lat_max = accel - self.alat_max
    cons_a_lat_min = self.alat_min - accel

    cons_dict = dict(
        cons_road_l=cons_road_l, cons_road_r=cons_road_r,
        cons_v_min=cons_v_min, cons_v_max=cons_v_max,
        cons_a_lat_max=cons_a_lat_max, cons_a_lat_min=cons_a_lat_min
    )

    # Obstacle constraint
    if self.obs_list is not None:
      footprint_traj = footprint.move2state(states[[0, 1, 3], :])
      assert (len(
          self.obs_list[0]
      ) == num_steps), ("The length of obstacles and states do not match!")

      cons_obs = np.empty((len(self.obs_list), num_steps))
      if get_obs_circ_index:
        obs_circ_idx = np.empty((2, len(self.obs_list), num_steps), dtype=int)
      for i, footprint_i in enumerate(footprint_traj):
        # obs_list_j is a list of obstacles.
        for j, obs_list_j in enumerate(self.obs_list):
          obs_j_i = obs_list_j[i]  # Gets the ith obstacle in obs_list_j.
          if get_obs_circ_index:
            min_dist, indices = footprint_i.dist2ellipse(
                obs_j_i, get_index=True
            )
            obs_circ_idx[:, j, i] = indices
          else:
            min_dist = footprint_i.dist2ellipse(obs_j_i)
          cons_obs[j, i] = footprint_i.b + obs_j_i.b + self.buffer - min_dist

      cons_dict["cons_obs"] = cons_obs

      if get_obs_circ_index:
        return cons_dict, obs_circ_idx
    # Returns anyway.
    return cons_dict

  def get_soft_cons_cost(
      self, footprint: Ellipse, states: np.ndarray, controls: np.ndarray,
      close_pts: Optional[np.ndarray] = None,
      slopes: Optional[np.ndarray] = None, cons_dict: Optional[dict] = None
  ) -> np.ndarray:
    """
    Gets the barrier cost of constraint function values given the interested
    states, the closest points on the centerline, the slope of their tangent
    lines, and the interested controls.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (4, N).
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

    if cons_dict is None:
      self._check_input(states, controls, close_pts, slopes)
      cons_dict = self.get_constraint(
          footprint, states, controls, close_pts, slopes
      )

    c_road_l, c_road_r = self._road_boundary_cost(
        cons_dict['cons_road_l'], cons_dict['cons_road_r']
    )
    c_road = c_road_l + c_road_r

    c_vel_min, c_vel_max = self._velocity_cost(
        cons_dict['cons_v_min'], cons_dict['cons_v_max']
    )
    c_vel = c_vel_min + c_vel_max

    c_lat_min, c_lat_max = self._lat_accec_cost(
        cons_dict['cons_a_lat_min'], cons_dict['cons_a_lat_max']
    )
    c_lat = c_lat_min + c_lat_max

    c_obs = 0.
    if self.obs_list is not None:
      barrier = self._obs_cost(cons_dict["cons_obs"])
      discount = self.gamma**(np.arange(states.shape[1]))
      c_obs = np.sum(barrier, axis=0)
      c_obs = c_obs * discount

    return c_road + c_vel + c_lat + c_obs

  def get_derivatives(
      self, footprint: Ellipse, states: np.ndarray, controls: np.ndarray,
      close_pts: np.ndarray, slopes: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of soft constraint cost given the
    interested states, the closest points on the centerline, the slope of their
    tangent lines, and the interested controls.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (4, N).
        controls (np.ndarray): of the shape (2, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array is of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector is of the shape (1, N).

    Returns:
        np.ndarray: c_x of the shape (4, N).
        np.ndarray: c_xx of the shape (4, 4, N).
        np.ndarray: c_u of the shape (2, N).
        np.ndarray: c_uu of the shape (2, 2, N).
        np.ndarray: c_ux of the shape (2, 4, N).
    """
    self._check_input(states, controls, close_pts, slopes)

    cons_dict, obs_circ_idx = self.get_constraint(
        footprint, states, controls, close_pts, slopes, get_obs_circ_index=True
    )

    # Road bound constraints.
    c_x_rd, c_xx_rd = self._road_boundary_derivative(
        states, slopes, cons_dict['cons_road_l'], cons_dict['cons_road_r']
    )

    # Velocity constraints.
    c_x_vel, c_xx_vel = self._velocity_bound_derivative(
        states, cons_dict['cons_v_min'], cons_dict['cons_v_max']
    )

    # Lateral acceleration constraints.
    c_x_lat, c_xx_lat, c_u_lat, c_uu_lat, c_ux_lat = (
        self._lat_accec_bound_derivative(
            states, controls, cons_dict['cons_a_lat_min'],
            cons_dict['cons_a_lat_max']
        )
    )

    # Obstacle constraints.
    c_x_obs, c_xx_obs = 0., 0.
    if self.obs_list is not None:
      c_x_obs, c_xx_obs = self._obs_derivative(
          footprint, states, cons_dict['cons_obs'], obs_circ_idx
      )

    # sum up
    c_x_cons = c_x_rd + c_x_lat + c_x_obs + c_x_vel
    c_xx_cons = c_xx_rd + c_xx_lat + c_xx_obs + c_xx_vel

    c_u_cons = c_u_lat
    c_uu_cons = c_uu_lat
    c_ux_cons = c_ux_lat

    return c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons

  def _road_boundary_cons(
      self, footprint: Ellipse, states: np.ndarray, close_pts: np.ndarray,
      slopes: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the road boundary constraint.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (4, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array is of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector is of the shape (1, N).

    Returns:
        np.ndarray: constarint fucntion value of the left road boundary.
        np.ndarray: constarint fucntion value of the road road boundary.
    """
    dx = states[0, :] - close_pts[0, :]
    dy = states[1, :] - close_pts[1, :]

    sr = np.sin(slopes)
    cr = np.cos(slopes)
    dis = sr*dx - cr*dy  # signed distance

    cons_road_l = -dis - (self.track_width_L - footprint.b)  # Left Bound
    cons_road_r = dis - (self.track_width_R - footprint.b)  # right bound

    return cons_road_l, cons_road_r

  def _road_boundary_cost(
      self, cons_road_l: np.ndarray, cons_road_r: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the constraint values of the road boundary to the corresponding
    barrier costs.

    Args:
        cons_road_l (np.ndarray): constarint value of the left road boundary.
        cons_road_r (np.ndarray): constarint value of the right road boundary.

    Returns:
        np.ndarray: barrier cost of the left road boundary.
        np.ndarray: barrier cost of the right road boundary.
    """
    barrier_l = self.q1_road * np.exp(
        np.clip(
            self.q2_road * cons_road_l, self.road_thr * self.q2_road,
            self.barrier_thr
        )
    )
    barrier_r = self.q1_road * np.exp(
        np.clip(
            self.q2_road * cons_road_r, self.road_thr * self.q2_road,
            self.barrier_thr
        )
    )
    return barrier_l, barrier_r

  def _velocity_cost(self, cons_v_min: np.ndarray,
                     cons_v_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the constraint values of the velocity limit to the corresponding
    barrier costs.

    Args:
        cons_v_min (np.ndarray): constarint value of the minimum velocity.
        cons_v_max (np.ndarray): constarint value of the maximum velocity.

    Returns:
        np.ndarray: barrier cost of the minimum velocity.
        np.ndarray: barrier cost of the maximum velocity.
    """
    barrier_v_min = self.q1_v * np.exp(
        np.clip(self.q2_v * cons_v_min, None, self.barrier_thr)
    )
    barrier_v_max = self.q1_v * np.exp(
        np.clip(self.q2_v * cons_v_max, None, self.barrier_thr)
    )
    return barrier_v_min, barrier_v_max

  def _lat_accec_cost(
      self, cons_a_lat_min: np.ndarray, cons_a_lat_max: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the constraint values of the lateral acceleration limit to the
    corresponding barrier costs.

    Args:
        cons_a_lat_min (np.ndarray): constarint value of the minimum lateral
            acceleration.
        cons_a_lat_max (np.ndarray): constarint value of the maximum lateral
            acceleration.

    Returns:
        np.ndarray: barrier cost of the minimum lateral acceleration.
        np.ndarray: barrier cost of the maximum lateral acceleration.
    """
    barrier_a_lat_min = self.q1_lat * (
        np.exp(np.clip(self.q2_lat * cons_a_lat_min, None, self.barrier_thr))
    )
    barrier_a_lat_max = self.q1_lat * (
        np.exp(np.clip(self.q2_lat * cons_a_lat_max, None, self.barrier_thr))
    )
    return barrier_a_lat_min, barrier_a_lat_max

  def _obs_cost(self, cons_obs: np.ndarray) -> np.ndarray:
    """
    Transforms the constraint values of obstacles to the corresponding
    barrier costs.

    Args:
        cons_obs (np.ndarray): of the shape (#obstacles, N).

    Returns:
        np.ndarray: barrier cost of the obstacle constraints, of the shape
            (#obstacles, N).
    """
    # Ignores and sets to -0.2 when self is far away from the obstacle.
    cons_clip = np.clip(
        self.q2_obs * cons_obs, -0.2 * self.q2_obs, self.barrier_thr
    )
    barrier = self.q1_obs * np.exp(cons_clip)
    return barrier

  def _road_boundary_derivative(
      self, states: np.ndarray, slopes: np.ndarray, cons_road_l: np.ndarray,
      cons_road_r: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of road boundary soft constraint cost.

    Args:
        states (np.ndarray): of the shape (4, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array is of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector is of the shape (1, N).
        cons_road_l (np.ndarray): constarint value of the left road boundary.
        cons_road_r (np.ndarray): constarint value of the right road boundary.

    Returns:
        np.ndarray: c_x of the shape (4, N).
        np.ndarray: c_xx of the shape (4, 4, N).
    """
    cons_road_l = cons_road_l.reshape(-1)
    cons_road_r = cons_road_r.reshape(-1)

    N = states.shape[-1]
    sr = np.sin(slopes).reshape(-1)
    cr = np.cos(slopes).reshape(-1)
    transform = np.array([sr, -cr])  # (2, N).

    # region: right bound
    idx_ignore = cons_road_r < self.road_thr

    # Jacobian
    c_x_r = np.zeros((4, N))
    c_xx_r = np.zeros((4, 4, N))
    _c_x_r, _c_xx_r = barrier_function(
        q1=self.q1_road, q2=self.q2_road, cons=cons_road_r, cons_dot=transform,
        cons_min=self.road_thr * self.q2_road, cons_max=self.barrier_thr
    )
    c_x_r[:2, :] = _c_x_r
    c_xx_r[:2, :2, :] = _c_xx_r

    # remove inactive
    c_x_r[:, idx_ignore] = 0
    c_xx_r[:, :, idx_ignore] = 0
    # endregion

    # region: Left Bound
    idx_ignore = cons_road_l < self.road_thr

    # Jacobian
    c_x_l = np.zeros((4, N))
    c_xx_l = np.zeros((4, 4, N))
    _c_x_l, _c_xx_l = barrier_function(
        q1=self.q1_road, q2=self.q2_road, cons=cons_road_l,
        cons_dot=-transform, cons_min=self.road_thr * self.q2_road,
        cons_max=self.barrier_thr
    )
    c_x_l[:2, :] = _c_x_l
    c_xx_l[:2, :2, :] = _c_xx_l

    # remove inactive
    c_x_l[:, idx_ignore] = 0
    c_xx_l[:, :, idx_ignore] = 0

    # endregion

    return c_x_r + c_x_l, c_xx_r + c_xx_l

  def _velocity_bound_derivative(
      self, states: np.ndarray, cons_v_min: np.ndarray, cons_v_max: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of velocity soft constraint cost.

    Args:
        states (np.ndarray): of the shape (4, N).
        cons_v_min (np.ndarray): constarint value of the minimum velocity.
        cons_v_max (np.ndarray): constarint value of the maximum velocity.

    Returns:
        np.ndarray: c_x of the shape (4, N).
        np.ndarray: c_xx of the shape (4, 4, N).
    """
    N = states.shape[1]
    transform = np.ones((1, N))
    c_x = np.zeros((4, N))
    c_xx = np.zeros((4, 4, N))

    _c_x_min, _c_xx_min = barrier_function(
        self.q1_v, self.q2_v, cons_v_min, -transform, cons_max=self.barrier_thr
    )
    _c_x_max, _c_xx_max = barrier_function(
        self.q1_v, self.q2_v, cons_v_max, transform, cons_max=self.barrier_thr
    )
    c_x[2, :] = _c_x_min + _c_x_max
    c_xx[2, 2, :] = _c_xx_min + _c_xx_max

    return c_x, c_xx

  def _lat_accec_bound_derivative(
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
    c_x = np.zeros((4, num_steps))
    c_xx = np.zeros((4, 4, num_steps))
    c_u = np.zeros((2, num_steps))
    c_uu = np.zeros((2, 2, num_steps))
    c_ux = np.zeros((2, 4, num_steps))

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

  def _obs_derivative(
      self, footprint: Ellipse, states: np.ndarray, cons_obs: np.ndarray,
      obs_circ_idx: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of obstacles soft constraint cost.

    Args:
        footprint (Ellipse): the footprint of the ego agent.
        states (np.ndarray): of the shape (4, N).
        cons_obs (np.ndarray): of the shape (#obstacles, N).
        obs_circ_idx (np.ndarray): the index of the closest circles of ego and
            obstacle. For example, x[:, i, j] = (idx_ego_circ, idx_obs_circ).

    Returns:
        np.ndarray: c_x of the shape (4, N).
        np.ndarray: c_xx of the shape (4, 4, N).
    """
    centers_with_wheelbase = footprint.center_local + self.wheelbase
    num_steps = states.shape[1]
    num_obs = len(self.obs_list)

    c_x = np.zeros(shape=(4, num_steps))
    c_xx = np.zeros(shape=(4, 4, num_steps))

    cons_flatten = cons_obs.reshape(-1, order='F')  # column-by-column
    cons_dot_concat = np.zeros(shape=(4, num_steps * num_obs))

    for i in range(num_steps):
      state = states[:, i]
      cos_theta = np.cos(state[3])
      sin_theta = np.sin(state[3])

      for j, obs_list_j in enumerate(self.obs_list):
        if cons_obs[j, i] < -0.2:  # ignore obstacle too far away.
          continue
        obs_j_i = obs_list_j[i]

        self_circ_idx, obs_j_circ_idx = obs_circ_idx[:, j, i]
        pos_along_major_axis = centers_with_wheelbase[self_circ_idx]
        self_center = (
            pos_along_major_axis * np.array([cos_theta, sin_theta]) + state[:2]
        )
        obs_center = obs_j_i.center[:, obs_j_circ_idx]
        diff = self_center - obs_center
        dist = np.linalg.norm(diff)
        cons_dot_concat[:2, i*num_obs + j] = -diff / dist
        cons_dot_concat[3, i*num_obs + j] = pos_along_major_axis / dist * (
            diff[0] * sin_theta - diff[1] * cos_theta
        )

    _c_x, _c_xx = barrier_function(
        q1=self.q1_obs, q2=self.q2_obs, cons=cons_flatten,
        cons_dot=cons_dot_concat, cons_min=-0.2 * self.q2_obs,
        cons_max=self.barrier_thr
    )
    for i in range(num_steps):
      start = i * num_obs
      end = (i+1) * num_obs
      c_x[:, i] = np.sum(_c_x[:, start:end], axis=-1)
      c_xx[:, :, i] = np.sum(_c_xx[:, :, start:end], axis=-1)

    return c_x, c_xx

  def _check_input(
      self, states: np.ndarray, controls: np.ndarray, close_pts: np.ndarray,
      slopes: np.ndarray
  ):
    """Checks the shape of the input arrays.

    Args:
        states (np.ndarray): of the shape (4, N).
        controls (np.ndarray): of the shape (2, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array is of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector is of the shape (1, N).
    """
    assert states.ndim == 2, "states dimension is not correct!"
    assert controls.ndim == 2, "controls dimension is not correct!"
    assert close_pts.ndim == 2, "close_pts dimension is not correct!"
    assert slopes.ndim == 2, "slopes dimension is not correct!"
