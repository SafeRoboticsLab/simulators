"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Tuple, Any
import numpy as np
from gym import spaces

from .constraints_bicycle_v2 import ConstraintsBicycleV2
from .base_race_car_single import BaseRaceCarSingleEnv


class RaceCarSingleEnvV2(BaseRaceCarSingleEnv):

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    super().__init__(config_env, config_agent)
    self.constraints = ConstraintsBicycleV2(
        config_env=config_env, config_agent=config_agent
    )

    # Ctrl cost.
    self.w_accel = config_env.W_ACCEL
    self.w_omega = config_env.W_OMEGA
    self.W_control[0, 0] = self.w_accel
    self.W_control[1, 1] = self.w_omega

    self.build_obs_rst_space(config_env, config_agent)
    self.seed(config_env.SEED)
    self.reset()

  def build_obs_rst_space(self, config_env: Any, config_agent: Any):
    # Reset Sample Space. Note that the first two dimension is in the local
    # frame and it needs to call track.local2global() to get the (x, y)
    # position in the global frame.
    low = np.zeros((self.state_dim,))
    low[1] = -config_env.TRACK_WIDTH_LEFT + config_agent.WIDTH * 3 / 4
    low[2] = config_agent.V_MIN / 0.8
    low[3] = -np.pi * 20 / 180
    low[4] = config_agent.DELTA_MIN * 0.8
    high = np.zeros((self.state_dim,))
    high[0] = 1.
    high[1] = config_env.TRACK_WIDTH_RIGHT - config_agent.WIDTH * 3 / 4
    high[2] = config_agent.V_MAX * 0.8
    high[3] = np.pi * 20 / 180
    high[4] = config_agent.DELTA_MAX * 0.8
    self.reset_sample_sapce = spaces.Box(
        low=np.float32(low), high=np.float32(high)
    )

    # Observation space.
    self.obs_type = getattr(config_env, "OBS_TYPE", "perfect")
    x_min, y_min = np.min(self.track.track_bound[2:, :], axis=1)
    x_max, y_max = np.max(self.track.track_bound[2:, :], axis=1)
    if self.obs_type == "perfect":
      low = np.zeros((self.state_dim,))
      low[0] = x_min
      low[1] = y_min
      low[2] = config_agent.V_MIN
      low[4] = config_agent.DELTA_MIN
      high = np.zeros((self.state_dim,))
      high[0] = x_max
      high[1] = y_max
      high[2] = config_agent.V_MAX
      high[3] = 2 * np.pi
      high[4] = config_agent.DELTA_MAX
    elif self.obs_type == "cos_sin":
      low = np.zeros((self.state_dim + 1,))
      low[0] = x_min
      low[1] = y_min
      low[2] = config_agent.V_MIN
      low[3] = -1.
      low[4] = -1.
      low[5] = config_agent.DELTA_MIN
      high = np.zeros((self.state_dim + 1,))
      high[0] = x_max
      high[1] = y_max
      high[2] = config_agent.V_MAX
      high[3] = 1.
      high[4] = 1.
      high[5] = config_agent.DELTA_MAX
    else:
      raise ValueError("Observation type {} is not supported!")
    self.observation_space = spaces.Box(
        low=np.float32(low), high=np.float32(high)
    )
    self.obs_dim = self.observation_space.low.shape[0]

  def _get_cost_state_derivative(
      self, states: np.ndarray, close_pts: np.ndarray, slopes: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Jacobian and Hessian of the cost function with respect to state.

    Args:
        states (np.ndarray): planned trajectory, (5, N).
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array should be of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector should be of the shape (1, N).

    Returns:
        np.ndarray: Jacobian.
        np.ndarray: Hessian.
    """
    ref_states, transform = self._get_ref_path_transform(close_pts, slopes)
    num_pts = close_pts.shape[1]
    zeros = np.zeros((num_pts))
    sr = np.sin(slopes).reshape(-1)
    cr = np.cos(slopes).reshape(-1)

    error = states - ref_states
    Q_trans = np.einsum(
        'abn, bcn->acn',
        np.einsum(
            'dan, ab -> dbn', transform.transpose(1, 0, 2), self.W_state
        ), transform
    )

    c_x = 2 * np.einsum('abn, bn->an', Q_trans, error)

    c_x_progress = -self.w_theta * np.array([cr, sr, zeros, zeros, zeros])
    c_x = c_x + c_x_progress
    c_xx = 2 * Q_trans

    return c_x, c_xx

  def _get_ref_path_transform(
      self, close_pts: np.ndarray, slopes: np.ndarray
  ) -> np.ndarray:
    """
    Gets the reference path and the transformation form the global frame to the
    local frame with the origin at the closest points.

    Args:
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array should be of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector should be of the shape (1, N).

    Returns:
        np.ndarray: reference path (x position, y position, and velocity)
        np.ndarray: transformation matrix from state error to contour and
            velocity error
    """
    num_pts = close_pts.shape[1]
    slopes = slopes.reshape(-1)
    zeros = np.zeros((num_pts))
    ones = np.ones((num_pts))
    sr = np.sin(slopes)
    cr = np.cos(slopes)
    transform = np.array([[sr, -cr, zeros, zeros, zeros],
                          [zeros, zeros, ones, zeros, zeros]])

    ref_states = np.zeros((self.agent.dyn.dim_x, num_pts))
    ref_states[0, :] = close_pts[0, :] + sr * self.track_offset
    ref_states[1, :] = close_pts[1, :] - cr * self.track_offset
    ref_states[2, :] = self.v_ref

    return ref_states, transform

  def report(self):
    print(
        "This is a Race Car simulator based on bicycle dynamics version 2 "
        + "and ellipse footprint."
    )
    super().report()
