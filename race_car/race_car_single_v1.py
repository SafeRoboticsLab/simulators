"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Tuple, Any, Optional, Union
import numpy as np
from gym import spaces
import torch

from .constraints_bicycle_v1 import ConstraintsBicycleV1
from .base_race_car_single import BaseRaceCarSingleEnv


class RaceCarSingleEnvV1(BaseRaceCarSingleEnv):

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    super().__init__(config_env, config_agent)
    self.constraints = ConstraintsBicycleV1(
        config_env=config_env, config_agent=config_agent
    )

    # Ctrl cost.
    self.w_accel = config_env.W_ACCEL
    self.w_delta = config_env.W_DELTA
    self.W_control[0, 0] = self.w_accel
    self.W_control[1, 1] = self.w_delta

  def build_obs_rst_space(self, config_env: Any, config_agent: Any):
    # Reset Sample Space. Note that the first two dimension is in the local
    # frame and it needs to call track.local2global() to get the (x, y)
    # position in the global frame.
    low = np.zeros((4,))
    low[1] = -config_env.TRACK_WIDTH_LEFT + config_agent.WIDTH * 3 / 4
    low[3] = -np.pi / 4
    high = np.zeros((4,))
    high[0] = 1.
    high[1] = config_env.TRACK_WIDTH_RIGHT - config_agent.WIDTH * 3 / 4
    high[2] = config_agent.V_MAX
    high[3] = np.pi / 4
    self.reset_sample_sapce = spaces.Box(
        low=np.float32(low), high=np.float32(high)
    )

    # Observation space.
    self.obs_type = getattr(config_env, "OBS_TYPE", "perfect")
    x_min, y_min = np.min(self.track.track_bound[2:, :], axis=1)
    x_max, y_max = np.max(self.track.track_bound[2:, :], axis=1)
    if self.obs_type == "perfect":
      low = np.zeros((4,))
      low[0] = x_min
      low[1] = y_min
      high = np.zeros((4,))
      high[0] = x_max
      high[1] = y_max
      high[2] = config_agent.V_MAX
      high[3] = 2 * np.pi
    elif self.obs_type == "cos_sin":
      low = np.zeros((5,))
      low[0] = x_min
      low[1] = y_min
      low[3] = -1.
      low[4] = -1.
      high = np.zeros((5,))
      high[0] = x_max
      high[1] = y_max
      high[2] = config_agent.V_MAX
      high[3] = 1.
      high[4] = 1.
    else:
      raise ValueError("Observation type {} is not supported!")
    self.observation_space = spaces.Box(
        low=np.float32(low), high=np.float32(high)
    )
    self.obs_dim = self.observation_space.low.shape[0]

  def reset(
      self, state: Optional[np.ndarray] = None, cast_torch: bool = False,
      **kwargs
  ) -> Union[np.ndarray, torch.FloatTensor]:
    """
    Resets the environment and returns the new state.

    Args:
        state (Optional[np.ndarray], optional): reset to this state if
            provided. Defaults to None.
        cast_torch (bool): cast state to torch if True.

    Returns:
        np.ndarray: the new state of the shape (4, ).
    """
    super().reset()
    if state is None:
      state = self.reset_sample_sapce.sample()
      state[:2], slope = self.track.local2global(state[:2], return_slope=True)
      direction = 1
      if self.rng.random() > 0.5:
        direction = -1
      state[3] = np.mod(direction*slope + state[3], 2 * np.pi)
    self.state = state.copy()

    obs = self.get_obs(state)
    if cast_torch:
      obs = torch.FloatTensor(obs)
    return obs

  def get_obs(self, state: np.ndarray) -> np.ndarray:
    """Gets the observation given the state.

    Args:
        state (np.ndarray): state of the shape (4, ).

    Returns:
        np.ndarray: observation. It can be the state or uses cos theta and
            sin theta to represent yaw.
    """
    if self.obs_type == 'perfect':
      return state
    else:
      _state = np.zeros(5)
      _state[:3] = state[:3]
      _state[3] = np.cos(state[3])
      _state[4] = np.sin(state[3])
      return _state

  def _get_cost_state_derivative(
      self, states: np.ndarray, close_pts: np.ndarray, slopes: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Jacobian and Hessian of the cost function with respect to state.

    Args:
        states (np.ndarray): planned trajectory, (4, N).
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
    ) - self.track_offset

    c_x = 2 * np.einsum('abn, bn->an', Q_trans, error)

    c_x_progress = -self.w_theta * np.array([cr, sr, zeros, zeros])
    c_x = c_x + c_x_progress
    c_xx = 2 * Q_trans

    return c_x, c_xx

  def report(self):
    print(
        "This is a Race Car simulator based on bicycle dynamics version 1 "
        + "and ellipse footprint."
    )
    super().report()
