"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Dict, Tuple, List, Any, Optional, Union
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from gym import spaces
import torch

from ..base_single_env import BaseSingleEnv
from ..ell_reach.ellipse import Ellipse
from .track import Track
from .constraints import Constraints
from .utils import get_centerline_from_traj


class RaceCarSingleEnv(BaseSingleEnv):
  """Implements an environment of a single Princeton Race Car.
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    assert config_env.NUM_AGENTS == 1, "This environment only has one agent!"
    super().__init__(config_env, config_agent)

    # Environment.
    self.track = Track(
        center_line=get_centerline_from_traj(config_env.TRACK_FILE),
        width_left=config_env.TRACK_WIDTH_LEFT,
        width_right=config_env.TRACK_WIDTH_RIGHT,
        loop=getattr(config_env, 'LOOP', True)
    )
    self.track_offset = config_env.TRACK_OFFSET
    self.constraints = Constraints(
        config_env=config_env, config_agent=config_agent
    )

    # Reset Sample Space. Note that the first two dimension is in the local
    # frame and it needs to call track.local2global() to get the (x, y)
    # position in the global frame.
    low = np.zeros((4,))
    low[1] = -config_env.TRACK_WIDTH_LEFT + config_agent.WIDTH * 3 / 4
    high = np.zeros((4,))
    high[0] = 1.
    high[1] = config_env.TRACK_WIDTH_RIGHT - config_agent.WIDTH * 3 / 4
    high[2] = config_agent.V_MAX
    high[3] = 2 * np.pi
    self.reset_sample_sapce = spaces.Box(
        low=np.float32(low), high=np.float32(high)
    )

    # Cost.
    self.w_vel = config_env.W_VEL
    self.w_contour = config_env.W_CONTOUR
    self.w_theta = config_env.W_THETA
    self.w_accel = config_env.W_ACCEL
    self.w_delta = config_env.W_DELTA
    self.v_ref = config_env.V_REF
    self.use_soft_cons_cost = config_env.USE_SOFT_CONS_COST
    self.W_state = np.array([[self.w_contour, 0], [0, self.w_vel]])
    self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])
    self.g_x_fail = config_env.G_X_FAIL
    self.target_amp = getattr(config_env, "TARGET_AMP", 1.)
    self.target_vel = getattr(config_env, "TARGET_VEL", 0.01)

    # Observation space.
    x_min, y_min = np.min(self.track.track_bound[2:, :], axis=1)
    x_max, y_max = np.max(self.track.track_bound[2:, :], axis=1)
    low = np.zeros((4,))
    low[0] = x_min
    low[1] = y_min
    high = np.zeros((4,))
    high[0] = x_max
    high[1] = y_max
    high[2] = config_agent.V_MAX
    high[3] = 2 * np.pi
    self.observation_space = spaces.Box(
        low=np.float32(low), high=np.float32(high)
    )
    self.observation_dim = self.observation_space.low.shape[0]
    self.visual_bounds = np.array([[x_min, x_max], [y_min, y_max]])
    self.visual_extent = np.array([
        self.visual_bounds[0, 0], self.visual_bounds[0, 1],
        self.visual_bounds[1, 0], self.visual_bounds[1, 1]
    ])
    self.seed(config_env.SEED)
    self.reset()

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
      # state[3] = slope  # random yaw as well.
    self.state = state.copy()

    if cast_torch:
      state = torch.FloatTensor(state)
    return state

  def get_samples(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gets state samples for value function plotting.

    Args:
        nx (int): the number of points along x-axis.
        ny (int): the number of points along y-axis.

    Returns:
        np.ndarray: a list of x-position.
        np.ndarray: a list of y-position.
    """
    xs = np.linspace(self.visual_bounds[0, 0], self.visual_bounds[0, 1], nx)
    ys = np.linspace(self.visual_bounds[1, 0], self.visual_bounds[1, 1], ny)
    return xs, ys

  def check_on_track(self, states: np.ndarray) -> np.ndarray:
    """Checks if the state is on the track (considering footprint).

    Args:
        states (np.ndarray): (x, y) positions, should ne (2, N).

    Returns:
        np.ndarray: a bool array of shape (N, ). True if the agent is on the
            track.
    """
    assert states.shape[0] == 2, "Shape should be (2, N)!"
    close_pts, slopes, _ = self.track.get_closest_pts(states)
    cons_road_l, cons_road_r = self.constraints._road_boundary_cons(
        self.agent.footprint, states, close_pts, slopes
    )

    flags = np.logical_and(cons_road_l <= 0, cons_road_r <= 0)
    return flags.reshape(-1)

  def render(
      self, ax: Optional[matplotlib.axes.Axes] = None, c_track: str = 'k',
      c_obs: str = 'r', c_ego: str = 'b', s: float = 12
  ):
    """Visualizes the current environment.

    Args:
        ax (Optional[matplotlib.axes.Axes], optional): the axes of matplotlib
            to plot if provided. Otherwise, we use the current axes. Defaults
            to None.
        c_track (str, optional): the color of the track. Defaults to 'k'.
        c_obs (str, optional): the color of the obstacles. Defaults to 'r'.
        c_ego (str, optional): the color of the ego agent. Defaults to 'b'.
        s (float, optional): the size of the ego agent point. Defaults to 12.
    """
    if ax is None:
      ax = plt.gca()
    self.track.plot_track(ax, c=c_track)
    ego = self.agent.footprint.move2state(self.state[[0, 1, 3]])
    ego.plot(ax, plot_center=False, color=c_ego)
    ax.scatter(self.state[0], self.state[1], c=c_ego, s=s)
    if self.constraints.obs_list is not None:
      for obs_list_j in self.constraints.obs_list:
        obs_list_j[0].plot(ax, color=c_obs, plot_center=False)

  def update_obs(self, obs_list: List[List[Ellipse]]):
    """Updates the obstacles.

    Args:
        obs_list (List[List[Ellipse]]): a list of ellipse lists. Each Ellipse
        in the list is an obstacle at each time step.
    """
    self.constraints.update_obs(obs_list)

  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Optional[dict] = None
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        float: the cost to minimize.
    """
    states_with_final, actions_with_final = self._reshape(
        state, action, state_nxt
    )
    close_pts, slopes, progress = self.track.get_closest_pts(
        states_with_final[:2, :], normalize_progress=True
    )
    ref_states, transform = self._get_ref_path_transform(close_pts, slopes)

    # State cost.
    error = states_with_final - ref_states
    Q_trans = np.einsum(
        'abn, bcn->acn',
        np.einsum(
            'dan, ab -> dbn', transform.transpose(1, 0, 2), self.W_state
        ), transform
    )
    c_contour = np.einsum(
        'an, an->n', error, np.einsum('abn, bn->an', Q_trans, error)
    )

    # Progress cost.
    c_progress = -self.w_theta * np.sum(progress)

    # Control cost.
    c_control = np.einsum(
        'an, an->n', actions_with_final,
        np.einsum('ab, bn->an', self.W_control, actions_with_final)
    )

    # Soft constraint cost.
    c_soft_cons = 0.
    if self.use_soft_cons_cost:
      if constraints is None:
        c_soft_cons = self.constraints.get_soft_cons_cost(
            footprint=self.agent.footprint, states=states_with_final,
            controls=actions_with_final, close_pts=close_pts, slopes=slopes
        )
      else:
        c_soft_cons = self.constraints.get_soft_cons_cost(
            footprint=self.agent.footprint, states=states_with_final,
            controls=actions_with_final, cons_dict=constraints
        )

    return np.sum(c_contour + c_control + c_soft_cons) + c_progress

  def get_constraints(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).

    Returns:
        Dict: each (key, value) pair is the name of a constraint function and
            an array consisting of values at each time step.
    """
    states_with_final, actions_with_final = self._reshape(
        state, action, state_nxt
    )
    close_pts, slopes, _ = self.track.get_closest_pts(states_with_final[:2, :])

    return self.constraints.get_constraint(
        footprint=self.agent.footprint, states=states_with_final,
        controls=actions_with_final, close_pts=close_pts, slopes=slopes
    )

  def get_target_margin(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all target margin functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).

    Returns:
        Dict: each (key, value) pair is the name and value of a target margin
            function.
    """
    states_with_final, actions_with_final = self._reshape(
        state, action, state_nxt
    )
    targets = {}
    target_vel_margin = states_with_final[2:3, :] - self.target_vel
    target_vel_margin[target_vel_margin < 0] *= self.target_amp
    targets['velocity'] = target_vel_margin
    return targets

  def get_done_and_info(
      self, constraints: Dict, targets: Optional[Dict] = None,
      final_only: bool = True, end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets.

    Args:
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.
        targets (Dict): each (key, value) pair is the name and value of a
            target margin function.

    Returns:
        bool: True if the episode ends.
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    if end_criterion is None:
      end_criterion = self.end_criterion

    done = False
    done_type = "not_raised"
    if self.cnt >= self.timeout:
      done = True
      done_type = "timeout"

    # Retrieves constraints / traget values.
    constraint_values = None
    for key, value in constraints.items():
      if constraint_values is None:
        num_pts = value.shape[1]
        constraint_values = value
      else:
        assert num_pts == value.shape[1], (
            "The length of constraint ({}) do not match".format(key)
        )
        constraint_values = np.concatenate((constraint_values, value), axis=0)
    target_values = None
    for key, value in targets.items():
      assert num_pts == value.shape[1], (
          "The length of target ({}) do not match".format(key)
      )
      if target_values is None:
        target_values = value
      else:
        target_values = np.concatenate((target_values, value), axis=0)

    # Gets info.
    g_x_list = np.max(constraint_values, axis=0)
    l_x_list = np.max(target_values, axis=0)
    if final_only:
      g_x = g_x_list[-1]
      l_x = l_x_list[-1]
      binary_cost = 1. if g_x > 0. else 0.
    else:
      g_x = g_x_list
      l_x = l_x_list
      binary_cost = 1. if np.any(g_x > 0.) else 0.

    # Gets done flag
    if end_criterion == 'failure':
      if final_only:
        failure = np.any(constraint_values[:, -1] > 0.)
      else:
        failure = np.any(constraint_values > 0.)
      if failure:
        done = True
        done_type = "failure"
        g_x = self.g_x_fail
    elif end_criterion == 'reach-avoid':
      if final_only:
        failure = g_x > 0.
        success = not failure and l_x <= 0.
      else:
        v_x_list = np.empty(shape=(num_pts,))
        v_x_list[num_pts
                 - 1] = max(l_x_list[num_pts - 1], g_x_list[num_pts - 1])
        for i in range(num_pts - 2, -1, -1):
          v_x_list[i] = max(g_x_list[i], min(l_x_list[i], v_x_list[i + 1]))
        inst = np.argmin(v_x_list)
        failure = np.any(constraint_values[:, :inst + 1] > 0.)
        success = not failure and (v_x_list[inst] <= 0)
      if success:
        done = True
        done_type = "success"
      elif failure:
        done = True
        done_type = "failure"
        g_x = self.g_x_fail
    elif end_criterion == 'timeout':
      pass
    else:
      raise ValueError("End criterion not supported!")

    # Gets info
    info = {
        "done_type": done_type,
        "g_x": g_x,
        "l_x": l_x,
        "binary_cost": binary_cost
    }
    return done, info

  def get_derivatives(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Jacobian and Hessian of the cost (possibly with soft constraint
    cost).

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).

    Returns:
        np.ndarray: c_x of the shape (4, N).
        np.ndarray: c_xx of the shape (4, 4, N).
        np.ndarray: c_u of the shape (2, N).
        np.ndarray: c_uu of the shape (2, 2, N).
        np.ndarray: c_ux of the shape (2, 4, N).
    """
    states_with_final, actions_with_final = self._reshape(
        state, action, state_nxt
    )
    close_pts, slopes, _ = self.track.get_closest_pts(
        states_with_final[:2, :], normalize_progress=True
    )

    c_x_cost, c_xx_cost = self._get_cost_state_derivative(
        states_with_final, close_pts, slopes
    )

    c_u_cost, c_uu_cost = self._get_cost_control_derivative(actions_with_final)

    q = c_x_cost
    Q = c_xx_cost

    r = c_u_cost
    R = c_uu_cost

    S = np.zeros((
        self.agent.dyn.dim_u, self.agent.dyn.dim_x, states_with_final.shape[1]
    ))

    if self.use_soft_cons_cost:
      c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons = (
          self.constraints.get_derivatives(
              footprint=self.agent.footprint, states=states_with_final,
              controls=actions_with_final, close_pts=close_pts, slopes=slopes
          )
      )
      q += c_x_cons
      Q += c_xx_cons
      r += c_u_cons
      R += c_uu_cons
      S += c_ux_cons

    return q, Q, r, R, S

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

    # shape [4xN]
    c_x = 2 * np.einsum('abn, bn->an', Q_trans, error)

    c_x_progress = -self.w_theta * np.array([cr, sr, zeros, zeros])
    c_x = c_x + c_x_progress
    c_xx = 2 * Q_trans

    return c_x, c_xx

  def _get_cost_control_derivative(
      self, controls: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Jacobian and Hessian of the cost function w.r.t the control.

    Args:
        controls (np.ndarray): planned control, (2, N).

    Returns:
        np.ndarray: Jacobian.
        np.ndarray: Hessian.
    """
    c_u = 2 * np.einsum('ab, bn->an', self.W_control, controls)
    c_uu = 2 * np.repeat(
        self.W_control[:, :, np.newaxis], controls.shape[1], axis=2
    )
    c_u[:, -1] = 0.
    c_uu[:, :, -1] = 0.
    return c_u, c_uu

  def _get_ref_path_transform(
      self, close_pts: np.ndarray, slopes: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets the reference path and the transformation form the global frame to the
    local frame with the origin at the closest points.

    Args:
        close_pts (np.ndarray): the position of the closest points on the
            centerline. This array should be of the shape (2, N).
        slopes (np.ndarray): the slope of of trangent line on those points.
            This vector should be of the shape (1, N).

    Returns:
        np.ndarray: _description_
    """
    num_pts = close_pts.shape[1]
    zeros = np.zeros((num_pts))
    ones = np.ones((num_pts))
    slopes = slopes.reshape(-1)

    transform = np.array([[np.sin(slopes), -np.cos(slopes), zeros, zeros],
                          [zeros, zeros, ones, zeros]])

    ref_states = np.zeros((self.agent.dyn.dim_x, num_pts))
    ref_states[0, :] = close_pts[0, :] + np.sin(slopes) * self.track_offset
    ref_states[1, :] = close_pts[1, :] - np.cos(slopes) * self.track_offset
    ref_states[2, :] = self.v_ref

    return ref_states, transform

  def _reshape(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenates states with state_nxt and appends dummy control after action.

    Args:
        state (np.ndarray): current states of the shape (4, N).
        action (np.ndarray): current actions of the shape (2, N).
        state_nxt (np.ndarray): next state or final state of the shape (4, ).

    Returns:
        np.ndarray: states with the final state.
        np.ndarray: action with the dummy control.
    """
    if state.ndim == 1:
      state = state[:, np.newaxis]
    if state_nxt.ndim == 1:
      state_nxt = state_nxt[:, np.newaxis]
    if action.ndim == 1:
      action = action[:, np.newaxis]
    assert state.shape[1] == action.shape[1], (
        "The length of states ({}) and ".format(state.shape[1]),
        "the length of actions ({}) don't match!".format(action.shape[1])
    )
    assert state_nxt.shape[1] == 1, "state_nxt should consist only 1 state!"

    states_with_final = np.concatenate((state, state_nxt), axis=1)
    actions_with_final = np.concatenate(
        (action, np.zeros((action.shape[0], 1))), axis=1
    )
    return states_with_final, actions_with_final

  def report(self):
    print(
        "This is a Race Car simulator based on bicycle dynamics and "
        + "ellipse footprint."
    )
