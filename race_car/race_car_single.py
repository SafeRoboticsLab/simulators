from typing import Dict, Tuple, List, Any, Optional
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from gym import spaces

from ..agent import Agent
from ..base_env import BaseEnv
from ..ell_reach.ellipse import Ellipse
from .track import Track
from .constraints import Constraints
from .utils import get_centerline_from_traj


class RaceCarSingleEnv(BaseEnv):
  """
  Implements an environment of a single Princeton Race Car.
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    assert config_env.NUM_AGENTS == 1, "This environment only has one agent!"
    super().__init__()

    # Environment.
    self.timeoff = config_env.TIMEOFF
    self.end_criterion = config_env.END_CRITERION

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

    # Action Space.
    action_space = np.array(config_agent.ACTION_RANGE)
    self.action_dim = action_space.shape[0]
    self.agent = Agent(config_agent, action_space)
    self.action_space = spaces.Box(
        low=action_space[:, 0], high=action_space[:, 1]
    )

    self.integrate_kwargs = {}
    if hasattr(config_agent, 'INTEGRATE_KWARGS'):
      self.integrate_kwargs = config_agent.INTEGRATE_KWARGS
      self.integrate_kwargs['noise'] = np.array(self.integrate_kwargs['noise'])

    # Observation Space. Note that the first two dimension is in the local
    # frame and it needs to call track.local2global() to get the (x, y)
    # position in the global frame.
    low = np.zeros((4, 1))
    low[1] = -config_env.TRACK_WIDTH_LEFT + config_agent.WIDTH * 3 / 4
    high = np.zeros((4, 1))
    high[0] = 1.
    high[1] = config_env.TRACK_WIDTH_RIGHT - config_agent.WIDTH * 3 / 4
    high[2] = config_agent.V_MAX
    high[3] = 2 * np.pi
    self.observation_space = spaces.Box(low=low, high=high)
    self.observation_dim = self.observation_space.low.shape
    self.reset()

  def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
    """Implements the step function in the environment.

    Args:
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].

    Returns:
        np.ndarray: next state.
        float: the reward that ctrl wants to maximize and dstb wants to
            minimize.
        bool: True if the episode ends.
        Dict[str, Any]]: additional information of the step, such as target
            margin and safety margin used in reachability analysis.
    """
    self.cnt += 1
    state_nxt, _ = self.agent.integrate_forward(
        state=self.state, control=action, **self.integrate_kwargs
    )
    constraints = self.get_constraints(self.state, action, state_nxt)
    cost = self.get_cost(self.state, action, state_nxt, constraints)
    done = self.get_done_flag(self.state, action, state_nxt, constraints)
    info = self.get_info(self.state, action, state_nxt, cost, constraints)

    return np.copy(state_nxt), -cost, done, info

  def reset(self, state: Optional[np.ndarray] = None) -> np.ndarray:
    self.cnt = 0
    if state is None:
      state = self.observation_space.sample()
      state[:2], slope = self.track.local2global(state[:2], return_slope=True)
      state[3] = slope
    self.state = state.copy()
    return state

  def render(
      self, ax: Optional[matplotlib.axes.Axes] = None, c_track: str = 'k',
      c_obs: str = 'r', c_ego: str = 'b', s: float = 12
  ):
    if ax is None:
      ax = plt.gca()
    self.track.plot_track(ax, c=c_track)
    ego = self.agent.footprint.move2state(self.state[[0, 1, 3], 0])
    ego.plot(ax, plot_center=False, color=c_ego)
    ax.scatter(self.state[0], self.state[1], c=c_ego, s=s)
    if self.constraints.obs_list is not None:
      for obs_list_j in self.constraints.obs_list:
        for obs_i_j in obs_list_j:
          obs_i_j.plot(ax, color=c_obs, plot_center=False)

  def update_obs(self, obs_list: List[List[Ellipse]]):
    self.constraints.update_obs(obs_list)

  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Optional[dict] = None
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current state.
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state or the final state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        float: the cost that ctrl wants to minimize and dstb wants to maximize.
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

    # with np.printoptions(precision=2, suppress=True):
    #   print("c_state", c_contour)
    #   print("c_constraint", c_soft_cons)
    #   print("c_control", c_control)
    #   print("c_progress", c_progress)
    return np.sum(c_contour + c_control + c_soft_cons) + c_progress

  def get_constraints(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current state.
        action (np.ndarray): a dictionary consists of 'ctrl' and 'dstb', which
            are accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
    """
    states_with_final, actions_with_final = self._reshape(
        state, action, state_nxt
    )
    close_pts, slopes, _ = self.track.get_closest_pts(states_with_final[:2, :])

    return self.constraints.get_constraint(
        footprint=self.agent.footprint, states=states_with_final,
        controls=actions_with_final, close_pts=close_pts, slopes=slopes
    )

  def get_done_flag(self, constraints: Dict) -> bool:
    """
    Gets the done flag given current state, current action, next state, and
    constraints.

    Args:
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        bool: True if the episode ends.
    """
    if self.cnt >= self.timeoff:
      return True
    if self.end_criterion == 'fail':
      for value in constraints.values():
        if value > 0.:
          return True
      return False

  def get_info(self, constraints: Dict) -> Dict:
    """
    Gets a dictionary to provide additional information of the step function
    given current state, current action, next state, cost, and constraints.

    Args:
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    pass

  def get_derivatives(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # with np.printoptions(precision=2, suppress=True):
    #   print("c_x_cost", c_x_cost)
    #   print("c_xx_cost", c_xx_cost)
    #   print("c_u_cost", c_u_cost)
    #   print("c_uu_cost", c_uu_cost)

    if self.use_soft_cons_cost:
      c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons = (
          self.constraints.get_derivatives(
              footprint=self.agent.footprint, states=states_with_final,
              controls=actions_with_final, close_pts=close_pts, slopes=slopes
          )
      )
      # with np.printoptions(precision=2, suppress=True):
      #   print("c_x_cons", c_x_cons)
      #   print("c_xx_cons", c_xx_cons)
      #   print("c_u_cons", c_u_cons)
      #   print("c_uu_cons", c_uu_cons)
      #   print("c_ux_cons", c_ux_cons)
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
