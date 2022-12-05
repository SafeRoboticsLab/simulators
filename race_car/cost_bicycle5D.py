from typing import Dict, List, Optional
from functools import partial
from jaxlib.xla_extension import DeviceArray
import numpy as np
import jax.numpy as jnp
import jax

from ..cost.base_cost import BaseCost, BarrierCost
from ..cost.box_cost import BoxObsBoxFootprintCost
from ..cost.quadratic_cost import QuadraticControlCost
from ..cost.half_space_cost import LowerHalfCost, UpperHalfCost
from ..cost.spline_cost import (
    BaseSplineCost, SplineBarrierCost, SplineRoadBoundaryCost, SplineYawCost
)


class Bicycle5DCost(BaseSplineCost):

  def __init__(self, config):
    super().__init__(config)

    # Racing cost parameters.
    self.v_ref: float = config.V_REF  # reference velocity.
    self.w_vel: float = config.W_VEL
    self.w_contour: float = config.W_CONTOUR
    self.w_theta: float = config.W_THETA
    self.w_accel: float = config.W_ACCEL
    self.w_omega: float = config.W_OMEGA
    self.wheelbase: float = config.WHEELBASE
    self.track_offset: float = config.TRACK_OFFSET

    # Soft constraint parameters.
    self.q1_yaw: float = config.Q1_YAW
    self.q2_yaw: float = config.Q2_YAW
    self.q1_road: float = config.Q1_ROAD
    self.q2_road: float = config.Q2_ROAD
    self.q1_obs: float = config.Q1_OBS
    self.q2_obs: float = config.Q2_OBS
    self.yaw_min: float = config.YAW_MIN
    self.yaw_max: float = config.YAW_MAX
    self.obs_spec = [np.asarray(x) for x in config.OBS_SPEC]
    self.obs_precision = list(getattr(config, "OBS_PRECISION", [31, 11]))
    self.barrier_clip_min: float = config.BARRIER_CLIP_MIN
    self.barrier_clip_max: float = config.BARRIER_CLIP_MAX
    self.buffer: float = getattr(config, "BUFFER", 0.)
    self.yaw_barrier_cost = SplineBarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_yaw, self.q2_yaw,
        SplineYawCost(config)
    )
    self.road_barrier_cost = SplineBarrierCost(
        self.barrier_clip_min, self.barrier_clip_max,
        self.q1_road, self.q2_road,
        SplineRoadBoundaryCost(config=config, buffer=self.buffer)
    )
    self.obs_barrier_cost: List[BarrierCost] = []
    for box_spec in self.obs_spec:
      self.obs_barrier_cost.append(
          BarrierCost(
              self.barrier_clip_min, self.barrier_clip_max, self.q1_obs,
              self.q2_obs,
              BoxObsBoxFootprintCost(
                  state_box_limits=self.state_box_limits, box_spec=box_spec,
                  precision=self.obs_precision, buffer=self.buffer
              )
          )
      )

    self.has_vel_constr: bool = config.HAS_VEL_CONSTR
    if self.has_vel_constr:
      self.q1_v: float = config.Q1_V
      self.q2_v: float = config.Q2_V
      self.v_min: float = config.V_MIN
      self.v_max: float = config.V_MAX
      self.vel_max_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
          UpperHalfCost(value=self.v_max, dim=2)
      )
      self.vel_min_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
          LowerHalfCost(value=self.v_min, dim=2)
      )

    self.has_delta_constr: bool = config.HAS_DELTA_CONSTR
    if self.has_delta_constr:
      self.q1_delta: float = config.Q1_DELTA
      self.q2_delta: float = config.Q2_DELTA
      self.delta_min: float = config.DELTA_MIN
      self.delta_max: float = config.DELTA_MAX
      self.delta_max_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_delta,
          self.q2_delta, UpperHalfCost(value=self.delta_max, dim=4)
      )
      self.delta_min_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_delta,
          self.q2_delta, LowerHalfCost(value=self.delta_min, dim=4)
      )

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)
        closest_pt (DeviceArray, vector shape)
        slope (DeviceArray, vector shape)
        theta (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    # control cost
    cost = self.w_accel * ctrl[0]**2 + self.w_omega * ctrl[1]**2

    # state cost
    sr = jnp.sin(slope[0])
    cr = jnp.cos(slope[0])
    offset = (
        sr * (state[0] - closest_pt[0]) - cr *
        (state[1] - closest_pt[1]) - self.track_offset
    )
    cost += self.w_contour * offset**2
    cost += self.w_vel * (state[2] - self.v_ref)**2
    cost -= self.w_theta * theta[0]

    # soft constraint cost
    cost += self.yaw_barrier_cost.get_stage_cost(
        state, ctrl, closest_pt, slope, theta, time_idx
    )
    cost += self.road_barrier_cost.get_stage_cost(
        state, ctrl, closest_pt, slope, theta, time_idx
    )
    for _obs_barrier_cost in self.obs_barrier_cost:
      _obs_barrier_cost: BarrierCost
      cost += _obs_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    if self.has_vel_constr:
      cost += self.vel_max_barrier_cost.get_stage_cost(state, ctrl, time_idx)
      cost += self.vel_min_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    if self.has_delta_constr:
      cost += self.delta_max_barrier_cost.get_stage_cost(state, ctrl, time_idx)
      cost += self.delta_min_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    return cost


class Bicycle5DConstraint(BaseSplineCost):

  def __init__(self, config):
    super().__init__(config)

    self.yaw_min = config.YAW_MIN
    self.yaw_max = config.YAW_MAX

    self.obs_spec = config.OBS_SPEC
    self.obs_precision = getattr(config, "OBS_PRECISION", [31, 11])
    if hasattr(config, "BUFFER"):
      self.buffer: float = config.BUFFER
      print(
          f"Adds buffer ({self.buffer}) around obstacles and road boundaries."
      )
    else:
      self.buffer = 0.

    self.yaw_constraint = SplineYawCost(config)

    self.road_constraint = SplineRoadBoundaryCost(
        config=config, buffer=self.buffer
    )
    self.obs_constraint = []
    for box_spec in self.obs_spec:
      self.obs_constraint.append(
          BoxObsBoxFootprintCost(
              state_box_limits=self.state_box_limits, box_spec=box_spec,
              precision=self.obs_precision, buffer=self.buffer
          )
      )

    self.has_vel_constr = config.HAS_VEL_CONSTR
    if self.has_vel_constr:
      self.v_min = config.V_MIN
      self.v_max = config.V_MAX
      self.vel_max_constraint = UpperHalfCost(value=self.v_max, dim=2)
      self.vel_min_constraint = LowerHalfCost(value=self.v_min, dim=2)

    self.has_delta_constr = config.HAS_DELTA_CONSTR
    if self.has_delta_constr:
      self.delta_min = config.DELTA_MIN
      self.delta_max = config.DELTA_MAX
      self.delta_max_constraint = UpperHalfCost(value=self.delta_max, dim=4)
      self.delta_min_constraint = LowerHalfCost(value=self.delta_min, dim=4)

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    cost = self.yaw_constraint.get_stage_cost(
        state, ctrl, closest_pt, slope, theta, time_idx
    )
    if self.has_vel_constr:
      cost = jnp.maximum(
          cost, self.vel_max_constraint.get_stage_cost(state, ctrl, time_idx)
      )
      cost = jnp.maximum(
          cost, self.vel_min_constraint.get_stage_cost(state, ctrl, time_idx)
      )

    if self.has_delta_constr:
      cost = jnp.maximum(
          cost,
          self.delta_max_constraint.get_stage_cost(state, ctrl, time_idx)
      )
      cost = jnp.maximum(
          cost,
          self.delta_min_constraint.get_stage_cost(state, ctrl, time_idx)
      )
    cost = jnp.maximum(
        cost,
        self.road_constraint.get_stage_cost(
            state, ctrl, closest_pt, slope, theta, time_idx
        )
    )
    for _obs_constraint in self.obs_constraint:
      _obs_constraint: BaseCost
      cost = jnp.maximum(
          cost, _obs_constraint.get_stage_cost(state, ctrl, time_idx)
      )
    return cost

  @partial(jax.jit, static_argnames='self')
  def get_cost_dict(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> Dict:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """

    yaw_cons = self.yaw_constraint.get_cost(
        state, ctrl, closest_pt, slope, theta, time_indices
    )
    road_cons = self.road_constraint.get_cost(
        state, ctrl, closest_pt, slope, theta, time_indices
    )
    obs_cons = -jnp.inf
    for _obs_constraint in self.obs_constraint:
      _obs_constraint: BaseCost
      obs_cons = jnp.maximum(
          obs_cons, _obs_constraint.get_cost(state, ctrl, time_indices)
      )

    info = dict(yaw_cons=yaw_cons, road_cons=road_cons, obs_cons=obs_cons)
    if self.has_vel_constr:
      info['vel_max_cons'] = self.vel_max_constraint.get_cost(
          state, ctrl, time_indices
      )
      info['vel_min_cons'] = self.vel_min_constraint.get_cost(
          state, ctrl, time_indices
      )
    if self.has_delta_constr:
      info['delta_max_cons'] = self.delta_max_constraint.get_cost(
          state, ctrl, time_indices
      )
      info['delta_min_cons'] = self.delta_min_constraint.get_cost(
          state, ctrl, time_indices
      )

    return info


class Bicycle5DSquareConstraint(BaseSplineCost):

  def __init__(self, config):
    super().__init__(config)
    self.constraint = Bicycle5DConstraint(config)

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    state_cost = self.constraint.get_stage_cost(
        state, ctrl, closest_pt, slope, theta, time_idx
    )
    state_cost = state_cost * jnp.abs(state_cost)
    return state_cost


class Bicycle5DReachabilityCost(BaseSplineCost):

  def __init__(self, config):
    super().__init__()
    self.constraint = Bicycle5DSquareConstraint(config)
    R = jnp.array([[config.W_ACCEL, 0.0], [0.0, config.W_OMEGA]])
    self.ctrl_cost = QuadraticControlCost(R=R, r=jnp.zeros(2))
    self.N = config.N

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    state_cost = self.constraint.get_stage_cost(
        state, ctrl, closest_pt, slope, theta, time_idx
    )
    ctrl_cost = self.ctrl_cost.get_stage_cost(state, ctrl, time_idx)
    return state_cost + ctrl_cost

  @partial(jax.jit, static_argnames='self')
  def get_traj_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> float:
    state_costs = self.constraint.get_cost(
        state, ctrl, closest_pt, slope, theta, time_indices
    )
    ctrl_costs = self.ctrl_cost.get_cost(state, ctrl, time_indices)
    # TODO: critical points version
    return (jnp.max(state_costs[1:]) + jnp.sum(ctrl_costs)).astype(float)


class Bicycle5DRefTrajCost(BaseSplineCost):

  def __init__(
      self, config, ref_traj: np.ndarray, ref_ctrl: Optional[np.ndarray] = None
  ):
    super().__init__(config)

    # Racing cost parameters.
    self.ref_traj = jnp.asarray(ref_traj)
    if ref_ctrl is None:
      self.ref_ctrl = jnp.zeros((2, ref_traj.shape[1]))
    else:
      assert ref_ctrl.shape[1] == ref_traj.shape[1]
      self.ref_ctrl = jnp.asarray(ref_ctrl)
    if isinstance(config.W_REF, float):
      self.w_ref: DeviceArray = jnp.eye(5) * config.W_REF
    else:
      assert isinstance(config.W_REF, list)
      assert len(config.W_REF) == 5
      self.w_ref: DeviceArray = jnp.diag(jnp.asarray(config.W_REF)).copy()
    self.w_accel: float = config.W_ACCEL
    self.w_omega: float = config.W_OMEGA
    self.R = jnp.asarray([[self.w_accel, 0.], [0., self.w_omega]])
    self.wheelbase: float = config.WHEELBASE

    # Soft constraint parameters.
    self.barrier_clip_min: float = config.BARRIER_CLIP_MIN
    self.barrier_clip_max: float = config.BARRIER_CLIP_MAX
    self.buffer: float = getattr(config, "BUFFER", 0.)
    self.has_yaw_constr: bool = config.HAS_YAW_CONSTR
    if self.has_yaw_constr:
      self.q1_yaw: float = config.Q1_YAW
      self.q2_yaw: float = config.Q2_YAW
      self.yaw_min: float = config.YAW_MIN
      self.yaw_max: float = config.YAW_MAX
      self.yaw_barrier_cost = SplineBarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_yaw,
          self.q2_yaw, SplineYawCost(config)
      )

    self.has_road_constr: bool = config.HAS_ROAD_CONSTR
    if self.has_road_constr:
      self.q1_road: float = config.Q1_ROAD
      self.q2_road: float = config.Q2_ROAD
      self.road_barrier_cost = SplineBarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_road,
          self.q2_road,
          SplineRoadBoundaryCost(config=config, buffer=self.buffer)
      )

    self.has_obs_constr: bool = config.HAS_OBS_CONSTR
    if self.has_obs_constr:
      self.q1_obs: float = config.Q1_OBS
      self.q2_obs: float = config.Q2_OBS
      self.obs_spec = [np.asarray(x) for x in config.OBS_SPEC]
      self.obs_precision = list(getattr(config, "OBS_PRECISION", [31, 11]))
      self.obs_barrier_cost: List[BarrierCost] = []
      for box_spec in self.obs_spec:
        self.obs_barrier_cost.append(
            BarrierCost(
                self.barrier_clip_min, self.barrier_clip_max, self.q1_obs,
                self.q2_obs,
                BoxObsBoxFootprintCost(
                    state_box_limits=self.state_box_limits, box_spec=box_spec,
                    precision=self.obs_precision, buffer=self.buffer
                )
            )
        )

    self.has_vel_constr: bool = config.HAS_VEL_CONSTR
    if self.has_vel_constr:
      self.q1_v: float = config.Q1_V
      self.q2_v: float = config.Q2_V
      self.v_min: float = config.V_MIN
      self.v_max: float = config.V_MAX
      self.vel_max_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
          UpperHalfCost(value=self.v_max, dim=2)
      )
      self.vel_min_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
          LowerHalfCost(value=self.v_min, dim=2)
      )

    self.has_delta_constr: bool = config.HAS_DELTA_CONSTR
    if self.has_delta_constr:
      self.q1_delta: float = config.Q1_DELTA
      self.q2_delta: float = config.Q2_DELTA
      self.delta_min: float = config.DELTA_MIN
      self.delta_max: float = config.DELTA_MAX
      self.delta_max_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_delta,
          self.q2_delta, UpperHalfCost(value=self.delta_max, dim=4)
      )
      self.delta_min_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_delta,
          self.q2_delta, LowerHalfCost(value=self.delta_min, dim=4)
      )

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)
        closest_pt (DeviceArray, vector shape)
        slope (DeviceArray, vector shape)
        theta (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """

    # state cost
    delta_x = state - self.ref_traj[:, time_idx].reshape(-1)
    Qx = jnp.einsum("i,ni->n", delta_x, self.w_ref)
    cost = jnp.einsum("n,n", delta_x, Qx)

    # control cost
    delta_u = ctrl - self.ref_ctrl[:, time_idx].reshape(-1)
    Ru = jnp.einsum("i,ni->n", delta_u, self.R)
    cost += jnp.einsum("n,n", delta_u, Ru)

    # soft constraint cost
    if self.has_yaw_constr:
      cost += self.yaw_barrier_cost.get_stage_cost(
          state, ctrl, closest_pt, slope, theta, time_idx
      )
    if self.has_road_constr:
      cost += self.road_barrier_cost.get_stage_cost(
          state, ctrl, closest_pt, slope, theta, time_idx
      )
    if self.has_obs_constr:
      for _obs_barrier_cost in self.obs_barrier_cost:
        _obs_barrier_cost: BarrierCost
        cost += _obs_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    if self.has_vel_constr:
      cost += self.vel_max_barrier_cost.get_stage_cost(state, ctrl, time_idx)
      cost += self.vel_min_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    if self.has_delta_constr:
      cost += self.delta_max_barrier_cost.get_stage_cost(state, ctrl, time_idx)
      cost += self.delta_min_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    return cost
