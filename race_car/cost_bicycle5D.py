from typing import Dict
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp
import jax

from ..cost.base_cost import BaseCost, BarrierCost
from ..cost.box_cost import BoxObsBoxFootprintCost
from ..cost.quadratic_cost import QuadraticControlCost
from ..cost.half_space_cost import LowerHalfCost, UpperHalfCost
from ..cost.spline_cost import (
    BaseSplineCost, SplineBarrierCost, SplineRoadBoundaryCost, SplineYawCost
)


#! We override all derivate functions in BaseCost as this cost depends on
#! information from Spline.
class Bicycle5DCost(BaseSplineCost):

  def __init__(self, config):
    super().__init__()
    # System parameters.
    self.state_box_limits = config.STATE_BOX_LIMITS

    # Racing cost parameters.
    self.v_ref = config.V_REF  # reference velocity.
    self.w_vel = config.W_VEL
    self.w_contour = config.W_CONTOUR
    self.w_theta = config.W_THETA
    self.w_accel = config.W_ACCEL
    self.w_omega = config.W_OMEGA
    self.wheelbase = config.WHEELBASE
    self.track_width_right = config.TRACK_WIDTH_RIGHT
    self.track_width_left = config.TRACK_WIDTH_LEFT
    self.track_offset = config.TRACK_OFFSET

    # Soft constraint parameters.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V
    self.q1_yaw = config.Q1_YAW
    self.q2_yaw = config.Q2_YAW
    self.q1_road = config.Q1_ROAD
    self.q2_road = config.Q2_ROAD
    self.q1_obs = config.Q1_OBS
    self.q2_obs = config.Q2_OBS
    self.q1_delta = config.Q1_DELTA
    self.q2_delta = config.Q2_DELTA
    self.v_min = config.V_MIN
    self.v_max = config.V_MAX
    self.yaw_min = config.YAW_MIN
    self.yaw_max = config.YAW_MAX
    self.delta_min = config.DELTA_MIN
    self.delta_max = config.DELTA_MAX
    self.obs_spec = config.OBS_SPEC
    self.obs_precision = getattr(config, "OBS_PRECISION", [31, 11])
    self.barrier_clip_min = config.BARRIER_CLIP_MIN
    self.barrier_clip_max = config.BARRIER_CLIP_MAX
    self.buffer = getattr(config, "BUFFER", 0.)
    self.vel_max_barrier_cost = BarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
        UpperHalfCost(value=self.v_max, dim=2)
    )
    self.vel_min_barrier_cost = BarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
        LowerHalfCost(value=self.v_min, dim=2)
    )
    self.yaw_barrier_cost = SplineBarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_yaw, self.q2_yaw,
        SplineYawCost(
            yaw_min=self.yaw_min, yaw_max=self.yaw_max,
            bidirectional=getattr(config, "BIDIRECTIONAL", True)
        )
    )
    self.delta_max_barrier_cost = BarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_delta,
        self.q2_delta, UpperHalfCost(value=self.delta_max, dim=4)
    )
    self.delta_min_barrier_cost = BarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_delta,
        self.q2_delta, LowerHalfCost(value=self.delta_min, dim=4)
    )
    self.road_barrier_cost = SplineBarrierCost(
        self.barrier_clip_min, self.barrier_clip_max,
        self.q1_road, self.q2_road,
        SplineRoadBoundaryCost(config=config, buffer=self.buffer)
    )
    self.obs_barrier_cost = []
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

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray
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
    cost += self.vel_max_barrier_cost.get_stage_cost(state, ctrl)
    cost += self.vel_min_barrier_cost.get_stage_cost(state, ctrl)
    cost += self.yaw_barrier_cost.get_stage_cost(
        state, ctrl, closest_pt, slope, theta
    )
    cost += self.delta_max_barrier_cost.get_stage_cost(state, ctrl)
    cost += self.delta_min_barrier_cost.get_stage_cost(state, ctrl)
    cost += self.road_barrier_cost.get_stage_cost(
        state, ctrl, closest_pt, slope, theta
    )
    for _obs_barrier_cost in self.obs_barrier_cost:
      _obs_barrier_cost: BarrierCost
      cost += _obs_barrier_cost.get_stage_cost(state, ctrl)
    return cost


class Bicycle5DConstraint(BaseSplineCost):

  def __init__(self, config):
    super().__init__()
    # System parameters.
    self.state_box_limits = config.STATE_BOX_LIMITS

    # Constraint parameters.
    self.track_width_right = config.TRACK_WIDTH_RIGHT
    self.track_width_left = config.TRACK_WIDTH_LEFT
    self.v_min = config.V_MIN
    self.v_max = config.V_MAX
    self.yaw_min = config.YAW_MIN
    self.yaw_max = config.YAW_MAX
    self.delta_min = config.DELTA_MIN
    self.delta_max = config.DELTA_MAX
    self.obs_spec = config.OBS_SPEC
    self.obs_precision = getattr(config, "OBS_PRECISION", [31, 11])
    self.buffer = getattr(config, "BUFFER", 0.)
    self.vel_max_constraint = UpperHalfCost(value=self.v_max, dim=2)
    self.vel_min_constraint = LowerHalfCost(value=self.v_min, dim=2)
    self.yaw_constraint = SplineYawCost(
        yaw_min=self.yaw_min, yaw_max=self.yaw_max,
        bidirectional=getattr(config, "BIDIRECTIONAL", True)
    )
    self.delta_max_constraint = UpperHalfCost(value=self.delta_max, dim=4)
    self.delta_min_constraint = LowerHalfCost(value=self.delta_min, dim=4)
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

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    cost = self.vel_max_constraint.get_stage_cost(state, ctrl)
    cost = jnp.maximum(
        cost, self.vel_min_constraint.get_stage_cost(state, ctrl)
    )
    cost = jnp.maximum(
        cost,
        self.yaw_constraint.get_stage_cost(
            state, ctrl, closest_pt, slope, theta
        )
    )
    cost = jnp.maximum(
        cost, self.delta_max_constraint.get_stage_cost(state, ctrl)
    )
    cost = jnp.maximum(
        cost, self.delta_min_constraint.get_stage_cost(state, ctrl)
    )
    cost = jnp.maximum(
        cost,
        self.road_constraint.get_stage_cost(
            state, ctrl, closest_pt, slope, theta
        )
    )
    for _obs_constraint in self.obs_constraint:
      _obs_constraint: BaseCost
      cost = jnp.maximum(cost, _obs_constraint.get_stage_cost(state, ctrl))
    return cost

  @partial(jax.jit, static_argnames='self')
  def get_cost_dict(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray
  ) -> Dict:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    vel_max_cons = self.vel_max_constraint.get_cost(state, ctrl)
    vel_min_cons = self.vel_min_constraint.get_cost(state, ctrl)
    yaw_cons = self.yaw_constraint.get_cost(
        state, ctrl, closest_pt, slope, theta
    )
    delta_max_cons = self.delta_max_constraint.get_cost(state, ctrl)
    delta_min_cons = self.delta_min_constraint.get_cost(state, ctrl)
    road_cons = self.road_constraint.get_cost(
        state, ctrl, closest_pt, slope, theta
    )

    obs_cons = -jnp.inf
    for _obs_constraint in self.obs_constraint:
      _obs_constraint: BaseCost
      obs_cons = jnp.maximum(obs_cons, _obs_constraint.get_cost(state, ctrl))

    return dict(
        vel_max_cons=vel_max_cons, vel_min_cons=vel_min_cons,
        yaw_cons=yaw_cons, delta_max_cons=delta_max_cons,
        delta_min_cons=delta_min_cons, road_cons=road_cons, obs_cons=obs_cons
    )


class Bicycle5DSquareConstraint(BaseSplineCost):

  def __init__(self, config):
    super().__init__()
    self.constraint = Bicycle5DConstraint(config)

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    state_cost = self.constraint.get_stage_cost(
        state, ctrl, closest_pt, slope, theta
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
      slope: DeviceArray, theta: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    state_cost = self.constraint.get_stage_cost(
        state, ctrl, closest_pt, slope, theta
    )
    ctrl_cost = self.ctrl_cost.get_stage_cost(state, ctrl)
    return state_cost + ctrl_cost

  @partial(jax.jit, static_argnames='self')
  def get_traj_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray
  ) -> float:
    state_costs = self.constraint.get_cost(
        state, ctrl, closest_pt, slope, theta
    )
    ctrl_costs = self.ctrl_cost.get_cost(state, ctrl)
    # TODO: critical points version
    return (jnp.max(state_costs[1:]) + jnp.sum(ctrl_costs)).astype(float)
