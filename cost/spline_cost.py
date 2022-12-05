from abc import ABC, abstractmethod
from typing import Optional
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from functools import partial

from .base_cost import BaseCost
from ..race_car.track import Track


class BaseSplineCost(ABC):

  def __init__(self, config):
    super().__init__()
    self.config = copy.deepcopy(config)

    # System parameters.
    self.state_box_limits = np.asarray(config.STATE_BOX_LIMITS)

    # Racing cost parameters.
    self.track_width_right: float = config.TRACK_WIDTH_RIGHT
    self.track_width_left: float = config.TRACK_WIDTH_LEFT

  @abstractmethod
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    raise NotImplementedError

  @partial(jax.jit, static_argnames='self')
  def get_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    obj = jax.vmap(self.get_stage_cost, in_axes=(1, 1, 1, 1, 1, 1))
    return obj(state, ctrl, closest_pt, slope, theta, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_traj_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    obj = jax.vmap(self.get_stage_cost, in_axes=(1, 1, 1, 1, 1, 1))
    costs = obj(state, ctrl, closest_pt, slope, theta, time_indices)
    return jnp.sum(costs).astype(float)

  @partial(jax.jit, static_argnames='self')
  def get_cx(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cx = jax.vmap(
        jax.jacfwd(self.get_stage_cost, argnums=0), in_axes=(1, 1, 1, 1, 1, 1),
        out_axes=1
    )
    return _cx(state, ctrl, closest_pt, slope, theta, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cu(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cu = jax.vmap(
        jax.jacfwd(self.get_stage_cost, argnums=1), in_axes=(1, 1, 1, 1, 1, 1),
        out_axes=1
    )
    return _cu(state, ctrl, closest_pt, slope, theta, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cxx(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cxx = jax.vmap(
        jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=0), argnums=0),
        in_axes=(1, 1, 1, 1, 1, 1), out_axes=2
    )
    return _cxx(state, ctrl, closest_pt, slope, theta, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cuu(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cuu = jax.vmap(
        jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=1), argnums=1),
        in_axes=(1, 1, 1, 1, 1, 1), out_axes=2
    )
    return _cuu(state, ctrl, closest_pt, slope, theta, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cux(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cux = jax.vmap(
        jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=1), argnums=0),
        in_axes=(1, 1, 1, 1, 1, 1), out_axes=2
    )
    return _cux(state, ctrl, closest_pt, slope, theta, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cxu(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    return self.get_cux(state, ctrl, closest_pt, slope, theta, time_indices).T

  @partial(jax.jit, static_argnames='self')
  def get_derivatives(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    return (
        self.get_cx(state, ctrl, closest_pt, slope, theta, time_indices),
        self.get_cu(state, ctrl, closest_pt, slope, theta, time_indices),
        self.get_cxx(state, ctrl, closest_pt, slope, theta, time_indices),
        self.get_cuu(state, ctrl, closest_pt, slope, theta, time_indices),
        self.get_cux(state, ctrl, closest_pt, slope, theta, time_indices),
    )


class SplineBarrierCost(BaseSplineCost):

  def __init__(
      self, clip_min: Optional[float], clip_max: Optional[float], q1: float,
      q2: float, cost: BaseSplineCost
  ):
    super().__init__(cost.config)
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.q1 = q1
    self.q2 = q2
    self.cost = cost

  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray,
      slope: DeviceArray, theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    _cost = self.cost.get_stage_cost(
        state, ctrl, closest_pt, slope, theta, time_idx
    )
    return self.q1 * jnp.exp(
        self.q2 * jnp.clip(a=_cost, a_min=self.clip_min, a_max=self.clip_max)
    )


class SplineRoadBoundaryCost(BaseSplineCost):

  def __init__(
      self, config, x_dim: int = 0, y_dim: int = 1, yaw_dim: int = 3,
      buffer: float = 0.
  ):
    super().__init__(config)
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.yaw_dim = yaw_dim

    self.state_box_limits = config.STATE_BOX_LIMITS
    self.offset = jnp.array(
        [[self.state_box_limits[0], self.state_box_limits[2]],
         [self.state_box_limits[0], self.state_box_limits[3]],
         [self.state_box_limits[1], self.state_box_limits[2]],
         [self.state_box_limits[1], self.state_box_limits[3]]]
    )

    self.track_width_right = config.TRACK_WIDTH_RIGHT
    self.track_width_left = config.TRACK_WIDTH_LEFT
    self.buffer = buffer

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
    # Rotates the footprint offset
    yaw = state[self.yaw_dim]
    rot_mat_state = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)],
                               [jnp.sin(yaw), jnp.cos(yaw)]])
    rot_offset = jnp.einsum("ik,jk->ji", rot_mat_state, self.offset)  # (4, 2)
    pos = rot_offset + state[jnp.array([self.x_dim, self.y_dim])]

    # Treats the closest point as the origin and rotates clockwise.
    rot_mat_closest = jnp.array([
        [jnp.cos(slope[0]), jnp.sin(slope[0])],
        [-jnp.sin(slope[0]), jnp.cos(slope[0])],
    ])
    rot_shift_pos = jnp.einsum(
        "ik,jk->ji", rot_mat_closest, (pos - closest_pt)
    )
    cost_left = jnp.max(rot_shift_pos[:, 1] - self.track_width_left)
    cost_right = jnp.max(-self.track_width_right - rot_shift_pos[:, 1])
    return jnp.maximum(cost_left, cost_right) + self.buffer


class SplineYawCost(BaseSplineCost):

  def __init__(self, config, yaw_dim: int = 3):
    super().__init__(config)
    self.yaw_dim = yaw_dim
    self.yaw_min = config.YAW_MIN
    self.yaw_max = config.YAW_MAX
    self.bidirectional = getattr(config, "BIDIRECTIONAL", True)

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
    yaw = state[self.yaw_dim]
    diff1 = yaw - slope[0]
    cost1_1 = jnp.maximum(diff1 - self.yaw_max, self.yaw_min - diff1)
    cost1_2 = jnp.maximum(
        diff1 - 2 * jnp.pi - self.yaw_max, self.yaw_min - diff1 + 2 * jnp.pi
    )
    cost1_3 = jnp.maximum(
        diff1 + 2 * jnp.pi - self.yaw_max, self.yaw_min - diff1 - 2 * jnp.pi
    )
    cost1 = jnp.minimum(cost1_1, cost1_2)
    cost1 = jnp.minimum(cost1, cost1_3)

    if self.bidirectional:
      slope2 = jnp.mod(slope[0], 2 * jnp.pi) - jnp.pi
      diff2 = yaw - slope2
      cost2_1 = jnp.maximum(diff2 - self.yaw_max, self.yaw_min - diff2)
      cost2_2 = jnp.maximum(
          diff2 - 2 * jnp.pi - self.yaw_max, self.yaw_min - diff2 + 2 * jnp.pi
      )
      cost2_3 = jnp.maximum(
          diff2 + 2 * jnp.pi - self.yaw_max, self.yaw_min - diff2 - 2 * jnp.pi
      )
      cost2 = jnp.minimum(cost2_1, cost2_2)
      cost2 = jnp.minimum(cost2, cost2_3)
      return jnp.minimum(cost1, cost2)
    else:
      return cost1
