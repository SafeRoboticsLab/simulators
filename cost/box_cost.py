import numpy as np
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from functools import partial

from .base_cost import BaseCost


class BoxObsCost(BaseCost):
  """
  We want s[i] < lb[i] or s[i] > ub[i].
  """

  def __init__(
      self, upper_bound: np.ndarray, lower_bound: np.ndarray,
      buffer: float = 0.
  ):
    super().__init__()
    self.ub = jnp.array(upper_bound)
    self.lb = jnp.array(lower_bound)
    self.n_dims = len(upper_bound)
    self.buffer = buffer

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    # signed distance to the box, negative inside
    sgn_dist = jnp.maximum(self.lb[0] - state[0], state[0] - self.ub[0])
    for i in range(1, self.n_dims):
      sgn_dist = jnp.maximum(sgn_dist, self.lb[i] - state[i])
      sgn_dist = jnp.maximum(sgn_dist, state[i] - self.ub[i])
    c = -sgn_dist
    return c + self.buffer


class BoxObsBoxFootprintCost(BaseCost):

  def __init__(
      self, state_box_limits: np.ndarray, box_spec: np.ndarray,
      precision: np.ndarray, x_dim: int = 0, y_dim: int = 1, yaw_dim: int = 3,
      buffer: float = 0.
  ):
    super().__init__()
    offset_xs = np.linspace(
        state_box_limits[0], state_box_limits[1], precision[0]
    )
    offset_ys = np.linspace(
        state_box_limits[2], state_box_limits[3], precision[1]
    )
    offset_xv, offset_yv = np.meshgrid(offset_xs, offset_ys, indexing='ij')
    offset = np.concatenate(
        (offset_xv[..., np.newaxis], offset_yv[..., np.newaxis]), axis=-1
    )
    self.offset = jnp.array(offset.reshape(-1, 2))
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.yaw_dim = yaw_dim

    # Box obstacle
    self.box_center = jnp.array([box_spec[0], box_spec[1]])
    self.box_yaw = box_spec[2]
    # rotate clockwise (to move the world frame to obstacle frame)
    self.obs_rot_mat = jnp.array([[
        jnp.cos(self.box_yaw), jnp.sin(self.box_yaw)
    ], [-jnp.sin(self.box_yaw), jnp.cos(self.box_yaw)]])
    self.box_halflength = box_spec[3]
    self.box_halfwidth = box_spec[4]
    self.buffer = buffer

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    yaw = state[self.yaw_dim]
    rot_mat = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)],
                         [jnp.sin(yaw), jnp.cos(yaw)]])
    rot_offset = jnp.einsum("ik,jk->ji", rot_mat, self.offset)
    pos = rot_offset + jnp.array([state[self.x_dim], state[self.y_dim]])
    pos_final = jnp.einsum(
        "ik,jk->ji", self.obs_rot_mat, pos - self.box_center
    )
    diff_x = jnp.minimum(
        self.box_halflength - pos_final[..., 0],
        pos_final[..., 0] + self.box_halflength
    )
    diff_y = jnp.minimum(
        self.box_halfwidth - pos_final[..., 1],
        pos_final[..., 1] + self.box_halfwidth
    )
    diff = jnp.minimum(diff_x, diff_y)
    c = jnp.max(diff)
    return c + self.buffer
