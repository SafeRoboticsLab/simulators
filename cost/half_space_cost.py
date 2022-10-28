import numpy as np
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from functools import partial

from .base_cost import BaseCost


class UpperHalfCost(BaseCost):
  """
  c = x[dim] - val
  """

  def __init__(self, value: float, dim: int):
    super().__init__()
    self.dim = dim
    self.value = value

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    c = state[self.dim] - self.value
    return c


class LowerHalfCost(BaseCost):
  """
  c = val - x[dim]
  """

  def __init__(self, value: float, dim: int):
    super().__init__()
    self.dim = dim
    self.value = value

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    c = self.value - state[self.dim]
    return c


class UpperHalfBoxFootprintCost(BaseCost):

  def __init__(
      self, dim: str, value: float, state_box_limits: np.ndarray,
      x_dim: int = 0, y_dim: int = 1, yaw_dim: int = 3
  ):
    super().__init__()
    if dim == 'x':
      self.dim = 0
      self.state_ret_dim = x_dim
    else:
      self.dim = 1
      self.state_ret_dim = y_dim

    self.value = value
    self.offset = jnp.array([[state_box_limits[0], state_box_limits[2]],
                             [state_box_limits[0], state_box_limits[3]],
                             [state_box_limits[1], state_box_limits[2]],
                             [state_box_limits[1], state_box_limits[3]]])
    self.yaw_dim = yaw_dim

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    yaw = state[self.yaw_dim]
    rot_mat = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)],
                         [jnp.sin(yaw), jnp.cos(yaw)]])
    rot_offset = jnp.einsum("ik,jk->ji", rot_mat, self.offset)  # (4, 2)
    pos = rot_offset[:, self.dim] + state[self.state_ret_dim]  # (4, )
    c = jnp.max(pos - self.value)
    return c


class LowerHalfBoxFootprintCost(BaseCost):

  def __init__(
      self, dim: str, value: float, state_box_limits: np.ndarray,
      x_dim: int = 0, y_dim: int = 1, yaw_dim: int = 3
  ):
    super().__init__()
    if dim == 'x':
      self.dim = 0
      self.state_ret_dim = x_dim
    else:
      self.dim = 1
      self.state_ret_dim = y_dim
    self.value = value
    self.offset = jnp.array([[state_box_limits[0], state_box_limits[2]],
                             [state_box_limits[0], state_box_limits[3]],
                             [state_box_limits[1], state_box_limits[2]],
                             [state_box_limits[1], state_box_limits[3]]])
    self.yaw_dim = yaw_dim

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    yaw = state[self.yaw_dim]
    rot_mat = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)],
                         [jnp.sin(yaw), jnp.cos(yaw)]])
    rot_offset = jnp.einsum("ik,jk->ji", rot_mat, self.offset)  # (4, 2)
    pos = rot_offset[:, self.dim] + state[self.state_ret_dim]  # (4, )
    c = jnp.max(self.value - pos)
    return c
