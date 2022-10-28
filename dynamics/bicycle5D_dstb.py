"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Tuple, Any, Dict
import numpy as np
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp

from .base_dstb_dynamics import BaseDstbDynamics


class BicycleDstb5D(BaseDstbDynamics):

  def __init__(self, config: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    super().__init__(config, action_space)
    self.dim_x = 5  # [x, y, v, psi, delta].

    # load parameters
    self.wheelbase: float = config.WHEELBASE  # vehicle chassis length

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """Clips the control and disturbance and computes one-step time evolution
    of the system.

    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].
        disturbance (DeviceArray): [x, y, v, psi, delta].

    Returns:
        DeviceArray: next state.
        DeviceArray: clipped control.
        DeviceArray: clipped disturbance.
    """
    # Clips the controller values between min and max accel and steer values.
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    dstb_clip = jnp.clip(
        disturbance, self.dstb_space[:, 0], self.dstb_space[:, 1]
    )
    state_nxt = self._integrate_forward(state, ctrl_clip, dstb_clip)
    state_nxt = state_nxt.at[3].set(
        jnp.mod(state_nxt[3] + jnp.pi, 2 * jnp.pi) - jnp.pi
    )
    return state_nxt, ctrl_clip, dstb_clip

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> DeviceArray:
    """
    Computes one-step time evolution of the system: x_+ = f(x, u, d).
    The discrete-time dynamics is as below
        x_k+1 = x_k + v_k cos(psi_k) dt + d0_k dt
        y_k+1 = y_k + v_k sin(psi_k) dt + d1_k dt
        v_k+1 = v_k + u0_k dt + d2_k dt
        psi_k+1 = psi_k + v_k tan(delta_k) / L dt + d3_k dt
        delta_k+1 = delta_k + u1_k dt + d4_k dt

    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].
        disturbance (DeviceArray): [x, y, v, psi, delta].

    Returns:
        DeviceArray: next state.
    """

    @jax.jit
    def _fwd_step(i, args):  # Euler method.
      _state, _ctrl, _dstb = args
      d_x = (state[2] * jnp.cos(state[3]) + _dstb[0]) * self.int_dt
      d_y = (state[2] * jnp.sin(state[3]) + _dstb[1]) * self.int_dt
      d_v = (_ctrl[0] + _dstb[2]) * self.int_dt
      d_psi = (
          state[2] * jnp.tan(state[4]) / self.wheelbase + _dstb[3]
      ) * self.int_dt
      d_delta = (_ctrl[1] + _dstb[4]) * self.int_dt
      return _state + jnp.array([d_x, d_y, d_v, d_psi, d_delta]), _ctrl, _dstb

    state_nxt = jax.lax.fori_loop(
        0, self.num_segment, _fwd_step, (state, control, disturbance)
    )[0]
    return state_nxt
