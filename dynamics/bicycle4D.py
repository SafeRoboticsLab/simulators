"""
Please contact the author(s) of this library if you have any questions.
Authors:  Zixu Zhang ( zixuz@princeton.edu )
          Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Tuple, Any
import numpy as np
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp

from .base_dynamics import BaseDynamics


class Bicycle4D(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    super().__init__(config, action_space)
    self.dim_x = 4  # [x, y, v, psi].

    # load parameters
    self.wheelbase: float = config.WHEELBASE  # vehicle chassis length

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """Clips the control and computes one-step time evolution of the system.

    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].

    Returns:
        DeviceArray: next state.
        DeviceArray: clipped control.
    """
    # Clips the controller values between min and max accel and steer values.
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    state_nxt = self._integrate_forward(state, ctrl_clip)
    state_nxt = state_nxt.at[3].set(
        jnp.mod(state_nxt[3] + jnp.pi, 2 * jnp.pi) - jnp.pi
    )
    return state_nxt, ctrl_clip

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: DeviceArray, control: DeviceArray
  ) -> DeviceArray:
    """Computes one-step time evolution of the system: x_+ = f(x, u).
    The discrete-time dynamics is as below:
        x_k+1 = x_k + v_k cos(psi_k) dt
        y_k+1 = y_k + v_k sin(psi_k) dt
        v_k+1 = v_k + u0_k dt
        psi_k+1 = psi_k + v_k tan(u1_k) / L dt

    Args:
        state (DeviceArray): [x, y, v, psi].
        control (DeviceArray): [accel, delta].

    Returns:
        DeviceArray: next state.
    """

    @jax.jit
    def _fwd_step(i, args):  # Euler method.
      _state, _ctrl = args

      d_x = (state[2] * jnp.cos(state[3])) * self.int_dt
      d_y = (state[2] * jnp.sin(state[3])) * self.int_dt
      d_v = _ctrl[0] * self.int_dt
      d_psi = (state[2] * jnp.tan(_ctrl[1]) / self.wheelbase) * self.int_dt
      return _state + jnp.array([d_x, d_y, d_v, d_psi]), _ctrl

    state_nxt = jax.lax.fori_loop(
        0, self.num_segment, _fwd_step, (state, control)
    )[0]
    return state_nxt
