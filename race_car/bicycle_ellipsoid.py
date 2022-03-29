from typing import Any, List, Union
import numpy as np

from ..ell_reach.ellipsoid import Ellipsoid
from ..ell_reach.ellipse import Ellipse
from .bicycle_dynamics import BicycleDynamics


class BicycleEllipsoid(BicycleDynamics):
  """Implements an agent with bicycle dynamics and ellipse footprint.
  """

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    super().__init__(config, action_space)
    ego_a = config.LENGTH / 2.0
    ego_b = config.WIDTH / 2.0
    ego_q = np.array([self.wheelbase / 2, 0])[:, np.newaxis]
    ego_Q = np.diag([ego_a**2, ego_b**2])
    self.footprint = Ellipse(q=ego_q, Q=ego_Q)

  def state2ell(self, state: np.ndarray) -> Union[Ellipse, List[Ellipse]]:
    if state.ndim == 1:
      theta = state[3]
      d = state[:2][:, np.newaxis]
      R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
      return self.footprint @ R + d
    else:
      assert state.ndim == 2, "State shape is incorrect!"
      ego_ell = []
      for i in range(state.shape[1]):
        theta = state[3, i]
        d = state[:2, i][:, np.newaxis]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        ego_ell.append(self.footprint @ R + d)
      return ego_ell
