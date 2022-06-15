"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import matplotlib

from .race_car_v1 import RaceCarEnvV1
from ..base_single_env import BaseSingleEnv


class RaceCarSingleEnvV1(BaseSingleEnv, RaceCarEnvV1):
  """
  A wrapper for an env with a single agent with bicycle dynamics v1.

  Args:
      BaseSingleEnv
      RaceCarEnvV1
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    assert config_env.NUM_AGENTS == 1, "This environment only has one agent!"
    BaseSingleEnv.__init__(self, config_env, config_agent)
    RaceCarEnvV1.__init__(self, config_env, config_agent)

  def seed(self, seed: int = 0):
    BaseSingleEnv.seed(self, seed)
    RaceCarEnvV1.seed(self, seed)

  def reset(
      self, state: Optional[np.ndarray] = None, cast_torch: bool = False,
      **kwargs
  ) -> Union[np.ndarray, torch.FloatTensor]:
    BaseSingleEnv.reset(self, state, cast_torch, **kwargs)
    return RaceCarEnvV1.reset(self, state, cast_torch, **kwargs)

  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Optional[dict] = None
  ) -> float:
    return RaceCarEnvV1._get_cost(self, state, action, state_nxt, constraints)

  def get_constraints(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    return RaceCarEnvV1._get_constraints(self, state, action, state_nxt)

  def get_target_margin(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    return RaceCarEnvV1._get_target_margin(self, state, action, state_nxt)

  def get_done_and_info(
      self, constraints: Dict, targets: Optional[Dict] = None,
      final_only: bool = True, end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    return RaceCarEnvV1._get_done_and_info(
        self, constraints, targets, final_only, end_criterion
    )

  def get_obs(self, state: np.ndarray) -> np.ndarray:
    return RaceCarEnvV1._get_obs(self, state)

  def render(
      self, ax: Optional[matplotlib.axes.Axes] = None, c_track: str = 'k',
      c_obs: str = 'r', c_ego: str = 'b', s: float = 12
  ):
    return RaceCarEnvV1._render(self, ax, c_track, c_obs, c_ego, s)

  def report(self):
    print(
        "This is a single Race Car simulator based on bicycle dynamics "
        + "version 2 and ellipse footprint."
    )
    RaceCarEnvV1._report(self)
