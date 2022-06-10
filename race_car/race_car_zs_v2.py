"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import matplotlib

from .race_car_v2 import RaceCarEnvV2
from ..base_zs_env import BaseZeroSumEnv
from ..utils import ActionZS


class RaceCarZeroSumEnvV2(BaseZeroSumEnv, RaceCarEnvV2):
  """
  A wrapper for a zero-sum game env with a physical agent of bicycle dynamics
  version 2.

  Args:
      BaseZeroSumEnv
      RaceCarEnvV2
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    assert config_env.NUM_AGENTS == 2, "This is a zero-sum game!"
    BaseZeroSumEnv.__init__(self, config_env, config_agent)
    RaceCarEnvV2.__init__(self, config_env, config_agent)

  def seed(self, seed: int = 0):
    BaseZeroSumEnv.seed(self, seed)
    RaceCarEnvV2.seed(self, seed)

  def reset(
      self, state: Optional[np.ndarray] = None, cast_torch: bool = False,
      **kwargs
  ) -> Union[np.ndarray, torch.FloatTensor]:
    BaseZeroSumEnv.reset(self, state, cast_torch, **kwargs)
    return RaceCarEnvV2.reset(self, state, cast_torch, **kwargs)

  def get_cost(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray,
      constraints: Optional[dict] = None
  ) -> float:
    return RaceCarEnvV2._get_cost(
        self, state, action['ctrl'], state_nxt, constraints
    )

  def get_constraints(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray
  ) -> Dict:
    return RaceCarEnvV2._get_constraints(
        self, state, action['ctrl'], state_nxt
    )

  def get_target_margin(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray
  ) -> Dict:
    return RaceCarEnvV2._get_target_margin(
        self, state, action['ctrl'], state_nxt
    )

  def get_done_and_info(
      self, constraints: Dict, targets: Optional[Dict] = None,
      final_only: bool = True, end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    return RaceCarEnvV2._get_done_and_info(
        self, constraints, targets, final_only, end_criterion
    )

  def get_obs(self, state: np.ndarray) -> np.ndarray:
    return RaceCarEnvV2._get_obs(self, state)

  def render(
      self, ax: Optional[matplotlib.axes.Axes] = None, c_track: str = 'k',
      c_obs: str = 'r', c_ego: str = 'b', s: float = 12
  ):
    return RaceCarEnvV2._render(self, ax, c_track, c_obs, c_ego, s)

  def report(self):
    print(
        "This is a zero-sum game environment of the Race Car simulator based"
        + "on bicycle dynamics version 2 and ellipse footprint."
    )
    RaceCarEnvV2._report(self)
