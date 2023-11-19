from typing import Tuple, Optional
import numpy as np
import time

from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):

  def __init__(self, id: str, action_range: np.ndarray, seed: int) -> None:
    super().__init__(id)
    self.action_range = np.array(action_range, dtype=np.float32)
    self.rng = np.random.default_rng(seed=seed)

  def get_action(self, obsrv: np.ndarray, num: Optional[int] = None,
                 **kwargs) -> Tuple[np.ndarray, dict]:
    if num is None:
      size = self.action_range.shape[0]
    else:
      size = (num, self.action_range.shape[0])

    time0 = time.time()
    action = self.rng.uniform(
        low=self.action_range[:, 0], high=self.action_range[:, 1], size=size
    )
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)
