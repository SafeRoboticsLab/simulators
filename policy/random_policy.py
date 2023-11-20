from typing import Tuple, Optional, Union
import numpy as np
import torch
import time

from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
  policy_type = "random"

  def __init__(self, id: str, action_range: np.ndarray, seed: int) -> None:
    super().__init__(id)
    self.action_range = np.array(action_range, dtype=np.float32)
    self.rng = np.random.default_rng(seed=seed)

  @property
  def is_stochastic(self) -> bool:
    return True

  def get_action(self, obsrv: np.ndarray, num: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
    if num is None:
      size = self.action_range.shape[0]
    else:
      size = (num, self.action_range.shape[0])

    time0 = time.time()
    action = self.rng.uniform(low=self.action_range[:, 0], high=self.action_range[:, 1], size=size)
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)

  def sample(self, obsrv: Union[np.ndarray, torch.Tensor],
             **kwargs) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    size = obsrv.shape[0]
    action = self.rng.uniform(low=self.action_range[:, 0], high=self.action_range[:, 1], size=size)
    log_prob_one = np.sum(np.log(1 / (self.action_range[:, 1] - self.action_range[:, 0])))

    if isinstance(obsrv, torch.Tensor):
      action = torch.from_numpy(action).to(obsrv.device)
      log_prob = torch.full(size=(size,), fill_value=log_prob_one, device=obsrv.device)
    else:
      log_prob = np.full(shape=(size,), fill_value=log_prob_one)

    return action, log_prob
