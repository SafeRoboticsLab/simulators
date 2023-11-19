# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Optional, Dict, Union
import time
import numpy as np
import torch

from .base_policy import BasePolicy


class NeuralNetworkControlSystem(BasePolicy):
  net: torch.nn.Module
  policy_type: str = "NNCS"

  def update_policy(self, nncs: NeuralNetworkControlSystem):
    self.net.load_state_dict(nncs.net.state_dict())

  def get_action(
      self, obsrv: Union[np.ndarray, torch.Tensor],
      append: Optional[Union[np.ndarray, torch.Tensor]] = None,
      latent: Optional[Union[np.ndarray, torch.Tensor]] = None, **kwargs
  ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict]:
    time0 = time.time()
    with torch.no_grad():
      action = self.net(obsrv, append=append, latent=latent)
    if isinstance(action, torch.Tensor):
      action = action.cpu().numpy()
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)
