from base_zs_env import BaseZeroSumEnv
from spirit_rl.spirit_pybullet_env import SpiritPybulletEnv
from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import matplotlib
from ..utils import ActionZS

class SpiritPybulletZeroSumEnv(BaseZeroSumEnv, SpiritPybulletEnv):
    def __init__(self, config_env: Any, config_agent: Any) -> None:
        assert config_env.NUM_AGENTS == 2, "This is a zero-sum game!"
        BaseZeroSumEnv.__init__(self, config_env, config_agent)
        SpiritPybulletEnv.__init__(self, config_env, config_agent)
    
    def seed(self, seed: int = 0):
        BaseZeroSumEnv.seed(self, seed)
        SpiritPybulletEnv.seed(self, seed)
    
    def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False, **kwargs) -> Union[np.ndarray, torch.FloatTensor]:
        BaseZeroSumEnv.reset(self, state, cast_torch, **kwargs)
        return SpiritPybulletEnv.reset(self, state, cast_torch, **kwargs)
    
    def get_cost(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray, constraints: Optional[Dict] = None) -> float:
        return SpiritPybulletEnv.get_cost(self, state, action['ctrl'], state_nxt, constraints)
    
    def get_constraints(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        return SpiritPybulletEnv.get_constraints(self, state, action['ctrl'], state_nxt)
    
    def get_target_margin(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        return SpiritPybulletEnv.get_target_margin(self, state, action, state_nxt)
    
    def get_done_and_info(
        self, constraints: Dict, targets: Optional[Dict] = None,
        final_only: bool = True, end_criterion: Optional[str] = None
    ) -> Tuple[bool, Dict]:
        return SpiritPybulletEnv._get_done_and_info(
            self, constraints, targets, final_only, end_criterion
        )

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return SpiritPybulletEnv._get_obs(self, state)
    
    def render(self):
        return super().render()
    
    def report(self):
        return super().report()

