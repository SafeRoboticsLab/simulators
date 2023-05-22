from ..base_zs_env import BaseZeroSumEnv
from .gvr_pybullet_env import GVRPybulletEnv
from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import matplotlib
from ..utils import ActionZS

class GVRPybulletZeroSumEnv(BaseZeroSumEnv, GVRPybulletEnv):
    """
    A wrapper for a zero-sum game env with a physical agent of using Spirit Pybullet dynamics

    Args:
        BaseZeroSumEnv
        GVRPybulletEnv
    """

    def __init__(self, config_env: Any, config_agent: Any) -> None:
        assert config_env.NUM_AGENTS == 2, "This is a zero-sum game!"
        BaseZeroSumEnv.__init__(self, config_env, config_agent)
        GVRPybulletEnv.__init__(self, config_env, config_agent)
    
    def seed(self, seed: int = 0):
        BaseZeroSumEnv.seed(self, seed)
        GVRPybulletEnv.seed(self, seed)
    
    def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False, **kwargs) -> Union[np.ndarray, torch.FloatTensor]:
        BaseZeroSumEnv.reset(self, state, cast_torch, **kwargs)
        self.agent.dyn.reset(**kwargs)
        obs = self.get_obs(None)

        self.state = obs.copy()
        
        if cast_torch:
            obs = torch.FloatTensor(obs)
        return obs
    
    def get_cost(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray, constraints: Optional[Dict] = None) -> float:
        return 0
    
    def get_constraints(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        return self.agent.dyn.get_constraints()
    
    def get_target_margin(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        return self.agent.dyn.get_target_margin()

    def get_done_and_info(self, state: np.ndarray, constraints: Dict, targets: Dict, final_only: bool = True, end_criterion: Optional[str] = None) -> Tuple[bool, Dict]:
        if end_criterion is None:
            end_criterion = self.end_criterion
        
        done = False
        done_type = "not_raised"
        if self.cnt >= self.timeout:
            done = True
            done_type = "timeout"
        
        g_x = max(list(constraints.values()))
        l_x = max(list(targets.values()))
        binary_cost = 1. if g_x > 0. else 0.

        #! FORCE EXPLORING FOR n STEPS BEFORE TERMINATING
        if self.cnt >= 300:
            # Gets done flag
            if end_criterion == 'failure':
                failure = g_x > 0
                if failure:
                    done = True
                    done_type = "failure"
            elif end_criterion == 'reach-avoid':
                failure = g_x > 0.
                success = not failure and l_x <= 0.
                
                if success:
                    done = True
                    done_type = "success"
                elif failure:
                    done = True
                    done_type = "failure"
            elif end_criterion == 'timeout':
                pass
            else:
                raise ValueError("End criterion not supported!")

        # Gets info
        info = {
            "done_type": done_type,
            "g_x": g_x,
            "l_x": l_x,
            "binary_cost": binary_cost
        }

        return done, info

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return self.agent.dyn.state
    
    def render(self):
        return super().render()
    
    def report(self):
        return super().report()

