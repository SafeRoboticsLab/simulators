from typing import Any, Tuple, Optional, Callable, List, Dict, Union
import numpy as np
from gym import spaces
import torch

from base_single_env import BaseSingleEnv

class SpiritPybulletEnv(BaseSingleEnv):
    def __init__(self, config_env: Any, config_agent: Any) -> None:
        assert config_env.NUM_AGENTS == 1, "This environment only has one agent!"
        super().__init__(config_env, config_agent)

    def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False, **kwargs) -> Union[np.ndarray, torch.FloatTensor]:
        super().reset()
        self.agent.dyn.reset()
        obs = self.get_obs()
        
        if cast_torch:
            obs = torch.FloatTensor(obs)
        return obs

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return self.agent.dyn.state
    
    def get_cost(self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray, constraints: Optional[Dict] = None) -> float:
        # TODO: Write this function
        return super().get_cost(state, action, state_nxt, constraints)

    def get_constraints(self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray) -> Dict:
        return self.agent.get_constraints()

    def get_target_margin(self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray) -> Dict:
        return super().get_target_margin()

    def get_done_and_info(self, constraints: Dict, targets: Dict, final_only: bool = True, end_criterion: Optional[str] = None) -> Tuple[bool, Dict]:
        
        if end_criterion is None:
            end_criterion = self.end_criterion
        
        done = False
        done_type = "not_raised"
        if self.cnt >= self.timeout:
            done = True
            done_type = "timeout"

        # Retrieves constraints / traget values.
        constraint_values = None
        for key, value in constraints.items():
            if constraint_values is None:
                num_pts = value.shape[1]
                constraint_values = value
            else:
                assert num_pts == value.shape[1], (
                    "The length of constraint ({}) do not match".format(key)
                )
                constraint_values = np.concatenate((constraint_values, value), axis=0)
        target_values = None
        for key, value in targets.items():
            assert num_pts == value.shape[1], (
                "The length of target ({}) do not match".format(key)
            )
            if target_values is None:
                target_values = value
            else:
                target_values = np.concatenate((target_values, value), axis=0)

        # Gets info.
        g_x_list = np.max(constraint_values, axis=0)
        l_x_list = np.max(target_values, axis=0)
        
        if final_only:
            g_x = g_x_list[-1]
            l_x = l_x_list[-1]
            binary_cost = 1. if g_x > 0. else 0.
        else:
            g_x = g_x_list
            l_x = l_x_list
            binary_cost = 1. if np.any(g_x > 0.) else 0.

        # Gets done flag
        if end_criterion == 'failure':
            if final_only:
                failure = np.any(constraint_values[:, -1] > 0.)
            else:
                failure = np.any(constraint_values > 0.)
            if failure:
                done = True
                done_type = "failure"
                g_x = self.g_x_fail
        elif end_criterion == 'reach-avoid':
            if final_only:
                failure = g_x > 0.
                success = not failure and l_x <= 0.
            else:
                v_x_list = np.empty(shape=(num_pts,))
                v_x_list[num_pts
                        - 1] = max(l_x_list[num_pts - 1], g_x_list[num_pts - 1])
                for i in range(num_pts - 2, -1, -1):
                    v_x_list[i] = max(g_x_list[i], min(l_x_list[i], v_x_list[i + 1]))
                inst = np.argmin(v_x_list)
                failure = np.any(constraint_values[:, :inst + 1] > 0.)
                success = not failure and (v_x_list[inst] <= 0)
            if success:
                done = True
                done_type = "success"
            elif failure:
                done = True
                done_type = "failure"
                g_x = self.g_x_fail
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
        if self.step_keep_constraints:
            info['constraints'] = constraints
        if self.step_keep_targets:
            info['targets'] = targets
        return done, info