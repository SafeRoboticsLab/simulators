from typing import TypeVar, TypedDict, List, Any, Optional
import numpy as np
from gym import spaces

from base_dynamics import BaseDynamics
# TODO: add other simulators.
from race_car.bicycle_dynamics import BicycleDynamics


# Type Hints
class ActionZS(TypedDict):
  ctrl: np.ndarray
  dstb: np.ndarray


GenericAction = TypeVar(
    'GenericAction', np.ndarray, List[np.ndarray], ActionZS
)

GenericState = TypeVar('GenericState', np.ndarray, List[np.ndarray])


# TODO: add other simulators.
def get_agent(dyn: str, config: Any, action_space: np.ndarray) -> BaseDynamics:
  if dyn == "Bicycle":
    return BicycleDynamics(config, action_space)


def build_obs_space(
    obs_spec: np.ndarray, obs_dim: Optional[tuple] = None
) -> spaces.Box:
  if obs_spec.ndim == 2:  # e.g., state.
    obs_space = spaces.Box(low=obs_spec[:, 0], high=obs_spec[:, 1])
  elif obs_spec.ndim == 4:  # e.g., RGB-D.
    obs_space = spaces.Box(low=obs_spec[:, :, :, 0], high=obs_spec[:, :, :, 1])
  else:  # Each dimension shares the same min and max.
    assert obs_spec.ndim == 1, "Unsupported obs space spec!"
    assert obs_dim is not None, "Obs. dim is not provided"
    obs_space = spaces.Box(low=obs_spec[0], high=obs_spec[1], shape=obs_dim)
  return obs_space


def concatenate_obs(observations: List[np.ndarray]) -> np.ndarray:
  base_shape = observations[0].shape[1:]
  flags = np.array([x.shape[1:] == base_shape for x in observations])
  assert np.all(flags), (
      "The obs. of each agent should be the same except the first dim!"
  )
  return np.concatenate(observations)
