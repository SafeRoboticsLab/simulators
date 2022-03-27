from typing import TypeVar, TypedDict, List, Any
import numpy as np

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


def concatenate_obs(observations: List[np.ndarray]) -> np.ndarray:
  base_shape = observations[0].shape[1:]
  flags = np.array([x.shape[1:] == base_shape for x in observations])
  assert np.all(flags), (
      "The obs. of each agent should be the same except the first dim!"
  )
  return np.concatenate(observations)
