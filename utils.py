from typing import TypeVar, TypedDict, List, Any
import numpy as np

from base_dynamics import BaseDynamics
# TODO: add other simulators.
from race_car.bicycle_dynamics import BicycleDynamics


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
