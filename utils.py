from typing import TypeVar, TypedDict, List
import numpy as np


class ActionZS(TypedDict):
  ctrl: np.ndarray
  dstb: np.ndarray


GenericAction = TypeVar(
    'GenericAction', np.ndarray, List[np.ndarray], ActionZS
)

GenericState = TypeVar('GenericState', np.ndarray, List[np.ndarray])
