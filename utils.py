"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from __future__ import annotations
from typing import TypeVar, TypedDict, List, Any, Optional, Union, Tuple
import numpy as np
from gym import spaces
import torch
import pickle


def save_obj(obj, filename, protocol=None):
  if protocol is None:
    protocol = pickle.HIGHEST_PROTOCOL
  with open(filename + '.pkl', 'wb') as f:
    pickle.dump(obj, f, protocol=protocol)


def load_obj(filename):
  with open(filename + '.pkl', 'rb') as f:
    return pickle.load(f)


# Type Hints
class ActionZS(TypedDict):
  ctrl: np.ndarray
  dstb: np.ndarray


GenericAction = TypeVar(
    'GenericAction', np.ndarray, List[np.ndarray], ActionZS
)

GenericState = TypeVar(
    'GenericState', torch.FloatTensor, np.ndarray, List[torch.FloatTensor],
    List[np.ndarray]
)


# Observation.
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


# Math Operator.
def barrier_function(
    q1: float, q2: float, cons: np.ndarray | float, cons_dot: np.ndarray,
    cons_min: Optional[float] = None, cons_max: Optional[float] = None
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
  clip = not (cons_min is None and cons_max is None)
  if clip:
    tmp = np.clip(q2 * cons, cons_min, cons_max)
  else:
    tmp = q2 * cons
  b = q1 * (np.exp(tmp))

  if isinstance(cons, np.ndarray):
    b = b.reshape(-1)
    assert b.shape[0] == cons_dot.shape[1], (
        "The shape of cons and cons_dot don't match!"
    )
    b_dot = np.einsum('n,an->an', q2 * b, cons_dot)
    b_ddot = np.einsum(
        'n,abn->abn', (q2**2) * b, np.einsum('an,bn->abn', cons_dot, cons_dot)
    )
  elif isinstance(cons, float):
    cons_dot = cons_dot.reshape(-1, 1)  # Transforms to column vector.
    b_dot = q2 * b * cons_dot
    b_ddot = (q2**2) * b * np.einsum('ik,jk->ij', cons_dot, cons_dot)
  else:
    raise TypeError("The type of cons is not supported!")
  return b_dot, b_ddot
