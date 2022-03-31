from typing import Optional
import csv
import numpy as np


def get_centerline_from_traj(filepath: str) -> np.ndarray:
  """
  Gets the centerline of the track from the trajectory data. We currently only
  support 2D track.

  Args:
      filepath (str): the path to file consisting of the centerline position.

  Returns:
      np.ndarray: centerline, of the shape (2, N).
  """
  x = []
  y = []
  with open(filepath) as f:
    spamreader = csv.reader(f, delimiter=',')
    for i, row in enumerate(spamreader):
      if i > 0:
        x.append(float(row[0]))
        y.append(float(row[1]))

  return np.array([x, y])
