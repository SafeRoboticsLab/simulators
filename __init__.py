from .config.utils import load_config

from .agent import Agent

from .base_single_env import BaseSingleEnv
from .base_zs_env import BaseZeroSumEnv

from .race_car.track import Track
from .race_car.constraints_bicycle_v1 import ConstraintsBicycleV1
from .race_car.constraints_bicycle_v2 import ConstraintsBicycleV2
from .race_car.race_car_single_v1 import RaceCarSingleEnvV1
from .race_car.race_car_single_v2 import RaceCarSingleEnvV2
from .race_car.race_car_zs_v2 import RaceCarZeroSumEnvV2

from .ell_reach.ellipse import Ellipse
from .ell_reach.plot_ellipsoids import plot_ellipsoids

from .utils import save_obj

import gym

gym.envs.register(  # no time limit imposed
    id='RaceCarSingleEnv-v1',
    entry_point=RaceCarSingleEnvV1,
)

gym.envs.register(  # no time limit imposed
    id='RaceCarSingleEnv-v2',
    entry_point=RaceCarSingleEnvV2,
)

gym.envs.register(  # no time limit imposed
    id='RaceCarZeroSumEnv-v2',
    entry_point=RaceCarZeroSumEnvV2,
)
