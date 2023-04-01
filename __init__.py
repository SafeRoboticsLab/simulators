from simulators.agent import Agent

from simulators.spirit_rl.spirit_pybullet_env import SpiritPybulletEnv
from simulators.spirit_rl.spirit_pybullet_single import SpiritPybulletSingleEnv
from simulators.spirit_rl.spirit_pybullet_zs import SpiritPybulletZeroSumEnv
from simulators.base_single_env import BaseSingleEnv
from simulators.base_zs_env import BaseZeroSumEnv

from simulators.race_car.track import Track
from simulators.race_car.race_car_single import RaceCarSingle5DEnv
from simulators.race_car.race_car_dstb import RaceCarDstb5DEnv
from simulators.race_car.cost_bicycle5D import (
    Bicycle5DCost, Bicycle5DConstraint, Bicycle5DReachabilityCost,
    Bicycle5DRefTrajCost
)

from simulators.ell_reach.ellipse import Ellipse
from simulators.ell_reach.plot_ellipsoids import plot_ellipsoids

from simulators.cost.quadratic_cost import QuadraticCost
from simulators.cost.half_space_cost import (
    UpperHalfCost, LowerHalfCost, UpperHalfBoxFootprintCost,
    LowerHalfBoxFootprintCost
)
from simulators.cost.base_cost import BarrierCost, BaseCost
from simulators.cost.box_cost import BoxObsCost, BoxObsBoxFootprintCost

from simulators.dynamics.bicycle4D import Bicycle4D
from simulators.dynamics.bicycle5D import Bicycle5D

from simulators.policy.base_policy import BasePolicy
from simulators.policy.nn_policy import NeuralNetworkControlSystem
from simulators.policy.ilqr_policy import iLQR
from simulators.policy.ilqr_spline_policy import iLQRSpline
from simulators.policy.linear_policy import LinearPolicy

from simulators.vec_env.subproc_vec_env import SubprocVecEnv
from simulators.vec_env.vec_env import VecEnvBase

from simulators.utils import (
    save_obj, load_obj, PrintLogger, parallel_apply, parallel_iapply
)

import gym

gym.envs.register(  # no time limit imposed
    id='RaceCarSingle5DEnv-v1',
    entry_point=RaceCarSingle5DEnv,
)

gym.envs.register(  # no time limit imposed
    id='RaceCarDstb5DEnv-v1',
    entry_point=RaceCarDstb5DEnv,
)
