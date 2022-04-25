"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from IPython.display import Image

from simulators import RaceCarSingleEnv, load_config
from simulators.ell_reach.ellipse import Ellipse
from simulators.ell_reach.plot_ellipsoids import plot_ellipsoids
from utils import save_obj

# region: Sets environment
# Loads config.
config_path = os.path.join("simulators", "race_car", "race_car_env.yaml")
config = load_config(config_path)
config_env = config['environment']
config_agent = config['agent']
config_solver = config['solver']

# Constructs a static obstacle.
ego_a = config_agent.LENGTH / 2.0
ego_b = config_agent.WIDTH / 2.0
ego_q = np.array([0, 5.6])[:, np.newaxis]
ego_Q = np.diag([ego_a**2, ego_b**2])
static_obs = Ellipse(q=ego_q, Q=ego_Q)

env = RaceCarSingleEnv(config_env, config_agent)

pos0, psi0 = env.track.interp([2])  # The position and yaw on the track.
x_cur = np.array([pos0[0], pos0[1], 0, psi0[-1]])
env.reset(x_cur)

static_obs_list = [static_obs for _ in range(2)]
env.constraints.update_obs([static_obs_list])
# endregion

# region: Constructs placeholder and initializes iLQR
config_env_imaginary = copy.deepcopy(config_env)
config_env_imaginary.INTEGRATE_KWARGS = config_agent.INTEGRATE_KWARGS
config_env_imaginary.USE_SOFT_CONS_COST = config_agent.USE_SOFT_CONS_COST
env_imaginary = RaceCarSingleEnv(config_env_imaginary, config_agent)
static_obs_list = [static_obs for _ in range(config_solver.N)]
env_imaginary.constraints.update_obs([static_obs_list])
env.agent.init_policy(
    policy_type="iLQR", env=env_imaginary, config=config_solver
)
init_control = np.zeros((2, config_solver.N - 1))

max_iter_receding = config_solver.MAX_ITER_RECEDING

fig_folder = os.path.join("experiments", "ilqr", "figure")
fig_prog_folder = os.path.join(fig_folder, "progress")
os.makedirs(fig_prog_folder, exist_ok=True)
# endregion

# region: Runs iLQR
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
c_track = 'k'
c_obs = 'r'
c_ego = 'b'


def rollout_step_callback(
    env, state_hist, action_hist, plan_hist, step_hist, *args, **kwargs
):
  solver_info = plan_hist[-1]
  ax.cla()

  # track.
  env.track.plot_track(ax, c=c_track)
  plot_ellipsoids(
      ax, static_obs_list[0:1], arg_list=[dict(c=c_obs, linewidth=1.)],
      dims=[0, 1], N=50, plot_center=False, use_alpha=True
  )

  # agent.
  ego = env.agent.footprint.move2state(solver_info['states'][[0, 1, 3], 0])
  plot_ellipsoids(
      ax, [ego], arg_list=[dict(c=c_ego)], dims=[0, 1], N=50, plot_center=False
  )

  # plan.
  ax.plot(
      solver_info['states'][0, :], solver_info['states'][1, :], linewidth=2,
      c=c_ego
  )

  # history.
  states = np.array(state_hist).T  # last one is the next state.
  sc = ax.scatter(
      states[0, :-1], states[1, :-1], s=24, c=states[2, :-1], cmap=cm.jet,
      vmin=0, vmax=config_agent.V_MAX, edgecolor='none', marker='o'
  )
  cbar = fig.colorbar(sc, ax=ax)
  cbar.set_label(r"velocity [$m/s$]", size=20)
  fig.savefig(
      os.path.join(fig_prog_folder,
                   str(states.shape[1] - 1) + ".png"), dpi=200
  )
  cbar.remove()

  print(
      "[{}]: solver returns status {} and uses {:.3f}.".format(
          states.shape[1] - 1, solver_info['status'], solver_info['t_process']
      ), end='\r'
  )


def rollout_episode_callback(
    env, state_hist, action_hist, plan_hist, step_hist, *args, **kwargs
):
  ax.cla()

  # track.
  env.track.plot_track(ax, c=c_track)
  plot_ellipsoids(
      ax, static_obs_list[0:1], arg_list=[dict(c=c_obs, linewidth=1.)],
      dims=[0, 1], N=50, plot_center=False, use_alpha=True
  )

  # agent.
  ego = env.agent.footprint.move2state(state_hist[-1][[0, 1, 3]])
  plot_ellipsoids(
      ax, [ego], arg_list=[dict(c=c_ego)], dims=[0, 1], N=50, plot_center=False
  )

  states = np.array(state_hist).T
  sc = ax.scatter(
      states[0, :], states[1, :], s=24, c=states[2, :], cmap=cm.jet, vmin=0,
      vmax=config_agent.V_MAX, edgecolor='none', marker='o'
  )
  cbar = fig.colorbar(sc, ax=ax)
  cbar.set_label(r"velocity [$m/s$]", size=20)
  fig.savefig(os.path.join(fig_folder, "final.png"), dpi=200)
  cbar.remove()
  t_process = 0.

  for solver_info in plan_hist:
    t_process += solver_info['t_process']
  print("\n\n --> Planning uses {:.3f}.".format(t_process))

  final_step_info = step_hist[-1]
  if final_step_info["done_type"] == "failure":
    print("The rollout fails!")
    constraint_dict = env.get_constraints(
        state=state_hist[-2], action=action_hist[-1], state_nxt=state_hist[-1]
    )
    for key, value in constraint_dict.items():
      print(key, ":", value[:, -1])


end_criterion = "failure"
# end_criterion = "timeout"
nominal_states, result, traj_info = env.simulate_one_trajectory(
    T_rollout=max_iter_receding, end_criterion=end_criterion,
    reset_kwargs=dict(state=x_cur),
    rollout_step_callback=rollout_step_callback,
    rollout_episode_callback=rollout_episode_callback
)
print("result", result)
nominal_ctrls = traj_info['action_hist']
A, B = env.agent.get_dyn_jacobian(
    nominal_states=nominal_states[:-1, :].T, nominal_controls=nominal_ctrls.T
)
print(A.shape, B.shape)
dict_for_minimax = {
    "nominal_states": nominal_states,
    "nominal_ctrls": nominal_ctrls,
    "dyn_x": A,
    "dyn_u": B
}
save_obj(dict_for_minimax, "dict_minimax")
# endregion

# region: Visualizes
gif_path = os.path.join(fig_folder, 'rollout.gif')
with imageio.get_writer(gif_path, mode='I') as writer:
  for i in range(len(nominal_states) - 1):
    filename = os.path.join(fig_prog_folder, str(i + 1) + ".png")
    image = imageio.imread(filename)
    writer.append_data(image)
Image(open(gif_path, 'rb').read(), width=400)
# endregion
