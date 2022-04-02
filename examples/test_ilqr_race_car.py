"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from IPython.display import Image

from simulators import RaceCarSingleEnv, load_config, iLQR
from simulators.ell_reach.ellipse import Ellipse
from simulators.ell_reach.plot_ellipsoids import plot_ellipsoids

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

static_obs_list = [static_obs for _ in range(config_solver.N)]
env.constraints.update_obs([static_obs_list])
# endregion

# region: Constructs placeholder and initializes iLQR
solver = iLQR(env, config_solver)
init_control = np.zeros((2, config_solver.N - 1))

t_total = 0.
max_iter_receding = config_solver.MAX_ITER_RECEDING
state_hist = np.zeros((4, max_iter_receding))
control_hist = np.zeros((2, max_iter_receding))
plan_hist = [{} for _ in range(max_iter_receding)]

c_track = 'k'

fig_folder = os.path.join("figure")
fig_prog_folder = os.path.join(fig_folder, "progress")
os.makedirs(fig_prog_folder, exist_ok=True)
# endregion

# region: Runs iLQR
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for i in range(max_iter_receding):
  # Plans the trajectory by using iLQR.
  states, controls, state_final, t_process, status = (
      solver.solve(state_init=x_cur, controls=init_control)
  )

  # Executes the first control.
  x_cur, _ = env.agent.integrate_forward(
      state=x_cur, control=controls[:, 0], **env.integrate_kwargs
  )
  print(
      "[{}]: solver returns status {} and uses {:.3f}.".format(
          i, status, t_process
      ), end='\r'
  )
  t_total += t_process

  # Records planning history, states and controls.
  plan_hist[i]['states'] = states
  plan_hist[i]['controls'] = controls
  plan_hist[i]['state_final'] = state_final

  state_hist[:, i] = states[:, 0]
  control_hist[:, i] = controls[:, 0]

  # Updates the nominal control signal for next receding horizon (The first
  # control is executed).
  init_control[:, :-1] = controls[:, 1:]
  init_control[:, -1] = 0.

  # Plots the current progress.
  ax.cla()
  env.track.plot_track(ax, c=c_track)

  plot_ellipsoids(
      ax, static_obs_list[0:1], arg_list=[dict(c='r', linewidth=1.)],
      dims=[0, 1], N=50, plot_center=False, use_alpha=True
  )
  ego = env.agent.footprint.move2state(states[[0, 1, 3], 0])
  plot_ellipsoids(
      ax, [ego], arg_list=[dict(c='b')], dims=[0, 1], N=50, plot_center=False
  )

  ax.plot(states[0, :], states[1, :], linewidth=2, c='b')
  sc = ax.scatter(
      state_hist[0, :i + 1], state_hist[1, :i + 1], s=24,
      c=state_hist[2, :i + 1], cmap=cm.jet, vmin=0, vmax=config_agent.V_MAX,
      edgecolor='none', marker='o'
  )
  cbar = fig.colorbar(sc, ax=ax)
  cbar.set_label(r"velocity [$m/s$]", size=20)
  fig.savefig(os.path.join(fig_prog_folder, str(i) + ".png"), dpi=200)
  cbar.remove()

print("\n\n --> Planning uses {:.3f}.".format(t_total))
# endregion

# region: Visualizes
gif_path = os.path.join(fig_folder, 'rollout.gif')
with imageio.get_writer(gif_path, mode='I') as writer:
  for i in range(max_iter_receding):
    filename = os.path.join(fig_prog_folder, str(i) + ".png")
    image = imageio.imread(filename)
    writer.append_data(image)
Image(open(gif_path, 'rb').read(), width=400)
# endregion
