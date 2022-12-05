"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
from typing import Dict
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from IPython.display import Image
import argparse
from shutil import copyfile
import jax
from jax import numpy as jnp

from simulators import (
    RaceCarSingle5DEnv, load_config, PrintLogger, Bicycle5DCost,
    Bicycle5DReachabilityCost, Bicycle5DRefTrajCost
)

jax.config.update('jax_platform_name', 'cpu')


def main(config_file):
  # region: Sets environment
  config = load_config(config_file)
  cfg_env = config['environment']
  cfg_agent = config['agent']
  cfg_solver = config['solver']
  cfg_cost = config['cost']
  cfg_cost.N = cfg_solver.N

  env = RaceCarSingle5DEnv(cfg_env, cfg_agent, cfg_cost)
  env.step_keep_constraints = True
  x_cur = np.array(getattr(cfg_solver, "INIT_STATE", [0., 0., 1., 0., 0.]))
  env.reset(x_cur)

  # endregion

  # region: Constructs placeholder and initializes iLQR
  max_iter_receding = cfg_solver.MAX_ITER_RECEDING
  ref_traj = None
  #! hacky
  config_ilqr_cost = copy.deepcopy(cfg_cost)
  config_ilqr_cost.BUFFER = getattr(cfg_solver, "BUFFER", 0.)
  if cfg_cost.COST_TYPE == "Reachability":
    policy_type = "iLQRReachabilitySpline"
    cost = Bicycle5DReachabilityCost(config_ilqr_cost)
    env.cost = cost  #! hacky
  else:
    policy_type = "iLQRSpline"
    if cfg_solver.USE_TRAJ_COST:
      len_ref_traj = max_iter_receding + cfg_solver.N
      center_line = env.track.center_line_data
      dist = np.linalg.norm(center_line - x_cur[:2, np.newaxis], axis=0)
      start_idx = np.argmin(dist)
      start_idx = min(
          start_idx, env.track.center_line_data.shape[1] - len_ref_traj - 1
      )
      print(center_line[:, start_idx])

      ref_traj = np.zeros((5, len_ref_traj))
      ref_traj[:2, :] = (
          env.track.center_line_data[:2, start_idx:start_idx + len_ref_traj]
      )
      ref_traj[2, :] = 1.5

      cfg_traj_cost = config['traj_cost']
      cfg_traj_cost.STATE_BOX_LIMITS = cfg_agent.BOX_LIMIT
      cfg_traj_cost.WHEELBASE = cfg_agent.WHEELBASE
      cfg_traj_cost.TRACK_WIDTH_LEFT = cfg_env.TRACK_WIDTH_LEFT
      cfg_traj_cost.TRACK_WIDTH_RIGHT = cfg_env.TRACK_WIDTH_RIGHT
      cfg_traj_cost.OBS_SPEC = cfg_env.OBS_SPEC
      cost = Bicycle5DRefTrajCost(cfg_traj_cost, jnp.asarray(ref_traj))
      env.cost = cost
    else:
      cost = Bicycle5DCost(config_ilqr_cost)

  env.agent.init_policy(
      policy_type=policy_type, config=cfg_solver, cost=cost, track=env.track,
      ref_traj=ref_traj
  )
  fig_folder = os.path.join(cfg_solver.OUT_FOLDER, "figure")
  fig_prog_folder = os.path.join(fig_folder, "progress")
  os.makedirs(fig_prog_folder, exist_ok=True)
  copyfile(config_file, os.path.join(cfg_solver.OUT_FOLDER, 'config.yaml'))
  sys.stdout = PrintLogger(os.path.join(cfg_solver.OUT_FOLDER, 'log.txt'))
  sys.stderr = PrintLogger(os.path.join(cfg_solver.OUT_FOLDER, 'log.txt'))
  # endregion

  # region: Runs iLQR
  # Warms up jit
  env.agent.policy.get_action(obs=x_cur, state=x_cur, time_idx=0)

  print("\n== iLQR starts ==")
  env.report()
  c_track = 'k'
  c_obs = 'r'
  c_ego = 'g'

  def rollout_step_callback(
      env: RaceCarSingle5DEnv, state_hist, action_hist, plan_hist, step_hist,
      time_idx, *args, **kwargs
  ):
    solver_info = plan_hist[-1]
    fig, ax = plt.subplots(
        1, 1, figsize=(cfg_solver.FIG_SIZE_X, cfg_solver.FIG_SIZE_Y)
    )
    ax.axis(env.visual_extent)
    ax.set_aspect('equal')

    # track, obstacles, footprint
    env.track.plot_track(ax, c=c_track)
    env.render_obs(ax=ax, c=c_obs)
    env.render_footprint(ax=ax, state=state_hist[-1], c=c_ego, lw=1.5)
    ego_fut = env.agent.footprint.move2state(
        solver_info['states'][[0, 1, 3], -1]
    )
    ego_fut.plot(ax, color=c_ego, lw=1.5, alpha=.5)
    if cfg_solver.CMAP:
      env.render_state_cost_map(
          ax=ax, nx=cfg_solver.CMAP_RES_X, ny=cfg_solver.CMAP_RES_Y,
          vmin=cfg_solver.CMAP_MIN, vmax=cfg_solver.CMAP_MAX,
          vel=state_hist[-1][2], yaw=state_hist[-1][3],
          delta=state_hist[-1][4], time_idx=time_idx
      )

    # plan.
    ax.plot(
        solver_info['states'][0, :], solver_info['states'][1, :], linewidth=1.,
        c=c_ego
    )
    # history.
    states = np.array(state_hist).T  # last one is the next state.
    sc = ax.scatter(
        states[0, :-1], states[1, :-1], s=24, c=states[2, :-1], cmap=cm.jet,
        vmin=cfg_agent.V_MIN, vmax=cfg_agent.V_MAX, edgecolor='none',
        marker='o'
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"velocity [$m/s$]", size=20)
    fig.savefig(
        os.path.join(fig_prog_folder,
                     str(states.shape[1] - 1) + ".png"), dpi=200
    )
    cbar.remove()
    plt.close('all')

    print(
        "[{}]: solver returns status {}, cost {:.1e}, and uses {:.3f}.".format(
            states.shape[1] - 1, solver_info['status'], solver_info['J'],
            solver_info['t_process']
        ), end=' -> '
    )
    ctrl = action_hist[-1]
    print(f"ctrl: [{ctrl[0]:.2e}, {ctrl[1]:.2e}]", end='\r')
    # with np.printoptions(prrecision=2, suppress=False):
    #   print(np.asarray(solver_info['K_closed_loop'][..., 0]))

  def rollout_episode_callback(
      env, state_hist, action_hist, plan_hist, step_hist, *args, **kwargs
  ):
    fig, axes = plt.subplots(
        2, 1, figsize=(cfg_solver.FIG_SIZE_X, 2 * cfg_solver.FIG_SIZE_Y)
    )

    for ax in axes:
      # track, obstacles, footprint
      env.track.plot_track(ax, c=c_track)
      env.render_obs(ax=ax, c=c_obs)
      env.render_footprint(ax=ax, state=state_hist[-1], c=c_ego)
      ax.axis(env.visual_extent)
      ax.set_aspect('equal')

    states = np.array(state_hist).T
    # ctrls = np.array(action_hist).T
    # action_space = np.array(cfg_agent.ACTION_RANGE, dtype=np.float32)

    ax = axes[0]
    sc = ax.scatter(
        states[0, :], states[1, :], s=24, c=states[2, :], cmap=cm.jet,
        vmin=cfg_agent.V_MIN, vmax=cfg_agent.V_MAX, edgecolor='none',
        marker='o'
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"velocity [$m/s$]", size=20)

    ax = axes[1]
    sc = ax.scatter(
        states[0, :], states[1, :], s=24, c=states[4, :], cmap=cm.jet,
        vmin=cfg_agent.DELTA_MIN, vmax=cfg_agent.DELTA_MAX, edgecolor='none',
        marker='o'
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"steering (rad)", size=20)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_folder, "final.png"), dpi=200)
    cbar.remove()
    plt.close('all')
    t_process = 0.

    for solver_info in plan_hist:
      t_process += solver_info['t_process']
    print("\n\n --> Planning uses {:.3f}.".format(t_process))

  end_criterion = "failure"
  # end_criterion = "timeout"
  nominal_states, result, traj_info = env.simulate_one_trajectory(
      T_rollout=max_iter_receding, end_criterion=end_criterion,
      reset_kwargs=dict(state=x_cur),
      rollout_step_callback=rollout_step_callback,
      rollout_episode_callback=rollout_episode_callback
  )
  print("result", result)
  print(traj_info['step_hist'][-1]["done_type"])
  constraints: Dict = traj_info['step_hist'][-1]['constraints']
  for k, v in constraints.items():
    print(f"{k}: {v[0, 1]:.1e}")

  # endregion

  # region: Visualizes
  gif_path = os.path.join(fig_folder, 'rollout.gif')
  frame_skip = getattr(cfg_solver, "FRAME_SKIP", 1)
  with imageio.get_writer(gif_path, mode='I') as writer:
    for i in range(len(nominal_states) - 1):
      if frame_skip != 1 and i % frame_skip != 0:
        continue
      filename = os.path.join(fig_prog_folder, str(i + 1) + ".png")
      image = imageio.imread(filename)
      writer.append_data(image)
  Image(open(gif_path, 'rb').read(), width=400)
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("simulators", "race_car", "race_car_straight.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
