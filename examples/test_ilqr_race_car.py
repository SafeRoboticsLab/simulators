# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------from typing import Dict
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio.v2 as imageio
from IPython.display import Image
import argparse
from shutil import copyfile
import jax
from jax import numpy as jnp
from omegaconf import OmegaConf

from simulators import (
    RaceCarSingle5DEnv, PrintLogger, Bicycle5DCost, Bicycle5DReachabilityCost,
    Bicycle5DRefTrajCost
)

jax.config.update('jax_platform_name', 'cpu')


def main(config_file: str):
  cfg = OmegaConf.load(config_file)

  # region: Sets environment
  cfg.cost.plan_horizon = cfg.solver.plan_horizon

  env = RaceCarSingle5DEnv(cfg.environment, cfg.agent, cfg.cost)
  env.step_keep_constraints = True
  x_cur = np.array(getattr(cfg.solver, "init_state", [0., 0., 1., 0., 0.]))
  env.reset(x_cur)

  # endregion

  # region: Constructs placeholder and initializes iLQR
  max_iter_receding = cfg.solver.max_iter_receding
  ref_traj = None
  # ! hacky
  config_ilqr_cost = copy.deepcopy(cfg.cost)
  config_ilqr_cost.buffer = getattr(cfg.solver, "buffer", 0.)
  if cfg.cost.cost_type == "Reachability":
    policy_type = "iLQRReachabilitySpline"
    cost = Bicycle5DReachabilityCost(config_ilqr_cost)
    env.cost = cost  # ! hacky
  else:
    policy_type = "iLQRSpline"
    if cfg.solver.use_traj_cost:
      len_ref_traj = max_iter_receding + cfg.solver.plan_horizon
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

      cfg.traj_cost.state_box_limit = cfg.agent.state_box_limit
      cfg.traj_cost.wheelbase = cfg.agent.wheelbase
      cfg.traj_cost.track_width_left = cfg.environment.track_width_left
      cfg.traj_cost.track_width_right = cfg.environment.track_width_right
      cfg.traj_cost.obs_spec = cfg.environment.obs_spec
      cost = Bicycle5DRefTrajCost(cfg.traj_cost, jnp.asarray(ref_traj))
      env.cost = cost
    else:
      cost = Bicycle5DCost(config_ilqr_cost)

  env.agent.init_policy(
      policy_type=policy_type, cfg=cfg.solver, cost=cost, track=env.track,
      ref_traj=ref_traj
  )
  fig_folder = os.path.join(cfg.solver.out_folder, "figure")
  fig_prog_folder = os.path.join(fig_folder, "progress")
  os.makedirs(fig_prog_folder, exist_ok=True)
  copyfile(config_file, os.path.join(cfg.solver.out_folder, 'config.yaml'))
  sys.stdout = PrintLogger(os.path.join(cfg.solver.out_folder, 'log.txt'))
  sys.stderr = PrintLogger(os.path.join(cfg.solver.out_folder, 'log.txt'))
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
        1, 1, figsize=(cfg.solver.fig_size_x, cfg.solver.fig_size_y)
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
    if cfg.solver.cmap:
      env.render_state_cost_map(
          ax=ax, nx=cfg.solver.cmap_res_x, ny=cfg.solver.cmap_res_y,
          vmin=cfg.solver.cmap_min, vmax=cfg.solver.cmap_max,
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
        vmin=cfg.agent.v_min, vmax=cfg.agent.v_max, edgecolor='none',
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
        2, 1, figsize=(cfg.solver.fig_size_x, 2 * cfg.solver.fig_size_y)
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
    # action_space = np.array(cfg.agent.action_range, dtype=np.float32)

    ax = axes[0]
    sc = ax.scatter(
        states[0, :], states[1, :], s=24, c=states[2, :], cmap=cm.jet,
        vmin=cfg.agent.v_min, vmax=cfg.agent.v_max, edgecolor='none',
        marker='o'
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"velocity [$m/s$]", size=20)

    ax = axes[1]
    sc = ax.scatter(
        states[0, :], states[1, :], s=24, c=states[4, :], cmap=cm.jet,
        vmin=cfg.agent.delta_min, vmax=cfg.agent.delta_max, edgecolor='none',
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
  frame_skip = getattr(cfg.solver, "frame_skip", 1)
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
