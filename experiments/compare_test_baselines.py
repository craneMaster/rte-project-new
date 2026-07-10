"""Compare decentralized test loss: learned margins vs untrained vs centralized optimum.

Runs the same held-out test trajectories through:
  - OURS:    receding-horizon decentralized MPC with trained line margins (from checkpoint)
  - UNTRAINED: same controller with initial (reset) margins
  - OPTIMAL: single-segment centralized open-loop planner (2b / get_central_full_traj, num_cycles=1)

Usage:
  python experiments/compare_test_baselines.py \\
      --checkpoint paper_experiments/manual/checkpoint_....pt \\
      --test_skew_mag 25 --test_seed 0 --num_traj 10
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.multiprocessing as mp

import dQPTH
import evals
import grid_pkg
import controller_utils

torch.set_default_dtype(torch.double)

BATT_COST = 100
CURT_CHANGE_COST = 0.01
CURT_NET_COST = 1
BUS_SLACK_COST = 1e8
LINE_SLACK_COST = 1e2
T, H = 20, 5
NUM_AGENTS = 3


def _make_grid(control_delay):
    bus_with_curt = torch.load("data/case118_gen_data.pt", weights_only=True)[:, 0].int()
    bus_with_batt = torch.tensor([10 * i + 2 for i in range(12)], dtype=torch.int)
    return grid_pkg.Grid(
        bus_with_curt, bus_with_batt, 15,
        "data/case118_line_data.pt", "data/case118_bus_data.pt",
        "data/case118_gen_data.pt", "data/case118_ptdf_data.pt",
        control_delay=control_delay,
    )


def _make_layer(pool):
    settings = dQPTH.build_settings(
        solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    return dQPTH.dQPTH_layer(settings=settings, pool=pool)


def _decentralized_loss(grid, controllers, layer, test_traj):
    grid.restore_baseline_init()
    ckpt = evals.evaluate_on_traj(
        grid, controllers, layer, test_traj, T, H,
        BATT_COST, CURT_CHANGE_COST, CURT_NET_COST, BUS_SLACK_COST, LINE_SLACK_COST,
    )
    return ckpt["total_losses"]


def _optimal_loss(grid, test_traj):
    grid.restore_baseline_init()
    ckpt = evals.get_central_full_traj(
        grid, T, H, test_traj, BATT_COST, CURT_CHANGE_COST, CURT_NET_COST,
        BUS_SLACK_COST, LINE_SLACK_COST, num_cycles=1,
    )
    return ckpt["total_losses"]


def _fmt_losses(losses):
    total, econ, viol = [float(x) for x in losses]
    return total, econ, viol


def _print_row(label, losses, opt_total=None):
    total, econ, viol = _fmt_losses(losses)
    ratio = total / opt_total if opt_total and opt_total > 0 else float("nan")
    print(f"  {label:10}  total={total:10.1f}  econ={econ:10.1f}  viol={viol:8.2f}  ratio_vs_opt={ratio:6.2f}")


def main(args):
    ckpt = torch.load(args.checkpoint, weights_only=False)
    if "line_max_changes" not in ckpt or "line_min_changes" not in ckpt:
        raise KeyError(f"Checkpoint missing line_max_changes / line_min_changes: {args.checkpoint}")

    control_delay = int(ckpt.get("control_delay", args.control_delay))
    line_terminal_cost = float(ckpt.get("line_terminal_cost", 0.0))
    batt_curt_terminal_cost = float(ckpt.get("batt_curt_terminal_cost", 0.0))

    grid = _make_grid(control_delay)
    partition = evals.training_partition()
    controllers = controller_utils.create_split_constraint_controllers(
        grid, NUM_AGENTS, partition, T, H,
        BATT_COST, CURT_CHANGE_COST, CURT_NET_COST, BUS_SLACK_COST, LINE_SLACK_COST,
        line_terminal_cost=line_terminal_cost, batt_curt_terminal_cost=batt_curt_terminal_cost,
    )

    learned_max = ckpt["line_max_changes"].detach()
    learned_min = ckpt["line_min_changes"].detach()

    test_data = torch.load(
        f"data/scenario_generation/test_trajs_with_mismatch_rad{args.radius}/"
        f"test_trajs_with_mismatch_ps{args.ps}_offset{args.offset}_tsm{args.test_skew_mag}.pt"
    )
    all_test = test_data[(args.ps, args.offset, args.test_skew_mag, args.test_seed)]
    num_traj = min(args.num_traj, all_test.shape[0])

    pool = mp.Pool(NUM_AGENTS)
    layer = _make_layer(pool)

    ours = torch.zeros(num_traj, 3)
    untrained = torch.zeros(num_traj, 3)
    optimal = torch.zeros(num_traj, 3)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test: ps={args.ps} offset={args.offset} skew_mag={args.test_skew_mag} "
          f"seed={args.test_seed}  num_traj={num_traj}  control_delay={control_delay}")
    print(f"{'traj':>4}  comparison (total / econ / viol)")
    print("-" * 72)

    for j in range(num_traj):
        test_traj = all_test[j]

        controller_utils.assign_line_limit_changes(controllers, learned_max, learned_min)
        ours[j] = _decentralized_loss(grid, controllers, layer, test_traj)

        for c in controllers:
            c.reset_params()
        untrained[j] = _decentralized_loss(grid, controllers, layer, test_traj)

        if not args.skip_optimal:
            optimal[j] = _optimal_loss(grid, test_traj)

        opt_total = float(optimal[j, 0]) if not args.skip_optimal else None
        print(f"{j:4d}")
        _print_row("OURS", ours[j], opt_total)
        _print_row("UNTRAINED", untrained[j], opt_total)
        if not args.skip_optimal:
            _print_row("OPTIMAL", optimal[j])

    pool.close()
    pool.terminate()

    def _summary(name, losses):
        m = losses.mean(dim=0)
        print(f"  {name:10}  mean_total={m[0]:10.1f}  mean_econ={m[1]:10.1f}  mean_viol={m[2]:8.2f}")

    print("-" * 72)
    print("Means:")
    _summary("OURS", ours)
    _summary("UNTRAINED", untrained)
    if not args.skip_optimal:
        _summary("OPTIMAL", optimal)
        gap = ours[:, 0].mean() - optimal[:, 0].mean()
        ratio = ours[:, 0].mean() / optimal[:, 0].mean()
        improve = untrained[:, 0].mean() - ours[:, 0].mean()
        print(f"  OURS vs OPTIMAL:     gap={gap:.1f}  ratio={ratio:.3f}")
        print(f"  OURS vs UNTRAINED:   improvement={improve:.1f}  "
              f"({100 * improve / untrained[:, 0].mean():.1f}% lower)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare test trajectories: learned vs untrained vs optimal")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to training checkpoint with line_max_changes / line_min_changes")
    parser.add_argument("--ps", type=float, default=2.0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--test_skew_mag", type=int, default=25)
    parser.add_argument("--test_seed", type=int, default=0)
    parser.add_argument("--num_traj", type=int, default=10,
                        help="Number of test trajectories (matches training test loop default)")
    parser.add_argument("--control_delay", type=int, default=1,
                        help="Fallback if not stored in checkpoint")
    parser.add_argument("--skip_optimal", action="store_true",
                        help="Skip slow centralized solves (decentralized only)")
    main(parser.parse_args())
