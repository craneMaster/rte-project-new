"""
Reproduce a rollout and decode the QP constraints that conflict when Gurobi
declares INFEASIBLE.

For every agent at every rollout step we:
  1) Build the agent's QP via get_matrices_dQPTH.
  2) Solve it with Gurobi using the same params as production.
  3) On infeasibility, compute the IIS, then translate every conflicting row
     into a label of the form:
       step <i>: batt_charge[k] >= <bound>
       step <i>: dynamics for curt[k]   (RHS = <forced value>)
       step <i>: line_flow[k] + slack_min >= <bound>
     so it's obvious which physical bound or dynamics row failed and by how
     much.

Run:
  python experiments/debug_gurobi_qp.py --control_delay 2 --H 5 --t_stop 4
"""
import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import gurobipy
from gurobipy import GRB

import grid_pkg
import controller_utils
from dQPTH_helpers.gurobi_ws import gurobi_status_name

torch.set_default_dtype(torch.double)


def decode_row(idx, controller):
    """Map a Gurobi row index (G added first, then A) to a human-readable label.

    Returns (kind, label, rhs_unscaled) where kind is "ineq" or "eq".
    """
    H = controller.H
    nB = controller.num_batt
    nC = controller.num_curt
    nL = controller.num_lines
    bus_ineq = controller.bus_ineq_dim          # 4*nB + 2*nC
    ineq_dim = controller.ineq_dim
    bus_state_dim = controller.bus_state_dim    # 2*nB + nC

    H_bus_ineq = H * bus_ineq
    H_line_block = H * nL

    if idx < ineq_dim:
        if idx < H_bus_ineq:
            step, local = divmod(idx, bus_ineq)
            if local < nB:
                k = local
                return ("ineq",
                        f"step {step}: batt_charge[{k}] >= {controller.batt_charge_min_limits[k].item():.4f}",
                        controller.batt_charge_min_limits[k].item())
            if local < 2 * nB:
                k = local - nB
                return ("ineq",
                        f"step {step}: batt_charge[{k}] <= {controller.batt_charge_max_limits[k].item():.4f}",
                        controller.batt_charge_max_limits[k].item())
            if local < 3 * nB:
                k = local - 2 * nB
                return ("ineq",
                        f"step {step}: batt_power[{k}] >= {controller.batt_power_min_limits[k].item():.4f}",
                        controller.batt_power_min_limits[k].item())
            if local < 4 * nB:
                k = local - 3 * nB
                return ("ineq",
                        f"step {step}: batt_power[{k}] <= {controller.batt_power_max_limits[k].item():.4f}",
                        controller.batt_power_max_limits[k].item())
            if local < 4 * nB + nC:
                k = local - 4 * nB
                return ("ineq",
                        f"step {step}: curt[{k}] >= {controller.curt_min_limits[k].item():.4f}",
                        controller.curt_min_limits[k].item())
            k = local - 4 * nB - nC
            return ("ineq",
                    f"step {step}: curt[{k}] <= {controller.curt_max_limits[k].item():.4f}",
                    controller.curt_max_limits[k].item())

        if idx < H_bus_ineq + H_line_block:
            rel = idx - H_bus_ineq
            step, k = divmod(rel, nL)
            return ("ineq",
                    f"step {step}: line_flow[{k}] - slack_max <= {controller.line_upper_limits[k].item():.4f}",
                    controller.line_upper_limits[k].item())

        rel = idx - H_bus_ineq - H_line_block
        step, k = divmod(rel, nL)
        return ("ineq",
                f"step {step}: -line_flow[{k}] - slack_min <= -{controller.line_lower_limits[k].item():.4f}",
                -controller.line_lower_limits[k].item())

    eq_idx = idx - ineq_dim
    if eq_idx < H * bus_state_dim:
        step, local = divmod(eq_idx, bus_state_dim)
        if local < nB:
            return ("eq", f"step {step}: dynamics row batt_charge[{local}]", None)
        if local < 2 * nB:
            return ("eq", f"step {step}: dynamics row batt_power[{local - nB}]", None)
        return ("eq", f"step {step}: dynamics row curt[{local - 2 * nB}]", None)

    rel = eq_idx - H * bus_state_dim
    step, k = divmod(rel, nL)
    return ("eq", f"step {step}: dynamics row line_flow[{k}]", None)


def diagnose_qp(controller, agent_idx, t, disturbances, warm_basis=None):
    """Build and solve the QP. On infeasibility, decode IIS rows.

    If ``warm_basis`` is a (VBasis, CBasis) tuple from a previous solve, it is
    installed before optimizing so we mimic production's warm-start behaviour.

    Returns ``(ok, sol_x, status_name, new_basis)``.
    """
    cfg = controller.get_matrices_dQPTH(disturbances[:, controller.buses_in_area], t)
    P = cfg["scipy_scaled_Q"]
    q = cfg["q"].detach().numpy()
    G = cfg["scipy_scaled_G"]
    h_scaled = cfg["h"].detach().numpy()
    A = cfg["scipy_scaled_A"]
    b_scaled = cfg["b"].detach().numpy()

    E_vec = controller.E_vec.detach().numpy()
    F_vec = controller.F_vec.detach().numpy()
    b_unscaled = b_scaled / np.where(E_vec == 0, 1.0, E_vec)
    h_unscaled = h_scaled / np.where(F_vec == 0, 1.0, F_vec)

    model = gurobipy.Model()
    model.setParam(GRB.Param.OutputFlag, 0)
    n_var = P.shape[0]
    x = model.addMVar(n_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    if G.shape[0] > 0:
        model.addMConstr(G, x, GRB.LESS_EQUAL, h_scaled)
    if A.shape[0] > 0:
        model.addMConstr(A, x, GRB.EQUAL, b_scaled)
    model.setObjective(0.5 * (x @ P @ x) + q @ x, sense=GRB.MINIMIZE)
    model.setParam("Method", 1)
    model.setParam("OptimalityTol", 1e-9)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("NumericFocus", 3)
    model.update()
    if warm_basis is not None:
        try:
            model.setAttr("VBasis", model.getVars(), warm_basis[0])
            model.setAttr("CBasis", model.getConstrs(), warm_basis[1])
        except gurobipy.GurobiError:
            pass
    model.optimize()

    status_name = gurobi_status_name(model.status)
    new_basis = None
    try:
        new_basis = (model.getAttr(GRB.Attr.VBasis), model.getAttr(GRB.Attr.CBasis))
    except gurobipy.GurobiError:
        pass

    if model.status == GRB.OPTIMAL:
        return True, x.X, status_name, new_basis

    print(f"\n  ✗ Agent {agent_idx} infeasible at t={t} ({status_name})")
    print(f"    nVar={n_var}, nEq={A.shape[0] if A is not None else 0}, "
          f"nIneq={G.shape[0] if G is not None else 0}")

    try:
        model.computeIIS()
    except gurobipy.GurobiError as exc:
        print(f"    IIS error: {exc}")
        return False, None, status_name, new_basis

    constrs = list(model.getConstrs())
    iis_rows = [i for i, c in enumerate(constrs) if c.IISConstr]
    print(f"    IIS contains {len(iis_rows)} conflicting rows:\n")

    for row in iis_rows:
        kind, label, _ = decode_row(row, controller)
        if kind == "eq":
            eq_idx = row - controller.ineq_dim
            forced = b_unscaled[eq_idx]
            print(f"      R{row:5d} [eq]   {label}")
            print(f"                       forced RHS (unscaled) = {forced: .6e}")
        else:
            bound = h_unscaled[row]
            print(f"      R{row:5d} [ineq] {label}")
            print(f"                       bound (unscaled)      = {bound: .6e}")

    eq_iis = [r for r in iis_rows if r >= controller.ineq_dim]
    ineq_iis = [r for r in iis_rows if r < controller.ineq_dim]
    if eq_iis and ineq_iis:
        print(f"\n    Conflict pattern: {len(eq_iis)} dynamics row(s) pin a value "
              f"that violates {len(ineq_iis)} hard bound(s).")
    elif len(ineq_iis) >= 2:
        print(f"\n    Conflict pattern: two hard bounds on overlapping variables "
              f"(one ≤ and one ≥) are mutually inconsistent.")
    return False, None, status_name, new_basis


def replay(args):
    bus_with_curt = torch.load("data/case118_gen_data.pt", weights_only=True)[:, 0].int()
    bus_with_batt = torch.tensor([10 * i + 2 for i in range(12)], dtype=torch.int)
    grid = grid_pkg.Grid(
        bus_with_curt, bus_with_batt, 15,
        "data/case118_line_data.pt", "data/case118_bus_data.pt",
        "data/case118_gen_data.pt", "data/case118_ptdf_data.pt",
        control_delay=args.control_delay,
    )

    partition = [
        [i for i in range(0, 42)] + [112, 113, 114, 116],
        [i for i in range(42, 69)] + [115],
        [i for i in range(69, 112)] + [117],
    ]
    controllers = controller_utils.create_split_constraint_controllers(
        grid, 3, partition, args.T, args.H, 100, 0.01, 1, 1e8, 1e2,
    )

    train_trajs = torch.load(f"data/scenario_generation/train_trajs_rad{args.radius}.pt")
    traj = train_trajs[(args.ps, args.offset, args.torch_seed)][args.traj]

    grid.reset_state()
    for c in controllers:
        c.zero_line_state()
        c.update_bus_state()

    warm_mode = "ON" if args.warm_start else "OFF"
    print(f"Replaying rollout: traj={args.traj}, H={args.H}, "
          f"control_delay={args.control_delay}, t_stop={args.t_stop}, "
          f"warm_start={warm_mode}\n")

    bases = [None] * len(controllers)

    for t in range(args.t_stop + 1):
        if t + args.H > traj.shape[0]:
            print(f"\nTrajectory exhausted at t={t} (need {t + args.H} rows, "
                  f"have {traj.shape[0]}).")
            return

        noise = traj[t:t + args.H, :]
        print(f"=== t = {t} ===")

        all_curt = torch.zeros(args.H, grid.num_curt)
        all_batt = torch.zeros(args.H, grid.num_batt)
        any_failed = False

        for i, c in enumerate(controllers):
            warm = bases[i] if args.warm_start else None
            ok, sol_x, status, new_basis = diagnose_qp(c, i, t, noise, warm_basis=warm)
            bases[i] = new_basis
            if not ok:
                any_failed = True
                continue
            sol_t = torch.tensor(sol_x)
            bus_states, _, actions_curt, actions_batt, *_ = c.interpret_z(sol_t, update_state=True)
            all_curt[:, c.curt_idx] = actions_curt
            all_batt[:, c.batt_idx] = actions_batt
            u0_curt = actions_curt[0].detach().numpy()
            u0_batt = actions_batt[0].detach().numpy()
            x0_curt = bus_states[0][c.bus_state_curt_idx].detach().numpy()
            print(f"  ✓ Agent {i}: {status}  "
                  f"|u0_curt|_∞={np.max(np.abs(u0_curt)):.3f}  "
                  f"|u0_batt|_∞={np.max(np.abs(u0_batt)):.3f}  "
                  f"|x0_curt|_∞={np.max(np.abs(x0_curt)):.3f}  "
                  f"x0_curt[0]={x0_curt[0]:.3f}")

        if any_failed:
            print(f"\nStopped at t={t} due to infeasibility.")
            return

        grid.update_state(all_curt[0], all_batt[0], noise[0])
        if args.refresh_bus_state:
            for c in controllers:
                c.update_bus_state()
        print()

    print("Reached t_stop without infeasibility.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_delay", type=int, default=2)
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--t_stop", type=int, default=4,
                        help="Last rollout step to simulate (inclusive).")
    parser.add_argument("--traj", type=int, default=4)
    parser.add_argument("--ps", type=float, default=4.0)
    parser.add_argument("--offset", type=int, default=180)
    parser.add_argument("--torch_seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--warm_start", action="store_true",
                        help="Carry the Gurobi basis across rollout steps (mimics production).")
    parser.add_argument("--refresh_bus_state", action="store_true",
                        help="Call update_bus_state() after each plant step (off by default to match train_with_rollout).")
    args = parser.parse_args()
    replay(args)


if __name__ == "__main__":
    main()
