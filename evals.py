import sys
sys.path.append("../../")

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import time

import grid_pkg
import controller_pkg
import controller_utils
import dQPTH

torch.set_default_dtype(torch.double)
torch.set_printoptions(threshold=10000)
NUMPY_SEED = 0
TORCH_SEED = 0
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)

HOT_LINE_VIOLATION_LINES = (53, 95, 29)


def training_partition():
    """Agent bus partition used by train_with_rollout and eval (must match at test time)."""
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    return [nodes_1, nodes_2, nodes_3]

def _make_optimizer(param_groups, optimizer_type, lr):
    if optimizer_type in ('clipped_gd', 'sgd'):
        return optim.SGD(param_groups, lr=lr)
    elif optimizer_type == 'adam':
        return optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")


def _make_scheduler(optimizer, lr_schedule, schedule_kwargs, total_steps):
    if lr_schedule is None:
        return None
    kw = schedule_kwargs or {}
    if lr_schedule == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kw)
    elif lr_schedule == 'step_decay':
        return optim.lr_scheduler.StepLR(optimizer, step_size=kw.get('step_size', 30), gamma=kw.get('gamma', 0.5))
    elif lr_schedule == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode=kw.get('mode', 'min'),
                                                    factor=kw.get('factor', 0.5),
                                                    patience=kw.get('patience', 10))
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}")


def _step_scheduler(scheduler, loss):
    """Step scheduler, passing loss for ReduceLROnPlateau."""
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()


def _project_params(controller_dqp_list, max_target_sum=0, min_target_sum=0):
    """Project line_max/min_change so their sum across agents equals the target."""
    n = len(controller_dqp_list)
    with torch.no_grad():
        max_params = torch.stack([controller.line_max_change.data for controller in controller_dqp_list])
        min_params = torch.stack([controller.line_min_change.data for controller in controller_dqp_list])
        max_correction = (max_params.sum(dim=0) - max_target_sum) / n
        min_correction = (min_params.sum(dim=0) - min_target_sum) / n
        for controller in controller_dqp_list:
            controller.line_max_change.data -= max_correction
            controller.line_min_change.data -= min_correction


def _sync_relative_line_state(grid, controller_dqp_list):
    """Set MPC line state to flow delta from episode init (matches zero_line_state margins)."""
    init_flows = grid.init_state[grid.state_line_flow_idx]
    delta = grid.state[grid.state_line_flow_idx] - init_flows
    with torch.no_grad():
        for controller in controller_dqp_list:
            controller.line_state = delta.clone()


def _sync_mpc_from_grid(grid, controller_dqp_list):
    """Refresh controller bus state and line state for the next MPC solve."""
    _sync_relative_line_state(grid, controller_dqp_list)
    for controller in controller_dqp_list:
        controller.update_bus_state()


def _sync_cycle_start_references(grid, controller_dqp_list):
    """Align battery targets and batt/curt terminal-cost baselines with grid.init_state.

    Line-flow terminal cost stays anchored to the nominal construction baseline (set at controller
    init); it is intentionally NOT updated here so multi-cycle handoff drift is pulled back toward
    a realizable operating point rather than re-anchored to the drifted state.
    """
    with torch.no_grad():
        grid.target_batt_charges = grid.init_state[grid.state_batt_charge_idx].clone()
        for controller in controller_dqp_list:
            controller.initial_batt_charges = grid.init_state[grid.state_batt_charge_idx][controller.batt_idx].clone()
            controller.initial_curtailments = grid.init_state[grid.state_curt_idx][controller.curt_idx].clone()
            controller.target_charges = grid.target_batt_charges[controller.batt_idx]
            controller.update_cost_targets()


def _rollout_net_curt(grid):
    """Absolute net curtailment level (penalized toward zero, matching MPC objective)."""
    return grid.state[grid.state_curt_idx]


def _curt_loss_parts(grid, action_curt, net_curt, curt_change_cost, curt_net_cost):
    """Return curtailment action and net-level penalty terms separately.

    The change penalty is based on the action that ACTUALLY took effect this step
    (``grid.effective_curt``), not the freshly-predicted ``action_curt``. Under control_delay the
    predicted action does not hit the grid until later (and is zero during the warm-up), so
    penalizing the issued action makes the cost match what physically happened. Callers must invoke
    this right after ``grid.update_state(...)`` so ``grid.effective_curt`` is current.
    """
    issued_curt = grid.effective_curt
    curt_change_loss = curt_change_cost * torch.sum(issued_curt ** 2)
    curt_net_loss = curt_net_cost * torch.sum(net_curt ** 2)
    return curt_change_loss, curt_net_loss


def _curt_loss(grid, action_curt, net_curt, curt_change_cost, curt_net_cost):
    """Penalize curtailment actions and absolute net curtailment level."""
    curt_change_loss, curt_net_loss = _curt_loss_parts(
        grid, action_curt, net_curt, curt_change_cost, curt_net_cost)
    return curt_change_loss + curt_net_loss


def _line_violations_loss(line_violations, line_slack_cost):
    """Sum of squared line limit violations."""
    return line_slack_cost * torch.sum(line_violations ** 2)


def _is_controlled_rollout_step(t, grid, *, training_cycle=None):
    """True once effective actions apply (after control_delay warm-up since reset)."""
    return t >= grid.control_delay


def _format_loss_breakdown(mpc_terminal, batt, curt_change, curt_net, bus_viol, line_viol):
    """Format rollout loss components for logging."""
    def _val(x):
        return float(x.item() if hasattr(x, "item") else x)

    mpc_terminal = _val(mpc_terminal)
    batt = _val(batt)
    curt_change = _val(curt_change)
    curt_net = _val(curt_net)
    bus_viol = _val(bus_viol)
    line_viol = _val(line_viol)
    total = mpc_terminal + batt + curt_change + curt_net + bus_viol + line_viol
    return (
        f"total={total:.4f} mpc_terminal={mpc_terminal:.4f} batt={batt:.4f} "
        f"curt_change={curt_change:.4f} curt_net={curt_net:.4f} "
        f"bus_viol={bus_viol:.4f} line_viol={line_viol:.4f}"
    )


# Fraction of line limit; handoff resets t=0 margin split when headroom falls below this.
HEADROOM_RESET_FRAC = 0.05


def _clamp_init_line_flows(grid):
    """Legacy: project init_state line flows onto [-limit, limit]. Prefer leaving init_state unchanged."""
    with torch.no_grad():
        limits = grid.line_data[:, 5]
        flows = grid.init_state[grid.state_line_flow_idx]
        clamped = torch.clamp(flows, -limits, limits)
        if not torch.equal(flows, clamped):
            grid.init_state = grid.init_state.clone()
            grid.init_state[grid.state_line_flow_idx] = clamped


def _line_margin_targets_from_flows(flows, limits):
    """Return raw upper/lower headroom from line flows (may be negative if overloaded)."""
    return limits - flows, -limits - flows


def _update_line_margin_targets(grid, total_max_margin, total_min_margin):
    """Set projection targets from actual init_state flows without mutating init_state.

    Matches legacy comm-limited-congestion-mgmt-main: upper margin is limit - flow,
    lower margin is -limit - flow (typically negative at nominal operating points).
    """
    flows = grid.init_state[grid.state_line_flow_idx]
    limits = grid.line_data[:, 5]
    raw_max, raw_min = _line_margin_targets_from_flows(flows, limits)
    total_max_margin[0] = raw_max
    total_min_margin[0] = raw_min
    return raw_max, raw_min


def _margin_overloaded(raw_max, raw_min):
    """True when any line flow is outside [-limit, limit] at the current init_state."""
    # Upper overload: flow above +limit (raw_max < 0). Lower overload: flow below -limit (raw_min > 0).
    return bool(((raw_max < 0) | (raw_min > 0)).any())


def _margin_headroom_needs_reset(raw_max, raw_min, limits):
    """True when any line is overloaded or t=0 headroom is too tight for rescaling."""
    threshold = HEADROOM_RESET_FRAC * limits
    # Nominal flows have raw_max > 0 and raw_min < 0.
    tight = (raw_max < threshold) | (raw_min > -threshold)
    return _margin_overloaded(raw_max, raw_min) or bool(tight.any())


def _reset_t0_margin_split(controller_dqp_list, total_max_margin, total_min_margin):
    """Re-initialize t=0 split margins from current headroom targets."""
    n = len(controller_dqp_list)
    with torch.no_grad():
        split_max = total_max_margin[0] / n
        split_min = total_min_margin[0] / n
        for controller in controller_dqp_list:
            controller.line_max_change.data[1:].zero_()
            controller.line_min_change.data[1:].zero_()
            controller.line_max_change.data[0].copy_(split_max)
            controller.line_min_change.data[0].copy_(split_min)


def _reproject_line_margins(grid, controller_dqp_list, total_max_margin, total_min_margin, *,
                            repair_boundary=False, auto_repair_overloaded=False):
    """Refresh margin targets from init_state and project split line limits across agents.

    By default, learned per-agent schedules are kept and only corrected so their sums match
    the new headroom (used on cycle handoff so margins carry across cycles).

    When repair_boundary is True, drop future increments and re-split t=0 equally from current
    headroom. When auto_repair_overloaded is True, the same reset runs only if any line is
    overloaded at init_state — plant flows/state are never mutated.
    """
    raw_max, raw_min = _update_line_margin_targets(grid, total_max_margin, total_min_margin)
    overloaded = _margin_overloaded(raw_max, raw_min)
    do_repair = repair_boundary or (auto_repair_overloaded and overloaded)
    with torch.no_grad():
        if do_repair:
            if auto_repair_overloaded and overloaded and not repair_boundary:
                n_ol = int(((raw_max < 0) | (raw_min > 0)).sum())
                print(
                    f"Cycle-start margin repair: {n_ol} overloaded line(s); "
                    f"re-splitting t=0 headroom (plant state unchanged)",
                    flush=True)
            for controller in controller_dqp_list:
                controller.line_max_change.data[1:].zero_()
                controller.line_min_change.data[1:].zero_()
            _reset_t0_margin_split(controller_dqp_list, total_max_margin, total_min_margin)
    _project_params(controller_dqp_list, max_target_sum=total_max_margin, min_target_sum=total_min_margin)
    return raw_max, raw_min


def _disturbance_horizon(disturbances, global_t, H, *, wrap=True):
    """Return H disturbance rows starting at global_t.

    When ``wrap`` is False, raises if the horizon extends past the end of ``disturbances``.
    """
    n = disturbances.shape[0]
    idx = global_t + torch.arange(H, device=disturbances.device, dtype=torch.long)
    if wrap:
        idx = idx % n
    elif int(idx[-1]) >= n:
        raise ValueError(
            f"disturbance horizon global_t={global_t}, H={H} exceeds trajectory length {n}")
    return disturbances[idx]


def multi_cycle_noise_length(num_cycles, T, H):
    """Total pre-generated noise rows for ``num_cycles`` overlapping T-step segments."""
    return num_cycles * T + H


def cycle_noise_window(full_disturbances, cycle, T, H):
    """Return the (T+H)-row noise segment for training cycle ``cycle`` (0-based).

    Cycle 0 uses rows [0, T+H); cycle 1 uses [T, 2T+H); etc., overlapping by H steps.
    """
    start = cycle * T
    end = start + T + H
    n = full_disturbances.shape[0]
    if end > n:
        raise ValueError(
            f"cycle {cycle} needs noise rows [{start}, {end}) but trajectory has length {n}; "
            f"regenerate data with length >= {end} (e.g. multi_cycle_noise_length(num_cycles, T, H))")
    return full_disturbances[start:end]


def _validate_disturbance_trajectories(all_traj, num_cycles, T, H, *, label="disturbance"):
    """Ensure stored trajectories are long enough for multi-cycle overlapping windows."""
    required = multi_cycle_noise_length(num_cycles, T, H)
    if all_traj.shape[1] < required:
        raise ValueError(
            f"{label} trajectories have length {all_traj.shape[1]} but "
            f"{num_cycles} cycles with T={T}, H={H} require {required} rows "
            f"(regenerate with: python data/scenario_generation/regenerate_multicycle_trajs.py)")


def _noise_horizon(disturbances, global_t, H, *, wrap=False):
    """Return H-step noise forecast from ``disturbances`` (no scaling)."""
    return _disturbance_horizon(disturbances, global_t, H, wrap=wrap)


def _reanchor_curt_to_nominal(grid):
    """Reset net curtailment in init_state to the nominal construction baseline.

    NOTE: no longer used by train_with_rollout, which now hands over the full terminal state each
    cycle (sustainable-system design). Kept for diagnostic scripts. Re-anchoring curtailment alone
    while leaving drifted line flows produces a physically inconsistent state; prefer a full reset
    to baseline_init_state if a nominal restart is ever needed.
    """
    with torch.no_grad():
        nominal_curt = grid.baseline_init_state[grid.state_curt_idx]
        grid.init_state = grid.init_state.clone()
        grid.init_state[grid.state_curt_idx] = nominal_curt.clone()


def _begin_traj_rollout(grid, controller_dqp_list, *, reset_grid=True, resync_line_state_each_step=False):
    """Initialize or continue a rollout from the current (or reset) grid state."""
    if reset_grid:
        grid.reset_state()
    if resync_line_state_each_step:
        _sync_mpc_from_grid(grid, controller_dqp_list)
    else:
        for controller in controller_dqp_list:
            controller.zero_line_state()
            controller.update_bus_state()


def _rollout_terminal_handoff(grid, controller_dqp_list, train_disturbances, T, H, dQPTH_layer, *, resync_line_state_each_step=True, cycle=0):
    """Roll out one T-step segment; return terminal state and matching delay buffers (no grad)."""
    cycle_disturbances = cycle_noise_window(train_disturbances, cycle, T, H)
    with torch.no_grad():
        _begin_traj_rollout(grid, controller_dqp_list, reset_grid=True, resync_line_state_each_step=resync_line_state_each_step)
        for t in range(T):
            if resync_line_state_each_step:
                _sync_mpc_from_grid(grid, controller_dqp_list)
            noise = _noise_horizon(cycle_disturbances, t, H)
            pred_actions_curt, pred_actions_batt, _, _, _, _ = \
                controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=True)
            grid.update_state(pred_actions_curt[0], pred_actions_batt[0], noise[0])
        return grid.state.detach().clone(), grid.snapshot_action_buffers()


def _prepare_cycle_handoff_state(grid, terminal_state, action_buffers=None, cycle_start_state=None):
    """Seed the next cycle from a terminal plant state and its delay-action buffers."""
    grid.set_cycle_handoff(terminal_state, action_buffers)
    return grid.init_state


def _aggregate_line_limit_schedule(source_controller_list, schedule_len):
    """Sum per-agent margin schedules into one centralized schedule.

    Split controllers store per-step margin *increments*; cumulative sums across
    agents give the total line headroom the centralized planner should enforce.
    """
    if source_controller_list is None:
        return None, None
    line_max = torch.stack([c.line_max_change.data for c in source_controller_list]).sum(dim=0)
    line_min = torch.stack([c.line_min_change.data for c in source_controller_list]).sum(dim=0)
    n = line_max.shape[0]
    if n >= schedule_len:
        return line_max[:schedule_len].clone(), line_min[:schedule_len].clone()
    pad = schedule_len - n
    zeros = torch.zeros(pad, line_max.shape[1], dtype=line_max.dtype, device=line_max.device)
    return torch.cat([line_max, zeros], dim=0), torch.cat([line_min, zeros], dim=0)


# @profile
def get_central_full_traj(grid, T, H, disturbances, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                          line_terminal_cost=0.0, batt_curt_terminal_cost=0.0, num_cycles=1, soft_bus_constraints=True,
                          source_controller_list=None):
    """
    Simulates what a fully centralized planner would do with full knowledge of future disturbances.

    Plans ``num_cycles * T`` steps in one open-loop solve (e.g. 5 cycles × 20 steps → one 100-step QP),
    then rolls out the plan with load drifts. When ``num_cycles > 1``, economic costs are accumulated
    per cycle segment (cycle-relative curtailment / battery targets reset every ``T`` steps during rollout).

    Args:
        grid (Grid object):
            defines the grid that control problem is set up on
        T (int):
            trajectory length per cycle
        H (int):
            unused; the planner horizon is set to ``num_cycles * T``
        disturbances (Torch tensor of shape (n, grid.num_buses) or (1, n, grid.num_buses)):
            load drifts (wraps if the plan needs rows past the end)
        num_cycles (int):
            number of consecutive T-step segments in the plan
        source_controller_list (list of Split_Constraint_Controller, optional):
            When provided, sum these agents' learned ``line_max/min_change`` schedules
            and apply them to the single centralized controller (instead of default margins).
    Returns:
        ckpt (dict): Contains run results, keys are:
            actions_curt (Torch tensor of shape (num_cycles * T, grid.num_curt))
            actions_batt (Torch tensor of shape (num_cycles * T, grid.num_batt))
            viol_per_step (Torch tensor of shape (num_cycles * T,))
            curt_cost_per_step (Torch tensor of shape (num_cycles * T,))
            total_losses (Torch tensor of shape (3,))
            num_cycles (int)
            T_per_cycle (int)
    """
    if num_cycles < 1:
        raise ValueError(f"num_cycles must be >= 1, got {num_cycles}")

    dist = disturbances[0] if disturbances.dim() == 3 else disturbances
    total_steps = num_cycles * T
    plan_horizon = total_steps
    if dist.shape[0] < total_steps:
        raise ValueError(
            f"disturbance trajectory has length {dist.shape[0]} but open-loop rollout needs "
            f"{total_steps} rows (num_cycles={num_cycles}, T={T})")

    partition = [[i for i in range(grid.num_buses)]]
    num_agents = 1
    controller_dqp_list = controller_utils.create_split_constraint_controllers(
        grid, num_agents, partition, plan_horizon, plan_horizon, batt_cost, curt_change_cost,
        curt_net_cost, bus_slack_cost, line_slack_cost, line_terminal_cost=line_terminal_cost,
        batt_curt_terminal_cost=batt_curt_terminal_cost, soft_bus_constraints=soft_bus_constraints)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
    schedule_len = 2 * plan_horizon
    agg_max, agg_min = _aggregate_line_limit_schedule(source_controller_list, schedule_len)
    if agg_max is not None:
        controller_utils.assign_line_limit_changes(
            controller_dqp_list, agg_max.unsqueeze(0), agg_min.unsqueeze(0))
    else:
        for controller in controller_dqp_list:
            controller.reset_params()
    _sync_cycle_start_references(grid, controller_dqp_list)

    noise = _disturbance_horizon(dist, 0, total_steps, wrap=False)
    print(f"solving {total_steps}-step plan ({num_cycles} cycles × {T} steps)")
    start_time = time.time()
    actions_curt, actions_batt, _, _, _, _ = \
        controller_utils.get_next_action(grid, controller_dqp_list, noise, 0, dQPTH_layer, verbose=False, update_state=False)
    print(f"solving takes time {time.time() - start_time}")
    print(f"rolling out {total_steps} steps with drifts")

    viol_per_step = torch.zeros(total_steps)
    curt_cost_per_step = torch.zeros(total_steps)
    batt_cost_per_step = torch.zeros(total_steps)

    for t in range(total_steps):
        if num_cycles > 1 and t > 0 and t % T == 0:
            grid.init_state = grid.state.clone()
            _sync_cycle_start_references(grid, controller_dqp_list)

        action_curt = actions_curt[t]
        action_batt = actions_batt[t]
        grid.update_state(action_curt, action_batt, noise[t, :])

        net_curt = grid.state[grid.state_curt_idx]
        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
        batt_charge_violations = bus_violations[:2*grid.num_batt]
        batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
        curt_violations = bus_violations[4*grid.num_batt:]
        scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
        scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
        scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
        bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                + torch.sum(scaled_batt_power_violations ** 2)
                                                + torch.sum(scaled_curt_violations ** 2))
        line_flows = grid.state[grid.state_line_flow_idx]
        line_limits = grid.line_data[:,5]
        line_violations = torch.relu(torch.abs(line_flows) - line_limits)
        line_violations_loss = _line_violations_loss(line_violations, line_slack_cost)
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_change_loss, curt_net_loss = _curt_loss_parts(
            grid, action_curt, net_curt, curt_change_cost, curt_net_cost)
        curt_cost_per_step[t] = curt_change_loss
        if _is_controlled_rollout_step(t, grid):
            curt_cost_per_step[t] = curt_cost_per_step[t] + curt_net_loss
            viol_per_step[t] = bus_violations_loss + line_violations_loss
            batt_cost_per_step[t] = batt_loss

    curt_cost_per_step = curt_cost_per_step.detach()
    batt_cost_per_step = batt_cost_per_step.detach()
    viol_per_step = viol_per_step.detach()
    total_loss = torch.sum(curt_cost_per_step) + torch.sum(batt_cost_per_step) + torch.sum(viol_per_step)
    total_losses = torch.tensor([total_loss, torch.sum(curt_cost_per_step), torch.sum(viol_per_step)])
    print(total_losses)
    ckpt = {
        "actions_curt": actions_curt.detach(),
        "actions_batt": actions_batt.detach(),
        "viol_per_step": viol_per_step,
        "curt_cost_per_step": curt_cost_per_step,
        "batt_cost_per_step": batt_cost_per_step,
        "total_losses": total_losses,
        "num_cycles": num_cycles,
        "T_per_cycle": T,
    }

    pool.close()
    pool.join()

    return ckpt


def evaluate_base_on_traj(grid, base_controller_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost, *, cycle=0):
    """
    Gets cost associated with running base MPCs on a test trajectory.

    ``cycle`` selects the overlapping noise window within a multi-cycle trajectory
    (same convention as ``evaluate_on_traj`` / ``run_dec`` single-cycle uses cycle=0).
    """
    cycle_disturbances = cycle_noise_window(test_traj, cycle, T, H)
    grid.reset_state()
    for controller in base_controller_list:
        controller.update_line_state()
        controller.update_bus_state()
    # Drop any warm-start bases from split-constraint QPs (different z_dim / nConstr).
    dQPTH_layer.clear_cache()
    actions_curt = torch.zeros(T, grid.num_curt)
    actions_batt = torch.zeros(T, grid.num_batt)
    viol_per_step = torch.zeros(T)
    curt_cost_per_step = torch.zeros(T)
    batt_cost_per_step = torch.zeros(T)
    total_loss = 0
    total_econ_loss = 0
    total_viol_loss = 0
    total_train_loss = 0

    for t in range(T):
        for controller in base_controller_list:
            controller.update_line_state()
            controller.update_bus_state()
        noise = _noise_horizon(cycle_disturbances, t, H)
        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
            controller_utils.get_next_action_base(grid, base_controller_list, noise, t, dQPTH_layer, verbose=False, update_state=True)

        action_curt = pred_actions_curt[0]
        action_batt = pred_actions_batt[0]
        grid.update_state(action_curt, action_batt, noise[0,:])

        # print(f"pred bus viol {torch.sum(pred_bus_slacks[0] ** 2)} pred line max viol {torch.sum(pred_line_max_slacks[:,0]**2)} " +
        #       f"pred line min viol {torch.sum(pred_line_min_slacks[:,0]**2)}" )
        net_curt = grid.state[grid.state_curt_idx]
        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
        line_flows = grid.state[grid.state_line_flow_idx]
        line_limits = grid.line_data[:,5]
        line_violations = torch.relu(torch.abs(line_flows) - line_limits)
        line_violations_loss = _line_violations_loss(line_violations, line_slack_cost)
        # line_violations_loss = line_slack_cost * (torch.sum(pred_line_max_slacks[:,0]**2) + torch.sum(pred_line_min_slacks[:,0]**2))
        # line_violations_loss = line_slack_cost * torch.sum(scaled_line_violations ** 2)
        # print(torch.sum(scaled_line_violations), torch.sum(line_violations))
        # print(f"action batt {torch.sum(action_batt ** 2)}, action curt {torch.sum(action_curt ** 2)}, curt_net {torch.sum(net_curt ** 2)}")
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_change_loss, curt_net_loss = _curt_loss_parts(
            grid, action_curt, net_curt, curt_change_cost, curt_net_cost)
        total_loss += curt_change_loss
        total_train_loss += curt_change_loss
        total_econ_loss += curt_change_loss
        curt_cost_per_step[t] = curt_change_loss
        if _is_controlled_rollout_step(t, grid):
            total_loss += batt_loss + curt_net_loss + line_violations_loss
            total_train_loss += batt_loss + curt_net_loss + _line_violations_loss(line_violations, line_slack_cost)
            total_econ_loss += batt_loss + curt_net_loss
            total_viol_loss += line_violations_loss
            viol_per_step[t] = line_violations_loss
            batt_cost_per_step[t] = batt_loss

        actions_curt[t] = action_curt
        actions_batt[t] = action_batt

    total_losses = torch.tensor([total_loss.detach(), total_econ_loss.detach(), total_viol_loss.detach()])

    ckpt = {
        "actions_curt": actions_curt.detach(),
        "actions_batt": actions_batt.detach(),
        "viol_per_step": viol_per_step.detach(),
        "curt_cost_per_step": curt_cost_per_step.detach(),
        "batt_cost_per_step": batt_cost_per_step.detach(),
        "total_losses": total_losses
    }

    return ckpt

# @profile
def evaluate_on_traj(grid, controller_dqp_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost, *, cycle=0):
    """
    Gets cost associated with running MPC with configured limits on a test trajectory.

    cycle selects the overlapping noise window within a multi-cycle trajectory.
    """
    cycle_disturbances = cycle_noise_window(test_traj, cycle, T, H)
    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
    total_max_margin = torch.zeros(T + H, grid.num_lines)
    total_min_margin = torch.zeros(T + H, grid.num_lines)
    _sync_cycle_start_references(grid, controller_dqp_list)
    _reproject_line_margins(grid, controller_dqp_list, total_max_margin, total_min_margin)

    test_total_loss = torch.zeros(3)
    # Drop any warm-start bases from base-MPC QPs (different z_dim / nConstr).
    dQPTH_layer.clear_cache()

    actions_curt = torch.zeros(T, grid.num_curt)
    actions_batt = torch.zeros(T, grid.num_batt)
    viol_per_step = torch.zeros(T)
    curt_cost_per_step = torch.zeros(T)
    batt_cost_per_step = torch.zeros(T)
    total_loss = 0
    total_econ_loss = 0
    total_viol_loss = 0
    total_train_loss = 0

    for t in range(T):
        noise = _noise_horizon(cycle_disturbances, t, H)
        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks, _ = \
            controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=True)

        action_curt = pred_actions_curt[0]
        action_batt = pred_actions_batt[0]
        grid.update_state(action_curt, action_batt, noise[0, :])

        # print(f"pred bus viol {torch.sum(pred_bus_slacks[0] ** 2)} pred line max viol {torch.sum(pred_line_max_slacks[:,0]**2)} " +
        #       f"pred line min viol {torch.sum(pred_line_min_slacks[:,0]**2)}" )
        net_curt = grid.state[grid.state_curt_idx]
        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
        batt_charge_violations = bus_violations[:2*grid.num_batt]
        batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
        curt_violations = bus_violations[4*grid.num_batt:]
        scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
        scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
        scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
        bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                + torch.sum(scaled_batt_power_violations ** 2)
                                                + torch.sum(scaled_curt_violations ** 2))
        line_flows = grid.state[grid.state_line_flow_idx]
        line_limits = grid.line_data[:,5]
        line_violations = torch.relu(torch.abs(line_flows) - line_limits)
        line_violations_loss = _line_violations_loss(line_violations, line_slack_cost)
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_change_loss, curt_net_loss = _curt_loss_parts(
            grid, action_curt, net_curt, curt_change_cost, curt_net_cost)

        # bus violations no longer included after I changed code to not allow slacks to be set on bus constraints
        total_loss += curt_change_loss
        total_train_loss += curt_change_loss
        total_econ_loss += curt_change_loss
        curt_cost_per_step[t] = curt_change_loss
        if _is_controlled_rollout_step(t, grid):
            total_loss += batt_loss + curt_net_loss + line_violations_loss
            total_train_loss += batt_loss + curt_net_loss + _line_violations_loss(line_violations, line_slack_cost)
            total_econ_loss += batt_loss + curt_net_loss
            total_viol_loss += line_violations_loss
            viol_per_step[t] = line_violations_loss
            batt_cost_per_step[t] = batt_loss

    total_losses = torch.tensor([total_loss.detach(), total_econ_loss.detach(), total_viol_loss.detach()])

    ckpt = {
        "actions_curt": actions_curt.detach(),
        "actions_batt": actions_batt.detach(),
        "viol_per_step": viol_per_step.detach(),
        "curt_cost_per_step": curt_cost_per_step.detach(),
        "batt_cost_per_step": batt_cost_per_step.detach(),
        "total_losses": total_losses
    }

    return ckpt


def train_with_surrogate(grid, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost, epochs, lr,
                         optimizer_type='clipped_gd', max_grad_norm=30, lr_schedule=None, schedule_kwargs=None, batch_size=None,
                         line_terminal_cost=0.0, batt_curt_terminal_cost=0.0):
    """
        Learns line impact allocations over multiple scenarios. Computes approximate gradients via a surrogate on the
        loss incurred by the MPC rollout for a given set of impact allocations.

        optimizer_type: 'clipped_gd' | 'sgd' | 'adam'
        lr_schedule:    None | 'cosine' | 'step_decay' | 'plateau'
        schedule_kwargs: e.g. {'step_size': 30, 'gamma': 0.5} for step_decay
    """
    num_agents = 3
    partition = training_partition()

    # surrogate solves all T steps altogether.
    # We set T=H, H=T since this gets us the size-T prediction window while also setting us up to learn limits for T+H steps.
    # TODO: do this in a way that is less hacky, also think about the fact that size-T prediction window uses only T limits, while
    # MPC rollout uses T+H
    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, H, T, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost, line_terminal_cost=line_terminal_cost,
                                                            batt_curt_terminal_cost=batt_curt_terminal_cost,
                                                            soft_bus_constraints=True)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    num_traj = all_train_traj.shape[0]
    if batch_size is None:
        batch_size = num_traj
    steps_per_epoch = (num_traj + batch_size - 1) // batch_size

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    param_groups = [{'params': [controller.line_max_change, controller.line_min_change]} for controller in controller_dqp_list]
    optimizer = _make_optimizer(param_groups, optimizer_type, lr)
    scheduler = _make_scheduler(optimizer, lr_schedule, schedule_kwargs, epochs * steps_per_epoch)

    total_max_margin = torch.zeros(T+H, grid.num_lines)
    total_min_margin = torch.zeros(T+H, grid.num_lines)
    _reproject_line_margins(grid, controller_dqp_list, total_max_margin, total_min_margin)

    run_failed = False
    fail_message = ""

    total_losses = torch.zeros(epochs)
    total_econ_losses = torch.zeros(epochs)
    total_viol_losses = torch.zeros(epochs)
    total_train_losses = torch.zeros(epochs)
    total_terminal_losses = torch.zeros(epochs)
    lr_reduction_epochs = []  # (epoch, batch_start, new_lr) each time plateau fires

    for epoch in tqdm(range(epochs)):
        epoch_total_loss = 0
        epoch_econ_loss = 0
        epoch_viol_loss = 0
        epoch_train_loss = 0
        epoch_terminal_loss = 0
        epoch_mpc_terminal = 0
        epoch_batt = 0
        epoch_curt_change = 0
        epoch_curt_net = 0
        epoch_bus_viol = 0
        epoch_line_viol = 0
        perm = torch.randperm(num_traj)

        for batch_start in tqdm(range(0, num_traj, batch_size)):
            batch_idx = perm[batch_start:batch_start + batch_size]
            print(f"batch starting at {batch_start}, batch_idx {batch_idx}")
            actual_batch_size = len(batch_idx)
            optimizer.zero_grad()
            batch_total_loss = 0

            for traj in batch_idx:
                train_disturbances = all_train_traj[traj]

                total_loss = 0
                total_econ_loss = 0
                total_viol_loss = 0
                total_train_loss = 0
                traj_mpc_terminal = 0
                traj_batt = 0
                traj_curt_change = 0
                traj_curt_net = 0
                traj_bus_viol = 0
                traj_line_viol = 0
                grid.reset_state()
                for controller in controller_dqp_list:
                    controller.zero_line_state()
                    controller.update_bus_state()

                start_time = time.time()
                pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks, mpc_terminal_loss = \
                        controller_utils.get_next_action(grid, controller_dqp_list, train_disturbances[0:T,:], 0, dQPTH_layer, verbose=False, update_state=True)
                total_loss += mpc_terminal_loss
                total_train_loss += mpc_terminal_loss
                total_econ_loss += mpc_terminal_loss
                traj_mpc_terminal += mpc_terminal_loss.detach()
                for t in range(T):
                    noise = train_disturbances[t]
                    action_curt = pred_actions_curt[t]
                    action_batt = pred_actions_batt[t]
                    grid.update_state(action_curt, action_batt, noise)

                    net_curt = grid.state[grid.state_curt_idx]
                    bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
                    batt_charge_violations = bus_violations[:2*grid.num_batt]
                    batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
                    curt_violations = bus_violations[4*grid.num_batt:]
                    scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
                    scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
                    scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
                    bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                        + torch.sum(scaled_batt_power_violations ** 2)
                                                        + torch.sum(scaled_curt_violations ** 2))
                    line_flows = grid.state[grid.state_line_flow_idx]
                    line_limits = grid.line_data[:,5]
                    line_violations = torch.relu(torch.abs(line_flows) - line_limits)
                    line_violations_loss = _line_violations_loss(line_violations, line_slack_cost)
                    batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
                    batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
                    curt_change_loss, curt_net_loss = _curt_loss_parts(
                        grid, action_curt, net_curt, curt_change_cost, curt_net_cost)
                    total_loss += curt_change_loss
                    total_train_loss += curt_change_loss
                    total_econ_loss += curt_change_loss
                    traj_curt_change += curt_change_loss.detach()
                    if _is_controlled_rollout_step(t, grid):
                        total_loss += batt_loss + curt_net_loss + bus_violations_loss + line_violations_loss
                        total_train_loss += batt_loss + curt_net_loss + bus_violations_loss + _line_violations_loss(line_violations, line_slack_cost)
                        total_econ_loss += batt_loss + curt_net_loss
                        total_viol_loss += bus_violations_loss + line_violations_loss
                        traj_batt += batt_loss.detach()
                        traj_curt_net += curt_net_loss.detach()
                        traj_bus_viol += bus_violations_loss.detach()
                        traj_line_viol += line_violations_loss.detach()
                start_time = time.time()
                (total_loss / actual_batch_size).backward()
                print(f"backward pass for one traj takes time {time.time() - start_time}")

                print(_format_loss_breakdown(
                    traj_mpc_terminal, traj_batt, traj_curt_change, traj_curt_net, traj_bus_viol, traj_line_viol), flush=True)

                batch_total_loss += total_loss.detach()
                epoch_total_loss += total_loss.detach()
                epoch_econ_loss += total_econ_loss.detach()
                epoch_viol_loss += total_viol_loss.detach()
                epoch_train_loss += total_train_loss.detach()
                epoch_terminal_loss += traj_mpc_terminal
                epoch_mpc_terminal += traj_mpc_terminal
                epoch_batt += traj_batt
                epoch_curt_change += traj_curt_change
                epoch_curt_net += traj_curt_net
                epoch_bus_viol += traj_bus_viol
                epoch_line_viol += traj_line_viol

            if optimizer_type == 'clipped_gd':
                all_params = [controller.line_max_change for controller in controller_dqp_list] + \
                             [controller.line_min_change for controller in controller_dqp_list]
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

            optimizer.step()
            _project_params(controller_dqp_list, max_target_sum=total_max_margin, min_target_sum=total_min_margin)

        prev_lr = optimizer.param_groups[0]['lr']
        _step_scheduler(scheduler, epoch_total_loss / num_traj)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            lr_reduction_epochs.append((epoch, new_lr))

        print(
            f"Epoch {epoch+1}/{epochs}, {_format_loss_breakdown(epoch_mpc_terminal, epoch_batt, epoch_curt_change, epoch_curt_net, epoch_bus_viol, epoch_line_viol)}",
            flush=True)
        total_losses[epoch] = epoch_total_loss
        total_econ_losses[epoch] = epoch_econ_loss
        total_viol_losses[epoch] = epoch_viol_loss
        total_train_losses[epoch] = epoch_train_loss
        total_terminal_losses[epoch] = epoch_terminal_loss

    line_max_changes = torch.zeros(num_agents, T+H, grid.num_lines)
    line_min_changes = torch.zeros(num_agents, T+H, grid.num_lines)
    for i in range(num_agents):
        line_max_changes[i] = controller_dqp_list[i].line_max_change
        line_min_changes[i] = controller_dqp_list[i].line_min_change

    ckpt = dict()
    ckpt["line_max_changes"] = line_max_changes
    ckpt["line_min_changes"] = line_min_changes
    ckpt["total_losses_per_iter"] = total_losses
    ckpt["total_econ_losses_per_iter"] = total_econ_losses
    ckpt["total_viol_losses_per_iter"] = total_viol_losses
    ckpt["total_train_losses_per_iter"] = total_train_losses
    ckpt["total_terminal_losses_per_iter"] = total_terminal_losses
    ckpt["lr_reduction_epochs"] = lr_reduction_epochs
    ckpt["run_failed"] = run_failed
    ckpt["fail_message"] = fail_message
    ckpt["all_train_traj"] = all_train_traj
    pool.close()
    return ckpt


def _assert_soft_bus_controllers(controller_list, *, label):
    """Fail fast if any controller lacks soft bus slacks (needed for multi-cycle feasibility)."""
    bad = [
        i for i, c in enumerate(controller_list)
        if not getattr(c, "soft_bus_constraints", False)
        or getattr(c, "bus_slack_start_idx", None) is None
    ]
    if bad:
        raise RuntimeError(
            f"{label}: controllers {bad} are missing soft bus constraints "
            f"(soft_bus_constraints / bus_slack_start_idx required for cycle feasibility)")


def _restore_cycle_start(grid, cycle_start_state, cycle_start_target_batt_charges,
                         cycle_start_action_buffers=None):
    """Restore plant init_state and matching delay buffers (physics-consistent handoff)."""
    grid.target_batt_charges = cycle_start_target_batt_charges.detach().clone()
    if cycle_start_action_buffers is not None:
        grid.set_cycle_handoff(cycle_start_state, cycle_start_action_buffers)
    else:
        grid.init_state = cycle_start_state.detach().clone()
    grid.reset_state()


def _reset_grid_for_cycle_test(grid, controller_dqp_list, total_max_margin, total_min_margin,
                               cycle_start_state, cycle_start_target_batt_charges,
                               cycle_start_action_buffers=None):
    """Reset to this cycle's start state, keep learned line limits, and refresh MPC references."""
    _restore_cycle_start(
        grid, cycle_start_state, cycle_start_target_batt_charges, cycle_start_action_buffers)
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
    _sync_cycle_start_references(grid, controller_dqp_list)
    # Keep learned margins; only re-anchor totals to this cycle-start headroom.
    _reproject_line_margins(grid, controller_dqp_list, total_max_margin, total_min_margin)


def _reset_grid_for_base_cycle_test(grid, base_controller_list, cycle_start_state,
                                    cycle_start_target_batt_charges,
                                    cycle_start_action_buffers=None):
    """Reset to this cycle's start state for the run_dec base-MPC baseline."""
    _restore_cycle_start(
        grid, cycle_start_state, cycle_start_target_batt_charges, cycle_start_action_buffers)
    for controller in base_controller_list:
        controller.update_line_state()
        controller.update_bus_state()
    _sync_cycle_start_references(grid, base_controller_list)


def _ensure_cycle_start_feasible(grid, controller_dqp_list, total_max_margin, total_min_margin,
                                 cycle_start_state, cycle_start_target_batt_charges,
                                 cycle_start_action_buffers, dQPTH_layer, H, *, cycle=0):
    """Prepare a cycle-start operating point whose soft-bus MPC model is solvable.

    Leaves the physical plant state and delay buffers untouched; only margin parameters and
    QP warm-start caches may change. Logs plant-box feasibility for diagnostics.
    """
    _restore_cycle_start(
        grid, cycle_start_state, cycle_start_target_batt_charges, cycle_start_action_buffers)
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
    _sync_cycle_start_references(grid, controller_dqp_list)
    # Prefer carrying learned margins; only wipe t=0/future increments if headroom is
    # overloaded or the cold probe below fails.
    _reproject_line_margins(
        grid, controller_dqp_list, total_max_margin, total_min_margin,
        auto_repair_overloaded=True)
    dQPTH_layer.clear_cache()

    feasible, infeas_idx, violations = grid.check_state_feasible()
    if not feasible:
        print(
            f"Cycle {cycle + 1} start plant has {len(infeas_idx)} box-constraint violation(s) "
            f"(max={float(violations.max()) if len(violations) else 0:.4g}); "
            f"keeping physical state and relying on soft bus/line slacks",
            flush=True)

    # Cold probe: one MPC solve from this start with zero forecast noise.
    noise = torch.zeros(H, grid.num_buses)
    try:
        controller_utils.get_next_action(
            grid, controller_dqp_list, noise, 0, dQPTH_layer, verbose=False, update_state=False)
    except Exception as e:
        print(
            f"Cycle {cycle + 1} start MPC probe failed ({e}); "
            f"forcing equal t=0 margin split and retrying",
            flush=True)
        _reproject_line_margins(
            grid, controller_dqp_list, total_max_margin, total_min_margin,
            repair_boundary=True)
        dQPTH_layer.clear_cache()
        controller_utils.get_next_action(
            grid, controller_dqp_list, noise, 0, dQPTH_layer, verbose=False, update_state=False)
    finally:
        # Probe must not leave the plant advanced / buffers mutated when update_state=False,
        # but re-seed from the cycle snapshot for safety.
        _restore_cycle_start(
            grid, cycle_start_state, cycle_start_target_batt_charges, cycle_start_action_buffers)
        for controller in controller_dqp_list:
            controller.zero_line_state()
            controller.update_bus_state()
        _sync_cycle_start_references(grid, controller_dqp_list)
        dQPTH_layer.clear_cache()


def _run_cycle_tests(grid, controller_dqp_list, dQPTH_layer, cycle_test_trajs, T, H,
                     batt_cost, curt_change_cost, curt_net_cost,
                     bus_slack_cost, line_slack_cost, total_max_margin, total_min_margin,
                     cycle_start_state, cycle_start_target_batt_charges, *,
                     cycle_start_action_buffers=None,
                     compare_optimal=True, compare_dec=False, base_controller_list=None,
                     line_terminal_cost=0.0, batt_curt_terminal_cost=0.0,
                     cycle=0):
    """Evaluate learned split-constraint MPC from this cycle's start state.

    Optionally compares against:
      - centralized open-loop planner (``compare_optimal``)
      - fixed base decentralized MPC from ``run_dec`` (``compare_dec``)

    Each trajectory is rolled out from ``cycle_start_state`` (not the construction baseline)
    with the current learned line limits. Returns per-trajectory losses plus the terminal grid
    state and delay-action buffers from the last learned test rollout (next-cycle handoff).
    """
    if compare_dec and base_controller_list is None:
        raise ValueError("compare_dec=True requires base_controller_list")

    num_traj = cycle_test_trajs.shape[0]
    our_losses = torch.zeros(num_traj, 3)
    optimal_losses = torch.zeros(num_traj, 3) if compare_optimal else None
    dec_losses = torch.zeros(num_traj, 3) if compare_dec else None
    handoff_terminal_state = None
    handoff_action_buffers = None

    with torch.no_grad():
        for j in range(num_traj):
            test_traj = cycle_test_trajs[j]
            try:
                _reset_grid_for_cycle_test(
                    grid, controller_dqp_list, total_max_margin, total_min_margin,
                    cycle_start_state, cycle_start_target_batt_charges,
                    cycle_start_action_buffers)
                dQPTH_layer.clear_cache()
                ckpt_ours = evaluate_on_traj(
                    grid, controller_dqp_list, dQPTH_layer, test_traj, T, H,
                    batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                    cycle=cycle)
                our_losses[j] = ckpt_ours["total_losses"]
                # Capture before dec/optimal comparisons mutate the grid / clear delay buffers.
                handoff_terminal_state = grid.state.detach().clone()
                handoff_action_buffers = grid.snapshot_action_buffers()
            except Exception as e:
                print(
                    f"Cycle {cycle + 1} ours test traj {j} failed: {e}",
                    flush=True)
                our_losses[j] = -1
                dQPTH_layer.clear_cache()

            if compare_dec:
                try:
                    _reset_grid_for_base_cycle_test(
                        grid, base_controller_list,
                        cycle_start_state, cycle_start_target_batt_charges,
                        cycle_start_action_buffers)
                    dQPTH_layer.clear_cache()
                    ckpt_dec = evaluate_base_on_traj(
                        grid, base_controller_list, dQPTH_layer, test_traj, T, H,
                        batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                        cycle=cycle)
                    dec_losses[j] = ckpt_dec["total_losses"]
                except Exception as e:
                    print(
                        f"Cycle {cycle + 1} dec (base) test traj {j} failed: {e}",
                        flush=True)
                    dec_losses[j] = -1
                    dQPTH_layer.clear_cache()

            if compare_optimal:
                try:
                    _reset_grid_for_cycle_test(
                        grid, controller_dqp_list, total_max_margin, total_min_margin,
                        cycle_start_state, cycle_start_target_batt_charges,
                        cycle_start_action_buffers)
                    cycle_disturbances = cycle_noise_window(test_traj, cycle, T, H)
                    # Preserve cycle-start seed across the central planner (it rolls the plant).
                    saved_init = grid.init_state.detach().clone()
                    saved_buffers = grid.init_action_buffers
                    ckpt_central = get_central_full_traj(
                        grid, T, H, cycle_disturbances, batt_cost, curt_change_cost, curt_net_cost,
                        bus_slack_cost, line_slack_cost,
                        line_terminal_cost=line_terminal_cost, batt_curt_terminal_cost=batt_curt_terminal_cost,
                        num_cycles=1, soft_bus_constraints=True,
                        source_controller_list=controller_dqp_list)
                    optimal_losses[j] = ckpt_central["total_losses"]
                    grid.init_state = saved_init
                    grid.init_action_buffers = saved_buffers
                    grid.reset_state()
                except Exception as e:
                    print(
                        f"Cycle {cycle + 1} optimal test traj {j} failed: {e}",
                        flush=True)
                    optimal_losses[j] = -1
                    dQPTH_layer.clear_cache()

    def _sum_valid(losses):
        ok = losses[:, 0] >= 0
        if not bool(ok.any()):
            return float("nan"), float("nan"), float("nan")
        tot = losses[ok].sum(dim=0)
        return float(tot[0]), float(tot[1]), float(tot[2])

    our_total, our_econ, our_viol = _sum_valid(our_losses)
    result = {
        "our_losses_per_traj": our_losses,
        "our_total": our_total,
        "our_econ": our_econ,
        "our_viol": our_viol,
    }
    if compare_dec:
        dec_total, dec_econ, dec_viol = _sum_valid(dec_losses)
        result.update({
            "dec_losses_per_traj": dec_losses,
            "dec_total": dec_total,
            "dec_econ": dec_econ,
            "dec_viol": dec_viol,
            "dec_gap": our_total - dec_total if our_total == our_total and dec_total == dec_total else float("nan"),
            "dec_ratio": (our_total / dec_total) if (dec_total == dec_total and dec_total != 0) else float("nan"),
        })
    if compare_optimal:
        opt_total, opt_econ, opt_viol = _sum_valid(optimal_losses)
        result.update({
            "optimal_losses_per_traj": optimal_losses,
            "optimal_total": opt_total,
            "optimal_econ": opt_econ,
            "optimal_viol": opt_viol,
            "gap": our_total - opt_total if our_total == our_total and opt_total == opt_total else float("nan"),
            "ratio": (our_total / opt_total) if (opt_total == opt_total and opt_total != 0) else float("nan"),
        })
    return handoff_terminal_state, handoff_action_buffers, result


def _optimal_cycle_comparison(grid, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                              line_terminal_cost=0.0, batt_curt_terminal_cost=0.0, *,
                              controller_dqp_list=None, cycle=0):
    """Run the centralized planner for a single T-step cycle on every training trajectory.

    Each solve starts from the current ``grid.init_state`` (the cycle-start state), so the
    comparison uses the same initial operating point as the just-trained decentralized cycle.
    ``grid`` references are saved and restored so the caller's training state is unaffected.

    Returns a (num_traj, 3) tensor of centralized ``[total, econ, viol]`` losses per trajectory.
    """
    saved_init_state = grid.init_state.clone()
    saved_state = grid.state.clone()
    saved_targets = grid.target_batt_charges.clone()
    saved_action_buffers = (
        grid.snapshot_action_buffers() if grid.init_action_buffers is not None else None
    )
    saved_init_action_buffers = grid.init_action_buffers

    num_traj = all_train_traj.shape[0]
    optimal_losses = torch.zeros(num_traj, 3)
    for traj in range(num_traj):
        # Restore the cycle-start init so every centralized solve sees the same starting point.
        grid.init_state = saved_init_state.clone()
        grid.target_batt_charges = saved_targets.clone()
        grid.init_action_buffers = saved_init_action_buffers
        cycle_disturbances = cycle_noise_window(all_train_traj[traj], cycle, T, H)
        ckpt_central = get_central_full_traj(
            grid, T, H, cycle_disturbances, batt_cost, curt_change_cost, curt_net_cost,
            bus_slack_cost, line_slack_cost,
            line_terminal_cost=line_terminal_cost, batt_curt_terminal_cost=batt_curt_terminal_cost,
            num_cycles=1, soft_bus_constraints=True,
            source_controller_list=controller_dqp_list)
        optimal_losses[traj] = ckpt_central["total_losses"]

    grid.init_state = saved_init_state
    grid.state = saved_state
    grid.target_batt_charges = saved_targets
    grid.init_action_buffers = saved_init_action_buffers
    grid.restore_action_buffers(saved_action_buffers)
    return optimal_losses


def train_with_rollout(grid, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost, epochs, lr,
                      optimizer_type='clipped_gd', max_grad_norm=30, lr_schedule=None, schedule_kwargs=None, batch_size=None,
                      line_terminal_cost=0.0, batt_curt_terminal_cost=0.0, num_cycles=1,
                      cycle_test_trajs=None, compare_optimal_each_cycle=True, compare_dec_each_cycle=True):
    """
        Learns line impact allocations over multiple scenarios.

        Each cycle:
        1. Train for ``epochs`` on T-step rollouts.
        2. Reset the grid to this cycle's start state while keeping learned line limits
           (totals are reprojected to the current headroom; the per-agent schedule is carried).
        3. Roll out on ``cycle_test_trajs`` (if provided) and optionally compare to the
           centralized planner and/or the fixed base decentralized MPC (``run_dec``).
        4. Seed the next cycle from the terminal state (+ delay-action buffers) of the last
           learned test rollout, carrying learned margins forward (reprojected to the
           handoff headroom).
    """
    if num_cycles < 1:
        raise ValueError(f"num_cycles must be >= 1, got {num_cycles}")

    T_horizon = T
    num_agents = 3
    partition = training_partition()

    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T_horizon, H, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost, line_terminal_cost=line_terminal_cost,
                                                            batt_curt_terminal_cost=batt_curt_terminal_cost,
                                                            soft_bus_constraints=True)
    _assert_soft_bus_controllers(controller_dqp_list, label="learned split MPC")
    # Soft-bus base for every cycle so delayed/congested starts stay QP-feasible while
    # leaving plant handoff state unchanged (same cost weights; bus limits as penalties).
    base_controller_list = None
    if compare_dec_each_cycle:
        base_controller_list = controller_utils.create_base_controllers(
            grid, num_agents, partition, T, H, batt_cost, curt_change_cost,
            curt_net_cost, bus_slack_cost, line_slack_cost,
            soft_bus_constraints=True)
        for controller in base_controller_list:
            controller.init_matrices()
        _assert_soft_bus_controllers(base_controller_list, label="base (dec) MPC")
        print(
            "Base (dec) MPC uses soft bus constraints on every cycle",
            flush=True)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    num_traj = all_train_traj.shape[0]
    if batch_size is None:
        batch_size = num_traj
    steps_per_epoch = (num_traj + batch_size - 1) // batch_size

    total_max_margin = torch.zeros(T_horizon + H, grid.num_lines)
    total_min_margin = torch.zeros(T_horizon + H, grid.num_lines)

    run_failed = False
    fail_message = ""

    total_losses = torch.full((num_cycles, epochs), float("nan"))
    total_econ_losses = torch.full((num_cycles, epochs), float("nan"))
    total_viol_losses = torch.full((num_cycles, epochs), float("nan"))
    total_train_losses = torch.full((num_cycles, epochs), float("nan"))
    total_terminal_losses = torch.full((num_cycles, epochs), float("nan"))
    lr_reduction_epochs = []
    optimal_comparison = {}
    dec_comparison = {}
    cycle_test_results = {}

    if cycle_test_trajs is None and num_cycles > 1:
        cycle_test_trajs = all_train_traj
    run_cycle_tests = cycle_test_trajs is not None and num_cycles > 1

    _validate_disturbance_trajectories(all_train_traj, num_cycles, T, H, label="training")
    if cycle_test_trajs is not None:
        _validate_disturbance_trajectories(cycle_test_trajs, num_cycles, T, H, label="cycle test")

    for cycle in range(num_cycles):
        # Each cycle tracks its own LOCALLY-predicted line flows (line_state is seeded to 0 against
        # this cycle's init reference and then evolves via each agent's local model). Force-feeding
        # the real GLOBAL flow delta (relative sync) breaks the communication-limited design: an
        # agent's area-restricted model cannot reproduce cross-area/PTDF-coupled flows, so the
        # learned margins get applied to flows the agent never predicted and curtailment explodes.
        # tmp_resync_test.py shows this inflates single-cycle curt_net ~42x (1.1e3 -> 4.7e4).
        resync_line_state = False
        if cycle == 0:
            grid.reset_state()
            for controller in controller_dqp_list:
                controller.zero_line_state()
                controller.update_bus_state()
                controller.reset_params()
            _sync_cycle_start_references(grid, controller_dqp_list)
            if resync_line_state:
                _sync_mpc_from_grid(grid, controller_dqp_list)
            cycle_start_state = grid.init_state.detach().clone()
            cycle_start_target_batt_charges = grid.target_batt_charges.detach().clone()
            cycle_start_action_buffers = None
        else:
            # Start from the terminal state left by the previous cycle's test rollout.
            grid.reset_state()
            for controller in controller_dqp_list:
                controller.zero_line_state()
                controller.update_bus_state()
            _sync_cycle_start_references(grid, controller_dqp_list)
            dQPTH_layer.clear_cache()
            cycle_start_state = grid.init_state.detach().clone()
            cycle_start_target_batt_charges = grid.target_batt_charges.detach().clone()
            cycle_start_action_buffers = (
                grid.snapshot_action_buffers() if grid.init_action_buffers is not None else None)
        # Soft-bus MPC feasibility check / margin repair at the start of every cycle.
        _ensure_cycle_start_feasible(
            grid, controller_dqp_list, total_max_margin, total_min_margin,
            cycle_start_state, cycle_start_target_batt_charges, cycle_start_action_buffers,
            dQPTH_layer, H, cycle=cycle)
        if base_controller_list is not None:
            _assert_soft_bus_controllers(
                base_controller_list, label=f"base (dec) MPC at cycle {cycle + 1}")
        print(
            f"Cycle {cycle + 1}/{num_cycles}",
            flush=True)

        if cycle == 0:
            param_groups = [{'params': [controller.line_max_change, controller.line_min_change]} for controller in controller_dqp_list]
            optimizer = _make_optimizer(param_groups, optimizer_type, lr)
        else:
            # Plateau decays carry across cycles otherwise; restore the configured lr each cycle.
            for group in optimizer.param_groups:
                group["lr"] = lr
        scheduler = _make_scheduler(optimizer, lr_schedule, schedule_kwargs, epochs * steps_per_epoch)

        print(
            f"Starting training cycle {cycle + 1}/{num_cycles} ({epochs} epochs, lr={lr})",
            flush=True)
        for epoch in tqdm(range(epochs), desc=f"cycle {cycle + 1}/{num_cycles}"):
            epoch_total_loss = 0
            epoch_econ_loss = 0
            epoch_viol_loss = 0
            epoch_train_loss = 0
            epoch_terminal_loss = 0
            epoch_mpc_terminal = 0
            epoch_batt = 0
            epoch_curt_change = 0
            epoch_curt_net = 0
            epoch_bus_viol = 0
            epoch_line_viol = 0
            perm = torch.randperm(num_traj)

            for batch_start in tqdm(range(0, num_traj, batch_size)):
                batch_idx = perm[batch_start:batch_start + batch_size]
                print(f"batch starting at {batch_start}, batch_idx {batch_idx}")
                actual_batch_size = len(batch_idx)
                optimizer.zero_grad()
                batch_total_loss = 0

                for traj in batch_idx:
                    train_disturbances = all_train_traj[traj]
                    cycle_disturbances = cycle_noise_window(train_disturbances, cycle, T, H)

                    total_loss = 0
                    total_econ_loss = 0
                    total_viol_loss = 0
                    total_train_loss = 0
                    traj_mpc_terminal = 0
                    traj_batt = 0
                    traj_curt_change = 0
                    traj_curt_net = 0
                    traj_bus_viol = 0
                    traj_line_viol = 0
                    _begin_traj_rollout(
                        grid, controller_dqp_list, reset_grid=True,
                        resync_line_state_each_step=resync_line_state)
                    # Drop previous-traj bases so t=0 is cold; avoids stale warm-starts across
                    # handoff / traj noise changes (control_delay amplifies RHS jumps).
                    dQPTH_layer.clear_cache()

                    start_time = time.time()
                    for t in range(T):
                        if resync_line_state:
                            _sync_mpc_from_grid(grid, controller_dqp_list)
                        noise = _noise_horizon(cycle_disturbances, t, H)
                        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks, mpc_terminal_loss = \
                            controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=True)
                        action_curt = pred_actions_curt[0]
                        action_batt = pred_actions_batt[0]
                        grid.update_state(action_curt, action_batt, noise[0])

                        net_curt = _rollout_net_curt(grid)
                        curt_change_loss, curt_net_loss = _curt_loss_parts(
                            grid, action_curt, net_curt, curt_change_cost, curt_net_cost)
                        total_loss += curt_change_loss
                        total_train_loss += curt_change_loss
                        total_econ_loss += curt_change_loss
                        traj_curt_change += curt_change_loss.detach()

                        if not _is_controlled_rollout_step(t, grid):
                            continue

                        total_loss += mpc_terminal_loss
                        total_train_loss += mpc_terminal_loss
                        total_econ_loss += mpc_terminal_loss
                        traj_mpc_terminal += mpc_terminal_loss.detach()
                        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
                        batt_charge_violations = bus_violations[:2*grid.num_batt]
                        batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
                        curt_violations = bus_violations[4*grid.num_batt:]
                        scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
                        scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
                        scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
                        bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                            + torch.sum(scaled_batt_power_violations ** 2)
                                                            + torch.sum(scaled_curt_violations ** 2))
                        line_flows = grid.state[grid.state_line_flow_idx]
                        line_limits = grid.line_data[:,5]
                        line_violations = torch.relu(torch.abs(line_flows) - line_limits)
                        line_violations_loss = _line_violations_loss(line_violations, line_slack_cost)
                        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
                        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
                        total_loss += batt_loss + curt_net_loss + bus_violations_loss + line_violations_loss
                        total_train_loss += batt_loss + curt_net_loss + bus_violations_loss + _line_violations_loss(line_violations, line_slack_cost)
                        total_econ_loss += batt_loss + curt_net_loss
                        total_viol_loss += bus_violations_loss + line_violations_loss
                        traj_batt += batt_loss.detach()
                        traj_curt_net += curt_net_loss.detach()
                        traj_bus_viol += bus_violations_loss.detach()
                        traj_line_viol += line_violations_loss.detach()
                    start_time = time.time()
                    (total_loss / actual_batch_size).backward()
                    print(f"backward pass for one traj takes time {time.time() - start_time}")
                    # print(f"forward pass takes time {time.time() - start_time}")

                    print(_format_loss_breakdown(
                        traj_mpc_terminal, traj_batt, traj_curt_change, traj_curt_net, traj_bus_viol, traj_line_viol), flush=True)

                    batch_total_loss += total_loss.detach()
                    epoch_total_loss += total_loss.detach()
                    epoch_econ_loss += total_econ_loss.detach()
                    epoch_viol_loss += total_viol_loss.detach()
                    epoch_train_loss += total_train_loss.detach()
                    epoch_terminal_loss += traj_mpc_terminal
                    epoch_mpc_terminal += traj_mpc_terminal
                    epoch_batt += traj_batt
                    epoch_curt_change += traj_curt_change
                    epoch_curt_net += traj_curt_net
                    epoch_bus_viol += traj_bus_viol
                    epoch_line_viol += traj_line_viol

                if optimizer_type == 'clipped_gd':
                    all_params = [controller.line_max_change for controller in controller_dqp_list] + \
                                 [controller.line_min_change for controller in controller_dqp_list]
                    torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

                optimizer.step()
                _project_params(controller_dqp_list, max_target_sum=total_max_margin, min_target_sum=total_min_margin)

            prev_lr = optimizer.param_groups[0]['lr']
            _step_scheduler(scheduler, epoch_total_loss / num_traj)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < prev_lr:
                lr_reduction_epochs.append((cycle, epoch, new_lr))

            print(
                f"Cycle {cycle + 1}/{num_cycles}, epoch {epoch+1}/{epochs}, "
                f"{_format_loss_breakdown(epoch_mpc_terminal, epoch_batt, epoch_curt_change, epoch_curt_net, epoch_bus_viol, epoch_line_viol)}",
                flush=True)
            total_losses[cycle, epoch] = epoch_total_loss
            total_econ_losses[cycle, epoch] = epoch_econ_loss
            total_viol_losses[cycle, epoch] = epoch_viol_loss
            total_train_losses[cycle, epoch] = epoch_train_loss
            total_terminal_losses[cycle, epoch] = epoch_terminal_loss

        cycle_test_handoff_state = None
        cycle_test_handoff_buffers = None
        if run_cycle_tests:
            print(
                f"Cycle {cycle + 1}/{num_cycles}: resetting to cycle start state with learned limits, "
                f"running {cycle_test_trajs.shape[0]} test rollouts",
                flush=True)
            cycle_test_handoff_state, cycle_test_handoff_buffers, cycle_test_result = _run_cycle_tests(
                grid, controller_dqp_list, dQPTH_layer, cycle_test_trajs, T, H,
                batt_cost, curt_change_cost, curt_net_cost,
                bus_slack_cost, line_slack_cost, total_max_margin, total_min_margin,
                cycle_start_state, cycle_start_target_batt_charges,
                cycle_start_action_buffers=cycle_start_action_buffers,
                compare_optimal=compare_optimal_each_cycle,
                compare_dec=compare_dec_each_cycle,
                base_controller_list=base_controller_list,
                line_terminal_cost=line_terminal_cost, batt_curt_terminal_cost=batt_curt_terminal_cost,
                cycle=cycle)
            cycle_test_results[cycle] = cycle_test_result
            msg = (
                f"Cycle {cycle + 1} test: our_total={cycle_test_result['our_total']:.4f} "
                f"(econ={cycle_test_result['our_econ']:.4f}, viol={cycle_test_result['our_viol']:.4f})"
            )
            if compare_dec_each_cycle:
                msg += (
                    f" | dec_total={cycle_test_result['dec_total']:.4f} "
                    f"dec_gap={cycle_test_result['dec_gap']:.4f} "
                    f"dec_ratio={cycle_test_result['dec_ratio']:.4f}"
                )
                dec_comparison[cycle] = {
                    "learned_total": cycle_test_result["our_total"],
                    "learned_econ": cycle_test_result["our_econ"],
                    "learned_viol": cycle_test_result["our_viol"],
                    "dec_total": cycle_test_result["dec_total"],
                    "dec_econ": cycle_test_result["dec_econ"],
                    "dec_viol": cycle_test_result["dec_viol"],
                    "dec_losses_per_traj": cycle_test_result["dec_losses_per_traj"],
                    "gap": cycle_test_result["dec_gap"],
                    "ratio": cycle_test_result["dec_ratio"],
                }
            if compare_optimal_each_cycle:
                msg += (
                    f" | optimal_total={cycle_test_result['optimal_total']:.4f} "
                    f"gap={cycle_test_result['gap']:.4f} ratio={cycle_test_result['ratio']:.4f}"
                )
                optimal_comparison[cycle] = {
                    "learned_total": cycle_test_result["our_total"],
                    "learned_econ": cycle_test_result["our_econ"],
                    "learned_viol": cycle_test_result["our_viol"],
                    "optimal_total": cycle_test_result["optimal_total"],
                    "optimal_econ": cycle_test_result["optimal_econ"],
                    "optimal_viol": cycle_test_result["optimal_viol"],
                    "optimal_losses_per_traj": cycle_test_result["optimal_losses_per_traj"],
                    "gap": cycle_test_result["gap"],
                    "ratio": cycle_test_result["ratio"],
                }
            print(msg, flush=True)
            _sync_mpc_from_grid(grid, controller_dqp_list)
            dQPTH_layer.clear_cache()

        if cycle < num_cycles - 1:
            if cycle_test_handoff_state is not None:
                _prepare_cycle_handoff_state(
                    grid, cycle_test_handoff_state, cycle_test_handoff_buffers, cycle_start_state)
                handoff_source = "test rollout terminal state"
            else:
                terminal_state, terminal_buffers = _rollout_terminal_handoff(
                    grid, controller_dqp_list, all_train_traj[0], T, H, dQPTH_layer,
                    resync_line_state_each_step=resync_line_state, cycle=cycle)
                _prepare_cycle_handoff_state(
                    grid, terminal_state, terminal_buffers, cycle_start_state)
                handoff_source = "training reference rollout terminal state"
            n_issued = (
                len(grid.init_action_buffers["issued_curt"])
                if grid.init_action_buffers is not None else 0
            )
            n_pending = (
                len(grid.init_action_buffers["pending_curt"])
                if grid.init_action_buffers is not None else 0
            )
            _sync_cycle_start_references(grid, controller_dqp_list)
            # Same sum-projection as before; overloaded t=0 repair (if needed) runs in
            # _ensure_cycle_start_feasible at the start of the next cycle only.
            _reproject_line_margins(grid, controller_dqp_list, total_max_margin, total_min_margin)
            dQPTH_layer.clear_cache()
            print(
                f"Cycle {cycle + 1} complete; next cycle starts from {handoff_source} "
                f"({n_issued} issued, {n_pending} pending action buffer(s); "
                f"carrying learned line margins)",
                flush=True)

    line_max_changes = torch.zeros(num_agents, T_horizon + H, grid.num_lines)
    line_min_changes = torch.zeros(num_agents, T_horizon + H, grid.num_lines)
    for i in range(num_agents):
        line_max_changes[i] = controller_dqp_list[i].line_max_change
        line_min_changes[i] = controller_dqp_list[i].line_min_change

    ckpt = dict()
    ckpt["line_max_changes"] = line_max_changes
    ckpt["line_min_changes"] = line_min_changes
    ckpt["num_cycles"] = num_cycles
    ckpt["epochs_per_cycle"] = epochs
    ckpt["T_per_cycle"] = T
    ckpt["T_horizon"] = T_horizon
    if num_cycles == 1:
        ckpt["total_losses_per_iter"] = total_losses[0]
        ckpt["total_econ_losses_per_iter"] = total_econ_losses[0]
        ckpt["total_viol_losses_per_iter"] = total_viol_losses[0]
        ckpt["total_train_losses_per_iter"] = total_train_losses[0]
        ckpt["total_terminal_losses_per_iter"] = total_terminal_losses[0]
    else:
        ckpt["total_losses_per_iter"] = total_losses
        ckpt["total_econ_losses_per_iter"] = total_econ_losses
        ckpt["total_viol_losses_per_iter"] = total_viol_losses
        ckpt["total_train_losses_per_iter"] = total_train_losses
        ckpt["total_terminal_losses_per_iter"] = total_terminal_losses
    ckpt["lr_reduction_epochs"] = lr_reduction_epochs
    ckpt["optimal_comparison"] = optimal_comparison
    ckpt["dec_comparison"] = dec_comparison
    ckpt["cycle_test_results"] = cycle_test_results
    ckpt["run_failed"] = run_failed
    ckpt["fail_message"] = fail_message
    ckpt["all_train_traj"] = all_train_traj
    pool.close()
    return ckpt


def _cycle_plot_colors(num_cycles):
    """Return ``num_cycles`` visually distinct colors for line plots."""
    import matplotlib.pyplot as plt

    if num_cycles <= 10:
        cmap = plt.cm.tab10
        return [cmap(i) for i in range(num_cycles)]
    if num_cycles <= 20:
        cmap = plt.cm.tab20
        return [cmap(i) for i in range(num_cycles)]
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    if num_cycles <= len(palette):
        return palette[:num_cycles]
    cmap = plt.cm.hsv
    return [cmap(i / num_cycles) for i in range(num_cycles)]


def plot_cycle_epoch_losses(ckpt, *, save_path=None, show=False):
    """Plot total training loss for each epoch within each cycle."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = ckpt.get("total_losses_per_iter")
    if total is None:
        raise KeyError("checkpoint missing total_losses_per_iter")

    total = total.detach().cpu().float() if torch.is_tensor(total) else torch.tensor(total, dtype=torch.float32)
    if total.ndim == 1:
        total = total.unsqueeze(0)

    num_cycles, epochs_per_cycle = total.shape
    epochs_per_cycle = ckpt.get("epochs_per_cycle", epochs_per_cycle)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, epochs_per_cycle + 1)
    colors = _cycle_plot_colors(num_cycles)
    for cycle in range(num_cycles):
        ax.plot(
            x, np.maximum(total[cycle].numpy(), 1e-6),
            marker="o", markersize=4, label=f"Cycle {cycle + 1}", color=colors[cycle],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss (log scale)")
    ax.set_yscale("log")
    ax.set_title("Training loss by cycle and epoch")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def eval_limits_surrogate(grid, line_max_changes, line_min_changes, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                          line_terminal_cost=0.0, batt_curt_terminal_cost=0.0):
    """
        Learns line impact allocations over multiple scenarios. Computes approximate gradients via a surrogate on the
        loss incurred by the MPC rollout for a given set of impact allocations.

        optimizer_type: 'clipped_gd' | 'sgd' | 'adam'
        lr_schedule:    None | 'cosine' | 'step_decay' | 'plateau'
        schedule_kwargs: e.g. {'step_size': 30, 'gamma': 0.5} for step_decay
    """
    print("hi", flush=True)
    num_agents = 3
    partition = training_partition()

    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, H, T, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost, line_terminal_cost=line_terminal_cost,
                                                            batt_curt_terminal_cost=batt_curt_terminal_cost,
                                                            soft_bus_constraints=True)
    print("hi 2", flush=True)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    num_traj = all_train_traj.shape[0]

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    with torch.no_grad():
        for i in range(num_agents):
            controller_dqp_list[i].line_max_change.copy_(line_max_changes[i])
            controller_dqp_list[i].line_min_change.copy_(line_min_changes[i])

    total_max_margin = torch.zeros(T+H, grid.num_lines)
    total_min_margin = torch.zeros(T+H, grid.num_lines)
    total_max_margin[0] =  grid.line_data[:,5] - grid.init_state[0:grid.num_lines]
    total_min_margin[0] = -grid.line_data[:,5] - grid.init_state[0:grid.num_lines]


    losses_per_traj = torch.zeros(num_traj)
    print("hi 3", flush=True)
    for traj in tqdm(range(num_traj)):
        train_disturbances = all_train_traj[traj]

        total_loss = 0
        total_econ_loss = 0
        total_viol_loss = 0
        total_train_loss = 0
        grid.reset_state()
        for controller in controller_dqp_list:
            controller.zero_line_state()
            controller.update_bus_state()

        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks, _ = \
                controller_utils.get_next_action(grid, controller_dqp_list, train_disturbances[0:T,:], 0, dQPTH_layer, verbose=False, update_state=True)
        for t in range(T):
            noise = train_disturbances[t]
            action_curt = pred_actions_curt[t]
            action_batt = pred_actions_batt[t]
            grid.update_state(action_curt, action_batt, noise)

            net_curt = grid.state[grid.state_curt_idx]
            bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
            batt_charge_violations = bus_violations[:2*grid.num_batt]
            batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
            curt_violations = bus_violations[4*grid.num_batt:]
            scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
            scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
            scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
            bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                + torch.sum(scaled_batt_power_violations ** 2)
                                                + torch.sum(scaled_curt_violations ** 2))
            line_flows = grid.state[grid.state_line_flow_idx]
            line_limits = grid.line_data[:,5]
            line_violations = torch.relu(torch.abs(line_flows) - line_limits)
            line_violations_loss = _line_violations_loss(line_violations, line_slack_cost)
            batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
            batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
            curt_change_loss, curt_net_loss = _curt_loss_parts(
                grid, action_curt, net_curt, curt_change_cost, curt_net_cost)
            total_loss += curt_change_loss
            total_train_loss += curt_change_loss
            total_econ_loss += curt_change_loss
            if _is_controlled_rollout_step(t, grid):
                total_loss += batt_loss + curt_net_loss + bus_violations_loss + line_violations_loss
                total_train_loss += batt_loss + curt_net_loss + bus_violations_loss + _line_violations_loss(line_violations, line_slack_cost)
                total_econ_loss += batt_loss + curt_net_loss
                total_viol_loss += bus_violations_loss + line_violations_loss
        print(f"surrogate, done with traj {traj}", flush=True)
        losses_per_traj[traj] = total_loss.detach()

    pool.close()
    return losses_per_traj.detach()


def eval_limits_rollout(grid, line_max_changes, line_min_changes, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                        line_terminal_cost=0.0, batt_curt_terminal_cost=0.0):
    """
        Learns line impact allocations over multiple scenarios. Computes approximate gradients via a surrogate on the
        loss incurred by the MPC rollout for a given set of impact allocations.

        optimizer_type: 'clipped_gd' | 'sgd' | 'adam'
        lr_schedule:    None | 'cosine' | 'step_decay' | 'plateau'
        schedule_kwargs: e.g. {'step_size': 30, 'gamma': 0.5} for step_decay
    """
    num_agents = 3
    partition = training_partition()

    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, H, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost, line_terminal_cost=line_terminal_cost,
                                                            batt_curt_terminal_cost=batt_curt_terminal_cost,
                                                            soft_bus_constraints=True)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    num_traj = all_train_traj.shape[0]

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    with torch.no_grad():
        for i in range(num_agents):
            controller_dqp_list[i].line_max_change.copy_(line_max_changes[i])
            controller_dqp_list[i].line_min_change.copy_(line_min_changes[i])

    total_max_margin = torch.zeros(T+H, grid.num_lines)
    total_min_margin = torch.zeros(T+H, grid.num_lines)
    total_max_margin[0] =  grid.line_data[:,5] - grid.init_state[0:grid.num_lines]
    total_min_margin[0] = -grid.line_data[:,5] - grid.init_state[0:grid.num_lines]

    losses_per_traj = torch.zeros(num_traj)

    for traj in tqdm(range(num_traj)):
        train_disturbances = all_train_traj[traj]

        total_loss = 0
        total_econ_loss = 0
        total_viol_loss = 0
        total_train_loss = 0
        grid.reset_state()
        for controller in controller_dqp_list:
            controller.zero_line_state()
            controller.update_bus_state()

        for t in range(T):
            noise = train_disturbances[t:t+H,:]
            pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks, _ = \
                controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=True)
            action_curt = pred_actions_curt[0]
            action_batt = pred_actions_batt[0]
            grid.update_state(action_curt, action_batt, noise[0])

            net_curt = grid.state[grid.state_curt_idx]
            bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
            batt_charge_violations = bus_violations[:2*grid.num_batt]
            batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
            curt_violations = bus_violations[4*grid.num_batt:]
            scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
            scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
            scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
            bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                + torch.sum(scaled_batt_power_violations ** 2)
                                                + torch.sum(scaled_curt_violations ** 2))
            line_flows = grid.state[grid.state_line_flow_idx]
            line_limits = grid.line_data[:,5]
            line_violations = torch.relu(torch.abs(line_flows) - line_limits)
            line_violations_loss = _line_violations_loss(line_violations, line_slack_cost)
            batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
            batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
            curt_change_loss, curt_net_loss = _curt_loss_parts(
                grid, action_curt, net_curt, curt_change_cost, curt_net_cost)
            total_loss += curt_change_loss
            total_train_loss += curt_change_loss
            total_econ_loss += curt_change_loss
            if _is_controlled_rollout_step(t, grid):
                total_loss += batt_loss + curt_net_loss + bus_violations_loss + line_violations_loss
                total_train_loss += batt_loss + curt_net_loss + bus_violations_loss + _line_violations_loss(line_violations, line_slack_cost)
                total_econ_loss += batt_loss + curt_net_loss
                total_viol_loss += bus_violations_loss + line_violations_loss

        losses_per_traj[traj] = total_loss.detach()
        print(f"rollout, done with traj {traj}", flush=True)
    pool.close()
    return losses_per_traj.detach()
