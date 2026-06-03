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
import evals

torch.set_default_dtype(torch.double)
torch.set_printoptions(threshold=10000)
NUMPY_SEED = 0
TORCH_SEED = 0
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)

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


# @profile
def get_central_full_traj(grid, T, H, disturbances, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """
    Simulates what a fully centralized planner would do with full knowledge of future disturbances for T steps.

    Args:
        grid (Grid object):
            defines the grid that control problem is set up on
        T (int):
            trajectory length
        H (int):
            horizon of MPC (does not apply for this function)
        disturbances (Torch tensor of shape (1, T+H, grid.num_buses)):
            future disturbances
        batt_cost (float):
            battery cost
        curt_change_cost (float):
            cost per step associated with changing curtailment orders
        curt_net_cost (float):
            cost per step associated with net curtailment
        slack_cost (float):
            cost associated with constraint violations
    Returns:
        ckpt (dict): Contains run results, keys are:
            actions_curt (Torch tensor of shape (T, grid.num_curt)):
                curtailment actions that would be taken
            actions_batt (Torch tensor of shape (T, grid.num_batt)):
                battery actions that would be taken
            viol_per_step (Torch tensor of shape (T)):
                squared violation at each step
            curt_cost_per_step (Torch tensor of shape (T)):
                curtailment cost incurred at each step
    """
    partition = [[i for i in range(grid.num_buses)]]
    num_agents = 1
    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, T, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost, active_eps=1)
    pool = mp.Pool(processes=num_agents)
    dqp_eps = 1e-3
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True, eps_active=dqp_eps)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    noise = disturbances[0:T]
    t = 0
    print("solving")
    start_time = time.time()
    actions_curt, actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
        controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=False)
    print(f"solving takes time {time.time() - start_time}")
    print("done solving")
    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    viol_per_step = torch.zeros(T)
    curt_cost_per_step = torch.zeros(T)
    batt_cost_per_step = torch.zeros(T)

    for t in range(T):
        action_curt = actions_curt[t]
        action_batt = actions_batt[t]
        grid.update_state(action_curt, action_batt, noise[t,:])

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
        scaled_line_violations = line_violations / grid.line_data[:,5]
        line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
        viol_per_step[t] = bus_violations_loss + line_violations_loss
        curt_cost_per_step[t] = curt_loss
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
        "total_losses": total_losses
    }

    pool.close()

    return ckpt


def get_multi_agent_vanilla_MPC(grid, T, H, disturbances, batt_cost, curt_change_cost, curt_net_cost, slack_cost):
    """
    Simulates what a collection of non-communicating controllers would do.

    Args:
        grid (Grid object):
            defines the grid that control problem is set up on
        T (int):
            trajectory length
        H (int):
            horizon of MPC
        disturbances (Torch tensor of shape (1, T+H, grid.num_buses)):
            future disturbances
        batt_cost (float):
            battery cost
        curt_change_cost (float):
            cost per step associated with changing curtailment orders
        curt_net_cost (float):
            cost per step associated with net curtailment
        slack_cost (float):
            cost associated with constraint violations
    Returns:
        ckpt (dict): Contains run results, keys are:
            actions_curt (Torch tensor of shape (T, grid.num_curt)):
                curtailment actions that would be taken
            actions_batt (Torch tensor of shape (T, grid.num_batt)):
                battery actions that would be taken
            viol_per_step (Torch tensor of shape (T)):
                squared violation at each step
            curt_cost_per_step (Torch tensor of shape (T)):
                curtailment cost incurred at each step
    """
    num_agents = 3
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    partition = [nodes_1, nodes_2, nodes_3]
    base_controller_list = controller_utils.create_base_controllers(grid, num_agents, partition, T, H, batt_cost, curt_change_cost,
                                                 curt_net_cost, slack_cost)
    pool = mp.Pool(processes=num_agents)
    dqp_eps = 1e-3
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True, eps_active=dqp_eps)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    grid.reset_state()
    for controller in base_controller_list:
        controller.update_line_state()
        controller.update_bus_state()

    actions_curt = torch.zeros(T, grid.num_curt)
    actions_batt = torch.zeros(T, grid.num_batt)
    viol_per_step = torch.zeros(T)
    curt_cost_per_step = torch.zeros(T)
    batt_charge_viol_scale = (torch.mean(grid.line_limits) / torch.mean(grid.batt_charge_max_limits)) ** 2
    batt_power_viol_scale = (torch.mean(grid.line_limits) / torch.mean(grid.batt_power_max_limits)) ** 2
    curt_viol_scale = (torch.mean(grid.line_limits) / torch.mean(grid.curt_max_limits)) ** 2

    for t in tqdm(range(T)):
        noise = disturbances[t:t+H]
        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
            controller_utils.get_next_action_base(grid, base_controller_list, noise, t, verbose=False, update_state=True)

        action_curt = pred_actions_curt[0]
        action_batt = pred_actions_batt[0]
        actions_curt[t] = action_curt.detach()
        actions_batt[t] = action_batt.detach()
        grid.update_state(action_curt, action_batt, noise[0,:])

        net_curt = grid.state[grid.state_curt_idx]
        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
        batt_charge_violations = bus_violations[:2*grid.num_buses]
        batt_power_violations = bus_violations[2*grid.num_buses:4*grid.num_buses]
        curt_violations = bus_violations[4*grid.num_buses:]
        scaled_bus_viols = batt_charge_viol_scale * torch.sum(batt_charge_violations ** 2) \
                        + batt_power_viol_scale * torch.sum(batt_power_violations ** 2) \
                        + curt_viol_scale * torch.sum(curt_violations ** 2)
        line_flows = grid.state[grid.state_line_flow_idx]
        line_limits = grid.line_data[:,5]
        line_violations = torch.relu(torch.abs(line_flows) - line_limits)

        curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
        viol_loss = slack_cost * (scaled_bus_viols + torch.sum(line_violations ** 2))

        curt_cost_per_step[t] = curt_loss.detach()
        viol_per_step[t] = viol_loss.detach()

    curt_cost_per_step = curt_cost_per_step.detach()
    viol_per_step = viol_per_step.detach()
    total_loss = torch.sum(curt_cost_per_step) + torch.sum(viol_per_step)

    ckpt = {
        "actions_curt": actions_curt.detach(),
        "actions_batt": actions_batt.detach(),
        "viol_per_step": viol_per_step.detach(),
        "curt_cost_per_step": curt_cost_per_step.detach(),
        "total_loss": total_loss.detach()
    }
    pool.close()
    return ckpt


def evaluate_base_on_traj(grid, base_controller_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """
    Gets cost associated with running base MPCs on a test trajectory.
    """
    grid.reset_state()
    for controller in base_controller_list:
        controller.update_line_state()
        controller.update_bus_state()
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
        noise = test_traj[t:t+H,:]
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
        line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
        # line_violations_loss = line_slack_cost * (torch.sum(pred_line_max_slacks[:,0]**2) + torch.sum(pred_line_min_slacks[:,0]**2))
        # line_violations_loss = line_slack_cost * torch.sum(scaled_line_violations ** 2)
        # print(torch.sum(scaled_line_violations), torch.sum(line_violations))
        # print(f"action batt {torch.sum(action_batt ** 2)}, action curt {torch.sum(action_curt ** 2)}, curt_net {torch.sum(net_curt ** 2)}")
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
        total_loss += batt_loss + curt_loss + line_violations_loss
        total_train_loss += batt_loss + curt_loss + line_slack_cost * torch.sum(line_violations ** 2)
        total_econ_loss += batt_loss + curt_loss
        total_viol_loss += line_violations_loss
        viol_per_step[t] = line_violations_loss
        curt_cost_per_step[t] = curt_loss
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
def evaluate_on_traj(grid, controller_dqp_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """
    Gets cost associated with running MPC with configured limits on a test trajectory.
    """
    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()

    test_total_loss = torch.zeros(3)
    dQPTH_layer.reset_cache()

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
        noise = test_traj[t:t+H,:]
        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
            controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=True)

        action_curt = pred_actions_curt[0]
        action_batt = pred_actions_batt[0]
        grid.update_state(action_curt, action_batt, noise[0,:])

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
        line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)

        # bus violations no longer included after I changed code to not allow slacks to be set on bus constraints
        total_loss += batt_loss + curt_loss + line_violations_loss
        total_train_loss += batt_loss + curt_loss + line_slack_cost * torch.sum(line_violations ** 2)
        total_econ_loss += batt_loss + curt_loss
        total_viol_loss += line_violations_loss
        viol_per_step[t] = line_violations_loss
        curt_cost_per_step[t] = curt_loss
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
                         optimizer_type='clipped_gd', max_grad_norm=30, lr_schedule=None, schedule_kwargs=None, batch_size=None):
    """
        Learns line impact allocations over multiple scenarios. Computes approximate gradients via a surrogate on the
        loss incurred by the MPC rollout for a given set of impact allocations.

        optimizer_type: 'clipped_gd' | 'sgd' | 'adam'
        lr_schedule:    None | 'cosine' | 'step_decay' | 'plateau'
        schedule_kwargs: e.g. {'step_size': 30, 'gamma': 0.5} for step_decay
    """
    num_agents = 3
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    partition = [nodes_1, nodes_2, nodes_3]

    # surrogate solves all T steps altogether.
    # We set T=H, H=T since this gets us the size-T prediction window while also setting us up to learn limits for T+H steps.
    # TODO: do this in a way that is less hacky, also think about the fact that size-T prediction window uses only T limits, while
    # MPC rollout uses T+H
    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, H, T, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost)
    pool = mp.Pool(processes=num_agents)
    dqp_eps = 1e-3
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True, eps_active=dqp_eps)
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
    total_max_margin[0] =  grid.line_data[:,5] - grid.init_state[0:grid.num_lines]
    total_min_margin[0] = -grid.line_data[:,5] - grid.init_state[0:grid.num_lines]

    run_failed = False
    fail_message = ""

    total_losses = torch.zeros(epochs)
    total_econ_losses = torch.zeros(epochs)
    total_viol_losses = torch.zeros(epochs)
    total_train_losses = torch.zeros(epochs)
    lr_reduction_epochs = []  # (epoch, batch_start, new_lr) each time plateau fires

    for epoch in tqdm(range(epochs)):
        epoch_total_loss = 0
        epoch_econ_loss = 0
        epoch_viol_loss = 0
        epoch_train_loss = 0
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
                grid.reset_state()
                for controller in controller_dqp_list:
                    controller.zero_line_state()
                    controller.update_bus_state()

                start_time = time.time()
                pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
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
                    line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
                    batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
                    batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
                    curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
                    total_loss += batt_loss + curt_loss + bus_violations_loss + line_violations_loss
                    total_train_loss += batt_loss + curt_loss + bus_violations_loss + line_slack_cost * torch.sum(line_violations ** 2)
                    total_econ_loss += batt_loss + curt_loss
                    total_viol_loss += bus_violations_loss + line_violations_loss
                start_time = time.time()
                (total_loss / actual_batch_size).backward()
                print(f"backward pass for one traj takes time {time.time() - start_time}")
                

                print(total_loss, total_train_loss)

                batch_total_loss += total_loss.detach()
                epoch_total_loss += total_loss.detach()
                epoch_econ_loss += total_econ_loss.detach()
                epoch_viol_loss += total_viol_loss.detach()
                epoch_train_loss += total_train_loss.detach()

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

        print(f"Epoch {epoch+1}/{epochs}, loss over all trajs={epoch_total_loss.detach():.4f}, viol={epoch_viol_loss.detach():.4f}", flush=True)
        total_losses[epoch] = epoch_total_loss
        total_econ_losses[epoch] = epoch_econ_loss
        total_viol_losses[epoch] = epoch_viol_loss
        total_train_losses[epoch] = epoch_train_loss

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
    ckpt["lr_reduction_epochs"] = lr_reduction_epochs
    ckpt["run_failed"] = run_failed
    ckpt["fail_message"] = fail_message
    ckpt["all_train_traj"] = all_train_traj
    pool.close()
    return ckpt


def train_with_rollout(grid, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost, epochs, lr,
                      optimizer_type='clipped_gd', max_grad_norm=30, lr_schedule=None, schedule_kwargs=None, batch_size=None):
    """
        Learns line impact allocations over multiple scenarios.
    """
    num_agents = 3
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    partition = [nodes_1, nodes_2, nodes_3]

    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, H, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost)
    pool = mp.Pool(processes=num_agents)
    dqp_eps = 1e-3
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True, eps_active=dqp_eps)
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
    total_max_margin[0] =  grid.line_data[:,5] - grid.init_state[0:grid.num_lines]
    total_min_margin[0] = -grid.line_data[:,5] - grid.init_state[0:grid.num_lines]

    run_failed = False
    fail_message = ""

    total_losses = torch.zeros(epochs)
    total_econ_losses = torch.zeros(epochs)
    total_viol_losses = torch.zeros(epochs)
    total_train_losses = torch.zeros(epochs)
    lr_reduction_epochs = []

    for epoch in tqdm(range(epochs)):
        epoch_total_loss = 0
        epoch_econ_loss = 0
        epoch_viol_loss = 0
        epoch_train_loss = 0
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
                grid.reset_state()
                for controller in controller_dqp_list:
                    controller.zero_line_state()
                    controller.update_bus_state()

                start_time = time.time()
                for t in range(T):
                    noise = train_disturbances[t:t+H,:]
                    pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
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
                    line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
                    batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
                    batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
                    curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
                    total_loss += batt_loss + curt_loss + bus_violations_loss + line_violations_loss
                    total_train_loss += batt_loss + curt_loss + bus_violations_loss + line_slack_cost * torch.sum(line_violations ** 2)
                    total_econ_loss += batt_loss + curt_loss
                    total_viol_loss += bus_violations_loss + line_violations_loss
                start_time = time.time()
                (total_loss / actual_batch_size).backward()
                print(f"backward pass for one traj takes time {time.time() - start_time}")
                # print(f"forward pass takes time {time.time() - start_time}")

                print(total_loss, total_train_loss)

                batch_total_loss += total_loss.detach()
                epoch_total_loss += total_loss.detach()
                epoch_econ_loss += total_econ_loss.detach()
                epoch_viol_loss += total_viol_loss.detach()
                epoch_train_loss += total_train_loss.detach()

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

        print(f"Epoch {epoch+1}/{epochs}, loss over all trajs={epoch_total_loss.detach():.4f}, viol={epoch_viol_loss.detach():.4f}", flush=True)
        total_losses[epoch] = epoch_total_loss
        total_econ_losses[epoch] = epoch_econ_loss
        total_viol_losses[epoch] = epoch_viol_loss
        total_train_losses[epoch] = epoch_train_loss

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
    ckpt["lr_reduction_epochs"] = lr_reduction_epochs
    ckpt["run_failed"] = run_failed
    ckpt["fail_message"] = fail_message
    ckpt["all_train_traj"] = all_train_traj
    pool.close()
    return ckpt


def eval_limits_surrogate(grid, line_max_changes, line_min_changes, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """
        Learns line impact allocations over multiple scenarios. Computes approximate gradients via a surrogate on the
        loss incurred by the MPC rollout for a given set of impact allocations.

        optimizer_type: 'clipped_gd' | 'sgd' | 'adam'
        lr_schedule:    None | 'cosine' | 'step_decay' | 'plateau'
        schedule_kwargs: e.g. {'step_size': 30, 'gamma': 0.5} for step_decay
    """
    print("hi", flush=True)
    num_agents = 3
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    partition = [nodes_1, nodes_2, nodes_3]

    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, H, T, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost)
    print("hi 2", flush=True)
    pool = mp.Pool(processes=num_agents)
    dqp_eps = 1e-3
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True, eps_active=dqp_eps)
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

        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
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
            line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
            batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
            batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
            curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
            total_loss += batt_loss + curt_loss + bus_violations_loss + line_violations_loss
            total_train_loss += batt_loss + curt_loss + bus_violations_loss + line_slack_cost * torch.sum(line_violations ** 2)
            total_econ_loss += batt_loss + curt_loss
            total_viol_loss += bus_violations_loss + line_violations_loss
        print(f"surrogate, done with traj {traj}", flush=True)
        losses_per_traj[traj] = total_loss.detach()

    pool.close()
    return losses_per_traj.detach()


def eval_limits_rollout(grid, line_max_changes, line_min_changes, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """
        Learns line impact allocations over multiple scenarios. Computes approximate gradients via a surrogate on the
        loss incurred by the MPC rollout for a given set of impact allocations.

        optimizer_type: 'clipped_gd' | 'sgd' | 'adam'
        lr_schedule:    None | 'cosine' | 'step_decay' | 'plateau'
        schedule_kwargs: e.g. {'step_size': 30, 'gamma': 0.5} for step_decay
    """
    num_agents = 3
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    partition = [nodes_1, nodes_2, nodes_3]

    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, H, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost)
    pool = mp.Pool(processes=num_agents)
    dqp_eps = 1e-3
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True, eps_active=dqp_eps)
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
            pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
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
            line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
            batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
            batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
            curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
            total_loss += batt_loss + curt_loss + bus_violations_loss + line_violations_loss
            total_train_loss += batt_loss + curt_loss + bus_violations_loss + line_slack_cost * torch.sum(line_violations ** 2)
            total_econ_loss += batt_loss + curt_loss
            total_viol_loss += bus_violations_loss + line_violations_loss

        losses_per_traj[traj] = total_loss.detach()
        print(f"rollout, done with traj {traj}", flush=True)
    pool.close()
    return losses_per_traj.detach()