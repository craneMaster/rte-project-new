"""
Filename: NAZA_utils.py
Author: James Chen
Date Created: 2025-07-29

Version History:
----------------
7/29/25 - Created file from RTE_utils.py
9/16/25 - Edited to work with NAZA_QPTH, missing a lot of version history updates oops
"""
import grid
import controller as NAZA
import numpy as np
import torch
import torch.nn as nn
import time
import cvxpy as cp

import torch

import numpy as np
import scipy as sp

from scipy.sparse import csc_matrix,csr_matrix,coo_matrix

def csc_torch_to_scipy(A):
    return csc_matrix((A.values().detach().numpy(),A.row_indices().numpy(),A.ccol_indices().numpy()),shape=A.size())

def csc_scipy_to_torch(A):
    return torch.sparse_csc_tensor(torch.tensor(A.indptr,dtype=torch.int64),torch.tensor(A.indices,dtype=torch.int64),torch.tensor(A.data),size=np.shape(A),dtype=torch.float64)


def coo_torch_to_scipy(A):
    # A must be coalesced
    return coo_matrix((A.values().detach().numpy(),(A._indices()[0,:].numpy(),A._indices()[1,:].numpy())),shape=A.size())

def coo_scipy_to_torch(A):
    i = torch.tensor(np.vstack((A.row,A.col)),dtype=torch.long)
    v = torch.tensor(A.data,dtype=torch.float64)
    return torch.sparse_coo_tensor(i,v,size=np.shape(A))

def initialize_torch_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    Qnp, qnp, Gnp, hnp, Anp, bnp = data["Q"][()], data["q"][()], data["G"][()], data["h"][()], data["A"][()], data["b"][()]

    Qnp, Gnp = Qnp.tocoo(), Gnp.tocoo()
    __to_coo = lambda M: torch.sparse_coo_tensor(torch.stack([torch.tensor(M.row), torch.tensor(M.col)]),
                                                 torch.tensor(M.data), M.shape, dtype=torch.float64, requires_grad=True)
    Q, G = __to_coo(Qnp).to_sparse_csc(), __to_coo(Gnp).to_sparse_csc()

    q = torch.tensor(qnp, requires_grad=True)
    h = torch.tensor(hnp, requires_grad=True)
    if Anp is not None:
        Anp = Anp.tocoo()
        A = __to_coo(Anp).to_sparse_csc()
        b = torch.tensor(bnp, requires_grad=True)

    return Q,q,G,h,A,b

class sparse_row_norm(torch.autograd.Function):
    '''
    Differentiate sparse row-wise 1 or 2 norm. This is not supported in torch 2.3.1 or external sparse_torch?
    '''

    @staticmethod
    def forward(ctx, A, p):
        '''
        '''

        assert(p == 1 or p == 2)

        ctx.p = p

        A = csc_torch_to_scipy(A) # TODO : convert to list of sparse matrices if sparse
        ctx.A = A

        N = np.expand_dims(sp.sparse.linalg.norm(A,ord=p,axis=1),-1)
        ctx.N = N

        return torch.tensor(N,dtype=torch.float64)

    @staticmethod
    def backward(ctx,grad_output):
        '''
        dN/dA ; gradient is sparse
        '''

        if ctx.p == 1:
            dN = ctx.A.sign().multiply(grad_output.numpy())
        elif ctx.p == 2:
            dN = ctx.A.multiply(grad_output.numpy() * np.power(ctx.N,-1))

        # return torch.tensor(dN.todense(),dtype=torch.float64), None # TODO : sparse or not?
        return csc_scipy_to_torch(csc_matrix(dN)), None


class sparse_row_normalize(torch.autograd.Function):
    '''
    Differentiate sparse row normalization. Insane this is not supported...
    '''

    @staticmethod
    def forward(ctx, A, N):
        '''
        '''

        N = N.numpy()
        ctx.N = N
        ctx.A = csc_torch_to_scipy(A)

        A = csr_matrix(ctx.A)
        A.data /= N[A.nonzero()[0],0] # normalize rows in-place
        A = csc_matrix(A)

        return csc_scipy_to_torch(A)

    @staticmethod
    def backward(ctx,grad_output):
        '''
        '''

        if grad_output.layout is torch.strided:
            dL = grad_output.numpy()
        else:
            dL = grad_output.to_dense().numpy()

        dA_norm = np.multiply(np.power(ctx.N, -1), dL)
        dN = -np.multiply(np.expand_dims(np.diag(ctx.A @ dL.T), -1), np.power(ctx.N, -2))

        return torch.tensor(dA_norm, dtype=torch.float64), torch.tensor(dN, dtype=torch.float64)

def test_normalization():

    # p = 2
    p = 1

    rng = np.random.default_rng()
    S = sp.sparse.random(3, 4, density=0.5, random_state=rng, format="csc")

    S = csc_scipy_to_torch(S)
    S.requires_grad_()

    D = S.clone().detach().to_dense().numpy()
    D = torch.tensor(D,dtype=torch.float64,requires_grad=True)

    print("Inputs: ")
    print(D)
    print(S.to_dense())
    print("\n")

    row_norm = sparse_row_norm.apply
    row_normalize = sparse_row_normalize.apply

    db = torch.rand(3,1,dtype=torch.float64,requires_grad=True)
    sb = db.clone().detach().requires_grad_()

    N_D = torch.unsqueeze(torch.linalg.vector_norm(D, ord=p, dim=-1),-1)
    D_new = torch.div(D, N_D)
    db_new = torch.div(db, N_D)

    N_S = row_norm(S,p)
    S_new = row_normalize(S,N_S)
    sb_new = torch.div(sb, N_S)

    # S = S.to_sparse_coo()
    # mat = torch_sparse.SparseTensor(row=S._indices()[0, :], col=S._indices()[1, :], value=S._values())

    print("Check outputs: ")
    print(N_D)
    print(N_S)
    print(db)
    print(sb)
    print("Check outputs: ")
    print("\n")

    print("Check gradients: ")
    l_D = D_new
    l_S = S_new
    l_D.backward(torch.ones(3,4))
    l_S.backward(torch.ones(3,4))
    print("Check gradients: ")
    print(D.grad)
    print(S.grad)

    return


def get_RTE_noise_values(file_path, grid, T, H, power_scale, bus_idx_gap, offset=0):
    """
    Generates a series of disturbance values over a collection of buses based on a set of power injection disturbances
    from an open-source RTE data source.

    Args:
        grid (Grid):
            Full grid that the disturbances are being generated for.
        T (int):
            Number of control timesteps.
        H (int):
            Size of prediction window.
        num_buses (int):
            Number of buses through network.
        power_scale (Double):
            Amount to scale disturbances by.
        bus_idx_gap (int):
            Quantity used to approximate strength of spatial correlation. Larger values mean pairs of buses have
            less correlated disturbances, especially as distance between buses increases.
        file_path (String):
            File location of RTE noise values.
        offset (int):
            Amount to shift entire trajectory of disturbance values.
    Returns:
        noise_values (Numpy array of size (T+H, num_buses)):
            A series of disturbance values.
    """
    data = np.loadtxt(file_path)
    vec = torch.tensor(data)
    disturbance_length = T + H
    all_disturbances = torch.zeros(disturbance_length, grid.num_buses)

    for i in range(grid.num_curt):
        bus = grid.bus_with_curt[i]
        bus_max_gen = grid.curt_max_limits[i]
        start_idx = offset + bus * bus_idx_gap
        idx = torch.arange(start_idx, start_idx + grid.delta_t*(disturbance_length+1), grid.delta_t, dtype=torch.long)

        if start_idx + grid.delta_t*(disturbance_length+1) >= vec.shape[0]:
            raise ValueError("Offset / bus gap is too large, index overflow occurred.")

        bus_power = bus_max_gen * vec[idx]
        bus_disturbances = torch.diff(bus_power)
        all_disturbances[:,bus] = bus_disturbances
    
    # normalize
    all_disturbances = power_scale * (all_disturbances / torch.mean(torch.abs(all_disturbances)))

    return all_disturbances


def get_iid_noise_values(T, H, num_buses, max_noise, max_trend=0):
    """
    Generates a series of iid disturbance values throughout the network.

    Args:
        T (int):
            Number of control timesteps.
        H (int):
            Size of prediction window.
        num_buses (int):
            Number of buses through network.
        max_noise (Double):
            Maximum noise magnitude at a bus.
        max_trend (Double):
            Maximum trend magnitude at a bus.
    Returns:
        noise_values: (numpy array) series of disturbance values (T+H, num_buses)
    """
    trend = 2 * max_trend * (torch.rand(num_buses) - 0.5)
    trend_values = trend.unsqueeze(0).repeat(T+H,1)
    noise_values = 2 * max_noise * (torch.rand(T+H,num_buses)-0.5)
    return trend_values + noise_values


def solve_NAZA(naza, noise, t, solver="MOSEK", verbose=False):
    """
    Performs NAZA forward pass with a particular solver. We do this since CvxpyLayers only works with ECOS or SCS.

    Args:
        naza (Split_Constraint_NAZA):
            NAZA controller.
        disturbances (Torch tensor of shape (H, num_buses)):
            Predictions of power injection noise at each bus in the NAZA control area.
        t (int):
            Current timestep.
        solver (String):
            Specifies which CVXPY solver to use, options are:
            - "OSQP"
            - "MOSEK"
    Returns:
            states (Torch tensor of shape (H, state_dim)):
                Next H predicted states based on this NAZA's localized dynamics.
            net_impacts (Torch tensor of shape (H, grid.num_lines)):
                Next H predicted cumulative net impacts on all grid lines.
            actions_curt (Torch tensor of shape (H, num_curt)):
                Next H curtailment actions.
            actions_batt (Torch tensor of shape (H, num_batt)):
                Next H battery actions.
            bus_slacks (Torch tensor of shape (H, slack_dim)):
                Next H bus slack values on H_limit.
            line_max_slacks (Torch tensor of shape (H, grid.num_lines)):
                Next H line max slack values.
            line_min_slacks (Torch tensor of shape (H, grid.num_lines)):
                Next H line min slack values.
    """
    init_state = naza.state.detach()
    init_net_impact = naza.net_impact.detach()
    line_max = naza.line_max[t:t+naza.H,:].detach()
    line_min = naza.line_min[t:t+naza.H,:].detach()

    states = [cp.Variable(naza.state_dim) for _ in range(naza.H+1)]
    net_impacts = [cp.Variable(naza.grid.num_lines) for _ in range(naza.H+1)]
    actions_curt = [cp.Variable(naza.num_curt) for _ in range(naza.H)]
    actions_batt = [cp.Variable(naza.num_batt) for _ in range(naza.H)]
    bus_slacks = [cp.Variable(naza.bus_slack_dim, nonneg=True) for _ in range(naza.H)]
    line_max_slacks = [cp.Variable(naza.grid.num_lines, nonneg=True) for _ in range(naza.H)]
    line_min_slacks = [cp.Variable(naza.grid.num_lines, nonneg=True) for _ in range(naza.H)]

    objective = 0
    constraints = []
    constraints.append(states[0] == init_state)
    constraints.append(net_impacts[0] == init_net_impact)

    for i in range(naza.H):
        constraints.append(states[i+1] == naza.A @ states[i] + naza.B_curt @ actions_curt[i] \
                            + naza.B_batt @ actions_batt[i] + naza.B_noise @ noise[i,:])
        constraints.append(net_impacts[i+1] == net_impacts[i] + naza.net_impact_B_curt @ actions_curt[i] \
                            + naza.net_impact_B_batt @ actions_batt[i] + naza.net_impact_B_noise @ noise[i,:])
        constraints.append(naza.H_x @ states[i+1][naza.num_lines:] <= naza.H_limit + bus_slacks[i])
        constraints.append(net_impacts[i+1] <= line_max[i] + line_max_slacks[i])
        constraints.append(net_impacts[i+1] >= line_min[i] - line_min_slacks[i])

    state_batt_charge_idx = slice(naza.num_lines, naza.num_lines + naza.num_batt)
    state_curt_idx = slice(naza.num_lines + 2*naza.num_batt, naza.num_lines + 2*naza.num_batt + naza.num_curt)

    naza_target_charges = naza.grid.target_batt_charges[naza.batt_idx]
    for i in range(naza.H):
        objective += cp.quad_form(states[i+1][state_batt_charge_idx] - naza_target_charges, naza.batt_cost * torch.eye(naza.num_batt))
        objective += naza.curt_change_cost * cp.sum_squares(actions_curt[i])
        objective += naza.curt_change_cost * cp.sum_squares(actions_batt[i])
        objective += naza.curt_net_cost * cp.sum_squares(states[i+1][state_curt_idx])
        objective += naza.slack_cost * cp.sum_squares(bus_slacks[i])
        objective += naza.slack_cost * cp.sum_squares(line_max_slacks[i])
        objective += naza.slack_cost * cp.sum_squares(line_min_slacks[i])

    problem = cp.Problem(cp.Minimize(objective), constraints)

    if solver.lower() == "osqp":
        problem.solve(verbose=verbose,
                    solver=cp.OSQP,
                    eps_abs=1e-8,
                    eps_rel=1e-8,
                    max_iter=20000,
                    warm_start=True,
                    polish=True)
    elif solver.lower() == "mosek":
        problem.solve(verbose=verbose,
                      solver=cp.MOSEK,
                      mosek_params={
                        'MSK_IPAR_LOG': 10,
                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 1000000,  # Interior-point max iterations
                        'MSK_IPAR_SIM_MAX_ITERATIONS': 1000000,      # Simplex max iterations (if used)
                      }
                    )
    elif solver.lower() == "ecos":
        problem.solve(verbose=verbose,
                      solver=cp.ECOS)
    elif solver.lower() == "scs":
        problem.solve(verbose=verbose,
                      solver=cp.SCS)

    # print("Status:", problem.status)

    opt_states = torch.zeros(naza.H, naza.state_dim)
    opt_net_impacts = torch.zeros(naza.H, naza.grid.num_lines)
    opt_actions_curt = torch.zeros(naza.H, naza.num_curt)
    opt_actions_batt = torch.zeros(naza.H, naza.num_batt)
    opt_bus_slacks = torch.zeros(naza.H, naza.bus_slack_dim)
    opt_line_max_slacks = torch.zeros(naza.H, naza.grid.num_lines)
    opt_line_min_slacks = torch.zeros(naza.H, naza.grid.num_lines)

    # print(type(states[i].value))
    for i in range(naza.H):
        opt_states[i,:] = torch.tensor(states[i].value)
        opt_net_impacts[i,:] = torch.tensor(net_impacts[i].value)
        opt_actions_curt[i,:] = torch.tensor(actions_curt[i].value)
        opt_actions_batt[i,:] = torch.tensor(actions_batt[i].value)
        opt_bus_slacks[i,:] = torch.tensor(bus_slacks[i].value)
        opt_line_max_slacks[i,:] = torch.tensor(line_max_slacks[i].value)
        opt_line_min_slacks[i,:] = torch.tensor(line_min_slacks[i].value)

    return opt_states, opt_net_impacts, opt_actions_curt, opt_actions_batt, opt_bus_slacks, opt_line_max_slacks, opt_line_min_slacks, objective.value


def create_base_nazas(grid, num_nazas, partition, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """
    Initializes base NAZA agents on a given grid given a desired partition of buses.

    Args:
        grid (Grid):
            Full grid that the agents act on.
        num_nazas (int):
            Number of split constraint NAZAs on the grid.
        partition (list of lists):
            Set of buses that each agent controls, agent i controls buses in partition[i]
        T (int):
            Number of control timesteps.
        H (int):
            Size of prediction window.
        batt_cost (Double):
            Cost on deviation of battery charge from target value. Must be non-negative.
        curt_change_cost (Double):
            Cost of changing amount of power curtailed at a bus. Must be non-negative.
        curt_net_cost (Double):
            Cost of net amount of power curtailed at a bus. Must be non-negative.
        slack_cost (Double):
            Slack penalty. Must be non-negative.
    Returns:
        naza_list: (list of Base_NAZA objects) agents fully initialized, in the order matching the partition
    Raises:
        ValueError: If the number of partitions does not match the number of agents.
        ValueError: If the partition is not a valid partition of the grid.
        ValueError: If costs are negative.
    """
    if not len(partition) == num_nazas:
        raise ValueError(f"Number of partitions {len(partition)} does not match number of agents {num_nazas}")
    if not is_partition(partition, [i for i in range(grid.num_buses)]):
        raise ValueError(f"Partition is not a valid partition of the grid")
    if curt_change_cost < 0:
            raise ValueError(f"curt_change_cost {curt_change_cost} is negative.")
    if curt_net_cost < 0:
        raise ValueError(f"init_curt_net_cost {curt_net_cost} is negative.")

    naza_list = list()
    for i in range(num_nazas):
        buses_in_area = torch.tensor(partition[i], dtype=torch.int)
        naza_list.append(NAZA.Base_NAZA(grid, H, buses_in_area, batt_cost, curt_change_cost, curt_net_cost,
                                        bus_slack_cost, line_slack_cost))
        
    return naza_list


def create_split_constraint_nazas(grid, num_nazas, partition, T, H, batt_cost, curt_change_cost, curt_net_cost,
                                  bus_slack_cost, line_slack_cost, active_eps=1e-5):
    """
    Initializes split constraint agents on a given grid given a desired partition of buses.

    Args:
        grid (Grid):
            Full grid that the agents act on.
        num_nazas (int):
            Number of split constraint NAZAs on the grid.
        partition (list of lists):
            Set of buses that each agent controls, agent i controls buses in partition[i]
        T (int):
            Number of control timesteps.
        H (int):
            Size of prediction window.
        batt_cost (Double):
            Cost on deviation of battery charge from target value. Must be non-negative.
        curt_change_cost (Double):
            Cost of changing amount of power curtailed at a bus. Must be non-negative.
        curt_net_cost (Double):
            Cost of net amount of power curtailed at a bus. Must be non-negative.
        bus_slack_cost (Double):
            Slack penalty on bus variables. Must be non-negative.
        line_slack_cost (Double):
            Slack penalty on line variables. Must be non-negative.
        eps_abs (Double):
            Tolerance to decide active set of constraints in dQP.
    Returns:
        naza_list: (list of Split_Constraint_NAZA objects) agents full initialized, in the order matching the partition
    Raises:
        ValueError: If the number of partitions does not match the number of agents.
        ValueError: If the partition is not a valid partition of the grid.
        ValueError: If costs are negative.
    """
    if not len(partition) == num_nazas:
        raise ValueError(f"Number of partitions {len(partition)} does not match number of agents {num_nazas}")
    if not is_partition(partition, [i for i in range(grid.num_buses)]):
        raise ValueError(f"Partition is not a valid partition of the grid")
    if curt_change_cost < 0:
            raise ValueError(f"curt_change_cost {curt_change_cost} is negative.")
    if curt_net_cost < 0:
        raise ValueError(f"init_curt_net_cost {curt_net_cost} is negative.")

    naza_list = list()

    max_margin = torch.zeros(T+H, grid.num_lines)
    min_margin = torch.zeros(T+H, grid.num_lines)
    max_margin[0] = ( grid.line_data[:,5] - grid.init_state[0:grid.num_lines]) / num_nazas
    min_margin[0] = (-grid.line_data[:,5] - grid.init_state[0:grid.num_lines]) / num_nazas

    for i in range(num_nazas):
        buses_in_area = torch.tensor(partition[i], dtype=torch.int)
        naza_list.append(NAZA.Split_Constraint_NAZA(grid, H, buses_in_area, max_margin, min_margin, batt_cost,
                                                    curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                                                    eps_abs=active_eps))
    return naza_list


def assign_line_limit_changes(naza_list, line_max_changes, line_min_changes):
    """
    Assigns the given line max/min changes to the NAZAs in the list.
    Args:
        naza_list (list of SplitConstraintNAZA objects):
            List of NAZAs whose line limits should be set.
        line_max_changes (Torch tensor of shape (num_agents, T+H, grid.num_lines)):
            Changes between timesteps for line max limits each agent should adhere to over the trajectory.
        line_min_changes (Torch tensor of shape (num_agents, T+H, grid.num_lines)):
            Changes between timesteps for line min limits each agent should adhere to over the trajectory.
    Returns:
        None
    Raises:
        ValueError: if number of agents does not coincide
    """
    num_agents = line_max_changes.shape[0]
    if not len(naza_list) == num_agents:
        raise ValueError(f"NAZA list has {len(naza_list)} NAZAs but line_max_vals has values for {num_agents} agents.")

    with torch.no_grad():
        for i in range(len(naza_list)):
            naza = naza_list[i]
            naza.line_max_change = nn.Parameter(line_max_changes[i].clone())
            naza.line_min_change = nn.Parameter(line_min_changes[i].clone())


def assign_line_limits(naza_list, line_max_vals, line_min_vals):
    """
    Assigns the given line limits to the NAZAs in the list.

    Args:
        naza_list (list of SplitConstraintNAZA objects):
            List of NAZAs whose line limits should be set.
        line_max_vals (Torch tensor of shape (num_agents, T+H, grid.num_lines)):
            Line max limits each agent should adhere to over the trajectory.
        line_min_vals (Torch tensor of shape (num_agents, T+H, grid.num_lines)):
            Line min limits each agent should adhere to over the trajectory.
    Returns:
        None
    Raises:
        ValueError: if number of agents does not coincide
    """
    num_agents = line_max_vals.shape[0]
    if not len(naza_list) == num_agents:
        raise ValueError(f"NAZA list has {len(naza_list)} NAZAs but line_max_vals has values for {num_agents} agents.")

    with torch.no_grad():
        for i in range(len(naza_list)):
            naza = naza_list[i]
            naza.line_max_change = nn.Parameter(line_max_vals.clone())
            naza.line_min_change = nn.Parameter(line_min_vals.clone())


def get_next_action(grid, naza_list, disturbances, t, dQPTH_layer, verbose=False, with_cvxpy=False, update_state=False):
    """
    Collects actions from all agents for the next prediction window, aggregates them into one single action to be taken
    on the entire grid.

    Args:
        grid (Grid):
            Full grid that the NAZAs act on.
        naza_list (List of Split_Constraint_NAZA objects):
            List of all NAZAs on the grid.
        disturbances (Torch tensor of shape (H, num_buses)):
            Predictions of power injection noise at each bus in the NAZA control area.
        t (int):
            Current timestep.
        update_state (bool):
            True if NAZA should update its own state as if the actions solved for are actually taken.
    Returns:
        pred_actions_curt (Torch tensor of shape (H, grid.num_curt)):
            Curtailment actions to be taken across the grid, aggregated across all NAZAs.
        pred_actions_batt (Torch tensor of shape (H, grid.num_batt)):
            Battery actions to be taken across the grid, aggregated across all NAZAs.
        pred_line_max_slacks (Torch tensor of shape (num_agents, H, grid.num_lines)):
            Each agent's predictions for the line max slack variables it would set along the horizon.
        pred_line_min_slacks (Torch tensor of shape (num_agents, H, grid.num_lines)):
            Each agent's predictions for the line min slack variables it would set along the horizon.
        pred_bus_slacks (Torch tensor of shape (H, 2*(2*grid.num_batt + grid.num_curt))):
            Aggregated bus slack variables (not including splits) across all agents.
    """
    H = naza_list[0].H
    pred_actions_curt = torch.zeros(H, grid.num_curt)
    pred_actions_batt = torch.zeros(H, grid.num_batt)
    num_agents = len(naza_list)
    pred_bus_slacks = torch.zeros(H, 2*(2*grid.num_batt + grid.num_curt))
    pred_line_max_slacks = torch.zeros(num_agents, H, grid.num_lines)
    pred_line_min_slacks = torch.zeros(num_agents, H, grid.num_lines)

    Q = []
    q = []
    G = []
    non_sparse_G = []
    h = []
    A = []
    b = []
    scipy_Q = []
    scipy_G = []
    scipy_A = []
    
    for i in range(num_agents):
        naza = naza_list[i]
        naza_disturbances = disturbances[:, naza.buses_in_area]
        config = naza.get_matrices_dQPTH(naza_disturbances, t)
        Q.append(config["csc_scaled_Q"])
        q.append(config["q"])
        G.append(config["csc_scaled_G"])
        non_sparse_G.append(config["G"])
        h.append(config["h"])
        A.append(config["csc_scaled_A"])
        b.append(config["b"])
        scipy_Q.append(config["scipy_scaled_Q"])
        scipy_G.append(config["scipy_scaled_G"])
        scipy_A.append(config["scipy_scaled_A"])
    try:
        results = dQPTH_layer(Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t)
    except Exception as e:
        print("Following error occurred when calling dQPTH_layer:")
        print(e)
    try:
        for i in range(num_agents):
            naza = naza_list[i]
            z = results[i]
            
            bus_states, line_states, actions_curt, actions_batt, bus_slacks, line_max_slacks, line_min_slacks, objective = \
                naza.interpret_z(z.flatten(), update_state=update_state)

            pred_actions_curt[:,naza.curt_idx] = actions_curt
            pred_actions_batt[:,naza.batt_idx] = actions_batt
            pred_line_max_slacks[i,:,:] = line_max_slacks
            pred_line_min_slacks[i,:,:] = line_min_slacks
            # pred_bus_slacks[:,naza.bus_slack_idx] = bus_slacks
    except Exception as e:
        print(e)
    return pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks


def get_next_action_base(grid, naza_list, disturbances, t, dQPTH_layer, verbose=False, with_cvxpy=False, update_state=False):
    """
    Collects actions from all agents for the next prediction window, aggregates them into one single action to be taken
    on the entire grid.

    Args:
        grid (Grid):
            Full grid that the NAZAs act on.
        naza_list (List of Split_Constraint_NAZA objects):
            List of all NAZAs on the grid.
        disturbances (Torch tensor of shape (H, num_buses)):
            Predictions of power injection noise at each bus in the NAZA control area.
        t (int):
            Current timestep.
        update_state (bool):
            True if NAZA should update its own state as if the actions solved for are actually taken.
    Returns:
        pred_actions_curt (Torch tensor of shape (H, grid.num_curt)):
            Curtailment actions to be taken across the grid, aggregated across all NAZAs.
        pred_actions_batt (Torch tensor of shape (H, grid.num_batt)):
            Battery actions to be taken across the grid, aggregated across all NAZAs.
        pred_line_max_slacks (Torch tensor of shape (num_agents, H, grid.num_lines)):
            Each agent's predictions for the line max slack variables it would set along the horizon.
        pred_line_min_slacks (Torch tensor of shape (num_agents, H, grid.num_lines)):
            Each agent's predictions for the line min slack variables it would set along the horizon.
        pred_bus_slacks (Torch tensor of shape (H, 2*(2*grid.num_batt + grid.num_curt))):
            Aggregated bus slack variables (not including splits) across all agents.
    """
    start_next_action = time.time()
    H = naza_list[0].H
    pred_actions_curt = torch.zeros(H, grid.num_curt)
    pred_actions_batt = torch.zeros(H, grid.num_batt)
    num_agents = len(naza_list)
    pred_bus_slacks = torch.zeros(H, 2*(2*grid.num_batt + grid.num_curt))
    pred_line_max_slacks = torch.zeros(num_agents, H, grid.num_lines)
    pred_line_min_slacks = torch.zeros(num_agents, H, grid.num_lines)

    Q = []
    q = []
    G = []
    non_sparse_G = []
    h = []
    A = []
    b = []
    scipy_Q = []
    scipy_G = []
    scipy_A = []
    
    for i in range(num_agents):
        naza = naza_list[i]
        naza_disturbances = disturbances[:, naza.buses_in_area]
        config = naza.get_matrices_dQPTH(naza_disturbances, t)
        Q.append(config["csc_scaled_Q"])
        q.append(config["q"])
        G.append(config["csc_scaled_G"])
        non_sparse_G.append(config["G"])
        h.append(config["h"])
        A.append(config["csc_scaled_A"])
        b.append(config["b"])
        scipy_Q.append(config["scipy_scaled_Q"])
        scipy_G.append(config["scipy_scaled_G"])
        scipy_A.append(config["scipy_scaled_A"])
    try:
        results = dQPTH_layer(Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t)
    except Exception as e:
        print("Following error occurred when calling dQPTH_layer:")
        print(e)
    try:
        for i in range(num_agents):
            naza = naza_list[i]
            z = results[i]
            
            bus_states, line_states, actions_curt, actions_batt, bus_slacks, line_max_slacks, line_min_slacks, objective = \
                naza.interpret_z(z.flatten(), update_state=update_state)

            pred_actions_curt[:,naza.curt_idx] = actions_curt
            pred_actions_batt[:,naza.batt_idx] = actions_batt
    except Exception as e:
        print(e)
    return pred_actions_curt, pred_actions_batt, None, None, None


def reset_states(grid, naza_list):
    """
    Resets all grid and NAZAs parameters to the values they had when originally initialized.

    Args:
        grid (Grid):
            Full grid that the NAZAs act on.
        naza_list (List of Split_Constraint_NAZA objects):
            List of all NAZAs on the grid.
    Returns:
        None
    """
    grid.reset_state()
    for naza in naza_list:
        naza.update_bus_state()
        naza.reset_line_state()


def update_naza_bus_states(naza_list):
    """
    Updates bus states of all NAZAs in the list.

    Args:
        naza_list (List of Split_Constraint_NAZA objects):
            List of all NAZAs on the grid.

    Returns:
        None
    """
    for naza in naza_list:
        naza.update_bus_state()


def is_partition(partition, full_list):
    """
    Checks if the given partition is a valid partition of the full_list.

    Args:
        partition: (list of lists) The partition to check.
        full_list: (list) The list to partition.

    Returns:
        bool: True if partition is a valid partition of full_list, False otherwise.
    """
    # Flatten the partition and check if it contains all elements of full_list
    flattened = [item for sublist in partition for item in sublist]
    return sorted(flattened) == sorted(full_list) and len(flattened) == len(set(flattened))


def find_cut_lines(grid, partition):
    """
    Identifies the lines connecting different areas in the grid.

    Args:
        grid: (Grid object)
        partition: (list of lists) Partition of nodes into different areas.

    Returns:
        cut_lines: (list) List of indices of cut lines
        cut_lines_data: (list) List of tuples containing line index, the two buses the line connects, and the two areas those
                               buses belong to.
    """
    # Initialize an empty list to store the indices of lines connecting different areas
    cut_lines = []
    cut_lines_data = []

    # Iterate through each line in the "line_data" array
    for idx, line in enumerate(grid.line_data):
        node1, node2 = int(line[0]), int(line[1])  # Convert to 0-based indexing
        # Check if the nodes belong to different areas
        for area1_idx, area1 in enumerate(partition):
            if node1 in area1:
                for area2_idx, area2 in enumerate(partition):
                    if area1_idx != area2_idx and node2 in area2:
                        cut_lines.append(idx)
                        cut_lines_data.append((idx, node1, node2, area1_idx, area2_idx))
                        break

    return cut_lines, cut_lines_data


def explain_infeasibility(grid, infeasible_idx, violations):
    """
    Returns a message explaining what specifically went infeasible.

    Args:
        grid: (Grid object)
        infeasible_idx: (list) index of constraint violation vector that was violated
        violations: (list) list of violation magnitude
    Returns:
        message: (String)
    """
    full_grid_params = grid.get_grid_data()
    full_line_data = full_grid_params["line_data"]
    bus_with_batt = full_grid_params["bus_with_batt"]
    bus_with_gen  = full_grid_params["bus_with_gen"]
    message = "BUSES ARE 0-INDEXED HERE:\n"

    for i in range(len(infeasible_idx)):
        idx_message = ""
        idx = infeasible_idx[i]
        if idx < grid.num_lines:
            line_num = idx
            idx_message += f"Line {line_num} from {full_line_data[line_num, 0]} to {full_line_data[line_num, 1]} "\
                           f"with limit of {grid.line_limits[line_num]} had min violated by {violations[i]}"
        elif idx >= grid.num_lines and idx < 2*grid.num_lines:
            line_num = idx - grid.num_lines
            idx_message += f"Line {line_num} from {full_line_data[line_num, 0]} to {full_line_data[line_num, 1]} "\
                           f"with limit of {grid.line_limits[line_num]} had max violated by {violations[i]}"
        elif idx >= 2*grid.num_lines and idx < 2*grid.num_lines + grid.num_batt:
            batt_num = idx - 2*grid.num_lines
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with min charge limit of "\
                           f"{grid.batt_charge_min_limits[batt_num]} had min violated by {violations[i]}"
        elif idx >= 2*grid.num_lines + grid.num_batt and idx < 2*(grid.num_lines + grid.num_batt):
            batt_num = idx - 2*grid.num_lines - grid.num_batt
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with max charge limit of "\
                           f"{grid.batt_charge_max_limits[batt_num]} had max violated by {violations[i]}"
        elif idx >= 2*(grid.num_lines + grid.num_batt) and idx < 2*(grid.num_lines + grid.num_batt) + grid.num_gen:
            gen_num = idx - 2*(grid.num_lines + grid.num_batt)
            idx_message += f"Curtailable generation {gen_num} at node {bus_with_gen[gen_num]} with min limit of "\
                           f"{grid.curt_min_limits[gen_num]} had min violated by {violations[i]}"
        elif idx >= 2*(grid.num_lines + grid.num_batt) + grid.num_gen and idx < 2*(grid.num_lines + grid.num_batt + grid.num_gen):
            gen_num = idx - 2*(grid.num_lines + grid.num_batt) - grid.num_gen
            idx_message += f"Curtailable generation {gen_num} at node {bus_with_gen[gen_num]} with max limit of "\
                           f"{grid.curt_max_limits[gen_num]} had max violated by {violations[i]}"
        elif idx >= 2*(grid.num_lines + grid.num_batt + grid.num_gen) and idx < 2*(grid.num_lines + grid.num_batt + grid.num_gen) + grid.num_batt:
            batt_num = idx - 2*(grid.num_lines + grid.num_batt + grid.num_gen)
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with min power injection limit of "\
                           f" {grid.batt_power_min_limits[batt_num]} had min violated by {violations[i]}"
        else:
            batt_num = idx - 2*(grid.num_lines + grid.num_batt + grid.num_gen) - grid.num_batt
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with max power injection limit of "\
                           f" {grid.batt_power_max_limits[batt_num]} had max violated by {violations[i]}"

        if not i == len(infeasible_idx) - 1:
            idx_message += "\n"
        message += idx_message

    return message