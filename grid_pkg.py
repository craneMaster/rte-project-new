import torch

class Grid:
    """Captures grid state and dynamics

    Tracks and updates line flows, batteries and curtailment in a grid with linearized dynamics.
    Agnostic to control strategy or communication structure, only considers grid physics.

    Attributes:
    state_dim (int):
        Dimension of the grid state vector
    bus_slack_dim (int):
        Dimension of the grid slack vector associated with bus-specific constraints.
    num_buses (int):
        Number of buses on the grid
    num_lines (int):
        Number of lines on the grid
    num_curt (int):
        Number of buses with curtailable generation on the grid
    num_batt (int):
        Number of batteries on the grid
    state (Torch matrix, shape (state_dim)):
        Vector containing current state of the grid, order is:
        - line flows
        - battery charge
        - battery power injection (positive if power goes from battery to grid)
        - curtailed power
    init_state (Torch matrix, shape (state_dim)):
        Initial state vector
    bus_with_curt (Torch matrix, shape (num_curt)):
        List of bus numbers with curtailable generation, contains int values
    bus_with_batt (Torch matrix, shape (num_batt)):
        List of bus numbers with a battery, contains int values
    delta_t (Double):
        Time between grid updates
    line_data (Torch matrix):
        Contains raw line data from MATPOWER test case
    bus_data (Torch matrix):
        Contains raw bus data from MATPOWER test case
    gen_data (Torch matrix):
        Contains raw generator data from MATPOWER test case
    M (Torch matrix, shape (num_lines, num_buses)):
        Sensitivity matrix for changes in line flow vs power injection, part of linearized dynamics
    M_curt (Torch matrix, shape (num_lines, num_curt)):
        Sensitivity matrix for changes in line flow vs curtailment, submatrix of M
    M_batt (Torch matrix, shape (num_lines, num_batt)):
        Sensitivity matrix for changes in line flow vs battery action, submatrix of M
    M_noise (Torch matrix, shape (num_lines, num_buses)):
        Sensitivity matrix for changes in line flow vs buses, submatrix of M
    A (Torch matrix, shape (state_dim, state_dim)):
        State transition dynamics, absent actions or noise
    B_curt (Torch matrix, shape (state_dim, num_curt)):
        Matrix that gives impact of curtailment actions on grid state
    B_batt (Torch matrix, shape (state_dim, num_batt)):
        Matrix that gives impact of battery actions on grid state
    B_noise (Torch matrix, shape (state_dim, num_buses)):
        Matrix that gives impact of noise at each bus on grid state
    H_x (Torch matrix, shape (2*state_dim, state_dim)):
        Matrix associated with state constraint, H_x @ state <= H_limit
    H_limit (Torch matrix, shape (2*state_dim)):
        Vector associated with state constraint, H_x @ state <= H_limit
    line_limits (Torch matrix, (num_line)):
        Maximum power flow magnitude across each line
    batt_charge_max_limits (Torch matrix, (num_batt)):
        Maximum charge for each battery
    batt_charge_min_limits (numpy array, (num_batt)):
        Minimum charge for each battery
    batt_power_max_limits (Torch matrix, (num_batt)):
        Maximum power injection for each battery
    batt_power_min_limits (Torch matrix, (num_batt)):
        Minimum power injection for each battery
    curt_max_values (Torch matrix, (num_curt)):
        Maximum amount of power that can be curtailed at each generator
    curt_min_values (Torch matrix, (num_curt)):
        Minimum amount of power that can be curtailed at each generator
    target_batt_charges (Torch matrix, (num_batt)):
        Vector containing target charge level for each battery.
    state_batt_charge_idx (Slice object):
        Indices for getting battery charges from overall state vector.
    state_line_flow_idx (Slice object):
        Indices for getting line flow from overall state vector.
    state_curt_idx (Slice object):
        Indices for getting curtailment from overall state vector.
    """

    def __init__(self, bus_with_curt, bus_with_batt, delta_t, line_data_loc, bus_data_loc, gen_data_loc, ptdf_data_loc,
                 control_delay=0):
        """
        Args:
            bus_with_curt (Torch matrix, shape (num_curt)):
                List of bus numbers with curtailable generation, contains int values
            bus_with_batt (Torch matrix, shape (num_batt)):
                List of bus numbers with a battery, contains int values
            delta_t (Double):
                Time between grid updates
            control_delay (int):
                Number of timesteps between issuing a control action and its effect on grid state.
                Zero means immediate effect (default).
            line_data_loc (String):
                Location of branch data file
            bus_data_loc (String):
                Location of bus data file
            gen_data_loc (String):
                Location of generator data file
            ptdf_data_loc (String):
                Location of ptdf data file. We currently use the ptdf matrix from MATPOWER for sensitivity matrix M.

        Notes:
            We assume that buses and lines are all 0-indexed. Data is stored in a .npy format, organized in the
            MATPOWER case format. See https://matpower.org/docs/ref/matpower5.0/case30.html for an example.
        """
        self.bus_with_curt = bus_with_curt
        self.bus_with_batt = bus_with_batt
        self.delta_t = float(delta_t)
        if control_delay < 0:
            raise ValueError(f"control_delay {control_delay} is negative.")
        self.control_delay = int(control_delay)
        self.line_data = torch.load(line_data_loc, weights_only=True)
        self.bus_data = torch.load(bus_data_loc, weights_only=True)
        self.gen_data = torch.load(gen_data_loc, weights_only=True)
        self.M = torch.load(ptdf_data_loc, weights_only=True)

        self.num_buses = self.bus_data.shape[0]
        self.num_lines = self.line_data.shape[0]
        self.num_curt = bus_with_curt.shape[0]
        self.num_batt = bus_with_batt.shape[0]
        self.state_dim = self.num_lines + self.num_curt + 2*self.num_batt
        self.bus_state_dim = self.num_curt + 2*self.num_batt
        self.bus_slack_dim = 2 * self.num_curt + 4 * self.num_batt

        init_line_flows = self.line_data[:, 13]
        init_batt_charges, init_batt_powers = self.init_batteries()
        init_curt_values = self.init_curt()
        self.set_matrices()

        self.init_state = torch.cat([init_line_flows, init_batt_charges, init_batt_powers, init_curt_values])
        self.baseline_init_state = self.init_state.clone()
        self.baseline_target_batt_charges = self.target_batt_charges.clone()
        # When set (multi-cycle handoff), reset_state() restores these delay queues with init_state.
        self.init_action_buffers = None
        self.state = self.init_state
        self._reset_action_buffers()

        self.state_line_flow_idx = slice(0, self.num_lines)
        self.state_batt_charge_idx = slice(self.num_lines, self.num_lines + self.num_batt)
        self.state_batt_power_idx = slice(self.num_lines + self.num_batt, self.num_lines + 2*self.num_batt)
        self.state_curt_idx = slice(self.num_lines + 2*self.num_batt, self.num_lines + 2*self.num_batt + self.num_curt)
        self.state_bus_state_idx = slice(self.num_lines, self.num_lines + 2*self.num_batt + self.num_curt)


    def check_state_feasible(self, tol=1e-3):
        """
        Checks if the current state is feasible.

        Args:
            tol: (float) maximum acceptable infeasibility

        Returns:
            bool: True if state is feasible, False otherwise
            infeasible_idx: (numpy array) indices where the state is infeasible
            violations: (numpy array) violation amounts

        """
        gap = self.H_x @ self.state - self.H_limit
        infeasible_idx = torch.where(gap > tol)[0]
        return torch.all(gap <= tol), infeasible_idx, gap[infeasible_idx]


    def add_agent(self, agent):
        """
        Adds an agent to the grid.

        Args:
            agent

        Returns:
            None
        """
        self.agents.append(agent)


    def set_matrices(self):
        """
        Function that constructs matrices associated with dynamics and constraints.

        Args:
            self

        Returns:
            None
        """
        self.M_curt = self.M[:, self.bus_with_curt]
        self.M_batt = self.M[:, self.bus_with_batt]
        self.M_noise = self.M
        nL = self.num_lines
        nB = self.num_batt
        nC = self.num_curt
        N = nL + 2*nB + nC
        self.A = torch.zeros((N, N))
        self.A[0:nL,          0:nL]          = torch.eye(nL)
        self.A[nL:nL+nB,      nL:nL+nB]      = torch.eye(nB)
        self.A[nL+nB:nL+2*nB, nL+nB:nL+2*nB] = torch.eye(nB)
        self.A[nL+2*nB:,      nL+2*nB:]      = torch.eye(nC)
        self.A[nL:nL+nB, nL+nB:nL+2*nB]      = -self.delta_t * torch.eye(nB) / 3600

        self.B_curt = torch.cat([self.M_curt,         torch.zeros(nB, nC),
                                 torch.zeros(nB, nC), torch.eye(nC)])
        self.B_batt = torch.cat([self.M_batt,  -self.delta_t * torch.eye(nB) / 3600,
                                 torch.eye(nB), torch.zeros(nC, nB)])
        self.B_noise = torch.cat([self.M_noise, torch.zeros(self.state_dim - nL, self.num_buses)])

        self.H_x = torch.zeros((2*nL+4*nB+2*nC,nL+2*nB+nC))
        self.H_x[0:nL, 0:nL] = -torch.eye(nL)

        rows = 2*nL + 4*nB + 2*nC
        cols = nL + 2*nB + nC
        self.H_x = torch.zeros((rows, cols))
        self.H_x[0:nL,                        0:nL]               = -torch.eye(nL)
        self.H_x[nL:2*nL,                     0:nL]               =  torch.eye(nL)
        self.H_x[2*nL:2*nL+nB,                nL:nL+nB]           = -torch.eye(nB)
        self.H_x[2*nL+nB:2*nL+2*nB,           nL:nL+nB]           =  torch.eye(nB)
        self.H_x[2*nL+2*nB:2*nL+3*nB,         nL+nB:nL+2*nB]      = -torch.eye(nB)
        self.H_x[2*nL+3*nB:2*nL+4*nB,         nL+nB:nL+2*nB]      =  torch.eye(nB)
        self.H_x[2*nL+4*nB:2*nL+4*nB+nC,      nL+2*nB:nL+2*nB+nC] = -torch.eye(nC)
        self.H_x[2*nL+4*nB+nC:2*nL+4*nB+2*nC, nL+2*nB:nL+2*nB+nC] =  torch.eye(nC)

        self.line_limits = self.line_data[:,5]
        self.H_limit = torch.cat([self.line_limits,           self.line_limits,           -self.batt_charge_min_limits, self.batt_charge_max_limits,
                                 -self.batt_power_min_limits, self.batt_power_max_limits, -self.curt_min_limits,        self.curt_max_limits])


    def init_batteries(self):
        """
        Creates min/max limits on battery charge and power injection, and initializes battery charge.

        Args:
            self

        Returns:
            init_batt_charges: (numpy array) initial values for the batt charge (num_batt,)
            init_batt_powers: (numpy array) initial values for battery power injection (num_batt,)
        """
        # assuming that batteries can collectively meet 5% of total average instantaneous demand
        total_demand = torch.sum(self.bus_data[:,2])
        max_batt_supply = total_demand / 20
        self.batt_power_max_limits = torch.ones(self.num_batt) * max_batt_supply / self.num_batt
        self.batt_power_min_limits = -self.batt_power_max_limits
        # self.batt_power_min_limits = -0 * torch.ones(self.num_batt) * (total_demand / 4) / self.num_batt

        # assuming that total battery power storage capacity can sustain max injection for 4 hours
        total_max = max_batt_supply * 4 * 60 * 60

        # assuming that all batteries have the same power storage capacity
        self.batt_charge_max_limits = torch.ones(self.num_batt) * (total_max / self.num_batt) / 3600    # conver to MWh
        self.batt_charge_min_limits = torch.zeros(self.num_batt)  # 'charge' actually means work or energy in physics

        # init_batt_charges = self.batt_charge_min_limits\
        #       + np.random.rand(self.num_batt) * (self.batt_charge_max_limits - self.batt_charge_min_limits)
        init_batt_charges = 0.5 * (self.batt_charge_max_limits + self.batt_charge_min_limits)
        init_batt_powers = 0.5 * (self.batt_power_max_limits + self.batt_power_min_limits)

        # self.target_batt_charges = torch.rand(self.num_batt) * (self.batt_charge_max_limits - self.batt_charge_min_limits) \
        #                             + self.batt_charge_min_limits
        self.target_batt_charges = init_batt_charges

        return init_batt_charges, init_batt_powers


    def init_curt(self):
        """
        Creates a list of curtailment limits.

        Args:
            self

        Returns:
            init_curt_values: (numpy array) initial values for the amount of power being curtailed (num_ges,)
        """
        self.curt_max_limits = self.gen_data[:,8]
        self.curt_min_limits = torch.zeros(self.num_curt)
        init_curt_values = 0.2 * self.curt_max_limits

        return init_curt_values


    def _reset_action_buffers(self):
        """Clear issued-action history and pending delay queue."""
        self._pending_actions_curt = []
        self._pending_actions_batt = []
        self._issued_actions_curt = []
        self._issued_actions_batt = []
        # Effective (actually-applied) action from the most recent update_state; zero until the
        # first controlled step. Losses penalize this rather than the freshly-predicted action so
        # the cost reflects what physically hit the grid under control_delay.
        self.effective_curt = torch.zeros(self.num_curt)
        self.effective_batt = torch.zeros(self.num_batt)


    def snapshot_action_buffers(self):
        """Deep-copy pending/issued delay queues and last effective actions."""
        return {
            "pending_curt": [a.detach().clone() for a in self._pending_actions_curt],
            "pending_batt": [a.detach().clone() for a in self._pending_actions_batt],
            "issued_curt": [a.detach().clone() for a in self._issued_actions_curt],
            "issued_batt": [a.detach().clone() for a in self._issued_actions_batt],
            "effective_curt": self.effective_curt.detach().clone(),
            "effective_batt": self.effective_batt.detach().clone(),
        }


    def restore_action_buffers(self, snapshot):
        """Restore delay queues from ``snapshot_action_buffers()`` (or clear if ``None``)."""
        if snapshot is None:
            self._reset_action_buffers()
            return
        self._pending_actions_curt = [a.clone() for a in snapshot["pending_curt"]]
        self._pending_actions_batt = [a.clone() for a in snapshot["pending_batt"]]
        self._issued_actions_curt = [a.clone() for a in snapshot["issued_curt"]]
        self._issued_actions_batt = [a.clone() for a in snapshot["issued_batt"]]
        self.effective_curt = snapshot["effective_curt"].clone()
        self.effective_batt = snapshot["effective_batt"].clone()


    def set_cycle_handoff(self, terminal_state, action_buffers=None):
        """Seed the next cycle from a terminal plant state and matching delay buffers."""
        self.init_state = terminal_state.detach().clone()
        if action_buffers is None:
            self.init_action_buffers = None
        else:
            # Snapshot once so later resets don't share live tensors with the capture site.
            self.init_action_buffers = {
                "pending_curt": [a.detach().clone() for a in action_buffers["pending_curt"]],
                "pending_batt": [a.detach().clone() for a in action_buffers["pending_batt"]],
                "issued_curt": [a.detach().clone() for a in action_buffers["issued_curt"]],
                "issued_batt": [a.detach().clone() for a in action_buffers["issued_batt"]],
                "effective_curt": action_buffers["effective_curt"].detach().clone(),
                "effective_batt": action_buffers["effective_batt"].detach().clone(),
            }


    def reset_state(self):
        """
        Resets state to its original value.

        Args:
            None
        Returns:
            None
        """
        self.state = self.init_state
        # Multi-cycle handoff: re-seed delay history so MPC past-action RHS and plant apply
        # stay consistent with the terminal snapshot (especially control_delay > 0).
        self.restore_action_buffers(self.init_action_buffers)


    def restore_baseline_init(self):
        """Reset init_state and battery targets to values from grid construction."""
        self.init_state = self.baseline_init_state.clone()
        self.target_batt_charges = self.baseline_target_batt_charges.clone()
        self.init_action_buffers = None
        self.reset_state()


    def get_past_actions(self, horizon_step):
        """
        Return the action that affects MPC horizon step ``horizon_step`` when it
        lies before the current planning window (i.e. ``horizon_step < control_delay``).

        At current time t with control_delay d, the issued queue stores actions
        oldest-first as [u_{t-L}, ..., u_{t-1}] where L = min(t, d). Horizon
        step i is driven by the effective action at grid time t+i, which is
        u_{t+i-d}. Its position in the queue is ``pos = L + i - d``; if
        ``pos < 0`` the action has not yet been issued and the contribution is
        zero. This warm-up handling matters when t < d: the queue is shorter
        than d, so naive index-by-horizon-step would alias future-effective
        actions onto early horizon rows and double-count them.
        """
        if self.control_delay == 0 or horizon_step >= self.control_delay:
            return torch.zeros(self.num_curt), torch.zeros(self.num_batt)
        queue_len = len(self._issued_actions_curt)
        pos = queue_len + horizon_step - self.control_delay
        if pos < 0 or pos >= queue_len:
            return torch.zeros(self.num_curt), torch.zeros(self.num_batt)
        return self._issued_actions_curt[pos], self._issued_actions_batt[pos]


    def update_state(self, action_curt, action_batt, noise):
        """
        Updates state of the entire grid based on all the actions and noise across the grid.

        Args:
            action_curt (Torch matrix, shape (num_curt)):
                Vector containing curtailment actions taken at each generator. Same order as bus_with_curt.
            action_batt (Torch matrix, shape (num_batt)):
                Vector containing action taken at each battery. Same order as bus_with_batt.
            noise (Torch matrix, shape (num_buses)):
                Vector containing power injection noise across the grid at that timestep.
        """
        if self.control_delay == 0:
            effective_curt, effective_batt = action_curt, action_batt
        else:
            self._pending_actions_curt.append(action_curt.clone())
            self._pending_actions_batt.append(action_batt.clone())
            if len(self._pending_actions_curt) > self.control_delay:
                effective_curt = self._pending_actions_curt.pop(0)
                effective_batt = self._pending_actions_batt.pop(0)
            else:
                effective_curt = torch.zeros_like(action_curt)
                effective_batt = torch.zeros_like(action_batt)

            self._issued_actions_curt.append(action_curt.clone())
            self._issued_actions_batt.append(action_batt.clone())
            if len(self._issued_actions_curt) > self.control_delay:
                self._issued_actions_curt.pop(0)
                self._issued_actions_batt.pop(0)

        # Expose the action that actually took effect this step (differentiable: it is a clone of an
        # earlier predicted action for control_delay > 0, or the current prediction for delay 0).
        self.effective_curt = effective_curt
        self.effective_batt = effective_batt
        self.state = self.A @ self.state + self.B_curt @ effective_curt + self.B_batt @ effective_batt + self.B_noise @ noise


    def get_grid_data(self):
        """
        Retrives all the information that an agent on the grid needs to know.

        Args:
            None

        Returns:
            grid_data: (dict)
        """
        grid_data = dict()
        grid_data["num_lines"] = self.num_lines
        grid_data["line_data"] = self.line_data
        grid_data["num_curt"] = self.num_curt
        grid_data["bus_with_curt"] = self.bus_with_curt
        grid_data["num_batt"] = self.num_batt
        grid_data["bus_with_batt"] = self.bus_with_batt
        grid_data["ptdf"] = self.M
        grid_data["delta_t"] = self.delta_t
        grid_data["batt_charge_min_limits"] = self.batt_charge_min_limits
        grid_data["batt_charge_max_limits"] = self.batt_charge_max_limits
        grid_data["batt_power_min_limits"] = self.batt_power_min_limits
        grid_data["batt_power_max_limits"] = self.batt_power_max_limits
        grid_data["curt_min_limits"] = self.curt_min_limits
        grid_data["curt_max_limits"] = self.curt_max_limits

        return grid_data


    def get_lines_from_buses(self, buses):
        """
        Reports the line index, "from" bus, and "to" bus corresponding to pairs of buses. If there is no line between
        a pair of buses, reports None for that pair.

        Args:
            buses: (list of tuples) tuple contains pair of bus numbers

        Returns:
            lines: (numpy array) contains rows with line index, "from" bus, and "to" bus, or None if no line connects
            the pair of buses
        """
        result = []
        for bus_pair in buses:
            bus1, bus2 = bus_pair
            match = torch.where(
                ((self.line_data[:, 0] == bus1) & (self.line_data[:, 1] == bus2)) |
                ((self.line_data[:, 0] == bus2) & (self.line_data[:, 1] == bus1))
            )[0]
            if match.size > 0:
                line_idx = match[0]
                from_bus = int(self.line_data[line_idx, 0])
                to_bus = int(self.line_data[line_idx, 1])
                result.append([line_idx, from_bus, to_bus])
            else:
                result.append([None, None, None])
        return torch.tensor(result) 
