import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

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

def str2bool(s):
    return s.lower() in {"1","true","t","yes","y"}

def build_tag(parser, args, *, skip=("job_id",), sep="_", bool_as_int=False, only_nondefaults=False):
    """
    Create a tag from all argparse args without hard-coding keys.

    - skip: names to exclude (e.g., ("job_id",))
    - sep: separator between parts
    - bool_as_int: if True -> booleans encoded as 1/0; else True/False
    - only_nondefaults: if True -> include only args whose value != default
    """
    def fmt(v):
        if isinstance(v, bool):
            return ("1" if v else "0") if bool_as_int else str(v)
        if isinstance(v, float):
            # compact float (no trailing zeros), but keep full precision
            return f"{v:g}"
        return str(v)

    # Build a dict of defaults to optionally filter unchanged values
    defaults = {}
    for action in parser._actions:
        if not hasattr(action, "dest") or action.dest in (None, "help"):
            continue
        defaults[action.dest] = action.default

    parts = []
    # Use parser._actions to preserve declaration order
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help" or dest in skip:
            continue

        val = getattr(args, dest)
        if only_nondefaults and dest in defaults and val == defaults[dest]:
            continue

        parts.append(f"{dest}{fmt(val)}")

    return sep.join(parts)

# @profile
def main(args):
    torch.set_default_dtype(torch.double)
    torch.set_printoptions(threshold=10000)
    io_disable = False
    # io_disable = not sys.stdout.isatty()

    # -------- sweep params --------
    ps = args.ps
    offset = args.offset
    test_skew_mag = args.test_skew_mag
    radius = args.radius
    job_id = args.job_id
    # ------------------------------
    folder = f"paper_experiments/{job_id}/"
    tag = build_tag(parser, args, skip=("job_id",), bool_as_int=False, only_nondefaults=False)
    print(f"Checkpoint for this run will be saved at {folder}checkpoint_{tag}.pt")
    os.makedirs(folder, exist_ok=True)

    line_data_loc = 'data/case118_line_data.pt'
    bus_data_loc = 'data/case118_bus_data.pt'
    gen_data_loc = 'data/case118_gen_data.pt'
    ptdf_data_loc = 'data/case118_ptdf_data.pt'

    bus_with_curt = torch.load(gen_data_loc, weights_only=True)[:,0].int()
    num_curt = bus_with_curt.shape[0]
    bus_with_batt = torch.tensor([10*i+2 for i in range(12)], dtype=torch.int)
    num_batt = bus_with_batt.shape[0]
    delta_t = 15

    grid = grid_pkg.Grid(bus_with_curt, bus_with_batt, delta_t, line_data_loc, bus_data_loc, gen_data_loc, ptdf_data_loc)
    T = 20
    H = 5
    
    train_trajs = torch.load(f'data/scenario_generation/train_trajs_rad{radius}.pt')
    test_trajs_with_mismatch = torch.load(f'data/scenario_generation/test_trajs_with_mismatch_rad{radius}/test_trajs_with_mismatch_ps{ps}_offset{offset}_tsm{test_skew_mag}.pt')

    batt_cost = 100
    curt_change_cost = 0.01
    curt_net_cost = 1
    bus_slack_cost = 1e8
    line_slack_cost = 1e2

    num_agents = 3
    nodes_1 = [i for i in range(0, 42)] + [112, 113, 114, 116]
    nodes_2 = [i for i in range(42, 69)] + [115]
    nodes_3 = [i for i in range(69, 112)] + [117]
    partition = [nodes_1, nodes_2, nodes_3]
    controller_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, H, batt_cost,
                                                                           curt_change_cost, curt_net_cost, bus_slack_cost,
                                                                           line_slack_cost)
    
    ckpt = dict()
    for k, v in vars(args).items():
        ckpt[k] = v

    num_train_traj = train_trajs["num_train_traj"]
    num_train_torch_seeds = train_trajs["num_torch_seeds"] // 2
    num_test_traj = test_trajs_with_mismatch["num_test_traj"]
    num_test_torch_seeds = test_trajs_with_mismatch["num_torch_seeds"]

    pool = mp.Pool(processes=num_agents)
    dqp_eps = 1
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    # for i in range(1):
    # for i in tqdm(range(num_test_torch_seeds)):
    for test_seed in tqdm(range(3)):
        all_test_trajs = test_trajs_with_mismatch[(ps, offset, test_skew_mag, test_seed)]
        optimal_losses = torch.zeros(num_test_traj,3)
        # for j in range(1):
        for j in tqdm(range(num_test_traj)):
            test_traj = all_test_trajs[j]
            ckpt_central = evals.get_central_full_traj(grid, T, H, test_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost)
            optimal_losses[j] = ckpt_central["total_losses"]

        ckpt[(test_seed, "optimal")] = optimal_losses

    pool.close()
    print("final save")
    torch.save(ckpt, f"{folder}checkpoint_{tag}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps", type=float, default=4.0)
    parser.add_argument("--offset", type=int, default=180)
    parser.add_argument("--test_skew_mag", type=int, default=25)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--job_id", type=str, default="manual") # <- from Slurm
    args = parser.parse_args()
    main(args)