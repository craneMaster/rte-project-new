import numpy as np
import torch.multiprocessing as mp

import torch
from torch import nn

import qpsolvers
from scipy.sparse import csc_matrix, bmat
import torch
import scipy.sparse as sp
import qdldl

import sys
import os
import warnings

src_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(src_dir)
sys.path.append(parent_dir)

from dQPTH_helpers.set_solver_tolerance import set_solver_tolerance
from dQPTH_helpers.gurobi_ws import (
    gurobi_solve_problem as gurobi_ws_solve_problem,
    diagnose_qp_failure,
    gurobi_status_name,
    qp_constraint_violations,
)

import dQPTH_helpers.sparse_helper as sparse_helper
import dQPTH_helpers.lin_solvers as lin_solvers
import time

class dQPTH_layer(nn.Module):
    '''solves and differentiates
    x^* = argmin_x 1/2 x^T Q x + q^T x
             s.t.  G x <= h
                   A x  = b
    (including dual variables mu^*,nu^*)

    Written by James Chen, based on dQP and qpth code, hence our name dQPTH

    We use much of the code from dQP, but adapt the backward pass from qpth, as we find that the Schur complement trick
    allows for better stability. (We do not claim this to be a limitation of dQP, but rather a limitation of the author's
    ability to implement and tune numerical algorithms.) For the forward pass we default to using Gurobi.

    dQP and qpth allow for batched QPs with the same size. We instead allow for multiprocessing of QPs of different
    sizes, and assign each QP to a different worker.
    '''
    def __init__(self,settings=None,workers=3,pool=None):
        super().__init__()

        if settings is None:
            settings = build_settings() # call with defaults

        for k, v in settings.items():
            setattr(self, k, v)

        self.dim = None
        self.nIneq = None
        self.nEq = None
        self.nBatch = None

        self.differentiate_QP = differentiate_QP.apply # set-up differentiation through active constraints

        # initialize numpy variables which are carried implicitly ; torch version is not stored
        self.x_star_np = None
        self.mu_star_np = None
        self.nu_star_np = None
        self.x_star_np_cache = dict()

        self.Q_np = None
        self.q_np = None
        self.G_np = None
        self.h_np = None
        self.A_np = None
        self.b_np = None

        self.backward_cache = dict()

        if pool == None:
            self.pool = mp.Pool(processes=workers)
            print(f"Persistent multiprocessing pool initialized with {workers} workers.")
        else:
            self.pool = pool

    def close(self):
        """Method to close the pool gracefully when done with the object."""
        print("Closing persistent multiprocessing pool.")
        self.pool.close()
        self.pool.join() # Wait for workers to finish current tasks and exit

    def forward(self, Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t):
        # check shapes, extract dim,nIneq,nEq,nBatch
        Q,q,G,h,A,b = self.get_shapes(Q, q, G, h, A, b)
        self.data_to_np(Q, q, G, h, A, b)
        self.solve(t)
        self.nu_star_with_inactive_np = self.nu_star_np.copy()
        x_star = self.setup_diff(Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t)

        return x_star


    def clear_cache(self):
        """
        Clears cache of warm-start points.
        """
        self.x_star_np_cache = dict()
        self.backward_cache = dict()


    def get_shapes(self,Q,q,G,h,A,b):
        '''
        extract dim,nIneq,nEq,nBatch and standardize shapes
        note, these checks will certainly overlap with checks in qpsolvers and their calls
        can find ways to improve the last dimension = 1 for vectors condition ; may or may not need to rewrite other code
        another option is just to let other parts fail and just extract, without checking
        '''
        # assume solve_type is sparse and that Q, G are lists
        self.nBatch = len(Q)
        self.nIneq = list()
        self.nEq = list()
        self.dim = list()
        for i in range(self.nBatch):
            nIneq, dim = G[i].size()
            self.nIneq.append(nIneq)
            self.dim.append(dim)

            if A[i] is not None and b[i] is not None:
                nEq, _ = A[i].size()
                self.nEq.append(nEq)
            else:
                self.nEq.append(0)

        return Q,q,G,h,A,b


    def data_to_np(self,Q,q,G,h,A,b):
        '''
        Convert torch QP parameters to numpy and scipy variables, accounting for sparsity
        Stores variables in the class and optionally saves them
        '''
        self.Q_np = []
        self.G_np = []
        if A is not None:
            self.A_np = []
            for i in range(self.nBatch):
                self.Q_np += [sparse_helper.csc_torch_to_scipy(Q[i])]
                self.G_np += [sparse_helper.csc_torch_to_scipy(G[i])]
                self.A_np += [sparse_helper.csc_torch_to_scipy(A[i])]
        else:
            self.A_np = None
            for i in range(self.nBatch):
                self.Q_np += [sparse_helper.csc_torch_to_scipy(Q[i])]
                self.G_np += [sparse_helper.csc_torch_to_scipy(G[i])]

        self.q_np = []
        self.h_np = []
        for i in range(self.nBatch):
            self.q_np += [q[i].detach().numpy()]
            self.h_np += [h[i].detach().numpy()]

        self.b_np = []
        for i in range(self.nBatch):
            if b[i] is not None:
                self.b_np += [b[i].detach().numpy()]
            else:
                self.b_np += [None]

        return


    def reset_cache(self):
        self.x_star_np_cache = dict()


    def solve(self, t):
        '''
        Solve the QP using qpsolvers
        data_to_np must be called before
        '''

        # initialize
        self.x_star_np = [None for i in range(self.nBatch)]
        if self.nEq[0] > 0:
            self.mu_star_np = [None for i in range(self.nBatch)]
        self.nu_star_np = [None for i in range(self.nBatch)]

        mp.set_start_method('spawn', force=True)
        processes = list()

        x_vals = []
        y_vals = []
        z_vals = []
        V_basis_vals = []
        C_basis_vals = []

        start_time = time.time()
        qp_solve_args = []
        for i in range(self.nBatch):
            if self.warm_start_from_previous and (i,0) in self.x_star_np_cache.keys(): # false if nBatch > 1
                if (i,t) in self.x_star_np_cache.keys():
                    initvals = self.x_star_np_cache[(i,t)]
                else:
                    initvals = self.x_star_np_cache[(i,t-1)]
            else:
                initvals = None
            if self.nEq[i] > 0:
                kwargs_main = {
                    "problem": qpsolvers.Problem(P=self.Q_np[i], q=self.q_np[i], G=self.G_np[i],
                                                 h=self.h_np[i], A=self.A_np[i], b=self.b_np[i]),
                    "solver": self.qp_solver,
                    "verbose": self.verbose,
                    "initvals": initvals
                }
            else:
                kwargs_main = {
                    "problem": qpsolvers.Problem(P=self.Q_np[i], q=self.q_np[i], G=self.G_np[i],
                                                    h=self.h_np[i], A=None, b=None),
                    "solver": self.qp_solver,
                    "verbose": self.verbose,
                    "initvals": initvals
                }
            qp_solve_args.append((self.qp_solver_keywords, kwargs_main))
        results = self.pool.starmap(call_qp_solver_pool_target, qp_solve_args)
        for i in range(self.nBatch):
            x, y, z, V_basis, C_basis, found, extras = results[i]

            if x is None or not found:
                self._report_solver_failure(i, t, qp_solve_args[i][1], extras)
                raise RuntimeError(
                    f"QP solver failed for agent {i} at rollout step t={t}: "
                    f"{extras.get('status_name', 'unknown')}"
                )

            self.x_star_np[i] = x
            # implicitly assumes if nBatch > 1 then duals are available:
            self.mu_star_np[i] = y
            self.nu_star_np[i] = z
            self.x_star_np_cache[(i,t)] = (V_basis, C_basis)
        return None


    def _report_solver_failure(self, agent_idx, t, kwargs_main, extras):
        problem = kwargs_main["problem"]
        initvals = kwargs_main.get("initvals")
        status_name = extras.get("status_name", gurobi_status_name(extras.get("status", -1)))
        eq_viol, ineq_viol = qp_constraint_violations(problem)

        print("\n" + "=" * 72)
        print(f"Gurobi/QP solve failed")
        print(f"  agent index : {agent_idx}")
        print(f"  rollout step: t={t}")
        print(f"  status      : {status_name} (code {extras.get('status', 'n/a')})")
        print(f"  problem size: nVar={problem.P.shape[0]}, nEq={0 if problem.A is None else problem.A.shape[0]}, "
              f"nIneq={0 if problem.G is None else problem.G.shape[0]}")
        print(f"  violations at x=0: max|Ax-b|={eq_viol:.3e}, max(Gx-h)={ineq_viol:.3e}")
        if initvals is not None:
            print("  warm start  : basis from previous step was supplied")
        else:
            print("  warm start  : none")

        diag = diagnose_qp_failure(problem, initvals=initvals, verbose=True)
        print(f"  diagnostic re-solve status: {diag['status_name']} (code {diag['status_code']})")
        if diag["iis_constraints"]:
            print(f"  IIS constraints ({len(diag['iis_constraints'])} in conflict):")
            for name in diag["iis_constraints"][:20]:
                print(f"    - {name}")
            if len(diag["iis_constraints"]) > 20:
                print(f"    ... and {len(diag['iis_constraints']) - 20} more")
        if "iis_error" in diag:
            print(f"  IIS error: {diag['iis_error']}")
        print("=" * 72 + "\n")


    def setup_diff(self, csc_Q, q, csc_G, G, h, csc_A, b, scipy_Q, scipy_G, scipy_A, t):
        '''
        Form the reduced KKT and set-up derivatives through x_star
        Sets non_differentiable check
        '''

        x_star_list = []
        for i in range(self.nBatch):
            zhats = torch.tensor(self.x_star_np[i])
            slacks = h[i] - G[i] @ zhats
            zhats = zhats.unsqueeze(0)
            lams = torch.tensor(self.nu_star_np[i]).unsqueeze(0)
            nus = torch.tensor(self.mu_star_np[i]).unsqueeze(0)
            if i not in self.backward_cache.keys():
                # print(f"making new one for i={i}, t={t}")
                self.backward_cache[i] = True

                eps = 1e-7
                Q_diag = scipy_Q[i].diagonal()
                Q_inv_diag = 1.0 / (Q_diag + eps)
                Q_inv = sp.diags(Q_inv_diag)

                eps = 1e-8
                s_safe = torch.clamp(slacks.flatten(), min=eps)
                lam_safe = torch.clamp(lams.flatten(), min=eps)

                # 2. Calculate the diagonal terms (s / lambda)
                # Note: QPTH calculates lams/slacks for the RHS, but the MATRIX needs slacks/lams
                d_vals = s_safe / lam_safe

                # 3. Create the sparse diagonal matrix
                # This goes into your Schur Complement calculation
                sparse_D_s = sp.diags(d_vals.detach().numpy())

                # 3. Create the sparse diagonal matrix
                # This goes into your Schur Complement calculation

                # 2. Pre-calculate Q_inv * Transposes (used in assembly and recovery)
                # Doing this once saves time
                invQ_GT = Q_inv @ scipy_G[i].T
                invQ_AT = Q_inv @ scipy_A[i].T

                # 3. Form the Schur Complement Blocks
                # Block 1,1: G * Q_inv * G^T + D_s
                S_GG = scipy_G[i] @ invQ_GT

                new_time = time.time()
                # Block 1,2: G * Q_inv * A^T
                S_GA = scipy_G[i] @ invQ_AT

                # Block 2,2: A * Q_inv * A^T (Equalities usually have 0 on diagonal, so we just start with this)
                S_AA = scipy_A[i] @ invQ_AT

                self.backward_cache[(i,"S_GG")] = S_GG
                self.backward_cache[(i,"S_GA")] = S_GA
                self.backward_cache[(i,"S_AA")] = S_AA
                # Assemble the full Schur Matrix S
                S_blocks = [
                    [S_GG + sparse_D_s, S_GA],
                    [S_GA.T,            S_AA]
                ]
                S = sp.bmat(S_blocks, format='csc')
                factor = qdldl.Solver(S)
                self.backward_cache[(i,"factor")] = factor
                self.backward_cache[(i, "indices")] = S.indices.copy()
                self.backward_cache[(i, "indptr")] = S.indptr.copy()

            else:
                S_GG = self.backward_cache[(i,"S_GG")]
                S_GA = self.backward_cache[(i,"S_GA")]
                S_AA = self.backward_cache[(i,"S_AA")]
                eps = 1e-8
                s_safe = torch.clamp(slacks.flatten(), min=eps)
                lam_safe = torch.clamp(lams.flatten(), min=eps)

                # 2. Calculate the diagonal terms (s / lambda)
                # Note: QPTH calculates lams/slacks for the RHS, but the MATRIX needs slacks/lams
                d_vals = s_safe / lam_safe
                sparse_D_s = sp.diags(d_vals.detach().numpy())
                S_blocks = [
                    [S_GG + sparse_D_s, S_GA],
                    [S_GA.T,            S_AA]
                ]
                S = sp.bmat(S_blocks, format='csc')
                factor = self.backward_cache[(i,"factor")]
                factor.update(S)
                cached_indices = self.backward_cache[(i, "indices")]
                cached_indptr = self.backward_cache[(i, "indptr")]

                # Check if structure matches
                if not (np.array_equal(S.indices, cached_indices) and
                        np.array_equal(S.indptr, cached_indptr)):

                    # DEBUG: Print details to help you find the culprit
                    print(f"ERROR: Sparsity pattern changed at batch index {i}!")
                    print(f"Old nnz: {len(cached_indices)}, New nnz: {len(S.indices)}")

            diff_params = {
                "x_star": torch.tensor(self.x_star_np[i]),
                "eq_dual_star": torch.tensor(self.mu_star_np[i]),
                "ineq_dual_star": torch.tensor(self.nu_star_np[i]),
                "scipy_Q": scipy_Q[i],
                "scipy_G": scipy_G[i],
                "scipy_A": scipy_A[i],
                "nonsparse_G": G[i],
                "dim": self.dim[i],
                "nEq": self.nEq[i],
                "nIneq": self.nIneq[i],
                "agent": i,
                "t": t,
                "warm_start_from_previous": self.warm_start_from_previous,
                "other_factor": self.backward_cache[(i,"factor")]
            }

            # torch.index_select and .values() only work with COO sparse matrices
            csc_Q_i = csc_Q[i].to_sparse_coo().coalesce().values()
            q_i = q[i]
            csc_G_i = csc_G[i].to_sparse_coo().coalesce().values()

            if self.nEq[i] > 0:
                csc_A_i = csc_A[i].to_sparse_coo().coalesce().values()
                b_i = b[i]
            else:
                csc_A_i = None
                b_i = None

            h_i = h[i]
            x_star = self.differentiate_QP(csc_Q_i, q_i, csc_G_i, h_i, csc_A_i, b_i, diff_params)

            x_star_list += [x_star]

        return x_star_list


def call_qp_solver_pool_target(qp_solver_keywords, kwargs_main):
    """
    Worker function for the multiprocessing pool.
    Returns results (numpy arrays/lists) which the pool gathers.
    """
    if kwargs_main.get("solver") == "gurobi":
        solution, bases = gurobi_ws_solve_problem(
            problem=kwargs_main["problem"],
            initvals=kwargs_main.get("initvals"),
            verbose=kwargs_main.get("verbose", False),
            **qp_solver_keywords,
        )
    else:
        full_kwargs = dict(**kwargs_main, **qp_solver_keywords)
        solution = qpsolvers.solve_problem(**full_kwargs)
        bases = (None, None)

    extras = dict(getattr(solution, "extras", {}) or {})
    if "status" in extras and "status_name" not in extras:
        extras["status_name"] = gurobi_status_name(extras["status"])
    return solution.x, solution.y, solution.z, bases[0], bases[1], solution.found, extras


class differentiate_QP(torch.autograd.Function):
    '''
    Differentiate the QP explicitly using the reduced KKT formed with the active constraints
    '''

    @staticmethod
    def forward(ctx, Q, q, G, h, A, b, params):
        '''
        Just return known solution as differentiable parameter. Store data for backwards.
        '''
        zhats = params["x_star"]
        ctx.zhats = zhats
        ctx.lams = params["ineq_dual_star"]
        ctx.nus = params["eq_dual_star"]
        ctx.scipy_Q = params["scipy_Q"]
        ctx.scipy_G = params["scipy_G"]
        ctx.scipy_A = params["scipy_A"]
        nonsparse_G = params["nonsparse_G"]
        ctx.slacks = h - nonsparse_G @ zhats
        ctx.other_factor = params["other_factor"]
        return zhats

    @staticmethod
    def backward(ctx, dl_dzhat):
        backward_start_time = time.time()
        zhats = ctx.zhats.unsqueeze(0)
        lams = ctx.lams.unsqueeze(0)
        nus = ctx.nus.unsqueeze(0)
        scipy_Q = ctx.scipy_Q
        scipy_G = ctx.scipy_G
        scipy_A = ctx.scipy_A
        slacks = ctx.slacks
        other_factor = ctx.other_factor
        eps = 1e-8
        s_safe = torch.clamp(slacks.flatten(), min=eps)
        lam_safe = torch.clamp(lams.flatten(), min=eps)

        # 2. Calculate the diagonal terms (s / lambda)
        # Note: QPTH calculates lams/slacks for the RHS, but the MATRIX needs slacks/lams
        d_vals = s_safe / lam_safe
        # 3. Create the sparse diagonal matrix
        # This goes into your Schur Complement calculation
        sparse_D_s = sp.diags(d_vals.numpy())

        # 1. Invert Q (trivial since diagonal)
        # Add eps to ensure stability
        eps = 1e-7
        Q_diag = scipy_Q.diagonal()
        Q_inv_diag = 1.0 / (Q_diag + eps)
        Q_inv = sp.diags(Q_inv_diag)

        # 2. Pre-calculate Q_inv * Transposes (used in assembly and recovery)
        # Doing this once saves time
        invQ_GT = Q_inv @ scipy_G.T
        invQ_AT = Q_inv @ scipy_A.T

        # 3. Form the Schur Complement Blocks
        # Block 1,1: G * Q_inv * G^T + D_s
        S_GG = scipy_G @ invQ_GT + sparse_D_s

        # Block 1,2: G * Q_inv * A^T
        S_GA = scipy_G @ invQ_AT

        # Block 2,2: A * Q_inv * A^T (Equalities usually have 0 on diagonal, so we just start with this)
        S_AA = scipy_A @ invQ_AT
        # Assemble the full Schur Matrix S
        S_blocks = [
            [S_GG, S_GA],
            [S_GA.T, S_AA]
        ]
        S = sp.bmat(S_blocks, format='csc')

        # 4. Form the RHS for the Schur system
        # We eliminate dx, so the new RHS depends on rhs_x
        # rhs_schur_1 = rhs_lam - G * Q_inv * rhs_x
        # rhs_schur_2 = rhs_nu  - A * Q_inv * rhs_x
        invQ_rhs_x = Q_inv @ dl_dzhat # Vector op
        rhs_schur_lam = -scipy_G @ invQ_rhs_x
        rhs_schur_nu  = -scipy_A @ invQ_rhs_x
        rhs_schur = np.concatenate([rhs_schur_lam, rhs_schur_nu])

        # 5. Solve for Dual Variables (dlam, dnu)
        # S is Symmetric. If D_s and Q are positive, S is Positive Definite.
        # You can use splu (LU) or try minres if it's large.
        factor = qdldl.Solver(S)
        d_dual = factor.solve(rhs_schur)
        d_dual2 = other_factor.solve(rhs_schur)
        n_ineq = scipy_G.shape[0]
        dlam = torch.tensor(d_dual[:n_ineq])
        dlam = dlam.unsqueeze(0)
        dnu  = torch.tensor(d_dual[n_ineq:])
        dnu = dnu.unsqueeze(0)

        # For our project, we only need dhs and dbs, calculating other gradients is expensive
        dQs = None
        dps = None
        dGs = None
        dhs = -dlam.mean(0)
        dAs = None
        dbs = -dnu.mean(0)
        grads = (dQs, dps, dGs, dhs, dAs, dbs, None)
        return grads


def build_settings(check_PSD=False,time=False,solve_type="dense",dual_available=None,normalize_constraints=False,empty_batch=True,warm_start_from_previous=False,omp_parallel=False,n_cpu=None, # general arguments
                   refine_active=False, # active arguments
                   qp_solver=None,verbose=False,qp_solver_keywords=None,eps_abs=1e-6,eps_rel=0, # qp solver arguments ... default to solver preference
                   lin_solver=None): # linear solver arguments

    available_qp_solvers = qpsolvers.available_solvers

    if verbose:
        print("Available QP solvers:\n" + str(qpsolvers.available_solvers))
        print("Available linear solvers:\n" + str(lin_solvers.get_dense_solvers() + lin_solvers.get_sparse_solvers() + qpsolvers.available_solvers))

    if solve_type == "dense":
        if qp_solver is None: # get first available qp solver
            if "cvxopt" in qpsolvers.dense_solvers:
                qp_solver = "cvxopt"
            else:
                qp_solver = qpsolvers.dense_solvers[0]
        if lin_solver is None:
            lin_solver = "scipy LU"
        # assert(qp_solver in qpsolvers.dense_solvers) # qpsolvers labeling of dense/sparse is not a strict classification
        assert(lin_solver in lin_solvers.get_dense_solvers())
    elif solve_type == "sparse":
        if qp_solver is None: # get first available qp solver
            if "gurobi" in qpsolvers.sparse_solvers:
                qp_solver = "gurobi"
            else:
                qp_solver = qpsolvers.sparse_solvers[0]
        if lin_solver is None:
            lin_solver = "scipy SPLU"
        # assert(qp_solver in qpsolvers.sparse_solvers)
        assert(lin_solver in lin_solvers.get_sparse_solvers())

    if dual_available is None:
        if qp_solver in ["clarabel","cvxopt","daqp","ecos","gurobi","highs","hpipm","mosek","osqp","piqp","proxqp","qpalm","qpoases","qpswift","quadprog","scs"]:
            if verbose:
                print("Solver is in base qpsolvers, duals are available.")
            dual_available = True
        else:
            raise("Solver is not in base qpsolvers, user must specify (T/F) if duals are available")


    if omp_parallel and n_cpu is None:
        n_cpu = os.cpu_count()
        print("No CPU count given, using all available: " + str(n_cpu))

    if qp_solver_keywords is None:
        qp_solver_keywords = {}
        qp_solver_keywords = set_solver_tolerance(qp_solver_keywords,qp_solver,eps_abs,eps_rel)

    else:
        print("eps_abs,eps_rel ignored if custom keywords given")

    if verbose:
        print("qp_solver: " + str(qp_solver))
        print("lin_solver: " + str(lin_solver))
        print("qp_solver_keywords: " + str(qp_solver_keywords))

    assert(isinstance(qp_solver_keywords,dict))

    settings = {
        "verbose" : verbose,
        "check_PSD" : check_PSD,
        "time" : time,
        "available_qp_solvers" : available_qp_solvers,
        "solve_type" : solve_type,
        "qp_solver": qp_solver,
        "dual_available": dual_available,
        "normalize_constraints": normalize_constraints,
        "empty_batch": empty_batch,
        "warm_start_from_previous": warm_start_from_previous,
        "qp_solver_keywords" : qp_solver_keywords,
        "lin_solver": lin_solver,
        "refine_active": refine_active,
        "omp_parallel" : omp_parallel,
        "n_cpu" : n_cpu
    }

    return settings