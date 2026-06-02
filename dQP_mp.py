import numpy as np
import scipy as sp
import multiprocessing
import torch.multiprocessing as mp

import torch
from torch import nn

# qp solver
import qpsolvers
from dQP_mp_helpers.gurobi_ws import gurobi_solve_problem as _gurobi_ws  # move to src.gurobi_ws once src/ is set up
import importlib
importlib.import_module("qpsolvers._internals").solve_function["gurobi"] = _gurobi_ws
from scipy.sparse import csc_matrix, bmat

import sys
import os
import warnings

# sys.path.append('../')

src_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(src_dir)
sys.path.append(parent_dir)

from dQP_mp_helpers.set_solver_tolerance import set_solver_tolerance

import dQP_mp_helpers.sparse_helper as sparse_helper
import dQP_mp_helpers.lin_solvers as lin_solvers
import time

# CPU parallelism
from joblib import Parallel, delayed


# Note, internal conventions for QP variables and parameters are different from the paper
# (P,q,C,d,A,b) = (Q,q,G,h,A,b)
# (z,\lambda,\mu) = (x,\mu,\nu)


class dQP_layer(nn.Module):
    ''' solves and differentiates
    x^* = argmin_x 1/2 x^T Q x + q^T x
             s.t.  G x <= h
                   A x  = b
    (including dual variables mu^*,nu^*)
    Q dim x dim ; q dim x 1 ; G nIneq x dim ; h nIneq x 1 ; A nEq x dim ; b nEq x 1
    if batched, first dimension is nBatch. Serial via for loops unless multiple CPUs/OMP is turned on

    input:
        torch parameters Q,q,G,h,A,b
        if sparse, need to be CSC type
    output:
        x_star,mu_star,nu_star,time

    see README for information about options

    internal variables
        -dim        : # dim
        -nEq        : # equalities
        -nIneq      : # inequalities
        -nBatch     : batch size, interpreted from input
        -active     : active set
        -nActive    : # active
        -A copy of Q,q,G,h,A,b,x_star,mu_star,nu_star as numpy variables denoted _np
        -r_pri_np    : primal residual h - Gx^*
        -A_reduce_np : equality/active inequality constraints
        -nEq_reduce  : # equality/active inequality constraints
        -KKT_A_np    : reduced KKT
        -KKT_A_np_factors   : factorizations of reduced KKT
        -KKT_b_np           : RHS of reduced KKT
        -non_differentiable : determines whether least-squares is used in differentiate_QP

        -differentiate_QP  : differentiates solution
        -sparse_row_norm   : differentiable matrix vecnorm
        -sparse_row_normalize :  differentiable matrix normalization by row

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

        self.nActive = None
        self.active = None
        self.nEq_reduce = None

        self.differentiate_QP = differentiate_QP.apply # set-up differentiation through active constraints

        # TODO : note that if these are not reset and A = None changes for different input, may cause problem
        self.N_A = None # torch variables
        self.N_G = None

        # initialize numpy variables which are carried implicitly ; torch version is not stored
        self.x_star_np = None
        self.mu_star_np = None
        self.nu_star_np = None
        self.nu_star_with_inactive_np = None
        self.x_star_np_cache = dict()

        self.Q_np = None
        self.q_np = None
        self.G_np = None
        self.h_np = None
        self.A_np = None
        self.b_np = None

        self.r_pri_np = None
        self.A_reduce_np = None
        self.KKT_A_np = None
        self.KKT_A_np_factors = None
        self.KKT_b_np = None

        self.non_differentiable = False # flag for non-differentiable weakly active constraints ; perform LSQ if True

        # custom differentiable sparse normalization functions
        self.sparse_row_norm = sparse_helper.sparse_row_norm.apply
        self.sparse_row_normalize = sparse_helper.sparse_row_normalize.apply

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

    def forward(self, Q, q, G, h, t, A=None, b=None):
        # check shapes, extract dim,nIneq,nEq,nBatch
        self.reset_parameters()
        Q,q,G,h,A,b = self.get_shapes(Q, q, G, h, A, b)

        if not self.training: # check if evaluation mode is turned on
            # warm-starts with previous problem's x_star stored in the class layer
            if self.warm_start_from_previous:  # false if nBatch > 1
                initvals = self.x_star_np_cache[t]
                # initvals = self.x_star_np
            else:
                initvals = None
            kwargs_fixed = dict(**{"solver": self.qp_solver, "verbose": self.verbose, "initvals": initvals},
                                **self.qp_solver_keywords)
            # do not normalize, this may be carried out in the solver if needed
            self.data_to_np(Q,q,G,h,A,b)
            x = call_single_qpsolvers(self.Q_np,self.q_np,self.G_np,self.h_np,self.A_np,self.b_np,kwargs_fixed)
            return torch.from_numpy(x.x),torch.from_numpy(x.y),torch.from_numpy(x.z), None, None # should time
        else:
            if self.time:
                if self.normalize_constraints:
                    # normalize
                    start_normalize = time.time()
                    G,h,A,b = self.normalize(G,h,A,b)
                    normalize_time = time.time() - start_normalize

                    print("### Time normalize: " + str(normalize_time))

                # convert and store in numpy as dense or sparse
                start_convert = time.time()
                self.data_to_np(Q,q,G,h,A,b)
                convert_time = time.time() - start_convert

                print("### Time conversion: " + str(convert_time))

                # solve and time QP ; solution is stored
                start_solve = time.time()
                self.solve(t)
                solve_time = time.time() - start_solve

                self.nu_star_with_inactive_np = self.nu_star_np.copy()

                print("### Time QP Solve: " + str(solve_time))

                # differentiate and time
                start_setup_diff = time.time()
                x_star, mu_star, nu_star = self.setup_diff(Q=Q,q=q,G=G,h=h,A=A,b=b)
                setup_diff_time = time.time() - start_setup_diff

                total_forward_time = normalize_time + convert_time + solve_time + setup_diff_time

                print("### Time Setup Differentiation Time: " + str(setup_diff_time))
            else:
                if self.normalize_constraints:
                    G, h, A, b = self.normalize(G, h, A, b)
                self.data_to_np(Q, q, G, h, A, b)
                start_time = time.time()
                self.solve(t)
                # print(f"solve takes time {time.time() - start_time}")
                self.nu_star_with_inactive_np = self.nu_star_np.copy()
                # ckpt["comb"] = self.get_x_mu_nu_star(t)
                x_star, mu_star, nu_star = self.setup_diff(t, Q=Q, q=q, G=G, h=h, A=A, b=b)

                solve_time = None
                total_forward_time = None

            nu_star_with_inactive = []
            for i in range(self.nBatch):
                if self.nActive[i] > 0:
                    new_nu_star = torch.tensor(self.nu_star_with_inactive_np[i])
                    new_nu_star[self.active[i]] = nu_star[i]
                    nu_star_with_inactive.append(new_nu_star)
                else:
                    nu_star_with_inactive.append(None)

            if self.normalize_constraints:
                if self.N_A is not None:
                        mu_star = torch.div(mu_star, self.N_A)
                nu_star_with_inactive = torch.div(nu_star_with_inactive,self.N_G)

            # print("this return is called")
            # print(x_star.shape)
            # print(mu_star.shape)
            # print(nu_star_with_inactive.shape)
            
            return x_star, mu_star, nu_star_with_inactive, solve_time, total_forward_time

    def clear_cache(self):
        """
        Clears cache of warm-start points.
        """
        self.x_star_np_cache = dict()
        self.backward_cache = dict()

    def reset_parameters(self):
        '''
        may not be necessary ; just a backup measure in case previously stored conditions are kept in error or not
        over-writtten
        currently think it might matter if nEq != 0 and then nEq = 0

        also ; in some cases, may want to store and re-use these
        '''
        self.dim,self.nIneq,self.nEq,self.nBatch = None, None, None, None
        self.nActive, self.active,self.nEq_reduce = None, None, None
        self.N_A, self.N_G = None, None
        self.x_star_np, self.mu_star_np, self.nu_star_np = None, None, None
        self.Q_np, self.q_np, self.G_np, self.h_np, self.A_np, self.b_np = None, None, None, None, None, None
        self.r_pri_np, self.A_reduce_np, self.KKT_A_np, self.KKT_A_np_factors, self.KKT_b_np = None, None, None, None, None
        self.non_differentiable = False

    # batched!
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
    
    # batched!
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

    # batched!
    def reset_cache(self):
        self.x_star_np_cache = dict()


    # batched!
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
                # initvals = self.x_star_np
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
            qp_solve_args.append((kwargs_main, self.qp_solver_keywords))
        results = self.pool.starmap(call_qp_solver_pool_target, qp_solve_args)

        for i in range(self.nBatch):
            x, y, z, V_basis, C_basis = results[i]

            if x is None:
                print("Solver failed to return a solution. Re-solving with verbose and exiting.")
                raise Exception("Exiting")

            self.x_star_np[i] = x
            # implicitly assumes if nBatch > 1 then duals are available:
            if self.nEq[i] > 0:
                self.mu_star_np[i] = y # TODO : make duals optional by setting = None throughout
            self.nu_star_np[i] = z
            self.x_star_np_cache[(i,t)] = (V_basis, C_basis)
        return None


    def setup_diff(self,t,Q,q,G,h,A,b):
        '''
        Form the reduced KKT and set-up derivatives through x_star
        Sets non_differentiable check
        '''

        x_mu_nu_star = self.get_x_mu_nu_star(t)

        x_star = []
        mu_star = []
        nu_star = []
        for i in range(self.nBatch):
            non_differentiable = (self.nEq_reduce[i] > self.dim[i] or  # quick check for linear dependence
                                        np.any(self.nu_star_np[i] < 1e-8))  # weakly active (note nu_star_np by now only contains active nu)

            diff_params = {
                "x_mu_nu_star": x_mu_nu_star[i],  # differentiable x_star
                "KKT_A_np": self.KKT_A_np[i],
                "KKT_A_np_factors": None,
                "solve_type": self.solve_type,
                "qp_solver": self.qp_solver,
                "lin_solver": self.lin_solver,
                "available_qp_solvers": self.available_qp_solvers,  # to check if lin_solver is a QP method
                "dim": self.dim[i],
                "nEq": self.nEq[i],
                "nIneq": self.nIneq[i],
                "nActive": self.nActive[i],
                "non_differentiable": non_differentiable,
                # internally converts csc --> coo to get indices
                "Q_pattern": None if (Q[i].layout is torch.strided) else self.Q_np[i].nonzero(),
                "G_pattern": None if (G[i].layout is torch.strided) else (self.G_np[i][self.active[i], :]).nonzero(),
                "A_pattern": None if ((A is None) or (A[i].layout is torch.strided)) else self.A_np[i].nonzero(),
                "agent": i,
                "t": t,
                "warm_start_from_previous": self.warm_start_from_previous
            }

            # torch.index_select and .values() only work with COO sparse matrices
            Q_i = Q[i].to_sparse_coo().coalesce().values()

            G_i = G[i].to_sparse_coo().coalesce()
            G_ind = G_i._indices()
            G_i = G_i.values()
            G_i = G_i[self.active[i][G_ind[0, :].numpy()]]

            if self.nEq[i] > 0:
                A_i = A[i].to_sparse_coo().coalesce().values()
                b_i = b[i]
            else:
                A_i = None
                b_i = None

            h_i = h[i][self.active[i]]

            x_mu_nu_star[i] = self.differentiate_QP(Q_i, q[i], G_i, h_i, A_i, b_i, diff_params, self.backward_cache)

            x_star += [x_mu_nu_star[i][0:self.dim[i]]]
            mu_star += [x_mu_nu_star[i][self.dim[i]:(self.nEq[i] + self.dim[i])]]
            nu_star += [x_mu_nu_star[i][(self.nEq[i] + self.dim[i]):]]

        return x_star,mu_star,nu_star


    # batched!
    def get_x_mu_nu_star(self, t):
        '''
        Constructs torch parameter x_mu_nu_star
        Solve for or retrieve mu_star via the external solver or fully/partially solving the reduced KKT
        Optional refinement of the active set
        '''

        # initial estimated active set before any changes
        self.get_active()
        # if self.verbose:
        # print("# Active: " + str(self.nActive), f" x has dim {self.dim}, ineq {self.nIneq}, eq {self.nEq}")

        # assume dual available
        # note, does not overwrite any of the solution despite hard-threshold projection onto constraints

        if self.refine_active:
            grad_f = (self.Q_np.dot(self.x_star_np) + self.q_np)
            # don't use mu_nu from refinement, just use to determine tolerances on active
            self.refine(grad_f)

        # mu_star_np, nu_star_np already set without solving ; forget inactive duals (will be 0 later)
 
        nu_star_temp = self.nu_star_np
        self.nu_star_np = []
        
        for i in range(self.nBatch):
            self.nu_star_np += [nu_star_temp[i][self.active[i]]]

        self.get_reduced_KKT()

        x_mu_nu_star = []
        for i in range(self.nBatch):
            x_mu_nu_star_temp = torch.tensor(self.x_star_np[i], dtype=torch.float64)
            if self.mu_star_np is not None:
                mu_star = torch.tensor(self.mu_star_np[i], dtype=torch.float64)
                x_mu_nu_star_temp = torch.concatenate((x_mu_nu_star_temp, mu_star))

            if self.nActive[i] > 0:
                nu_star = torch.tensor(self.nu_star_np[i], dtype=torch.float64)
                x_mu_nu_star_temp = torch.concatenate((x_mu_nu_star_temp, nu_star))
            x_mu_nu_star += [x_mu_nu_star_temp]

        return x_mu_nu_star


    # batched!
    def get_active(self):
        '''
        Determines the active constraints at the solution x_star_np
        '''

        self.get_r_pri()
        # one-sided check on residual
        self.nActive = [None for i in range(self.nBatch)]
        self.active = [None for i in range(self.nBatch)]
        self.nEq_reduce = [None for i in range(self.nBatch)]
        for i in range(self.nBatch):
            active = self.r_pri_np[i] < self.eps_active
            self.nActive[i] = np.sum(active, axis=-1)
            self.nEq_reduce[i] = self.nActive[i] + self.nEq[i]
            self.active[i] = active

        return None

    # batched!
    def get_r_pri(self):
        '''
        Determines the primal residual using stored x_star_np, G_np, h_np
        '''
        self.r_pri_np = [None for i in range(self.nBatch)]
        for i in range(self.nBatch):
            x,G,h = self.x_star_np[i], self.G_np[i], self.h_np[i]
            h_approx = G @ x
            self.r_pri_np[i] = h - h_approx
        return None


    # batched!
    def solve_KKT_for_dual(self):
        '''
        Get the current reduced KKT from stored active set and solve it for x_star, and active nu_star
        '''

        if self.time:
            start_reduced_KKT = time.time()

        if self.lin_solver in self.available_qp_solvers:
            QP_form = [self.Q_np, self.q_np, self.KKT_A_np[self.dim:, 0:self.dim], self.KKT_b_np[self.dim:]]
        else:
            QP_form = None

        # solve and store factorization for backwards if available TODO: Make sure that if KKT changes that A_np_factors changes too
        if self.solve_type == "dense":
            x_mu_reduced_np, self.KKT_A_np_factors = lin_solvers.dense_solve(self.KKT_A_np, self.KKT_b_np,
                                                                          linear_solver=self.lin_solver,
                                                                          QP_form=QP_form, x_warmstart=None)
        elif self.solve_type == "sparse":
            x_mu_reduced_np, self.KKT_A_np_factors = lin_solvers.sparse_solve(self.KKT_A_np, self.KKT_b_np,
                                                                           linear_solver=self.lin_solver,
                                                                           QP_form=QP_form, x_warmstart=None)

        if self.time:
            reduced_KKT_time = time.time() - start_reduced_KKT
            print("### Time 1st KKT solve: " + str(reduced_KKT_time))

        return x_mu_reduced_np


    # ignore since we don't refine
    def refine(self,grad_f):
        '''
        Refines the active set by simultaneously minimizing the primal and dual residuals w.r.t. different active sets;
        The solution is fixed
        Sets active, nActive, nEqreduce, and the reduced KKT
        Returns active mu,nu
        '''

        active = self.active
        self.get_r_pri()
        r_pri = self.r_pri_np

        # get ordering of numerically inactive constraints
        i_ord = np.argsort(r_pri, axis=0) # note: if using duals to determine active set by something like nu > eps, the selected duals may not be the least ordered primal
        iterate = True
        prev_tot_viol = 1e10
        prev_mu_nu_star = 0
        iter = 0

        while iterate:
            print("iter refinement: " + str(iter))

            # set-up reduced equality constraints
            if self.A_np is None:
                G_aset = self.G_np[active, :]
            else:
                if self.solve_type == "dense":
                    G_aset = np.vstack((self.A_np, self.G_np[active, :]))
                elif self.solve_type == "sparse":
                    G_aset = sp.sparse.vstack((self.A_np, self.G_np[active, :]))

            # solve for the dual variable
            if self.solve_type == "dense":
                mu_nu_star, r_dual = lin_solvers.dense_LSQ(G_aset.T, -grad_f, lsq_solver=self.qp_solver, eps_abs=self.eps_abs, eps_rel=self.eps_rel)
            elif self.solve_type == "sparse":
                mu_nu_star, r_dual = lin_solvers.sparse_LSQ(G_aset.T, -grad_f, lsq_solver=self.qp_solver, eps_abs=self.eps_abs, eps_rel=self.eps_rel)

            # check violation of KKT and update active set
            tot_viol = np.sqrt(np.linalg.norm(r_pri[i_ord[0:self.nActive]]) ** 2 + r_dual ** 2)
            print("\| r \|_2: " + str(tot_viol))

            if tot_viol < self.eps_active:
                iterate = False
            elif tot_viol < prev_tot_viol:
                print("Change constraints.")
                self.KKT_A_np_factors = None  # KKT has changed, release factorization
                prev_tot_viol = tot_viol
                prev_mu_nu_star = mu_nu_star
                if self.nActive < self.nIneq:
                    active[i_ord[self.nActive]] = True
                    self.nActive += 1
                else:
                    break
            else:
                print("Keep previous constraints.")
                self.nActive = self.nActive - 1  # revert to previous activity
                active[i_ord[self.nActive]] = False

                mu_nu_star = prev_mu_nu_star  # keep previous mu_star
                iterate = False

            iter += 1

        if self.verbose:
            print("# Active Refined: " + str(self.nActive))
        assert (self.nActive == np.sum(self.active))

        # update active set
        self.active = active
        self.nEq_reduce = self.nActive + self.nEq
        self.get_reduced_KKT()

        return mu_nu_star


    # batched!
    def get_reduced_KKT(self):
        '''
        Form the reduced KKT only in np form given the active constraints
        '''

        self.A_reduce_np = []
        self.KKT_A_np = []
        self.KKT_b_np = []

        for i in range(self.nBatch):
            # form the effective equality constraints

            if self.nEq_reduce[i] == 0:
                self.A_reduce_np += [None]
                self.KKT_A_np += [self.Q_np[i]]
                self.KKT_b_np += [-self.q_np[i]]
            else:
                if self.nEq[i] == 0:
                    # self.A_reduce_np += [self.G_np[i][self.active[i]] if not (self.nIneq == 1 and self.nActive[i] == 1) else self.G_np[i][self.active[i]]]
                    self.A_reduce_np += [self.G_np[i][self.active[i]]]

                else:
                    if self.solve_type == "dense":
                        # self.A_reduce_np += [np.vstack((self.A_np[i], self.G_np[i][self.active[i]] if not (self.nIneq == 1 and self.nActive[i] == 1) else self.G_np[i][self.active[i]]))]
                        self.A_reduce_np += [np.vstack((self.A_np[i], self.G_np[i][self.active[i]]))]

                    elif self.solve_type == "sparse":
                        # self.A_reduce_np += [sp.sparse.vstack((self.A_np[i], self.G_np[i][self.active[i]] if not (self.nIneq == 1 and self.nActive[i] == 1) else self.G_np[i][self.active[i]]), format="csc")]
                        self.A_reduce_np += [sp.sparse.vstack((self.A_np[i], self.G_np[i][self.active[i]]), format="csc")]

                # get np version for calculations, including sparse if necessary
                if self.solve_type == "dense":
                    self.KKT_A_np += [np.bmat([[self.Q_np[i], np.transpose(self.A_reduce_np[i])], [self.A_reduce_np[i], np.zeros((self.nEq_reduce[i],self.nEq_reduce[i]))]])]
                elif self.solve_type == "sparse":
                    self.KKT_A_np += [bmat([[self.Q_np[i], np.transpose(self.A_reduce_np[i])], [self.A_reduce_np[i], None]], format="csc")]
                if self.nEq == 0:
                    self.KKT_b_np += [np.concatenate((-self.q_np[i],self.h_np[i][self.active[i]]))]
                else:
                    self.KKT_b_np += [np.concatenate((-self.q_np[i],self.b_np[i],self.h_np[i][self.active[i]]))]

        return None

def call_qp_solver_mp(self, x_tensor, y_tensor, z_tensor, V_basis_tensor, C_basis_tensor, kwargs_main, qp_solver_keywords):
    """
    Worker function that writes its result directly to a shared memory tensor.
    """
    solution, bases = qpsolvers.solve_problem(**dict(**kwargs_main, **self.qp_solver_keywords))
    x_tensor.copy_(torch.tensor(solution.x))
    y_tensor.copy_(torch.tensor(solution.y))
    z_tensor.copy_(torch.tensor(solution.z))
    V_basis_tensor.copy_(torch.tensor(bases[0], dtype=torch.int32))
    C_basis_tensor.copy_(torch.tensor(bases[1], dtype=torch.int32))

    return

# Make this a standalone function or a static method
def call_qp_solver_pool_target(qp_solver_keywords, kwargs_main):
    """
    Worker function for the multiprocessing pool.
    Returns results (numpy arrays/lists) which the pool gathers.
    """
    # Note: We combine qp_solver_keywords and kwargs_main here
    full_kwargs = dict(**kwargs_main, **qp_solver_keywords)
    
    solution, bases = qpsolvers.solve_problem(**full_kwargs)
    
    # Return results as standard Python objects (numpy arrays/lists)
    # The pool handles pickling these back to the main process
    return solution.x, solution.y, solution.z, bases[0], bases[1]


def call_single_qpsolvers(Q,q,G,h,A,b,kwargs):
    kwargs_problem = {
        "problem": qpsolvers.Problem(P=Q, q=q, G=G,h=h, A=A, b=b)
    }

    solution, bases = qpsolvers.solve_problem(**dict(**kwargs_problem, **kwargs))

    if solution.x is None:
        print("Solver failed to return a solution. Re-solving with verbose and exiting.")
        kwargs["verbose"] = True
        qpsolvers.solve_problem(**dict(**kwargs_problem, **kwargs))
        raise Exception("Exiting")

    return solution

class differentiate_QP(torch.autograd.Function):
    '''
    Differentiate the QP explicitly using the reduced KKT formed with the active constraints
    '''

    @staticmethod
    def forward(ctx, Q, q, G, h, A, b, params, backward_cache):
        '''
        Just return known solution as differentiable parameter. Store data for backwards.
        '''
        ctx.KKT_A = params["KKT_A_np"]
        ctx.pre_factorization = params["KKT_A_np_factors"]
        ctx.solve_type = params["solve_type"]
        ctx.qp_solver = params["qp_solver"]
        ctx.lin_solver = params["lin_solver"]
        ctx.available_qp_solvers = params["available_qp_solvers"]
        ctx.dim = params["dim"]
        ctx.nEq = params["nEq"]
        ctx.nIneq = params["nIneq"]
        ctx.nActive = params["nActive"]
        ctx.non_diff = params["non_differentiable"] # have to not use torch keyword --> non_diff
        ctx.x_mu_nu_star = params["x_mu_nu_star"].detach().numpy()
        ctx.agent = params["agent"]
        ctx.t = params["t"]
        ctx.warm_start_from_previous = params["warm_start_from_previous"]
        ctx.backward_cache = backward_cache
        specific_requires_grad = [] # handle None case
        for var in [Q,q,G,h,A,b]:
            if var is None:
                specific_requires_grad += [False]
            else:
                specific_requires_grad += [var.requires_grad]

        ctx.specific_requires_grad = specific_requires_grad
        ctx.patterns = [params["Q_pattern"],params["G_pattern"],params["A_pattern"]]

        return params["x_mu_nu_star"] # just return x_star

    @staticmethod
    def backward(ctx,grad_output):
        '''
        Computes dl/dQ, dl/dq, dl/dG, dl/dh, dl/dA, dl/db
        '''

        # For geometry scaling, profile the backward through QP alone
        # t = time.time() # TODO: note that this should be removed in final product
        # For geometry scaling, profile the backward through QP alone

        KKT_A = ctx.KKT_A
        KKT_b = grad_output.numpy()

        t = ctx.t
        agent = ctx.agent
        warm_start_from_previous = ctx.warm_start_from_previous
        backward_cache = ctx.backward_cache

        initvals = None
        if warm_start_from_previous and (agent,0) in backward_cache.keys(): # false if nBatch > 1
            # initvals = self.x_star_np
            if (agent,t) in backward_cache.keys():
                initvals = backward_cache[(agent,t)]
            else:
                initvals = backward_cache[(agent,t-1)]

        # grad_b,_ = lin_solvers.sparse_LSQ(KKT_A, KKT_b, lsq_solver="scipy", eps_abs=1e-5, eps_rel=1e-5, x0=initvals)
        
        if ctx.lin_solver in ctx.available_qp_solvers:
            QP_form = [KKT_A[0:ctx.dim,0:ctx.dim],-KKT_b[0:ctx.dim],KKT_A[ctx.dim:,0:ctx.dim],KKT_b[ctx.dim:]]
        else:
            QP_form = None
        # print(f"new backward pass, KKT_A has shape {KKT_A.shape}")
        try:
            if ctx.non_diff:
                raise Exception("Weakly active anticipated. Back-propagate using least-squares.")
            else:
                with warnings.catch_warnings():
                    # print("attempted solve")
                    warnings.filterwarnings('error')
                    if ctx.solve_type == "dense":
                        grad_b,_ = lin_solvers.dense_solve(KKT_A, KKT_b,linear_solver=ctx.lin_solver,pre_factorization=ctx.pre_factorization,QP_form=QP_form)
                    elif ctx.solve_type == "sparse":
                        grad_b,_ = lin_solvers.sparse_solve(KKT_A, KKT_b,linear_solver=ctx.lin_solver,pre_factorization=ctx.pre_factorization,QP_form=QP_form,x_warmstart=initvals)
                        # print("actual sparse solve worked", ctx.t)
            assert(grad_b is not None)

        # if linear solve fails, do least-squares. This occurs when there are weakly active constraints.
        except Exception as e:
            print('Linear solve failed. Back-propagate using least-squares: ', repr(e))

            # print("Use QP solver to solve least-squares. Users can change solver and tolerances on L1135.")
            if ctx.solve_type == "dense":
                # grad_b,_ = lin_solvers.dense_LSQ(KKT_A, KKT_b, lsq_solver=ctx.qp_solver, eps_abs=1e-5, eps_rel=1e-5) # TODO: PIQP failed on sudoku... potentially because output empty arrays or vectors without dim 1 at end
                grad_b,_ = lin_solvers.dense_LSQ(KKT_A, KKT_b, lsq_solver="scipy", eps_abs=1e-5, eps_rel=1e-5)
            elif ctx.solve_type == "sparse":
                # print("did sparse solve", ctx.t)
                # grad_b,_ = lin_solvers.sparse_LSQ(KKT_A, KKT_b, lsq_solver=ctx.qp_solver, eps_abs=1e-5, eps_rel=1e-5)
                grad_b,_ = lin_solvers.sparse_LSQ(KKT_A, KKT_b, lsq_solver="scipy", eps_abs=1e-5, eps_rel=1e-5, x0=initvals)

        if len(grad_b.shape) == 2:
            grad_b = grad_b.squeeze(-1)

        x = ctx.x_mu_nu_star[0:ctx.dim]
        mu = ctx.x_mu_nu_star[ctx.dim:(ctx.dim+ctx.nEq)]
        nu = ctx.x_mu_nu_star[(ctx.dim+ctx.nEq):]
        dx =  -grad_b[0:ctx.dim]
        dmu = -grad_b[ctx.dim:(ctx.dim+ctx.nEq)]
        dnu = -grad_b[(ctx.dim+ctx.nEq):]

        grad_Q,grad_q,grad_G,grad_h,grad_A,grad_b = None,None,None,None,None,None
        if ctx.specific_requires_grad[0]:
            if ctx.patterns[0] is None:
                grad_Q = np.outer(dx,x)
                grad_Q = torch.from_numpy(1/2*(grad_Q + grad_Q.T))
            else:
                i_row = ctx.patterns[0][0]
                i_col = ctx.patterns[0][1]
                grad_Q = torch.from_numpy(1/2*(np.multiply(dx[i_row],x[i_col]) + np.multiply(x[i_row],dx[i_col]))).squeeze(-1)
                if grad_Q.dim() == 2:
                    grad_Q = grad_Q.squeeze(-1)
                elif grad_Q.dim() == 0:
                    grad_Q = grad_Q.unsqueeze(0)
        if ctx.specific_requires_grad[1]:
            grad_q = torch.from_numpy(dx)
        if ctx.specific_requires_grad[2]:
            if ctx.patterns[1] is None:
                grad_G = torch.from_numpy(np.outer(dnu, x) + np.outer(nu, dx))
            else:
                i_row = ctx.patterns[1][0]
                i_col = ctx.patterns[1][1]
                grad_G = torch.from_numpy(np.multiply(dnu[i_row],x[i_col]) + np.multiply(nu[i_row],dx[i_col])).squeeze(-1)
                if grad_G.dim() == 2:
                    grad_G = grad_G.squeeze(-1)
                elif grad_G.dim() == 0:
                    grad_G = grad_G.unsqueeze(0)
        if ctx.specific_requires_grad[3]:
            grad_h = torch.from_numpy(-dnu)
        if ctx.specific_requires_grad[4]:
            if ctx.patterns[2] is None:
                grad_A = torch.from_numpy(np.outer(dmu, x)) + np.outer(mu, dx)
            else:
                i_row = ctx.patterns[2][0]
                i_col = ctx.patterns[2][1]
                grad_A = torch.from_numpy(np.multiply(dmu[i_row],x[i_col]) + np.multiply(mu[i_row],dx[i_col])).squeeze(-1)
                if grad_A.dim() == 2:
                    grad_A = grad_A.squeeze(-1)
                elif grad_A.dim() == 0:
                    grad_A = grad_A.unsqueeze(0)

        if ctx.specific_requires_grad[5]:
            grad_b = torch.from_numpy(-dmu)

        # For geometry scaling, profile the backward through QP alone
        # t_diff = time.time() - t # TODO: note that this should be removed in final product
        # f = open("../experiments/geometry/results/profiling/t_diff.dat","w+")
        # print(t_diff,file=f)
        # For geometry scaling, profile the backward through QP alone

        return grad_Q,grad_q,grad_G,grad_h,grad_A,grad_b,None,None

def build_settings(check_PSD=False,time=False,solve_type="dense",dual_available=None,normalize_constraints=False,empty_batch=True,warm_start_from_previous=False,omp_parallel=False,n_cpu=None, # general arguments
                   eps_active=1e-5,refine_active=False, # active arguments
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
        "eps_active": eps_active,
        "eps_abs": eps_active,
        "eps_rel": eps_active,
        "lin_solver": lin_solver,
        "refine_active": refine_active,
        "omp_parallel" : omp_parallel,
        "n_cpu" : n_cpu
    }

    return settings