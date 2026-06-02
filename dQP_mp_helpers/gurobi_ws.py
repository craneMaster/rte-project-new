#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors
# Copyright 2021 Dustin Kenefake
#
# Modified from qpsolvers/solvers/gurobi_.py to support basis warm-starting.
# gurobi_solve_problem now accepts initvals=(VBasis, CBasis) and returns
# (Solution, bases) instead of Solution.

"""Solver interface for `Gurobi <https://www.gurobi.com/>`__."""

import time
from typing import Optional, Union

import gurobipy
import numpy as np
import scipy.sparse as spa
from gurobipy import GRB

from qpsolvers.problem import Problem
from qpsolvers.solution import Solution


def gurobi_solve_problem(
    problem: Problem,
    initvals=None,
    verbose: bool = False,
    **kwargs,
):
    """Solve a quadratic program using Gurobi with optional basis warm-start.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start basis as (VBasis, CBasis) lists returned from a previous
        solve, or None to start from scratch.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Tuple of (Solution, (VBasis, CBasis)).
    """
    model = gurobipy.Model()
    if not verbose:
        model.setParam(GRB.Param.OutputFlag, 0)

    P, q, G, h, A, b, lb, ub = problem.unpack()
    num_vars = P.shape[0]
    identity = spa.eye(num_vars)
    x = model.addMVar(
        num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS
    )

    ineq_constr, eq_constr, lb_constr, ub_constr = None, None, None, None
    if G is not None:
        ineq_constr = model.addMConstr(G, x, GRB.LESS_EQUAL, h)
    if A is not None:
        eq_constr = model.addMConstr(A, x, GRB.EQUAL, b)
    if lb is not None:
        lb_constr = model.addMConstr(identity, x, GRB.GREATER_EQUAL, lb)
    if ub is not None:
        ub_constr = model.addMConstr(identity, x, GRB.LESS_EQUAL, ub)
    objective = 0.5 * (x @ P @ x) + q @ x
    model.setObjective(objective, sense=GRB.MINIMIZE)
    model.setParam("Method", 1)
    model.setParam("OptimalityTol", 1e-9)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("NumericFocus", 3)
    model.update()
    if initvals is not None:
        model.setAttr("VBasis", model.getVars(), initvals[0])
        model.setAttr("CBasis", model.getConstrs(), initvals[1])
    model.optimize()
    bases = model.getAttr(GRB.Attr.VBasis), model.getAttr(GRB.Attr.CBasis)

    solution = Solution(problem)
    solution.extras["status"] = model.status
    solution.found = model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
    if solution.found:
        solution.x = x.X
        __retrieve_dual(solution, ineq_constr, eq_constr, lb_constr, ub_constr)
    else:
        model.update()
    return solution, bases


def __retrieve_dual(
    solution: Solution,
    ineq_constr: Optional[gurobipy.MConstr],
    eq_constr: Optional[gurobipy.MConstr],
    lb_constr: Optional[gurobipy.MConstr],
    ub_constr: Optional[gurobipy.MConstr],
) -> None:
    solution.z = -ineq_constr.Pi if ineq_constr is not None else np.empty((0,))
    solution.y = -eq_constr.Pi if eq_constr is not None else np.empty((0,))
    if lb_constr is not None and ub_constr is not None:
        solution.z_box = -ub_constr.Pi - lb_constr.Pi
    elif ub_constr is not None:
        solution.z_box = -ub_constr.Pi
    elif lb_constr is not None:
        solution.z_box = -lb_constr.Pi
    else:
        solution.z_box = np.empty((0,))
