"""``KinoSolver`` is an optimization model class that solves the scheduling
of parallel node-running for ``KinoRunner``, using a ``pyomo`` linear program.
"""
# pylint: disable=invalid-name
import typing
from itertools import product
from math import ceil
from typing import Dict, Tuple, Set

import pandas as pd
from pyomo.core.util import quicksum
from pyomo.environ import (
    AbstractModel,
    Binary,
    Constraint,
    NonNegativeIntegers,
    Objective,
    Param,
    RangeSet,
    Set,
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    Var,
    value,
)

from kedro.pipeline.node import Node


# Constraint: each job must be assigned exactly once
def _assign_constraint(model, j):
    return (
        quicksum(
            model.x[i, j, t]
            for i in model.workers
            for t in range(0, model.max_time - model.proc_time[j] + 1)
        )
        == 1
    )


# Constraint: job finishing time is start time + processing time
def _time_constraint(model, j):
    return model.done_time[j] == model.proc_time[j] + quicksum(
        t * model.x[i, j, t]
        for i in model.workers
        for t in range(0, model.max_time - model.proc_time[j] + 1)
    )


# Constraint: compute makespan from job finishing times
def _compute_makespan(model, j):
    return model.ms >= model.done_time[j]


# Constraint: each machine at each time can have no more than 1 job scheduled
def _conflict_constraint(model, i, t):
    return (
        quicksum(
            model.x[i, j, s]
            for j in model.nodes
            for s in model.times
            if s <= t < s + model.proc_time[j]
        )
        <= 1
    )


# Constraint: predecessor constraint for each job
def _predecessor_constraint(model, k, j):
    if model.dependency[j, k] == 1:
        return model.done_time[j] <= model.done_time[k] - model.proc_time[k]
    return model.done_time[j] >= 0


class KinoSolver:
    """``KinoSolver`` is an optimization model class that solves the scheduling
    of parallel node-running for ``KinoRunner``, using a ``pyomo`` linear program.
    """

    def __init__(self):
        self.model = AbstractModel()
        self._define_model_structure()

        self.instance = None
        self.opt = None
        self.opt_results = None
        self.optimal_objective = None
        self._schedule_dict = None
        self._node_worker = None

    def _define_model_structure(self):
        """Set up model structure incl. params, sets, variables, objective, constraints."""
        # Set of node names
        self.model.nodes = Set()

        # Construct RangeSet of workers from n_workers
        self.model.n_workers = Param(within=NonNegativeIntegers)
        self.model.workers = RangeSet(1, self.model.n_workers)

        # Processing time of each node
        self.model.proc_time = Param(self.model.nodes, within=NonNegativeIntegers)

        # Dependency of each node/job
        # Indices are defined as: Job i is prerequisite of Job j -> dependency[i,j] == 1
        self.model.dependency = Param(self.model.nodes, self.model.nodes, within=Binary)

        # max_time (T) and RangeSet of times as indices
        self.model.max_time = Param(within=NonNegativeIntegers)
        self.model.times = RangeSet(0, self.model.max_time)

        # Binary decision variable x
        self.model.x = Var(
            self.model.workers, self.model.nodes, self.model.times, within=Binary
        )

        # Finishing time of each job
        self.model.done_time = Var(self.model.nodes, within=NonNegativeIntegers)

        # Total Makespan (this will be the objective)
        self.model.ms = Var(within=NonNegativeIntegers)

        # Default sense is minimize so no need to set anything here
        self.model.OBJ = Objective(rule=lambda model: model.ms)

        # Define constraints
        self.model.assign_constraint = Constraint(
            self.model.nodes, rule=_assign_constraint
        )
        self.model.time_constraint = Constraint(self.model.nodes, rule=_time_constraint)
        self.model.compute_makespan = Constraint(
            self.model.nodes, rule=_compute_makespan
        )
        self.model.conflict_constraint = Constraint(
            self.model.workers, self.model.times, rule=_conflict_constraint
        )
        self.model.predecessor_constraint = Constraint(
            self.model.nodes, self.model.nodes, rule=_predecessor_constraint
        )

    def instantiate_from_data(
        self,
        nodes: typing.Set[Node],
        n_workers: int,
        proc_time: Dict[str, float],
        dependency: Dict[Node, typing.Set[Node]],
    ):
        """Create model instance from input data."""
        node_names = [n.name for n in nodes]

        dependency_w_names = {
            n.name: {prereq.name for prereq in dependency[n]} for n in dependency
        }

        dependency_grid = dict.fromkeys(product(node_names, node_names), 0)
        for node in dependency_w_names:
            for prereq in dependency_w_names[node]:
                dependency_grid[prereq, node] = 1

        proc_time_ceil = {n: ceil(proc_time[n]) for n in proc_time}

        data = {
            None: {
                "nodes": {None: node_names},
                "n_workers": {None: n_workers},
                "proc_time": proc_time_ceil,
                "dependency": dependency_grid,
                "max_time": {None: sum(proc_time_ceil.values())},
            }
        }

        self.instance = self.model.create_instance(data=data)

    def solve(
        self,
        solver_class: str = "glpk",
        options: str = "",
        verbose: bool = False,
    ):
        """Solve the linear program using the specified solver."""
        self.opt = SolverFactory(solver_class)

        self.opt_results = self.opt.solve(self.instance, options=options, tee=verbose)

        if (self.opt_results.solver.status == SolverStatus.ok) and (
            self.opt_results.solver.termination_condition
            == TerminationCondition.optimal
        ):
            self.optimal_objective = value(self.instance.ms)
            print(f"Solve finished. Optimal objective: {self.optimal_objective}")
        elif (
            self.opt_results.solver.termination_condition
            == TerminationCondition.infeasible
        ):
            raise ValueError("Problem is infeasible.")

    def get_schedule(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Return the optimized run schedule for each worker."""
        if (self.opt_results.solver.status != SolverStatus.ok) or (
            self.opt_results.solver.termination_condition
            != TerminationCondition.optimal
        ):
            raise ValueError("Model is not solved or is infeasible.")

        # Convert pyomo Var object to pandas dataframe
        x_dict = {(i, j, t): value(v) for (i, j, t), v in self.instance.x.items()}
        x_df = pd.Series(x_dict).reset_index()
        x_df.columns = ["worker", "job", "start_time", "value"]

        # Convert pyomo's 1-index to 0-index
        x_df["worker"] -= 1

        self._schedule_dict = (
            x_df[x_df["value"] == 1]
            .sort_values(["worker", "start_time"])
            .groupby("worker")
            .agg({"job": list})
            .to_dict()["job"]
        )

        self._node_worker = (
            x_df[x_df["value"] == 1][["job", "worker"]]
            .set_index("job")
            .to_dict()["worker"]
        )

        return self._schedule_dict, self._node_worker
