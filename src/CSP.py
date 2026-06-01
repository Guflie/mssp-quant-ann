"""
CSP model for satellite scheduling problem.

Each variable represents a satellite in orbit, and the values that can be 
assigned are a visibility slot, which is some disrcetized temporal interval 
in which some ground antena can serve the satellites.
A constraint between two variables means they share at least one Timeslot in
their domains, i.e. they conflict and that timeslot cannot be assigned to both.
The constraint graph is stored as a boolean upper-triangular numpy matrix
indexed by variable/satellite id.

Public API
----------
Timeslot                    := immutable (antenna_id, timeframe) value
Var                         := CSP variable with a Timeslot domain
CSP                         := constraint graph over Var objects
ac3                         := arc-consistency preprocessing (modifies domains in-place)
split_connected_components  := decompose a CSP into independent sub-problems
"""

from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np


@dataclass(frozen=True)
class Timeslot:
    """
    Immutable (antenna_id, timeframe) pair; it's the "value" assigned to Vars.
    """
    antenna_id :int
    timeframe :int


class Var:
    """
    A CSP variable representing a satellite.

    Attributes
    ----------
    id     : int            := unique identifier, used as matrix index in CSP
    domain : set[Timeslot]  := set of Timeslots the satellite can be scheduled at
    """
    id :int
    domain :set[Timeslot]

    def __init__(self, id :int) ->None:
        self.id = id
        self.domain = set()

    def add_value(self, val :Timeslot) ->None:
        """
        Add val to the variable's domain.
        """
        self.domain.add(val)

    def load_values(self, vals :set[Timeslot]) ->None:
        """
        Add the set of values to variable's domain (used for bulk loads).
        """
        self.domain = self.domain.union(vals)

    def trim_value(self, val :Timeslot) ->None:
        """
        Remove val from the domain; raises KeyError if absent.
        """
        self.domain.remove(val)


class CSP:
    """
    Constraint Satisfaction Problem over satellite scheduling variables (Diff formulation).

    Two variables are constrained when their domains share at least one Timeslot,
    meaning they conflict and cannot both be assigned that slot. The constraint
    graph is encoded as a boolean upper-triangular numpy matrix (const[i][j], i < j).

    Attributes
    ----------
    max_vars : int        := capacity of the constraint matrix (fixed at construction)
    vars     : list[Var]  := registered variables
    const    : np.ndarray := upper-triangular bool matrix; const[i][j] is True iff
                            vars i and j share at least one Timeslot in their domains
    """

    def __init__(self, max_vars :int):
        self.max_vars = max_vars    # used for initialize the NumPy matrix
        self.vars :list[Var] = []
        self.const = np.zeros([max_vars, max_vars], dtype=bool)
        # TODO: optimize this to waste less space

    def add_var(self, var :Var) ->None:
        """
        Add var to the problem; silently ignored if max_vars capacity is already reached.
        """
        if len(self.vars) < self.max_vars: 
            self.vars.append(var)
            # automatically enforce constraints between new variable and the old ones
            for v in (v for v in self.vars if v.id != var.id):
                self.add_constraint(v, var)


    def get_var_by_id(self, id :int) ->Var:
        """
        Return the Var with the given id, or a sentinel Var(id=-1) if not found.
        """
        for v in self.vars:
            if v.id == id: return v
        return Var(id=-1)

    @staticmethod
    def _order_params(u :Var, v :Var) ->list[Var]:
        """
        Return [min_id_var, max_id_var] to enforce upper-triangular matrix access to the 
        constraint matrix.
        """
        min_var :Var = min(u, v, key=lambda v: v.id)
        max_var :Var = max(u, v, key=lambda v: v.id)
        return [min_var, max_var]
    # TODO: this might be improved

    def add_constraint(self, u :Var, v :Var) ->None:
        """
        Add a constraint between u and v if their domains overlap.
        """
        if u.id == v.id:
            return
        if v.domain & u.domain:
            min_v, max_v = self._order_params(u, v)
            self.const[min_v.id][max_v.id] = True

    def remove_constraint(self, u :Var, v :Var) ->None:
        """
        Remove the constraint between u and v (no-op if they are the same variable).
        """
        if u.id == v.id: return
        min_v, max_v = self._order_params(u, v)
        self.const[min_v.id][max_v.id] = False

    def update_constrains(self) ->None:
        """
        Rebuild the full constraint matrix by scanning all variable pairs.
        """
        for v in self.vars:
            for u in self.vars:
                if v.id != u.id and v.domain & u.domain:
                    min_v, max_v = self._order_params(u, v)
                    self.const[min_v.id][max_v.id] = True

    def constraint_exists(self, u :Var, v :Var) ->bool:
        """
        Return True if a constraint exists between u and v, otherwise False.
        """
        if u.id == v.id: return False
        min_v, max_v = self._order_params(u, v)
        return self.const[min_v.id][max_v.id]

    def neighbors(self, u :Var) ->list[Var]:
        """
        Return all variables that share a constraint with u.
        """
        return [v for v in self.vars if u != v and self.constraint_exists(u, v)]

    def degree(self, v :Var) ->int:
        """
        Return the number of constraints involving v (row + column of the matrix).
        """
        return sum(self.const[v.id]) + sum(self.const[:, v.id])

    def num_constraints(self) ->int:
        """
        Return the total number of constraints in the CSP.
        """
        return int(self.const.sum())

    def compute_summary(self) ->dict[str, float]:
        """
        Return a dict with main statistics: variable count, domain sizes, constraint density.
        """
        summ: dict[str, float] = dict()
        summ["num_vars"] = len(self.vars)

        summ["avg_domain_size"] = float(np.mean([len(v.domain) for v in self.vars]))
        summ["sd_domain_size"] = float(np.std([len(v.domain) for v in self.vars]))
        summ["min_domain_size"] = float(np.min([len(v.domain) for v in self.vars]))
        summ["max_domain_size"] = float(np.max([len(v.domain) for v in self.vars]))
        
        summ["num_const"] = self.num_constraints()
        summ["avg_degree"] = float(np.mean([self.degree(v) for v in self.vars]))
        summ["sd_degree"] = float(np.std([self.degree(v) for v in self.vars]))
        summ["const_dens"] = (
            float(2 * np.sum(self.const) / (len(self.vars) ** 2 - len(self.vars)))
            if len(self.vars) > 1 else 0
        )
        return summ


def ac3(csp :CSP, debug=False) ->bool:
    """
    Enforce arc consistency (AC-3) on the CSP.

    For the Diff formulation, a timeslot in domain[u] loses support only when
    domain[v] contains no value different from it (i.e. domain[v] == {timeslot}).
    Modifies Var.domain in-place. Returns False if any domain is wiped out.
    """

    # will follow some function used by the ac3 algorithm
    def _revise(u :Var, v :Var) ->bool:
        """
        Remove timeslots from u.domain that have no support in v.domain; return True if revised.
        """
        revised :bool = False
        for color in list(u.domain):
            if not any(c != color for c in v.domain):
                u.domain.remove(color)
                revised = True
        return revised

    def _prune_disjoint_constraints(csp :CSP, var :Var) ->int:
        """
        Remove constraints between var and any neighbor whose domain is now disjoint from var's.

        In the Diff formulation, disjoint domains guarantee the constraint is never
        violated, so the arc is redundant. Returns the number of pruned constraints.
        """
        pruned = 0
        for w in csp.vars:
            if w.id != var.id and csp.constraint_exists(var, w) and not var.domain & w.domain:
                csp.remove_constraint(var, w)
                pruned += 1
        return pruned

    queue = deque(
        (u, v)
        for u in csp.vars
        for v in csp.vars
        if u.id != v.id and csp.constraint_exists(u, v)
    )
    while queue:
        u, v = queue.popleft()
        if _revise(u, v):
            if not u.domain: return False

            pruned = _prune_disjoint_constraints(csp, u)
            if debug: print(f"pruned {pruned} constraints") # TODO: switch this to log insted of bare print

            for w in csp.vars:
                if w.id != u.id and w.id != v.id and csp.constraint_exists(w, u):
                    queue.append((w, u))
    return True


def split_connected_components(csp :CSP) ->list[CSP]:
    """
    Decompose a CSP into independent sub-CSPs, one per connected component
    of the constraint graph. Variables with no constraints form singleton components.

    Each sub-CSP gets deep-copied Var objects (same ids, copied domains) so
    sub-problems can be solved independently without domain mutations leaking.
    The constraint matrix size is preserved (max_vars) so variable ids remain
    valid as matrix indices.
    """
    visited :set[int] = set()
    components :list[list[Var]] = []

    for start in csp.vars:
        if start.id in visited:
            continue
        component :list[Var] = []
        queue = deque([start])
        visited.add(start.id)
        while queue:
            var = queue.popleft()
            component.append(var)
            for neighbor in csp.neighbors(var):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append(neighbor)
        components.append(component)

    result :list[CSP] = []
    for component in components:
        sub = CSP(max_vars=csp.max_vars)
        for var in component:
            new_var = Var(var.id)
            new_var.domain = set(var.domain)
            sub.add_var(new_var)
        for u in sub.vars:
            for v in sub.vars:
                if u.id < v.id and csp.const[u.id][v.id]:
                    sub.const[u.id][v.id] = True
        result.append(sub)

    return result
