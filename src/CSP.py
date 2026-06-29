"""
CSP model for satellite scheduling problem.

Each variable represents a satellite in orbit, and the values that can be
assigned are a visibility slot, which is a disrcetized temporal interval
in which some ground antena can serve the satellites.
A constraint between two variables means they share at least one Timeslot in
their domains, i.e. they conflict and that timeslot cannot be assigned to both.
The constraint graph is stored as a boolean upper-triangular numpy matrix
indexed by variable/satellite id.

Public API
----------
Timeslot                    := immutable (antenna_id, timeframe) value
CSP_Variable                := CSP variable with a Timeslot domain
CSP                         := constraint graph over CSP_Variable objects
ac3                         := arc-consistency preprocessing (modifies domains in-place)
split_connected_components  := decompose a CSP into independent sub-problems
"""

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass(frozen=True)
class Timeslot:
    """
    Immutable (antenna_id, timeframe) pair; it's the "value" assigned to CSP_Variables.
    """
    antenna_id :int
    timeframe :int


class CSP_Variable:
    """
    A CSP variable representing a satellite.

    Attributes
    ----------
    id     : int            := unique identifier, used as matrix index in CSP
    name   : str            := name assigned to the satellite, for better interpretability
    domain : set[Timeslot]  := set of Timeslots the satellite can be scheduled at
    """
    id :int
    name :str
    domain :set[Timeslot]
    weight :float

    def __init__(self, id :int, weight: float = 1, name: str = "") ->None:
        self.id = id
        self.name = name
        self.domain = set()
        self.weight = weight

    def add_value(self, val :Timeslot) ->None:
        """
        Add val to the variable's domain.
        """
        self.domain.add(val)

    def load_values(self, vals :set[Timeslot]) ->None:
        """
        Add the set of values to variable's domain (used for bulk loads).
        """
        self.domain.update(vals)

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
    max_vars   : int                    := capacity of the constraint matrix (fixed at construction)
    vars       : list[CSP_Variable]     := registered variables
    const      : np.ndarray             := upper-triangular bool matrix; const[i][j] is True iff
                                           vars i and j share at least one Timeslot in their domains
    _var_by_id : dict[int,CSP_Variable] := O(1) lookup index kept in sync with vars
    """

    def __init__(self, max_vars :int):
        self.max_vars = max_vars    # used for initialize the NumPy matrix
        self.vars :list[CSP_Variable] = []
        self._var_by_id :dict[int, CSP_Variable] = {}
        self.const = np.zeros([max_vars, max_vars], dtype=bool)
        # TODO: optimize this to waste less space

    def add_var(self, var :CSP_Variable) ->None:
        """
        Add var to the problem; raises ValueError if max_vars capacity is already reached.
        """
        if len(self.vars) >= self.max_vars:
            raise ValueError(f"CSP capacity exceeded: max_vars={self.max_vars}")
        self.vars.append(var)
        self._var_by_id[var.id] = var
        # automatically enforce constraints with old variables as the new one is added
        for v in (v for v in self.vars if v.id != var.id):
            self.add_constraint(v, var)

    def get_var_by_id(self, id :int) ->CSP_Variable | None:
        """
        Return the CSP_Variable with the given id, or None if not found.
        """
        return self._var_by_id.get(id)

    @staticmethod
    def _order_params(u :CSP_Variable, v :CSP_Variable) ->tuple[CSP_Variable, CSP_Variable]:
        """
        Return (min_var_by_id, max_var_by_id) to enforce upper-triangular matrix access to the
        constraint matrix.
        """
        
        return min(u, v, key=lambda v: v.id), max(u, v, key=lambda v: v.id)

    def add_constraint(self, u :CSP_Variable, v :CSP_Variable) ->None:
        """
        Add a constraint between u and v if their domains overlap.
        """
        if u.id == v.id: return # no-op if the two parameters are the same variable
        if v.domain & u.domain:
            min_v, max_v = self._order_params(u, v)
            self.const[min_v.id][max_v.id] = True

    def remove_constraint(self, u :CSP_Variable, v :CSP_Variable) ->None:
        """
        Remove the constraint between u and v.
        """
        if u.id == v.id: return # no-op if the two parameters are the same variable
        min_v, max_v = self._order_params(u, v)
        self.const[min_v.id][max_v.id] = False

    def update_constraints(self) ->None:
        """
        Rebuild the full constraint matrix by scanning all variable pairs.
        """
        self.const[:] = False # resets stale constraints before rebuilding
        for v in self.vars:
            for u in self.vars:
                if v.id < u.id:
                    self.const[v.id][u.id] = bool(v.domain & u.domain)

    def constraint_exists(self, u :CSP_Variable, v :CSP_Variable) ->bool:
        """
        Return True if a constraint exists between u and v, otherwise False.
        """
        if u.id == v.id: return False
        min_v, max_v = self._order_params(u, v)
        return self.const[min_v.id][max_v.id]

    def neighbors(self, u :CSP_Variable) ->list[CSP_Variable]:
        """
        Return all variables that share a constraint with u.
        """
        return [v for v in self.vars if u.id != v.id and self.constraint_exists(u, v)]

    def degree(self, v :CSP_Variable) ->int:
        """
        Return the number of constraints involving v (row + column of the matrix).
        """
        return int(self.const[v.id].sum() + self.const[:, v.id].sum())

    def num_constraints(self) ->int:
        """
        Return the total number of constraints in the CSP.
        """
        return int(self.const.sum())

    def compute_summary(self) ->dict[str, float]:
        """
        Return a dict with main problem statistics: variable count, domain sizes, constraint density.
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
    domain[v] contains no value different from it (implicit constraint).
    Modifies CSP_Variable.domain in-place. Returns False if any domain is wiped out.
    """

    # ---------- sub-function declaration ------------------------------------------------------------
    def _revise(u :CSP_Variable, v :CSP_Variable) ->bool:
        """
        Remove timeslots from u.domain that have no support in v.domain; return True if any 
        domain has been revised.
        """
        revised :bool = False
        # to visualize the problem think about graph coloring problem: each Timeslot (pair 
        # Antenna-Timeframe) is color, and you try to color all variables with non-conflicting
        # color.
        for color in list(u.domain):
            if v.domain == {color}:
                u.domain.remove(color)
                revised = True
        return revised

    def _prune_disjoint_constraints(csp :CSP, var :CSP_Variable) ->int:
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
    # -----------------------------------------------------------------------------------------------

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
            if debug: print(f"pruned {pruned} constraints") # TODO: switch this to log instead of bare print

            for w in csp.vars:
                if w.id != u.id and w.id != v.id and csp.constraint_exists(w, u):
                    queue.append((w, u))
    return True


def split_connected_components(csp :CSP) ->list[CSP]:
    """
    Decompose a CSP into independent sub-CSPs, one per connected component
    of the constraint graph.

    Each sub-CSP gets deep-copied CSP_Variable objects (same ids, copied domains) so
    sub-problems can be solved independently without mutual influences.
    The constraint matrix size is preserved (max_vars) so variable ids remain
    valid as matrix indices.
    """
    visited :set[int] = set()
    components :list[list[CSP_Variable]] = []

    for start in csp.vars:
        if start.id in visited:
            continue
        component :list[CSP_Variable] = []
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
            new_var = CSP_Variable(var.id, var.weight, var.name)
            new_var.domain = set(var.domain)
            sub.add_var(new_var)
        result.append(sub)

    return result
