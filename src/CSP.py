"""
CSP model for satellite scheduling problem.

Each variable represents a satellite in orbit, and the values that can be
assigned are a contact slots, which is a disrcetized temporal interval
in which some ground antena can serve the satellites.
A constraint between two variables means they share at least one contact slot in
their domains, i.e. they conflict and that timeslot cannot be assigned to both.
The constraint graph is stored as a boolean upper-triangular numpy matrix
indexed by variable/satellite id.

Public API
----------
Timeslot                    := immutable (antenna_id, timeframe) value
CSP_Variable                := CSP variable with a Timeslot domain
CSP                         := constraint graph over CSP_Variable objects
ac3                         := arc-consistency preprocessing (modifies domains in-place,
                               optionally tolerating wiped-out variables)
split_connected_components  := decompose a CSP into independent sub-problems
"""

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass(frozen=True)
class Timeslot:
    """
    It's the "value" assigned to CSP_Variables, corresponding to a contact slot for that satellite.
    I keeps all the relevant information.
    Times are integer seconds from the scheduling horizon start: comparisons are exact, so a gap
    exactly equal to the reconfiguration time is never misread because of floating-point noise.
    """
    slot_id :str
    antenna_id :str
    reconfiguration_s :int
    start_s :int
    end_s :int

    @classmethod
    def from_hours(cls, slot_id :str, antenna_id :str, reconfiguration_hours :float,
                   start_hours :float, end_hours :float) -> Timeslot:
        """
        Build a Timeslot from times in hours (as stored in the slot CSVs), rounded to whole seconds.
        """
        to_s = lambda hours: round(hours * 3600)
        return cls(slot_id, antenna_id, to_s(reconfiguration_hours), to_s(start_hours), to_s(end_hours))

    @property
    def duration_s(self) -> int:
        return self.end_s - self.start_s

    @staticmethod
    def overlap(t1: Timeslot, t2: Timeslot) -> bool:
        """
        Returns whether the two timeslots t1 and t2 overlap or not.
        """
        # test whether the two timeslot share the same antenna
        antenna_test :bool = t1.antenna_id == t2.antenna_id

        # test if the two slots overlap temporarly
        overlap_test :bool = t1.end_s + t1.reconfiguration_s > t2.start_s \
                            if t1.start_s <= t2.start_s \
                            else t2.end_s + t2.reconfiguration_s > t1.start_s

        return antenna_test and overlap_test


class CSP_Variable:
    """
    A CSP variable representing a satellite.

    Attributes
    ----------
    id        : int              := unique identifier, used as matrix index in CSP
    name      : str              := name assigned to the satellite, for better interpretability
    domain    : set[Timeslot]    := set of Timeslots the satellite can be scheduled at
    """
    id :int
    name :str
    domain :set[Timeslot]

    def __init__(self, name: str = "") ->None:
        self.id = -1
        self.name = name
        self.domain = set()

    def add_value(self, val :Timeslot) ->None:
        """
        Add val Timeslot to the variable's domain.
        """
        self.domain.add(val)

    def load_values(self, vals :set[Timeslot]) ->None:
        """
        Add the set of values to variable's domain (used for bulk loads), with the default weight.
        """
        for val in vals:
            self.add_value(val)

    def trim_value(self, val :Timeslot) ->None:
        """
        Remove val from the domain; raises KeyError if absent.
        """
        self.domain.remove(val)

    def has_conflict(self, val :Timeslot) ->bool:
        """
        Returns whether the variable conflict with some value in the domain or not
        """
        for d_val in list(self.domain): 
            if Timeslot.overlap(d_val, val): return True
        return False


class CSP:
    """
    Constraint Satisfaction Problem over satellite scheduling variables (Diff formulation).

    Two variables are constrained when their domains share at least one overlapping Timeslot,
    meaning they conflict and cannot both be assigned that slot. The constraint
    graph is encoded as a boolean upper-triangular numpy matrix (const[i][j], i < j).

    Attributes
    ----------
    max_vars   : int                    := capacity of the constraint matrix (fixed at construction)
    vars       : list[CSP_Variable]     := registered variables
    const      : np.ndarray             := upper-triangular bool matrix; const[i][j] is True iff
                                           vars i and j overlap on at least one Timeslot in their domains
    _var_by_id : dict[int,CSP_Variable] := O(1) lookup index kept in sync with vars
    """

    def __init__(self, max_vars :int):
        self.max_vars = max_vars    # used for initialize the NumPy matrix
        self.vars :list[CSP_Variable] = []
        self._var_by_name :dict[str, CSP_Variable] = {}
        self.const = np.zeros([max_vars, max_vars], dtype=bool)
        self._var_count = 0 # use to assing to each variable the correct index

    def add_var(self, var :CSP_Variable) ->None:
        """
        Add var to the problem; raises ValueError if max_vars capacity is already reached.
        """
        if self._var_count >= self.max_vars:
            raise ValueError(f"CSP capacity exceeded: max_vars={self.max_vars}")
        
        var.id = self._var_count
        self._var_count += 1
        
        self.vars.append(var)
        self._var_by_name[var.name] = var
        # automatically enforce constraints with old variables as the new one is added
        for v in (v for v in self.vars if v.id != var.id):
            self.add_constraint(v, var)

    def get_var_by_name(self, name :str) ->CSP_Variable | None:
        """
        Return the CSP_Variable with the given id, or None if not found.
        """
        return self._var_by_name.get(name)

    @staticmethod
    def _order_params(u :CSP_Variable, v :CSP_Variable) ->tuple[CSP_Variable, CSP_Variable]:
        """
        Return (min_var_by_id, max_var_by_id) to enforce upper-triangular matrix access to the
        constraint matrix.
        """        
        return min(u, v, key=lambda v: v.id), max(u, v, key=lambda v: v.id)

    def add_constraint(self, u :CSP_Variable, v :CSP_Variable) ->bool:
        """
        Check whether the two variables overlap, and in case set the conflicting edge.
        Returns whether the edge has been added or not.
        """
        if u.id == v.id: return False
        for t1 in u.domain:
            for t2 in v.domain:
                if Timeslot.overlap(t1, t2):
                    min_v, max_v = self._order_params(u,v)
                    self.const[min_v.id, max_v.id] = True
                    return True
        return False
        
    def remove_constraint(self, u :CSP_Variable, v :CSP_Variable) ->bool:
        """
        Remove the constraint between u and v (check before that this is true).
        Returns whether the constraint has been removed or not.
        """
        if u.id == v.id: return False # no-op if the two parameters are the same variable
        min_v, max_v = self._order_params(u, v)
        for t1 in u.domain:
                    for t2 in v.domain:
                        if Timeslot.overlap(t1, t2): return False
        self.const[min_v.id][max_v.id] = False
        return True

    def update_constraints(self) ->None:
        """
        Rebuild the full constraint matrix by scanning all variable pairs.
        """
        self.const[:] = False # resets stale constraints before rebuilding
        for v in self.vars:
            for u in self.vars:
                if v.id < u.id: self.add_constraint(u, v)

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


def ac3(csp :CSP, debug=False, allow_wipeout=False) ->bool:
    """
    Enforce arc consistency (AC-3) on the CSP.

    For the Diff formulation, a timeslot in domain[u] loses support only when
    domain[v] contains no value different from it (implicit constraint).
    Modifies CSP_Variable.domain in-place. Returns False if any domain is wiped out.

    With allow_wipeout=True, a wiped-out variable does not stop the propagation:
    it loses all its constraints (it can no longer be scheduled) and AC-3 keeps
    pruning the rest of the problem.
    """

    # ---------- sub-function declaration ------------------------------------------------------------
    def _revise(u :CSP_Variable, v :CSP_Variable) ->bool:
        """
        Remove timeslots from u.domain that have no support in v.domain; return True if any 
        domain has been revised.
        A timeslot has no support when it conflicts with every timeslot of v: whatever v takes,
        the pair is infeasible. An empty v.domain (wiped-out variable) constrains nothing.
        """
        revised :bool = False
        for color in list(u.domain):
            if v.domain and all(Timeslot.overlap(color, other) for other in v.domain):
                u.trim_value(color)
                revised = True
        return revised

    def _prune_disjoint_constraints(csp :CSP, var :CSP_Variable) ->int:
        """
        Remove constraints between var and any neighbor none of whose timeslots conflicts any more
        with var's (remove_constraint checks it).

        Once no pair of values conflicts, the constraint can never be violated, so the arc is
        redundant. Returns the number of pruned constraints.
        """
        pruned = 0
        for w in csp.vars:
            if w.id != var.id and csp.constraint_exists(var, w) and csp.remove_constraint(var, w):
                pruned += 1
        return pruned
    # -----------------------------------------------------------------------------------------------

    queue = deque(
        (u, v)
        for u in csp.vars
        for v in csp.vars
        if u.id != v.id and csp.constraint_exists(u, v)
    )
    feasible = True
    while queue:
        u, v = queue.popleft()
        if _revise(u, v):
            if not u.domain:
                if not allow_wipeout: return False
                feasible = False

            pruned = _prune_disjoint_constraints(csp, u)
            if debug: print(f"pruned {pruned} constraints") # TODO: switch this to log instead of bare print

            for w in csp.vars:
                if w.id != u.id and w.id != v.id and csp.constraint_exists(w, u):
                    queue.append((w, u))
    return feasible


def split_connected_components(csp :CSP) ->list[CSP]:
    """
    Decompose a CSP into independent sub-CSPs, one per connected component
    of the constraint graph.

    Each sub-CSP gets copied CSP_Variable objects (same names, copied domains) so
    sub-problems can be solved independently without mutual influences.
    Each sub-CSP is sized to its component and add_var renumbers its variables from 0,
    so ids are local to a sub-CSP: use the variable name to match across CSPs.
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
        sub = CSP(max_vars=len(component))
        for var in component:
            new_var = CSP_Variable(var.name)
            new_var.domain = set(var.domain)
            sub.add_var(new_var)
        result.append(sub)

    return result
