"""
CSP model for satellite scheduling problem.

Each variable represents a satellite in orbit, and the values that can be assigned are its
contact slots: discretized temporal intervals in which some ground antenna can serve the satellite.
A satellite takes at most one of its slots (or stays unscheduled). Two satellites are constrained
when some pair of their slots conflicts: same antenna, and the slots overlap or leave less than
the reconfiguration time between them (Timeslot.overlap).

The conflicts are stored once, slot by slot, in a ConflictIndex shared by a CSP and by every
sub-problem derived from it (copy, subproblem): the constraints are read from the index and the
current domains, so removing values never requires recomputing overlaps, and sub-problems are
cheap to create.

Public API
----------
Timeslot                    := immutable contact slot value (slot_id, antenna, times in seconds)
CSP_Variable                := CSP variable with a Timeslot domain
ConflictIndex               := shared index of the conflicts between slots of different satellites
CSP                         := constraint graph over CSP_Variable objects

The reduction and decomposition algorithms working on this model are in CSP_decomposition.py.
"""

from dataclasses import dataclass
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
    id        : int              := identifier assigned by CSP.add_var, kept by copies and sub-problems
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
        Add val Timeslot to the variable's domain (before the variable is added to a CSP).
        """
        self.domain.add(val)

    def load_values(self, vals :set[Timeslot]) ->None:
        """
        Add the set of values to variable's domain (used for bulk loads).
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


class ConflictIndex:
    """
    Grow-only index of the conflicts between slots of different satellites, shared by a CSP and
    by every sub-problem derived from it. Overlaps are computed once, when a satellite's slots are
    registered, and only against slots on the same antenna (the only ones that can overlap).

    Attributes
    ----------
    conflicts : dict[Timeslot, set[Timeslot]] := slot -> slots of other satellites conflicting with it
    owner     : dict[Timeslot, str]           := slot -> name of the satellite it belongs to
    """

    def __init__(self) ->None:
        self.conflicts :dict[Timeslot, set[Timeslot]] = {}
        self.owner :dict[Timeslot, str] = {}
        self._by_antenna :dict[str, list[Timeslot]] = {}

    def register(self, name :str, slots :set[Timeslot]) ->None:
        """
        Add the slots of satellite name, computing their conflicts with the slots already
        registered. Slots already registered (e.g. by a sub-problem sharing the index) are skipped.
        """
        for t in slots:
            if t in self.owner:
                if self.owner[t] != name:
                    raise ValueError(f"{t.slot_id} already belongs to {self.owner[t]}, not {name}")
                continue
            self.owner[t] = name
            self.conflicts[t] = set()
            same_antenna = self._by_antenna.setdefault(t.antenna_id, [])
            for u in same_antenna:
                if self.owner[u] != name and Timeslot.overlap(t, u):
                    self.conflicts[t].add(u)
                    self.conflicts[u].add(t)
            same_antenna.append(t)


class CSP:
    """
    Constraint Satisfaction Problem over satellite scheduling variables.

    Two variables are constrained when some slot of one conflicts with some slot of the other.
    Constraints are not stored: they are read from the shared ConflictIndex, restricted to the
    slots still in the current domains, so they always reflect the domains (removing a value can
    only remove constraints) and need no update.
    Values must be loaded into a variable before add_var registers it; afterwards they can only be
    removed (trim_value, domain set operations).

    Attributes
    ----------
    vars         : list[CSP_Variable]      := registered variables
    index        : ConflictIndex           := conflicts between slots, shared with copies and sub-problems
    _var_by_name : dict[str, CSP_Variable] := O(1) lookup index kept in sync with vars
    _next_id     : int                     := id given to the next variable added without one
    """

    def __init__(self, index :ConflictIndex | None = None) ->None:
        self.vars :list[CSP_Variable] = []
        self._var_by_name :dict[str, CSP_Variable] = {}
        self.index = index if index is not None else ConflictIndex()
        self._next_id = 0

    def add_var(self, var :CSP_Variable) ->None:
        """
        Add var to the problem and register its slots in the conflict index; raises ValueError
        if a variable with the same name is already present. A variable without an id gets the
        next free one; copies keep the id of the original.
        """
        if var.name in self._var_by_name:
            raise ValueError(f"variable {var.name!r} is already in the CSP")
        if var.id < 0:
            var.id = self._next_id
        self._next_id = max(self._next_id, var.id + 1)
        self.vars.append(var)
        self._var_by_name[var.name] = var
        self.index.register(var.name, var.domain)

    def get_var_by_name(self, name :str) ->CSP_Variable | None:
        """
        Return the CSP_Variable with the given name, or None if not found.
        """
        return self._var_by_name.get(name)

    def conflicting_values(self, u :CSP_Variable, t :Timeslot) ->set[Timeslot]:
        """
        Return the values still in the other variables' domains that conflict with value t of u.
        """
        found = set()
        for c in self.index.conflicts[t]:
            v = self._var_by_name.get(self.index.owner[c])
            if v is not None and v is not u and c in v.domain:
                found.add(c)
        return found

    def neighbors(self, u :CSP_Variable) ->list[CSP_Variable]:
        """
        Return all variables that share a constraint with u (sorted by id).
        """
        names = set()
        for t in u.domain:
            for c in self.index.conflicts[t]:
                o = self.index.owner[c]
                v = self._var_by_name.get(o)
                if v is not None and v is not u and c in v.domain:
                    names.add(o)
        return sorted((self._var_by_name[n] for n in names), key=lambda v: v.id)

    def constraint_exists(self, u :CSP_Variable, v :CSP_Variable) ->bool:
        """
        Return True if a constraint exists between u and v, otherwise False.
        """
        if u is v:
            return False
        return any(c in v.domain for t in u.domain for c in self.index.conflicts[t])

    def degree(self, v :CSP_Variable) ->int:
        """
        Return the number of constraints involving v.
        """
        return len(self.neighbors(v))

    def num_constraints(self) ->int:
        """
        Return the total number of constraints in the CSP.
        """
        return sum(self.degree(v) for v in self.vars) // 2

    def copy(self) ->CSP:
        """
        Return an independent copy: new variables (same names and ids, copied domains) sharing
        the conflict index, so nothing is recomputed.
        """
        return self.subproblem(v.name for v in self.vars)

    def subproblem(self, names) ->CSP:
        """
        Return a copy restricted to the variables with the given names (in this CSP's order),
        sharing the conflict index.
        """
        keep = set(names)
        sub = CSP(self.index)
        for v in self.vars:
            if v.name in keep:
                new_var = CSP_Variable(v.name)
                new_var.id = v.id
                new_var.domain = set(v.domain)
                sub.add_var(new_var)
        return sub

    def key(self) ->frozenset:
        """
        Hashable identity of the sub-problem (variable names and domains), e.g. for caching.
        """
        return frozenset((v.name, frozenset(v.domain)) for v in self.vars)

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

        degrees = [self.degree(v) for v in self.vars]
        summ["num_const"] = sum(degrees) // 2
        summ["avg_degree"] = float(np.mean(degrees))
        summ["sd_degree"] = float(np.std(degrees))
        summ["const_dens"] = (
            float(2 * summ["num_const"] / (len(self.vars) ** 2 - len(self.vars)))
            if len(self.vars) > 1 else 0
        )
        return summ
