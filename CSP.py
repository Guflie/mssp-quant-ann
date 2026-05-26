from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass(frozen=True)
class Timeslot:
    antenna_id :int
    timeframe :int


class Var:
    """ 
        Implementation of a Constraint Satisfaction Problem (CSP) variable.
    
        ===== Attributes =====
        - domain: list of all possible values of the variable
        - id: variable id
    """
    id :int
    domain: set[Timeslot]

    def __init__(self, id :int) -> None:
        self.id = id
        self.domain = set()

    def add_value(self, val :Timeslot) ->None:
        """ Add the value to the variable domain """
        self.domain.add(val)

    def load_values(self, vals :set[Timeslot]) ->None:
        """ Bulk load of values in the variable domain """
        self.domain = self.domain.union(vals)

    def trim_value(self, val :Timeslot) ->None:
        """ Remove the value from the variable domain """
        self.domain.remove(val)


class CSP: # for now focus on the Diff forumulation, in which each constraint means a conflict bewteen satellites

    def __init__(self, max_vars : int):
        self.max_vars = max_vars
        self.vars : list[Var] = []
        self.const = np.zeros([max_vars, max_vars], dtype=bool) # upper-triangula matrix  containing variable constraints (handled in that way)


    def add_var(self, var : Var) ->None:
        if len(self.vars) < self.max_vars:
            self.vars.append(var)

    def get_var_by_id(self, id :int) ->Var:
        for v in self.vars:
            if v.id == id: return v
        return Var(id=-1)

    @staticmethod
    def _order_params(u : Var, v : Var) ->list[Var]:
        """ Re-order vars based on their Id, used to enforce triangual matrix """

        min_var :Var = min(u, v, key=lambda v: v.id) 
        max_var :Var = max(u, v, key=lambda v: v.id)
        return [min_var, max_var]

    def add_constraint(self, u: Var, v: Var) ->None:
        """
            Add a constraint between vars if it's legit
        """
        if u.id == v.id: return
        if v.domain & u.domain:
            min_v, max_v = self._order_params(u, v)
            self.const[min_v.id][max_v.id] = True
    
    def remove_constraint(self, u: Var, v: Var) ->None:
        """
            Remove constraint between vars
        """
        if u.id == v.id: return
        min_v, max_v = self._order_params(u, v)
        self.const[min_v.id][max_v.id] = False

    def update_constrains(self) ->None:
        """ 
            build up constraints among all the variables, making sure everything is ok
        """
        for v in self.vars:
            for u in self.vars:
                if v.id != u.id and v.domain & u.domain:
                    min_v, max_v = self._order_params(u, v)
                    self.const[min_v.id][max_v.id] = True

    def constraint_exists(self, u: Var, v: Var) ->bool:
        """
            Just check if the constraint exists or not
        """
        if u.id == v.id: return False # short-ciruit exit, avoid useless computation
        min_v, max_v = self._order_params(u, v)
        return self.const[min_v.id][max_v.id]

    def neighbors(self, u: Var) ->list[Var]:
        return [ v for v in self.vars if u != v and self.constraint_exists(u,v) ]

    def degree(self, v: Var) -> int:
        return sum(self.const[v.id]) + sum(self.const[:,v.id])

    # informative outoput methods

    def num_constraints(self) -> int:
        return int(self.const.sum())

    def compute_summary(self):
        summ :dict[str, float] = dict()
        summ["num_vars"]  =  len(self.vars)
        summ["avg_domain_size"] = sum([len(v.domain) for v in self.vars])/len(self.vars)\
            if len(self.vars) > 0 else 0
        summ["min_domain_size"] = min([len(v.domain) for v in self.vars])\
            if len(self.vars) > 0 else 0
        summ["max_domain_size"] = max([len(v.domain) for v in self.vars])\
            if len(self.vars) > 0 else 0
        summ["num_const"] = self.num_constraints()
        summ["avg_degree"] = float(np.mean([self.degree(v) for v in self.vars]))
        summ["const_dens"] = float(2*np.sum(self.const)/(len(self.vars)**2-len(self.vars)))\
            if len(self.vars) > 1 else 0
        return summ

from collections import deque

def ac3(csp : CSP, debug=False) -> bool: 
    """
    Enforce arc consistency (AC-3) on the CSP.

    For the Diff formulation, a color in domain[u] loses support only when
    domain[v] contains no value different from it (i.e. domain[v] == {color}).
    Modifies Var.domain in-place. Returns False if any domain is wiped out.
    """
    queue = deque(
        (u, v)
        for u in csp.vars
        for v in csp.vars
        if u.id != v.id and csp.constraint_exists(u, v)
    )
    while queue:
        u, v = queue.popleft()
        if _revise(u, v):
            if not u.domain: # = if the domain is empty
               return False

            pruned = prune_disjoint_constraints(csp, u)        
            if debug: print(f"pruned {pruned} constraints")

            for w in csp.vars:
                if w.id != u.id and w.id != v.id and csp.constraint_exists(w, u):
                    queue.append((w, u))
    return True

def _revise(u: Var, v: Var) -> bool:
    """
      Remove colors from u.domain that have no support in v.domain 
    """
    revised = False
    for color in list(u.domain):
        if not any(c != color for c in v.domain):
            u.domain.remove(color)
            revised = True
    return revised


def prune_disjoint_constraints(csp: CSP, var: Var) ->int:
    """
    Remove constraints between var and any neighbor whose domain is disjoint from var's.

    In the Diff formulation, disjoint domains guarantee the constraint is
    never violated, so the arc is redundant. Returns the number of pruned constraints.
    """
    pruned = 0
    for w in csp.vars: # optimize this, you can remove some iters
        if w.id != var.id and csp.constraint_exists(var, w) and not var.domain & w.domain:
            csp.remove_constraint(var, w)
            pruned += 1
    return pruned

def split_connected_components(csp: CSP) -> list[CSP]:
    """
    Decompose a CSP into independent sub-CSPs, one per connected component
    of the constraint graph. Variables with no constraints form singleton components.

    Each sub-CSP gets deep-copied Var objects (same ids, copied domains) so
    sub-problems can be solved independently without domain mutations leaking.
    The constraint matrix size is preserved (max_vars) so variable ids remain
    valid as matrix indices.
    """
    visited: set[int] = set()
    components: list[list[Var]] = []

    for v in csp.vars:
        if v.id in visited:
            continue
        
        component: list[Var] = []
        queue = deque([v])
        visited.add(v.id)
        while queue:
            var = queue.popleft()
            component.append(var)
            for neighbor in csp.neighbors(var):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append(neighbor)
        components.append(component)

    result: list[CSP] = []
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