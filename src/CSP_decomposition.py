"""
Reduction and decomposition algorithms for the satellite scheduling CSP (model in CSP.py).

Public API
----------
forward_check               := fix one variable and prune the conflicting values of its neighbors
                               (exact, modifies domains and constraints in-place)
dominance_reduction         := remove the timeslots dominated by another timeslot of the same
                               satellite (exact, modifies domains and constraints in-place)
split_connected_components  := decompose a CSP into independent sub-problems
ac3                         := arc-consistency preprocessing; no longer used in the pipeline,
                               since it is not exact when satellites may stay unscheduled
"""

from collections import deque
from CSP import CSP, CSP_Variable, Timeslot


def forward_check(csp :CSP, var :CSP_Variable, value :Timeslot | None) ->int:
    """
    Fix var to value (None = satellite left unscheduled) and propagate the choice.

    Every neighbor loses the timeslots that conflict with value; then the constraints
    left without any conflicting pair of values are removed, so the sub-problem can split
    further. Unlike AC-3, this pruning is exact even when satellites may stay unscheduled:
    the removed timeslots are infeasible given the fixed choice.
    Modifies domains and constraints in-place. Returns the number of pruned timeslots.
    """
    if value is not None and value not in var.domain:
        raise ValueError(f"{value.slot_id} is not in the domain of {var.name}")

    affected = csp.neighbors(var)
    var.domain = set() if value is None else {value}

    pruned = 0
    if value is not None:
        for w in affected:
            for t in [t for t in w.domain if Timeslot.overlap(value, t)]:
                w.trim_value(t)
                pruned += 1

    # remove_constraint drops an edge only when no conflicting pair is left
    for w in affected:
        csp.remove_constraint(var, w)
        for x in csp.neighbors(w):
            csp.remove_constraint(w, x)
    return pruned


def dominance_reduction(csp :CSP, slot_weight :dict[str, float]) ->int:
    """
    Remove dominated timeslots (neighbourhood substitutability), repeating until none is left.

    A timeslot a of a satellite is dominated by another timeslot b of the same satellite when
    weight(b) >= weight(a) and every timeslot of other satellites conflicting with b also conflicts
    with a: any schedule using a stays feasible, and is not worse, using b instead.
    Removing a is therefore exact, also when satellites may stay unscheduled.
    Ties (same weight, same conflicts) keep the timeslot with the smallest slot_id.
    Removing timeslots shrinks the conflicts of the other satellites, so new dominations can
    appear: the check is repeated until a full pass removes nothing.
    Modifies domains and constraints in-place. Returns the number of removed timeslots.
    """
    # conflicts[t] := timeslots of other satellites that overlap t
    conflicts :dict[Timeslot, set[Timeslot]] = {t: set() for v in csp.vars for t in v.domain}
    for v in csp.vars:
        for u in csp.neighbors(v):
            if u.id <= v.id:
                continue
            for a in v.domain:
                for b in u.domain:
                    if Timeslot.overlap(a, b):
                        conflicts[a].add(b)
                        conflicts[b].add(a)

    def _dominates(b :Timeslot, a :Timeslot) ->bool:
        wa, wb = slot_weight[a.slot_id], slot_weight[b.slot_id]
        if wb < wa or not conflicts[b] <= conflicts[a]:
            return False
        # on a tie only one of the two is removed
        return wb > wa or conflicts[b] < conflicts[a] or b.slot_id < a.slot_id

    removed_total = 0
    while True:
        removed :set[Timeslot] = set()
        for v in csp.vars:
            for a in v.domain:
                # a dominator removed in this same pass is skipped: whatever removed it also dominates a
                if any(b != a and b not in removed and _dominates(b, a) for b in v.domain):
                    removed.add(a)
        if not removed:
            break
        for v in csp.vars:
            v.domain -= removed
        for t in removed:
            for c in conflicts.pop(t):
                conflicts[c].discard(t)
        removed_total += len(removed)

    # remove_constraint drops an edge only when no conflicting pair is left
    for v in csp.vars:
        for u in csp.neighbors(v):
            csp.remove_constraint(v, u)
    return removed_total


# no longer used
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
