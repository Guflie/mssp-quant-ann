"""
CSP -> QUBO translator for the satellite scheduling problem.

Translation encoding
--------
One binary variable x[v, t] per (Var, Timeslot) pair:
    x[v, t] := 1  means satellite v is assigned timeslot t.

Two penalty terms are added to the QUBO energy:

    One-hot: enforce the choice of only one timeslot per satellite 

    Conflict: enforce te choice of different timeslots for different satellites 

The two constats penaly_conflict,penalty_onehot are arbitrary constants and help 
to manage the QUBO problem shape.
              
The QUBO is returned as a dict {(i, j): coeff} with i <= j for direct
compatibility with D-Wave simulator.

----------
build_index_map   : assign a unique integer index to every (var_id, Timeslot)
csp_to_qubo       : produce the QUBO dict and index map
decode_sample     : map a binary QUBO sample back to a CSP assignment
"""

from __future__ import annotations
from CSP import CSP, Timeslot



def build_index_map(csp :CSP) ->dict[tuple[int, Timeslot], int]:
    """
    Assign a unique integer QUBO index to every (var_id, Timeslot) pair.
    Ordering is deterministic: variables by id, timeslots by (antenna_id, timeframe).
    """
    index_map :dict[tuple[int, Timeslot], int] = {}
    idx = 0
    for v in sorted(csp.vars, key=lambda v: v.id):
        for t in sorted(v.domain, key=lambda t: (t.antenna_id, t.timeframe)):
            index_map[(v.id, t)] = idx
            idx += 1
    return index_map


def csp_to_qubo(
    csp :CSP,
    penalty_onehot :float = 10.0,   # TODO: make this penalty dependent from the number of timelsots for that variable
    penalty_conflict :float = 1.0,
    assignment_gain :float = 5.0,
) -> tuple[
    dict[tuple[int, int], float], 
    dict[tuple[int, Timeslot], int]
    ]:
    """
    Translate a CSP into a QUBO minimisation problem.

    Parameters
    ----------
    csp              := the CSP instance
    penalty_onehot   := weight of the one-hot constraint penalty
    penalty_conflict := weight of the conflict penalty

    Returns
    -------
    Q         := QUBO dict {(i, j): coeff} with i <= j.
    index_map := {(var_id, Timeslot): qubo_index} — needed to decode samples.
    """
    index_map = build_index_map(csp)
    Q: dict[tuple[int, int], float] = {}

    # add one variable to the QUBO
    def _add(i :int, j :int, val :float) ->None:
        if i > j:
            i, j = j, i
        key = (i, j)
        Q[key] = val

    # adding reward for choosing the assignment
    for v in csp.vars:
        for t in v.domain:
            i = index_map[(v.id, t)]
            _add(i, i, -assignment_gain)

    # adding conflict penalty between same timeslot and different satellite
    for v in csp.vars:
        for u in csp.neighbors(v):
            if u.id <= v.id:
                continue
            for t in v.domain & u.domain:
                i = index_map[(v.id, t)]
                j = index_map[(u.id, t)]
                _add(i, j, penalty_conflict)

    # adding one-hot penalty
    for v in csp.vars:
        domain = list(v.domain)
        for k in range(len(domain)):
            for l in range(k + 1, len(domain)):
                i = index_map[(v.id, domain[k])]
                j = index_map[(v.id, domain[l])]
                _add(i, j, penalty_onehot)

    return Q, index_map

def decode_sample(
    sample :dict[int, int],
    csp :CSP,
    index_map :dict[tuple[int, Timeslot], int],
) ->dict[int, Timeslot | None]:
    """
    Decode a binary QUBO sample into a CSP assignment.

    Returns {var_id: Timeslot} for variables assigned exactly one timeslot,
    or {var_id: None} when the sample assigns zero or multiple timeslots to
    a variable (indicating a constraint violation in the QUBO solution).

    Parameters
    ----------
    sample    : {qubo_index: 0 or 1} as returned by neal
    csp       : the original CSP (used to iterate over variables)
    index_map : as returned by csp_to_qubo
    """
    reverse :dict[int, tuple[int, Timeslot]] = {
        idx :(var_id, t) for (var_id, t), idx in index_map.items()
    }

    assigned: dict[int, list[Timeslot]] = {v.id: [] for v in csp.vars}
    for idx, val in sample.items():
        if val == 1 and idx in reverse:
            var_id, t = reverse[idx]
            assigned[var_id].append(t)

    return {
        var_id: (slots[0] if len(slots) == 1 else None)
        for var_id, slots in assigned.items()
    }
