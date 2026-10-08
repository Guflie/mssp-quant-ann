"""
CSP -> QUBO translator for the satellite scheduling problem.

Translation encoding
--------
One binary variable x[t] per Timeslot (each timeslot belongs to exactly one satellite):
    x[t] := 1  means the satellite owning t is assigned timeslot t.
QUBO variables are keyed by slot_id, so the QUBO and its decoded samples no longer need the CSP.

The reward term favours scheduling, each timeslot by its own (normalised) weight.
Two penalty terms are added to the QUBO energy:

    One-hot: a satellite takes at most one timeslot

    Conflict: two satellites cannot take timeslots that overlap on the same antenna
              (Timeslot.overlap, the same rule used to build the CSP constraints)

The penalties must exceed the largest reward a single timeslot can bring (assignment_gain,
since weights are normalised to [0, 1]); otherwise taking both timeslots of a violated
pair lowers the energy and the ground state is not a feasible schedule.

The QUBO is returned as a dict {(i, j): coeff} with i <= j for direct
compatibility with D-Wave simulator.

----------
build_index_map   : assign a unique integer index to every slot_id
csp_to_qubo       : produce the QUBO dict and index map
decode_sample     : map a binary QUBO sample back to the chosen slot_ids
"""

from __future__ import annotations
from CSP import CSP, Timeslot


def build_index_map(csp :CSP) ->dict[str, int]:
    """
    Assign a unique integer QUBO index to every slot_id in the CSP domains.
    Ordering is deterministic: by slot_id.
    """
    slot_ids = sorted(t.slot_id for v in csp.vars for t in v.domain)
    return {slot_id: idx for idx, slot_id in enumerate(slot_ids)}


def csp_to_qubo(
    csp :CSP,
    slot_weight :dict[str, float] | None = None,
    penalty_onehot :float = 10.0,
    penalty_conflict :float = 10.0,
    assignment_gain :float = 5.0,
) -> tuple[
    dict[tuple[int, int], float],
    dict[str, int]
    ]:
    """
    Translate a CSP into a QUBO minimization problem.

    Parameters
    ----------
    csp              := the CSP instance
    slot_weight      := {slot_id: weight}; None gives every timeslot the same weight
    penalty_onehot   := weight of the one-hot constraint penalty (must be > assignment_gain)
    penalty_conflict := weight of the conflict penalty (must be > assignment_gain)
    assignment_gain  := reward of scheduling the heaviest timeslot

    Returns
    -------
    Q         := QUBO dict {(i, j): coeff} with i <= j.
    index_map := {slot_id: qubo_index} — needed to decode samples.
    """
    index_map = build_index_map(csp)
    Q: dict[tuple[int, int], float] = {}

    # add a coefficient to the QUBO (summed, in case the same pair gets more terms)
    def _add(i :int, j :int, val :float) ->None:
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + val

    # find the max contact slot weight to normalize them during QUBO encoding (to avoid explodinf assignments)
    # (fixed) one-hot/conflict penalties regardless of the raw weight scale
    def _weight(t :Timeslot) ->float:
        return 1.0 if slot_weight is None else slot_weight[t.slot_id]
    max_weight = max((_weight(t) for v in csp.vars for t in v.domain), default=0)

    # adding reward for choosing the assignment
    for v in csp.vars:
        for t in v.domain:
            i = index_map[t.slot_id]
            normalized_weight = _weight(t) / max_weight if max_weight > 0 else 1.0
            _add(i, i, -assignment_gain*normalized_weight)

    # adding conflict penalty between overlapping timeslots of different satellites
    for v in csp.vars:
        for u in csp.neighbors(v):
            if u.id <= v.id:
                continue
            for t1 in v.domain:
                for t2 in u.domain:
                    if Timeslot.overlap(t1, t2):
                        _add(index_map[t1.slot_id], index_map[t2.slot_id], penalty_conflict)

    # adding penalty for selecting the same satellite two times in the scheduling
    for v in csp.vars:
        domain = list(v.domain)
        for k in range(len(domain)):
            for l in range(k + 1, len(domain)):
                i = index_map[domain[k].slot_id]
                j = index_map[domain[l].slot_id]
                _add(i, j, penalty_onehot)

    return Q, index_map

def decode_sample(
    sample :dict[int, int],
    index_map :dict[str, int],
) ->list[str]:
    """
    Decode a binary QUBO sample into the list of chosen slot_ids (sorted).

    One-hot and antenna-conflict violations are not checked here: validate the
    returned slots against the slot table (satellite, antenna, times).

    Parameters
    ----------
    sample    : {qubo_index: 0 or 1} as returned by neal
    index_map : as returned by csp_to_qubo
    """
    return sorted(slot_id for slot_id, idx in index_map.items() if sample.get(idx) == 1)
