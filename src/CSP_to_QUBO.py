"""
CSP -> QUBO translator for the satellite scheduling problem.

Translation encoding
--------
The reward term favours scheduling, each timeslot by its own (normalised) weight.
Every satellite takes at most one timeslot, and two satellites cannot take timeslots that
overlap on the same antenna (Timeslot.overlap, the same rule used to build the CSP constraints).
Two encodings of the satellite variables are available:

    One-hot: one binary variable x[t] per Timeslot (x[t] = 1 means the satellite takes t).
             "At most one" is a penalty on every pair of timeslots of the same satellite,
             i.e. d(d-1)/2 couplers for a domain of size d.

    Domain-wall (default): a chain of d binary variables b_1 .. b_d per satellite, read as
             1..1 0..0 ; the position of the last 1 is the chosen timeslot (all 0 = unscheduled).
             The timeslot indicator is x_k = b_k - b_{k+1} (b_{d+1} = 0), and a penalty on every
             "0 followed by 1" keeps the chain valid: d-1 couplers per satellite. Each conflict
             x_k * y_j expands into up to 4 couplers, partly cancelling when the conflicting
             timeslots are adjacent in the chain (timeslots are ordered by antenna, then start).
             See N. Chancellor, Quantum Sci. Technol. 4, 045004 (2019).

The conflict (and one-hot) penalties must exceed the largest reward a single timeslot can bring
(assignment_gain, since weights are normalised to [0, 1]); otherwise taking both timeslots of a
violated pair lowers the energy and the ground state is not a feasible schedule.
In a broken domain-wall chain some x_k = -1, and a -1 timeslot paired with a conflicting chosen
timeslot lowers the energy by penalty_conflict. The worst case, all the conflicts of a timeslot
chosen, gives the provably safe chain penalty assignment_gain + penalty_conflict * (max number of
conflicts of one timeslot), but it widens the coefficient range far beyond the rewards, which the
QPU rescaling then squeezes towards the noise level. By default (penalty_chain None) each satellite
gets instead assignment_gain + penalty_conflict * alpha + 1, with alpha the largest number of
conflicts of one of its timeslots that can be chosen together (they conflict among themselves):
the most a chain break can realistically gain. This level is validated empirically, not proven:
check decoded samples for broken chains.

The QUBO is returned as a dict {(i, j): coeff} with i <= j for direct
compatibility with D-Wave simulator.

----------
csp_to_qubo       : produce the QUBO dict and index map, with the chosen encoding
one_hot_qubo      : one-hot encoding
domain_wall_qubo  : domain-wall encoding
decode_sample     : map a binary QUBO sample back to the chosen slot_ids (both encodings)
"""

from __future__ import annotations
from typing import Literal
from CSP import CSP, Timeslot

# index_map := {slot_id: (bit, next_bit)}: the timeslot indicator is sample[bit] - sample[next_bit],
# with next_bit None when there is no next bit (always for one-hot, the chain end for domain-wall)
IndexMap = dict[str, tuple[int, int | None]]
Qubo = dict[tuple[int, int], float]


def csp_to_qubo(
    csp :CSP,
    slot_weight :dict[str, float] | None = None,
    encoding :Literal["domain_wall", "one_hot"] = "domain_wall",
    penalty_onehot :float = 10.0,
    penalty_conflict :float = 10.0,
    penalty_chain :float | None = None,
    assignment_gain :float = 5.0,
) -> tuple[Qubo, IndexMap]:
    """
    Translate a CSP into a QUBO minimization problem.

    Parameters
    ----------
    csp              := the CSP instance
    slot_weight      := {slot_id: weight}; None gives every timeslot the same weight
    encoding         := "domain_wall" (default) or "one_hot"
    penalty_onehot   := one-hot only: same-satellite pair penalty (must be > assignment_gain)
    penalty_conflict := conflict penalty (must be > assignment_gain)
    penalty_chain    := domain-wall only: one chain penalty for every satellite; None computes a
                        per-satellite value from its conflicts (see module doc)
    assignment_gain  := reward of scheduling the heaviest timeslot

    Returns
    -------
    Q         := QUBO dict {(i, j): coeff} with i <= j.
    index_map := {slot_id: (bit, next_bit)} — needed to decode samples.
    """
    if encoding == "domain_wall":
        return domain_wall_qubo(csp, slot_weight, penalty_conflict, penalty_chain, assignment_gain)
    if encoding == "one_hot":
        return one_hot_qubo(csp, slot_weight, penalty_onehot, penalty_conflict, assignment_gain)
    raise ValueError(f"unknown encoding: {encoding!r}")


def _normalized_weights(csp :CSP, slot_weight :dict[str, float] | None) ->dict[str, float]:
    """
    Slot weights scaled to [0, 1] by the largest one, so the reward stays comparable to the
    fixed penalties regardless of the raw weight scale (all 1 when slot_weight is None).
    """
    slot_ids = [t.slot_id for v in csp.vars for t in v.domain]
    if slot_weight is None:
        return {s: 1.0 for s in slot_ids}
    max_weight = max((slot_weight[s] for s in slot_ids), default=0)
    return {s: slot_weight[s] / max_weight if max_weight > 0 else 1.0 for s in slot_ids}


def _conflict_pairs(csp :CSP) ->list[tuple[Timeslot, Timeslot]]:
    """
    All pairs of timeslots of different satellites that overlap on the same antenna.
    """
    pairs = []
    for v in csp.vars:
        for u in csp.neighbors(v):
            if u.id <= v.id:
                continue
            for t1 in v.domain:
                for t2 in u.domain:
                    if Timeslot.overlap(t1, t2):
                        pairs.append((t1, t2))
    return pairs


def _add(Q :Qubo, i :int, j :int, val :float) ->None:
    """
    Add a coefficient to the QUBO (summed, as several terms may hit the same pair).
    """
    if i > j:
        i, j = j, i
    Q[(i, j)] = Q.get((i, j), 0.0) + val


def _drop_zeros(Q :Qubo) ->Qubo:
    """
    Remove the coefficients cancelled out by opposite terms (they would be empty couplers).
    """
    return {key: val for key, val in Q.items() if abs(val) > 1e-12}


def _chain_penalties(
    csp :CSP,
    conflicts :list[tuple[Timeslot, Timeslot]],
    penalty_conflict :float,
    assignment_gain :float,
) ->dict[str, float]:
    """
    Per-satellite chain penalty: assignment_gain + penalty_conflict * alpha + 1, where alpha is the
    largest number of conflicts of one of its timeslots that can be chosen together (pairwise not
    conflicting): the most a chain break can realistically turn into a reward.
    """
    neighbours :dict[Timeslot, list[Timeslot]] = {}
    for t1, t2 in conflicts:
        neighbours.setdefault(t1, []).append(t2)
        neighbours.setdefault(t2, []).append(t1)

    # the conflicts of a timeslot are on its antenna: an interval graph, where the greedy
    # choice by earliest end (reconfiguration included) gives the largest non-conflicting set
    def _alpha(slots :list[Timeslot]) ->int:
        chosen :list[Timeslot] = []
        for t in sorted(slots, key=lambda t: t.end_s + t.reconfiguration_s):
            if not chosen or not Timeslot.overlap(chosen[-1], t):
                chosen.append(t)
        return len(chosen)

    return {
        v.name: assignment_gain + penalty_conflict * max((_alpha(neighbours.get(t, [])) for t in v.domain), default=0) + 1.0
        for v in csp.vars
    }


def one_hot_qubo(
    csp :CSP,
    slot_weight :dict[str, float] | None = None,
    penalty_onehot :float = 10.0,
    penalty_conflict :float = 10.0,
    assignment_gain :float = 5.0,
) -> tuple[Qubo, IndexMap]:
    """
    One-hot encoding: one bit per timeslot, ordered by slot_id.
    """
    slot_ids = sorted(t.slot_id for v in csp.vars for t in v.domain)
    bit = {slot_id: idx for idx, slot_id in enumerate(slot_ids)}
    weight = _normalized_weights(csp, slot_weight)
    Q :Qubo = {}

    # adding reward for choosing the assignment
    for slot_id, i in bit.items():
        _add(Q, i, i, -assignment_gain * weight[slot_id])

    # adding conflict penalty between overlapping timeslots of different satellites
    for t1, t2 in _conflict_pairs(csp):
        _add(Q, bit[t1.slot_id], bit[t2.slot_id], penalty_conflict)

    # adding penalty for selecting the same satellite two times in the scheduling
    for v in csp.vars:
        domain = list(v.domain)
        for k in range(len(domain)):
            for l in range(k + 1, len(domain)):
                _add(Q, bit[domain[k].slot_id], bit[domain[l].slot_id], penalty_onehot)

    return _drop_zeros(Q), {slot_id: (i, None) for slot_id, i in bit.items()}


def domain_wall_qubo(
    csp :CSP,
    slot_weight :dict[str, float] | None = None,
    penalty_conflict :float = 10.0,
    penalty_chain :float | None = None,
    assignment_gain :float = 5.0,
) -> tuple[Qubo, IndexMap]:
    """
    Domain-wall encoding: one chain of bits per satellite (satellites by name, timeslots in the
    chain by antenna then start), bit k standing for "the wall is at or after timeslot k".
    """
    index_map :IndexMap = {}
    chains :list[tuple[str, list[int]]] = []     # (satellite name, chain bits)
    idx = 0
    for v in sorted(csp.vars, key=lambda v: v.name):
        order = sorted(v.domain, key=lambda t: (t.antenna_id, t.start_s, t.slot_id))
        bits = list(range(idx, idx + len(order)))
        for k, t in enumerate(order):
            index_map[t.slot_id] = (bits[k], bits[k + 1] if k + 1 < len(bits) else None)
        chains.append((v.name, bits))
        idx += len(order)

    # timeslot indicator x = b_k - b_{k+1}, as (bit, sign) terms
    def _indicator(slot_id :str) ->list[tuple[int, int]]:
        b, nxt = index_map[slot_id]
        return [(b, 1)] if nxt is None else [(b, 1), (nxt, -1)]

    conflicts = _conflict_pairs(csp)
    if penalty_chain is None:
        chain_penalty = _chain_penalties(csp, conflicts, penalty_conflict, assignment_gain)
    else:
        chain_penalty = {v.name: penalty_chain for v in csp.vars}

    weight = _normalized_weights(csp, slot_weight)
    Q :Qubo = {}

    # adding reward for choosing the assignment: -gain * w_k * (b_k - b_{k+1})
    for slot_id in index_map:
        for b, sign in _indicator(slot_id):
            _add(Q, b, b, -assignment_gain * weight[slot_id] * sign)

    # adding chain penalty on every "0 followed by 1": P * b_k * (1 - b_{k-1})
    for name, bits in chains:
        for prev, cur in zip(bits, bits[1:]):
            _add(Q, cur, cur, chain_penalty[name])
            _add(Q, prev, cur, -chain_penalty[name])

    # adding conflict penalty: P * x_k * y_j, expanded on the chain bits
    for t1, t2 in conflicts:
        for b1, s1 in _indicator(t1.slot_id):
            for b2, s2 in _indicator(t2.slot_id):
                _add(Q, b1, b2, penalty_conflict * s1 * s2)

    return _drop_zeros(Q), index_map


def decode_sample(
    sample :dict[int, int],
    index_map :IndexMap,
) ->list[str]:
    """
    Decode a binary QUBO sample into the list of chosen slot_ids (sorted), for both encodings:
    a timeslot is chosen when its indicator sample[bit] - sample[next_bit] equals 1.

    One-hot violations, broken domain-wall chains and antenna conflicts are not checked here:
    validate the returned slots against the slot table (satellite, antenna, times).

    Parameters
    ----------
    sample    : {qubo_index: 0 or 1} as returned by neal
    index_map : as returned by csp_to_qubo
    """
    def _bit(i :int | None) ->int:
        return 0 if i is None else sample.get(i, 0)
    return sorted(slot_id for slot_id, (b, nxt) in index_map.items() if _bit(b) - _bit(nxt) == 1)
