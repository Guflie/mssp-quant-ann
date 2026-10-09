"""
Reduction and decomposition algorithms for the satellite scheduling CSP (model in CSP.py).

All functions work on CSP objects: the conflicts between slots live in the CSP's shared
ConflictIndex, so reductions only remove values from the domains and sub-problems are cheap
copies (CSP.copy, CSP.subproblem) that never recompute overlaps.

Public API
----------
forward_check               := fix one variable and prune the conflicting values of its neighbors
                               (exact, modifies the domains in-place)
dominance_reduction         := remove the timeslots dominated by another timeslot of the same
                               satellite (exact, modifies the domains in-place)
split_connected_components  := decompose a CSP into independent sub-problems
decompose_csp               := exact decomposition tree into QPU-sized terminal sub-problems
                               (CSP counterpart of the MWIS separator decomposition), with budgets
ac3                         := arc-consistency preprocessing; no longer used in the pipeline,
                               since it is not exact when satellites may stay unscheduled
"""

import math
from collections import deque
from itertools import combinations
from typing import Any

import networkx as nx

from CSP import CSP, CSP_Variable, Timeslot
from CSP_to_QUBO import csp_to_qubo


def forward_check(csp :CSP, var :CSP_Variable, value :Timeslot | None) ->int:
    """
    Fix var to value (None = satellite left unscheduled) and propagate the choice.

    Every other variable loses the timeslots that conflict with value; the constraints left
    without any conflicting pair of values disappear with them (constraints are read from the
    domains), so the sub-problem can split further. Unlike AC-3, this pruning is exact even when
    satellites may stay unscheduled: the removed timeslots are infeasible given the fixed choice.
    Modifies the domains in-place. Returns the number of pruned timeslots.
    """
    if value is not None and value not in var.domain:
        raise ValueError(f"{value.slot_id} is not in the domain of {var.name}")

    blocked = set() if value is None else csp.conflicting_values(var, value)
    var.domain = set() if value is None else {value}
    for c in blocked:
        csp.get_var_by_name(csp.index.owner[c]).trim_value(c)
    return len(blocked)


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
    Modifies the domains in-place. Returns the number of removed timeslots.
    """
    # a conflict with a timeslot that is no longer in any domain cannot be violated, so it must
    # not count against a timeslot: live[t] = conflicts of t restricted to the current domains,
    # shrinking as slots are removed
    live = {t: csp.conflicting_values(v, t) for v in csp.vars for t in v.domain}

    def _dominates(b :Timeslot, a :Timeslot) ->bool:
        """True if b (same satellite as a) can replace a in any schedule without loss."""
        wa, wb = slot_weight[a.slot_id], slot_weight[b.slot_id]
        # b must be worth at least as much, and conflict with nothing a does not already conflict with
        if wb < wa or not live[b] <= live[a]:
            return False
        # strictly better in weight or in conflicts -> b dominates a.
        # On a tie (same weight, same conflicts) a and b would dominate each other and both would
        # be removed: the slot_id tie-break makes the relation asymmetric, so only one is removed
        return wb > wa or live[b] < live[a] or b.slot_id < a.slot_id

    removed_total = 0
    while True:
        # one pass: every decision is taken on the same snapshot of live, updated only at the end
        removed :set[Timeslot] = set()
        for v in csp.vars:
            # only timeslots of the same satellite are compared: a schedule using a can switch to b
            for a in v.domain:
                # a dominator removed in this same pass is skipped: whatever removed it also dominates a
                if any(b != a and b not in removed and _dominates(b, a) for b in v.domain):
                    removed.add(a)

        # fixpoint: nothing dominated is left
        if not removed:
            return removed_total

        for v in csp.vars:
            v.domain -= removed
        # drop the removed timeslots from the conflict sets: the other satellites now have fewer
        # conflicts, so new dominations can appear in the next pass.
        # Conflicts are symmetric, so this also removes t from live[c] before c (if removed too) is popped
        for t in removed:
            for c in live.pop(t):
                live[c].discard(t)
        removed_total += len(removed)


# no longer used
def ac3(csp :CSP, debug=False, allow_wipeout=False) ->bool:
    """
    Enforce arc consistency (AC-3) on the CSP.

    A timeslot in domain[u] loses support when it conflicts with every timeslot of a neighbour v.
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
    # -----------------------------------------------------------------------------------------------

    # constraints are read from the domains: once a pair of variables has no conflicting values
    # left, constraint_exists is False and the arc is no longer propagated
    queue = deque(
        (u, v)
        for u in csp.vars
        for v in csp.vars
        if u is not v and csp.constraint_exists(u, v)
    )
    feasible = True
    while queue:
        u, v = queue.popleft()
        if _revise(u, v):
            if not u.domain:
                if not allow_wipeout: return False
                feasible = False
            if debug: print(f"revised {u.name} against {v.name}") # TODO: switch this to log instead of bare print

            for w in csp.vars:
                if w is not u and w is not v and csp.constraint_exists(w, u):
                    queue.append((w, u))
    return feasible


def _adjacency(csp :CSP) ->dict[str, set[str]]:
    """
    Returns the adjacency graph for satellites: satellites are linked when they have a
    conflicting pair of timeslots (the CSP constraint graph, keyed by satellite name).
    """
    return {v.name: {u.name for u in csp.neighbors(v)} for v in csp.vars}


def _pieces(adjacency :dict[str, set[str]], removed :set[str] = set()) ->list[set[str]]:
    """
    Connected components of the satellite graph once the removed satellites are taken out.
    """
    seen = set(removed)
    pieces = []
    for start in adjacency:
        if start in seen:
            continue
        piece, stack = {start}, [start]
        seen.add(start)
        while stack:
            for nxt in adjacency[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    piece.add(nxt)
                    stack.append(nxt)
        pieces.append(piece)
    return pieces


def split_connected_components(csp :CSP) ->list[CSP]:
    """
    Decompose a CSP into independent sub-CSPs, one per connected component of the constraint graph,
    largest first (in timeslots, ties broken by the sorted satellite names).

    Each sub-CSP gets copied CSP_Variable objects (same names and ids, copied domains) sharing the
    conflict index, so sub-problems can be solved independently without mutual influences.
    """
    pieces = _pieces(_adjacency(csp))
    size = {v.name: len(v.domain) for v in csp.vars}
    pieces.sort(key=lambda p: (-sum(size[s] for s in p), sorted(p)))
    return [csp.subproblem(piece) for piece in pieces]


# ------------------------------------------------------------------------------------------------
# Exact decomposition tree (CSP counterpart of the MWIS separator decomposition)
# ------------------------------------------------------------------------------------------------

def exact_preprocess(csp :CSP, slot_weight :dict[str, float]) ->dict[str, Any]:
    """
    Exact reductions applied at every tree node: dominance, then every satellite none of whose
    timeslots conflicts with the rest takes its heaviest timeslot (CSP counterpart of the MWIS
    isolated nodes). Modifies csp's domains; the returned sub-problem leaves out the forced
    satellites and the satellites left with an empty domain.
    """
    dominated = dominance_reduction(csp, slot_weight)
    forced = {
        v.name: max(v.domain, key=lambda t: (slot_weight[t.slot_id], t.slot_id))
        for v in csp.vars
        if v.domain and not csp.neighbors(v)
    }
    return {
        "csp": csp.subproblem(v.name for v in csp.vars if v.domain and v.name not in forced),
        "forced_selected": {s: t.slot_id for s, t in forced.items()},
        "forced_objective": sum(slot_weight[t.slot_id] for t in forced.values()),
        "dominated_slots": dominated,
    }


def qubo_size(csp :CSP, encoding :str) ->tuple[int, int]:
    """
    (qubits, couplers) of the QUBO of a sub-problem with the given encoding.
    """
    Q, index_map = csp_to_qubo(csp, encoding=encoding)
    return len(index_map), sum(1 for i, j in Q if i != j)


def separator_states(csp :CSP, separator :tuple[str, ...]) ->list[dict[str, Timeslot | None]]:
    """
    Every assignment of the separator satellites (a timeslot or None = unscheduled) whose chosen
    timeslots do not conflict with each other: one branch each.
    """
    states :list[dict[str, Timeslot | None]] = [{}]
    for s in separator:
        options :list[Timeslot | None] = [*sorted(csp.get_var_by_name(s).domain, key=lambda t: t.slot_id), None]
        states = [
            {**state, s: t}
            for state in states
            for t in options
            if t is None or not any(u is not None and u in csp.index.conflicts[t] for u in state.values())
        ]
    return states


def find_separator(
    csp :CSP, adjacency :dict[str, set[str]], max_separator_size :int, max_candidate_pool :int, max_states :int,
) ->dict[str, Any] | None:
    """
    Pick the satellites to branch on, ranking every candidate cut by how small the sub-problems
    left after removing it are: the largest piece first (in timeslots, i.e. QUBO variables), then
    the balance of the split (sum of squared piece sizes), then the number of branches it creates
    (product of domain sizes + 1) and its size. All candidates compete on the same ranking:
    articulation points, every pair of satellites (graphs up to 200 satellites) and cuts of up to
    max_separator_size satellites among the max_candidate_pool highest-degree ones.
    The best-ranked cut whose feasible states fit max_states is returned with its states;
    None when no affordable candidate splits the graph.
    """
    size = {v.name: len(v.domain) for v in csp.vars}

    candidates :dict[tuple[str, ...], str] = {}
    graph = nx.Graph([(s, o) for s, nbrs in adjacency.items() for o in nbrs])
    graph.add_nodes_from(adjacency)
    for a in nx.articulation_points(graph):
        candidates[(a,)] = "articulation point"
    satellites = sorted(adjacency)
    if max_separator_size >= 2 and len(satellites) <= 200:
        for sep in combinations(satellites, 2):
            candidates.setdefault(sep, "separator pair")
    pool = sorted(satellites, key=lambda s: (-len(adjacency[s]), s))[:max_candidate_pool]
    for k in range(2, max_separator_size + 1):
        for sep in combinations(sorted(pool), k):
            candidates.setdefault(sep, f"degree-pool separator size {k}")

    ranked = []
    for sep, method in candidates.items():
        pieces = [sum(size[s] for s in piece) for piece in _pieces(adjacency, set(sep))]
        if len(pieces) <= 1:
            continue
        score = (max(pieces), sum(p * p for p in pieces), math.prod(size[s] + 1 for s in sep),
                 len(sep), tuple(sorted(sep)))
        ranked.append((score, sep, method))

    # the best cut is useless if its branches exceed the budget: take the best affordable one
    # (score[2] bounds the states from above; the exact count drops mutually conflicting choices)
    for score, sep, method in sorted(ranked):
        if score[2] > 50 * max_states:
            continue
        states = separator_states(csp, sep)
        if len(states) <= max_states:
            return {"separator": sep, "score": score, "method": method, "states": states}
    return None


def _build_node(csp :CSP, component_id :str, context :dict[str, Any], depth :int) ->dict[str, Any]:
    """
    Build one decomposition-tree node: exact preprocessing, then an independent sum, a terminal,
    a separator branching or an incomplete leaf when a budget is exceeded.

    Parameters
    ----------
    csp          := the sub-problem; owned by this node (its domains are reduced in-place)
    component_id := prefix of the node id
    context      := shared settings, budgets and mutable state (see decompose_csp)
    depth        := level of the node in the tree (the root is 0)

    Returns
    -------
    The node dict. Every node records its preprocessing (forced_selected, forced_objective,
    dominated_slots) and its node_type; leaves (terminal, incomplete) also record their remaining
    domains, independent_sum nodes their children, separator nodes one entry per branch with the
    separator assignment, its objective and the children left after forward checking.
    """
    # reuse the result if this exact sub-problem was already decomposed
    # (different branches often leave the same piece untouched)
    cache_key = csp.key()
    if cache_key in context["cache"]:
        context["cache_hits"] += 1
        return context["cache"][cache_key]

    node_number = context["next_node_number"]
    context["next_node_number"] += 1

    # ---------- exact reductions first: every decision below is taken on the reduced sub-problem
    input_qubits = sum(len(v.domain) for v in csp.vars)
    pre = exact_preprocess(csp, context["slot_weight"])
    reduced :CSP = pre["csp"]
    node :dict[str, Any] = {
        "decomposition_node_id": f"{component_id}_dnode_{node_number:04d}",
        "original_component_id": component_id,
        "depth": depth,
        "input_qubits": input_qubits,                                  # size before the preprocessing
        "qubits": sum(len(v.domain) for v in reduced.vars),            # size after it
        "forced_selected": pre["forced_selected"],                     # {satellite: slot_id} decided exactly
        "forced_objective": pre["forced_objective"],
        "dominated_slots": pre["dominated_slots"],
        "children": [],
        "branches": [],
    }
    # register the node before recursing so repeated sub-problems hit the cache
    context["cache"][cache_key] = node
    context["nodes"].append(node)

    # close the node as a leaf, keeping its remaining domains for the solver (or the fallback)
    def _leaf(node_type :str, reason :str) ->dict[str, Any]:
        node.update({"node_type": node_type, "reason": reason,
                     "domains": {v.name: sorted(t.slot_id for t in v.domain) for v in reduced.vars}})
        return node

    # ---------- nothing left after preprocessing: the forced choices are the optimum of this node
    if not reduced.vars:
        return _leaf("terminal", "resolved by exact preprocessing")


    # ----------- check for budget limits: stop here, the node needs a heuristic fallback
    if context["next_node_number"] > context["max_decomposition_nodes"]:
        return _leaf("incomplete", "decomposition node budget exceeded")
    if depth > context["max_recursion_depth"]:
        return _leaf("incomplete", "recursion depth exceeded")

    # split a sub-problem into its independent pieces (largest first) and build one child each
    # (split_connected_components returns them sorted by size, ties broken by the satellite names)
    def _children(sub :CSP) ->list[dict[str, Any]]:
        return [_build_node(piece, component_id, context, depth + 1) for piece in split_connected_components(sub)]

    # --------- check for components already disconnected (e.g. after dominance): solve each piece independently and sum
    adjacency = _adjacency(reduced)
    if len(_pieces(adjacency)) > 1:
        node["node_type"] = "independent_sum"
        node["children"] = _children(reduced)
        return node

    # small enough: this leaf is sent to the QPU
    # (the QUBO is built only when the qubits fit, to count its couplers)
    if node["qubits"] <= context["max_qubits"]:
        _, couplers = qubo_size(reduced, context["encoding"])
        node["couplers"] = couplers
        if couplers <= context["max_couplers"]:
            return _leaf("terminal", "fits QPU limits")

    # too big and connected: branch on the affordable separator leaving the smallest sub-problems
    remaining = context["max_separator_states"] - context["separator_state_count"]
    # every cut has at least 2 states (a timeslot or none): skip the search once the budget is spent
    found = None if remaining < 2 else find_separator(
        reduced, adjacency, context["max_separator_size"], context["max_candidate_pool"], remaining)
    if found is None:
        return _leaf("incomplete", "no separator within the remaining separator-state budget")
    separator, states = found["separator"], found["states"]
    context["separator_state_count"] += len(states)
    node.update({"node_type": "separator", "separator": separator,
                 "separator_method": found["method"], "separator_score": found["score"]})

    # each branch fixes the separator satellites and forward-checks their choice:
    # every timeslot conflicting with a chosen one is removed from the other satellites,
    # the separator satellites leave the sub-problem (decided by the branch), and satellites
    # left with no timeslot are dropped (they can only stay unscheduled in this branch)
    for state_index, state in enumerate(states):
        branch = reduced.copy()
        for s, t in state.items():
            forward_check(branch, branch.get_var_by_name(s), t)
        rest = branch.subproblem(v.name for v in branch.vars if v.name not in state and v.domain)
        node["branches"].append({
            "state_index": state_index,
            "assignment": {s: (None if t is None else t.slot_id) for s, t in state.items()},     # None = unscheduled
            "state_objective": sum(context["slot_weight"][t.slot_id] for t in state.values() if t is not None),
            "children": _children(rest),        # what is left may split into independent pieces
        })
    # the node's optimum is the best branch: state_objective + the optima of its children
    return node


def decompose_csp(
    csp :CSP,
    slot_weight :dict[str, float],
    component_id :str,
    max_qubits :int,
    max_couplers :int,
    encoding :str = "domain_wall",
    max_separator_size :int = 3,
    max_candidate_pool :int = 14,
    max_separator_states :int = 10_000,
    max_recursion_depth :int = 12,
    max_decomposition_nodes :int = 10_000,
) ->dict[str, Any]:
    """
    Exact decomposition of one CSP component into QPU-sized terminal sub-problems.

    Mirrors the MWIS separator decomposition on the satellite constraint graph. Every tree node
    applies the exact preprocessing (dominance, unconstrained satellites take their best timeslot)
    and then becomes:
        independent_sum := the satellites split into independent pieces, solved separately and summed
        terminal        := the QUBO of the chosen encoding fits max_qubits / max_couplers
        separator       := one branch per feasible state of the separator satellites (each takes
                           one of its timeslots or none); the choice is forward-checked and the
                           rest is decomposed recursively
        incomplete      := a budget was exceeded; needs a heuristic fallback
    The separator is the cut leaving the smallest sub-problems (see find_separator). Considering
    every state preserves the optimum when the decomposition completes. Budgets are counted per
    call, so decompose each connected component separately (as the MWIS pipeline does).

    Parameters
    ----------
    csp                     := the CSP component to decompose (not modified)
    slot_weight             := {slot_id: weight}, used by dominance and by the branch objectives
    component_id            := prefix of the tree node ids
    max_qubits              := QPU limit on the QUBO variables of a terminal
    max_couplers            := QPU limit on the QUBO couplers of a terminal
    encoding                := QUBO encoding the terminals are sized with ("domain_wall" or "one_hot")
    max_separator_size      := largest number of satellites in one separator
    max_candidate_pool      := highest-degree satellites the larger separators are drawn from
    max_separator_states    := budget on the branches created over the whole tree
    max_recursion_depth     := deepest tree level before a node is left incomplete
    max_decomposition_nodes := budget on the tree nodes

    Returns
    -------
    A dict with the tree root, the list of all its nodes, the terminal and incomplete leaves,
    the status ("complete" when no leaf is incomplete, i.e. the optimum is preserved) and
    usage statistics (separator states spent, cache hits, depth reached).
    """

    context = {
        # problem data, shared by every node (the conflicts live in the CSP's shared index)
        "slot_weight": slot_weight,
        "encoding": encoding,
        # terminal size limits and separator search settings
        "max_qubits": max_qubits,
        "max_couplers": max_couplers,
        "max_separator_size": max_separator_size,
        "max_candidate_pool": max_candidate_pool,
        # budgets: exceeding one turns the current node into an incomplete leaf
        "max_separator_states": max_separator_states,
        "max_recursion_depth": max_recursion_depth,
        "max_decomposition_nodes": max_decomposition_nodes,
        # mutable state, updated while the tree is built
        "next_node_number": 1,          # id of the next node, also counts the nodes created
        "cache": {},                    # sub-problem (CSP.key) -> node, to reuse repeated sub-problems
        "nodes": [],                    # every node, in creation order
        "cache_hits": 0,
        "separator_state_count": 0,     # branches created so far, checked against max_separator_states
    }
    # the root works on a copy, so the input csp is not modified;
    # satellites with an empty domain cannot be scheduled and are left out
    root = _build_node(csp.subproblem(v.name for v in csp.vars if v.domain), component_id, context, 0)

    # leaves: terminals go to the QPU, incomplete ones need a heuristic fallback
    nodes = context["nodes"]
    terminals = [n for n in nodes if n["node_type"] == "terminal"]
    incomplete = [n for n in nodes if n["node_type"] == "incomplete"]
    return {
        "component_id": component_id,
        "root": root,
        "nodes": nodes,
        "terminals": terminals,
        "incomplete": incomplete,
        "status": "complete" if not incomplete else "incomplete",
        "separator_state_count": context["separator_state_count"],
        "cache_hits": context["cache_hits"],
        "max_depth": max((n["depth"] for n in nodes), default=0),
    }
