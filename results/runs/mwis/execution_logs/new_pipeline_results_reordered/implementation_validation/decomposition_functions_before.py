def materialize_node_graph(node: dict[str, Any]) -> nx.Graph:
    """Recreate a leaf graph as a read-only view of the full conflict graph."""
    return conflict_graph.subgraph(node["graph_node_ids"])


def retain_leaf_graph(node: dict[str, Any], graph: nx.Graph) -> None:
    """Store only the IDs needed to recreate terminal or unresolved graphs."""
    node["graph_node_ids"] = tuple(graph.nodes)


# Profiling helper: count calls and total seconds per decomposition operation
def add_decomposition_time(profile: dict[str, Any], operation: str, started: float) -> None:
    totals = profile.setdefault(operation, {"calls": 0, "seconds": 0.0})
    totals["calls"] += 1
    totals["seconds"] += time.perf_counter() - started


def graph_fits_qpu_limits(graph: nx.Graph) -> bool:
    """
    Return whether both terminal-size limits are satisfied.
    """
    return (
        graph.number_of_nodes() <= MAX_QPU_TERMINAL_NODES
        and graph.number_of_edges() <= MAX_QPU_TERMINAL_COUPLERS
    )


def exact_preprocess(graph: nx.Graph) -> dict[str, Any]:
    """
    Select isolated nodes and solve them exactly
    """
    # Nodes with no conflicts are always worth selecting
    forced_selected = set(nx.isolates(graph))
    # A read-only induced view avoids copying the graph at every tree node.
    reduced = graph.subgraph(set(graph.nodes) - forced_selected)
    return {
        "graph": reduced,
        "forced_selected": forced_selected,
        "forced_objective": sum(graph.nodes[node]["weight"] for node in forced_selected),
    }


def residual_components_after_separator(
    graph: nx.Graph, separator: tuple[str, ...], profile: dict[str, Any] | None = None,
) -> list[set[str]]:
    """
    Extract the new sub-problems after removing the graph separator (avoiding copying the whoel graph).
    """    
    started = time.perf_counter()
    # Remove the separator and split what is left into connected pieces
    residual_nodes = set(graph.nodes) - set(separator)
    residual = graph.subgraph(residual_nodes)  # read-only view; connected_components only reads it
    pieces = [set(c) for c in nx.connected_components(residual)]
    if profile is not None:
        add_decomposition_time(profile, "separator_partition_scans", started)
    return pieces


def articulation_piece_sizes(graph: nx.Graph) -> dict[str, tuple[int, ...]]:
    """
    Return the size of the new sub-problems obtained after deleting the articulation node (the node that splits the graph).
    """
    # discovery: order in which the DFS first reaches each node
    # low: earliest discovery reachable from the node's subtree via one back edge
    # parent: node the DFS came from (to backtrack and skip the edge back to it)
    # subtree_size: number of nodes in the node's DFS subtree, itself included
    discovery, low, parent, subtree_size = {}, {}, {}, {}
    
    # Nodes of each connected component, keyed by its DFS root
    members_by_root = {}
    clock = 0

    # Iterative DFS avoids Python recursion limits on long components.
    for root in graph:
        # Start a new DFS only in components not visited yet
        if root in discovery:
            continue
        # Initialise the root of a new component
        clock += 1
        discovery[root] = low[root] = clock
        parent[root] = None
        subtree_size[root] = 1
        members_by_root[root] = [root]
        # Each entry: a node and its neighbours still to explore
        stack = [(root, iter(graph.neighbors(root)))]
        while stack:
            node, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                # Node finished: pass its size and low-link up to the parent
                stack.pop()
                ancestor = parent[node]
                if ancestor is not None:
                    subtree_size[ancestor] += subtree_size[node]
                    low[ancestor] = min(low[ancestor], low[node])
                continue

            # Unvisited neighbour: go deeper
            if neighbor not in discovery:
                clock += 1
                discovery[neighbor] = low[neighbor] = clock
                parent[neighbor] = node
                subtree_size[neighbor] = 1
                members_by_root[root].append(neighbor)
                stack.append((neighbor, iter(graph.neighbors(neighbor))))
            elif neighbor != parent[node]:
                # Already visited, not the parent: back edge to an earlier node
                low[node] = min(low[node], discovery[neighbor])

    # Low-link values tell which DFS child subtrees disconnect when a node is removed.
    result = {}
    for root, members in members_by_root.items():
        component_size = subtree_size[root]
        for node in members:
            # Child subtrees with no edge above node become separate pieces
            separated = [
                subtree_size[child]
                for child in graph.neighbors(node)
                if parent.get(child) == node and low[child] >= discovery[node]
            ]
            # Everything else (the ancestors' side) forms one more piece
            if node != root:
                remainder = component_size - 1 - sum(separated)
                if remainder:
                    separated.append(remainder)
            # Two or more pieces: node is an articulation point
            if len(separated) > 1:
                result[node] = tuple(sorted(separated))
    return result


# Subsets of the separator with no internal conflict: one branch each
def feasible_separator_states(graph: nx.Graph, separator: tuple[str, ...]) -> list[frozenset[str]]:
    states = []
    # Try every subset size, from none selected to all selected
    for size in range(len(separator) + 1):
        for selected in combinations(separator, size):
            selected = set(selected)
            # Keep the subset only if its nodes do not conflict with each other
            if all(not graph.has_edge(a, b) for a, b in combinations(selected, 2)):
                states.append(frozenset(selected))
    return states


def separator_piece_sizes(
    adjacency: dict[str, frozenset[str]], separator: tuple[str, ...],
    profile: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    """Compute exact residual component sizes using one cached adjacency table.

    Native set intersections avoid rebuilding and traversing filtered graph views
    for every candidate. Sizes suffice for ranking; the selected separator's
    actual branch graphs are still constructed by the unchanged recursion.
    """
    started = time.perf_counter()
    remaining = set(adjacency).difference(separator)
    sizes = []
    while remaining:
        queue = [remaining.pop()]
        size = 1
        for node in queue:
            reached = adjacency[node].intersection(remaining)
            remaining.difference_update(reached)
            queue.extend(reached)
            size += len(reached)
            if not remaining:
                break  # All remaining vertices have already been discovered.
        sizes.append(size)
    if profile is not None:
        add_decomposition_time(profile, "separator_partition_scans", started)
    return tuple(sizes)


# Rank a separator (lower tuple is better); None if it does not split the graph
def score_separator(
    graph: nx.Graph, separator: tuple[str, ...], pieces: list[set[str]] | None = None,
    profile: dict[str, Any] | None = None,
    adjacency: dict[str, frozenset[str]] | None = None,
) -> tuple | None:
    started = time.perf_counter()
    if pieces is None:
        if adjacency is None:
            adjacency = {node: frozenset(graph.neighbors(node)) for node in graph}
        sizes = separator_piece_sizes(adjacency, separator, profile)
    else:
        sizes = tuple(len(piece) for piece in pieces)
    # Not a real cut: the graph stays in one piece
    if len(sizes) <= 1:
        if profile is not None:
            add_decomposition_time(profile, "separator_scoring", started)
        return None
    # First limit the largest child, then prefer a more balanced split.
    score = (
        max(sizes),
        sum(size * size for size in sizes),
        len(feasible_separator_states(graph, separator)),
        len(separator),
        tuple(sorted(separator)),
    )
    if profile is not None:
        add_decomposition_time(profile, "separator_scoring", started)
    return score


def score_separator_piece_sizes(
    graph: nx.Graph, separator: tuple[str, ...], sizes: tuple[int, ...],
    profile: dict[str, Any] | None = None,
) -> tuple | None:
    """Score an articulation separator without rescanning the graph for its pieces."""
    started = time.perf_counter()
    if len(sizes) <= 1:
        if profile is not None:
            add_decomposition_time(profile, "separator_scoring", started)
        return None
    # Same ranking as score_separator
    score = (
        max(sizes),
        sum(size * size for size in sizes),
        len(feasible_separator_states(graph, separator)),
        len(separator),
        tuple(sorted(separator)),
    )
    if profile is not None:
        add_decomposition_time(profile, "separator_scoring", started)
    return score


# Pick the best separator: articulation points first, then larger cuts
def find_separator(graph: nx.Graph, profile: dict[str, Any] | None = None) -> dict[str, Any] | None:
    candidates = []

    # 1) Single-node cuts, all found in one DFS pass
    articulation_candidates = []
    operation_started = time.perf_counter()
    articulation_sizes = articulation_piece_sizes(graph)
    if profile is not None:
        add_decomposition_time(profile, "articulation_analysis", operation_started)
        profile["articulation_candidates"] = profile.get("articulation_candidates", 0) + len(articulation_sizes)
    for node, sizes in articulation_sizes.items():
        separator = (node,)
        score = score_separator_piece_sizes(graph, separator, sizes, profile)
        if score is not None:
            articulation_candidates.append((score, separator, "articulation point"))

    # Prefer the best-balanced one-node cut before searching larger cuts.
    # This can change the decomposition tree, but not the optimum: every
    # feasible selected/not-selected state of the chosen separator is solved.
    if articulation_candidates:
        score, separator, method = min(articulation_candidates, key=lambda item: item[0])
        return {"separator": separator, "score": score, "method": method}

    # Cache only the current induced graph, once for all candidate cuts.
    operation_started = time.perf_counter()
    adjacency = {node: frozenset(graph.neighbors(node)) for node in graph}
    if profile is not None:
        add_decomposition_time(profile, "separator_adjacency_construction", operation_started)

    # 2) Small graphs: try every pair of nodes
    node_ids = sorted(graph.nodes)
    if MAX_SEPARATOR_SIZE >= 2 and len(node_ids) <= 200:
        for separator in combinations(node_ids, 2):
            if profile is not None:
                profile["pair_candidates"] = profile.get("pair_candidates", 0) + 1
            score = score_separator(graph, separator, profile=profile, adjacency=adjacency)
            if score is not None:
                candidates.append((score, separator, "exhaustive separator pair"))

    # 3) Larger cuts drawn only from the highest-degree nodes
    operation_started = time.perf_counter()
    pool = [
        node for node, _ in sorted(
            graph.degree,
            key=lambda item: (-item[1], item[0]),
        )[:MAX_SEPARATOR_CANDIDATE_POOL]
    ]
    if profile is not None:
        add_decomposition_time(profile, "candidate_pool_construction", operation_started)
    for size in range(2, MAX_SEPARATOR_SIZE + 1):
        for separator in combinations(pool, size):
            if profile is not None:
                profile["degree_pool_candidates"] = profile.get("degree_pool_candidates", 0) + 1
            score = score_separator(graph, separator, profile=profile, adjacency=adjacency)
            if score is not None:
                candidates.append((score, separator, f"degree-pool separator size {size}"))

    # Keep the lowest score; None if nothing splits the graph
    if not candidates:
        return None
    score, separator, method = min(candidates)
    return {"separator": tuple(separator), "score": score, "method": method}


def build_decomposition_node(graph: nx.Graph, component_id: str, context: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Build one decomposition-tree node, splitting or recording a terminal/fallback.

    The selected separator's feasible states become branches. Independent pieces
    are solved separately; budget limits are recorded as incomplete leaves.
    """
    # Reuse the result if this exact node set was already decomposed
    cache_key = frozenset(graph.nodes)
    if cache_key in context["cache"]:
        context["cache_hits"] += 1
        return context["cache"][cache_key]

    # Unique ID for this tree node
    node_number = context["next_node_number"]
    context["next_node_number"] += 1
    node_id = f"{component_id}_dnode_{node_number:04d}"

    if node_number % 100 == 0:
        print(f"[MWIS timing] {component_id}: built {node_number:,} decomposition nodes (depth {depth})", flush=True)

    # Take isolated nodes for free and continue on the rest
    operation_started = time.perf_counter()
    preprocessing = exact_preprocess(graph)
    add_decomposition_time(context["profile"], "exact_preprocessing", operation_started)
    reduced = preprocessing["graph"]
    node = {
        "decomposition_node_id": node_id,
        "original_component_id": component_id,
        "depth": depth,
        "input_node_count": graph.number_of_nodes(),
        "input_edge_count": graph.number_of_edges(),
        "graph_node_count": reduced.number_of_nodes(),
        "graph_edge_count": reduced.number_of_edges(),
        "forced_selected": set(preprocessing["forced_selected"]),
        "forced_objective": preprocessing["forced_objective"],
        "children": [],
        "branches": [],
    }
    # Register the node before recursing so repeated subgraphs hit the cache
    context["cache"][cache_key] = node
    context["nodes"][node_id] = node

    # Budget limits: stop here and mark the node incomplete
    if context["next_node_number"] > MAX_DECOMPOSITION_NODES:
        node.update({"node_type": "incomplete", "reason": "decomposition node budget exceeded"})
        retain_leaf_graph(node, reduced)
        return node
    if depth > MAX_RECURSION_DEPTH:
        node.update({"node_type": "incomplete", "reason": "recursion depth exceeded"})
        retain_leaf_graph(node, reduced)
        return node
    # Nothing left after preprocessing
    if reduced.number_of_nodes() == 0:
        node.update({"node_type": "terminal", "terminal_reason": "resolved by exact preprocessing"})
        retain_leaf_graph(node, reduced)
        return node

    # Already disconnected: solve each piece independently and sum
    operation_started = time.perf_counter()
    natural_pieces = [set(c) for c in nx.connected_components(reduced)]
    add_decomposition_time(context["profile"], "natural_component_scans", operation_started)
    if len(natural_pieces) > 1:
        node.update({"node_type": "independent_sum"})
        node["children"] = [
            build_decomposition_node(reduced.subgraph(sorted(piece)), component_id, context, depth + 1)
            for piece in sorted(natural_pieces, key=lambda c: (-len(c), sorted(c)))
        ]
        return node

    # Small enough: this leaf is sent to the QPU
    if graph_fits_qpu_limits(reduced):
        node.update({"node_type": "terminal", "terminal_reason": "fits QPU limits"})
        retain_leaf_graph(node, reduced)
        return node

    # Too big and connected: look for a separator
    operation_started = time.perf_counter()
    separator_result = find_separator(reduced, context["profile"])
    add_decomposition_time(context["profile"], "separator_search", operation_started)
    if separator_result is None:
        node.update({"node_type": "incomplete", "reason": "no separator found within configured search limits"})
        retain_leaf_graph(node, reduced)
        return node

    # One branch per valid selection of separator nodes
    separator = tuple(separator_result["separator"])
    operation_started = time.perf_counter()
    states = feasible_separator_states(reduced, separator)
    add_decomposition_time(context["profile"], "separator_state_generation", operation_started)
    if context["separator_state_count"] + len(states) > MAX_SEPARATOR_STATES:
        node.update({"node_type": "incomplete", "reason": "separator-state budget exceeded"})
        retain_leaf_graph(node, reduced)
        return node

    context["separator_state_count"] += len(states)
    node.update({
        "node_type": "separator",
        "separator": separator,
        "separator_method": separator_result["method"],
        "separator_score": separator_result["score"],
    })

    # For each branch: fix the separator choice and recurse on what is left
    separator_set = set(separator)
    for state_index, selected_separator in enumerate(states):
        selected_separator = set(selected_separator)
        # Neighbours of selected separator nodes cannot be selected
        forced_neighbors = set()
        for selected_node in selected_separator:
            forced_neighbors.update(reduced.neighbors(selected_node))
        forced_neighbors.difference_update(separator_set)

        # Drop the separator and forced-out nodes, then split into independent pieces
        operation_started = time.perf_counter()
        branch_nodes = set(reduced.nodes) - separator_set - forced_neighbors
        branch_graph = reduced.subgraph(branch_nodes)
        branch_components = [set(c) for c in nx.connected_components(branch_graph)]
        add_decomposition_time(context["profile"], "branch_partition_scans", operation_started)
        children = [
            build_decomposition_node(branch_graph.subgraph(sorted(piece)), component_id, context, depth + 1)
            for piece in sorted(branch_components, key=lambda c: (-len(c), sorted(c)))
        ]
        node["branches"].append({
            "state_index": state_index,
            "selected_separator": selected_separator,
            "state_objective": sum(reduced.nodes[n]["weight"] for n in selected_separator),
            "children": children,
        })
    return node


# Decompose one natural component and collect its leaves
def build_component_decomposition(component: dict[str, Any]) -> dict[str, Any]:
    context = {"next_node_number": 1, "cache": {}, "nodes": {}, "cache_hits": 0, "separator_state_count": 0, "profile": {}}
    root = build_decomposition_node(component["graph"], component["component_id"], context)
    nodes = list(context["nodes"].values())
    # Terminal leaves go to the QPU; incomplete ones need a fallback
    terminals = [node for node in nodes if node["node_type"] == "terminal"]
    incomplete = [node for node in nodes if node["node_type"] == "incomplete"]
    return {
        "component_id": component["component_id"],
        "root": root,
        "nodes": nodes,
        "terminals": terminals,
        "incomplete": incomplete,
        "status": "complete" if not incomplete else "incomplete",
        "separator_state_count": context["separator_state_count"],
        "cache_hits": context["cache_hits"],
        "operation_profile": context["profile"],
        "max_depth": max((node["depth"] for node in nodes), default=0),
    }