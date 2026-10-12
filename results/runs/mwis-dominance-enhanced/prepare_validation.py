"""Adapt the existing independent baseline validator to reduced solver exports."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'results/_config.json').is_file())
LOG = ROOT / 'results/runs/mwis-dominance-enhanced/execution_logs/baseline_s200_g50'
CONTROL = ROOT / 'results/runs/mwis/execution_logs/new_pipeline_results_reordered/exp1_s200_g50'
source = (CONTROL / 'validate_run.py').read_text(encoding='utf-8')
source = source.replace('results/runs/mwis/', 'results/runs/mwis-dominance-enhanced/')
source = source.replace("    input_manifest = json.loads((LOG_DIR / 'input_manifest_entry.json').read_text(encoding='utf-8'))\n    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == input_manifest['sha256'], 'Frozen input changed'", "    preservation = json.loads((LOG_DIR / 'preserved_artifacts_manifest.json').read_text(encoding='utf-8'))\n    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in preservation.items()), 'An existing artifact changed'\n    expected_input_sha = '8c7f74c2e3b02e7c8158429b0fa9f21df7bbdc3445de913fdf563f8c3e560c8e'\n    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == expected_input_sha, 'Frozen baseline input changed'")
source = source.replace("PRODUCER / 'conflict_graph_edges.csv'", "PRODUCER / 'original_conflict_graph_edges.csv'", 1)
source = source.replace("PRODUCER / 'conflict_graph_nodes.csv'", "PRODUCER / 'original_conflict_graph_nodes.csv'", 1)
start = source.index('    # Exercise the actual comparison loader')
end = source.index('    report = {', start)
extra = '''    # Independent reduction proof: validate each saved witness using closed neighborhoods.
    import networkx as nx
    raw_graph = nx.Graph()
    raw_graph.add_nodes_from(by_id)
    raw_graph.add_edges_from(expected_edges)
    def audit_removals(input_ids, records):
        live = set(input_ids)
        for pass_index in sorted({int(row['pass']) for row in records}):
            batch = [row for row in records if int(row['pass']) == pass_index]
            already_removed = set()
            for row in batch:
                a, b = row['removed_slot'], row['witness_slot']
                assert a in live and b in live and a != b and b not in already_removed
                assert by_id[a]['satellite_id'] == by_id[b]['satellite_id']
                assert weight[b] >= weight[a]
                assert ((set(raw_graph.neighbors(b)) | {b}) & live) <= ((set(raw_graph.neighbors(a)) | {a}) & live)
                already_removed.add(a)
            live.difference_update(already_removed)
        return live
    removals = read_rows(PRODUCER / 'global_dominance_removals.csv')
    retained = audit_removals(by_id, removals)
    solver_nodes = {row['slot_id'] for row in read_rows(PRODUCER / 'conflict_graph_nodes.csv')}
    assert solver_nodes == retained
    assert all(abs(float(row['weight']) - weight[row['slot_id']]) < TOL for row in read_rows(PRODUCER / 'conflict_graph_nodes.csv'))
    solver_edges = {pair(row['source'], row['target']) for row in read_rows(PRODUCER / 'conflict_graph_edges.csv')}
    assert solver_edges == {edge for edge in expected_edges if set(edge) <= retained}
    reduced_graph = raw_graph.subgraph(retained)
    parts = sorted(nx.connected_components(reduced_graph), key=lambda c: (-len(c), sorted(c)))
    assert len(parts) == len(metadata['components'])
    components_table = read_rows(PRODUCER / 'components_summary.csv')
    for row, part in zip(components_table, parts):
        assert int(row['node_count']) == len(part)
        assert int(row['edge_count']) == reduced_graph.subgraph(part).number_of_edges()
    recursive_removed = 0
    for node in metadata['nodes']:
        records = node.get('dominance_removals', [])
        if records:
            input_ids = node['dominance_input_nodes']
            assert set(input_ids) <= retained
            assert reduced_graph.subgraph(input_ids).number_of_edges() == node['input_edge_count']
            survivors = audit_removals(input_ids, records)
            assert all(not (set(raw_graph.neighbors(slot)) & survivors) for slot in node['forced_selected'])
            assert reduced_graph.subgraph(survivors - set(node['forced_selected'])).number_of_edges() == node['qpu_edge_count']
            assert len(survivors) + len(records) == node['input_node_count']
            assert len(survivors) - len(node['forced_selected']) == node['qpu_node_count']
            recursive_removed += len(records)
    assert set(selected) <= retained, 'Dominated alternatives were incorrectly added back'

    # Independently rebuild the sampled pre-MILP tree and trace contact origins.
    nodes = {node['node_id']: node for node in metadata['nodes']}
    terminal_map = {row['terminal_id']: row for row in terminals}
    cache = {}
    def combine(base, children, extra=None):
        origins = dict(base)
        for child in children:
            assert not (set(origins) & set(child['origins'])), 'Duplicate contact in reconstruction'
            origins.update(child['origins'])
        if extra:
            assert not (set(origins) & set(extra))
            origins.update(extra)
        return origins
    def evaluate(node_id):
        if node_id in cache:
            return cache[node_id]
        node = nodes[node_id]
        base = {slot: 'exact preprocessing' for slot in node['forced_selected']}
        base_score = float(node['forced_objective'])
        assert abs(base_score - sum(weight[slot] for slot in base)) < TOL
        kind = node['node_type']
        if kind == 'terminal':
            terminal_id = node.get('terminal_id')
            sampled = json.loads(samples[terminal_id]['selected_nodes']) if terminal_id else []
            origins = combine(base, [], {slot: 'annealing' for slot in sampled})
            score = base_score + (float(samples[terminal_id]['objective']) if terminal_id else 0.0)
        elif kind == 'incomplete':
            h = node['heuristic_solution']
            origins = combine(base, [], {slot: 'greedy fallback' for slot in h['heuristic_selected_nodes']})
            score = float(h['objective'])
        elif kind == 'independent_sum':
            children = [evaluate(child) for child in node['children']]
            origins = combine(base, children)
            score = base_score + sum(c['objective'] for c in children)
        else:
            choices = []
            for branch in node['branches']:
                children = [evaluate(child) for child in branch['children']]
                origins = combine(base, children, {slot: 'separator assignment' for slot in branch['selected_separator']})
                score = base_score + float(branch['state_objective']) + sum(c['objective'] for c in children)
                choices.append({'origins': origins, 'objective': score})
            choice = max(choices, key=lambda c: (c['objective'], tuple(sorted(c['origins']))))
            origins, score = choice['origins'], choice['objective']
        assert abs(sum(weight[s] for s in origins) - score) < TOL
        result = {'origins': origins, 'objective': score}
        cache[node_id] = result
        return result
    pre_results = [evaluate(c['root_node_id']) for c in metadata['components']]
    pre_origins = combine({}, pre_results)
    before_milp = validate_selection(list(pre_origins))
    final_origins = {}
    provenance = {row['component_id']: row for row in read_rows(ANNEALER / 'component_provenance_summary.csv')}
    selected_set = set(selected)
    for component, part, result in zip(metadata['components'], parts, pre_results):
        cid = component['component_id']
        final_part = selected_set & part
        source_name = provenance[cid]['solution_source']
        if source_name.startswith('MILP'):
            final_origins.update({s: 'MILP' for s in final_part})
        else:
            assert final_part == set(result['origins']), 'Complete component reconstruction changed'
            final_origins.update(result['origins'])
    from collections import Counter
    origin_counts = dict(Counter(final_origins.values()))
    assert set(final_origins) == selected_set
    milp_rows = read_rows(ANNEALER / 'milp_component_results.csv')
    assert len(milp_rows) == int(final_summary['MILP incomplete components attempted'])
    assert all(row['component_id'] in {c['component_id'] for c in metadata['components'] if c['status'] == 'incomplete'} for row in milp_rows)
    true_gap = 12581.45 - before_milp
    assert true_gap >= -TOL and final_objective <= 12581.45 + TOL
    if final_summary['Status'] == 'hybrid_exact':
        assert abs(final_objective - 12581.45) < TOL
    record = {
        'Annealed objective before MILP': before_milp,
        'Hybrid gap (%)': float(final_summary['Hybrid schedule relative gap bound (%)']),
        'MILP proven optimal': int(final_summary['MILP components proven optimal']),
        'MILP attempted': int(final_summary['MILP incomplete components attempted']),
        'Incomplete components': int(final_summary['Components with incomplete separator decomposition (all states)']),
    }
'''
source = source[:start] + extra + source[end:]
source = source.replace("'run_name': RUN, 'validation_passed': True,", "'run_name': RUN, 'validation_passed': True,\n        'preserved_artifact_hashes_checked': len(preservation),\n        'global_dominance_removed': len(removals), 'recursive_dominance_removed_occurrences': recursive_removed,\n        'retained_solver_nodes': len(retained), 'retained_solver_edges': len(solver_edges),\n        'contact_origins': origin_counts, 'pre_milp_contact_origins': dict(Counter(pre_origins.values())),\n        'pre_milp_absolute_gap_against_certified_optimum': true_gap,\n        'pre_milp_relative_gap_against_certified_optimum_percent': 100 * true_gap / 12581.45,")
source = source.replace("'current comparison loader and metrics'", "'global and recursive closed-neighborhood dominance witnesses', 'reduced graph and component consistency', 'independent pre-MILP reconstruction and final provenance', 'preservation of existing sources, config, inputs and MWIS runs'")
(LOG / 'validate_run.py').write_text(source, encoding='utf-8')
print('Prepared independent baseline validator.')
