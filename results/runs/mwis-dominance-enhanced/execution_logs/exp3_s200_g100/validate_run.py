"""Independently validate the reordered active-count baseline exports without calling a solver."""
import ast
import csv
import gzip
import hashlib
import json
import math
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import pandas as pd

LOG_DIR = Path(__file__).resolve().parent
ROOT = next(parent for parent in LOG_DIR.parents if (parent / 'results/_config.json').is_file())
CONFIG = json.loads((LOG_DIR / 'config.json').read_text(encoding='utf-8'))
RUN = CONFIG['run_name_mwis']
PRODUCER = ROOT / 'results/runs/mwis-dominance-enhanced/pipeline_results' / RUN
ANNEALER = ROOT / 'results/runs/mwis-dominance-enhanced/qubo_annealing' / RUN
TOL = 1e-6


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as source:
        return list(csv.DictReader(source))


def metrics(path):
    return {row['metric']: row['value'] for row in read_rows(path)}


def pair(left, right):
    return tuple(sorted((left, right)))


def main():
    input_path = ROOT / CONFIG['candidate_slot_csv']
    preservation = json.loads((LOG_DIR / 'preserved_artifacts_manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in preservation.items()), 'An existing artifact changed'
    expected_input_sha = 'a64610ae97025fd1e6049a06a39bf78d4e2495b4ee3a481bb669b951d0c3653c'
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == expected_input_sha, 'Frozen gateway-scaling input changed'
    slots = read_rows(input_path)
    by_id = {row['slot_id']: row for row in slots}
    assert len(by_id) == len(slots) == 2033
    assert len({row['satellite_id'] for row in slots}) == 200
    assert len({row['station_id'] for row in slots}) == 100
    assert len({row['antenna_id'] for row in slots}) == 100
    weight = {slot: float(row['weight']) for slot, row in by_id.items()}
    milliweight = {slot: int(Decimal(row['weight']) * 1000) for slot, row in by_id.items()}
    times = {
        slot: tuple(int((Decimal(row[column]) * 3_600_000_000).to_integral_value())
                    for column in ['start_time_hours', 'end_time_hours', 'reconfiguration_time_hours'])
        for slot, row in by_id.items()
    }
    assert all(end - start == 240_000_000 and setup == 180_000_000 for start, end, setup in times.values())

    # Build all pair conflicts directly from the input, rather than trusting the export.
    expected_edges = set()
    for left, right in combinations(by_id, 2):
        a, b = by_id[left], by_id[right]
        conflict = a['satellite_id'] == b['satellite_id']
        if a['antenna_id'] == b['antenna_id']:
            a_start, a_end, a_setup = times[left]
            b_start, b_end, b_setup = times[right]
            earlier_end, later_start = (a_end, b_start) if a_start <= b_start else (b_end, a_start)
            conflict |= later_start - earlier_end < max(a_setup, b_setup)
        if conflict:
            expected_edges.add(pair(left, right))
    exported_edges = {pair(row['source'], row['target']) for row in read_rows(PRODUCER / 'original_conflict_graph_edges.csv')}
    assert exported_edges == expected_edges, 'Conflict graph differs from the scheduling constraints'
    graph_nodes = read_rows(PRODUCER / 'original_conflict_graph_nodes.csv')
    assert {row['slot_id'] for row in graph_nodes} == set(by_id)
    assert all(abs(float(row['weight']) - weight[row['slot_id']]) < TOL for row in graph_nodes)

    def validate_selection(ids):
        assert len(ids) == len(set(ids)), 'Duplicate selection'
        ids = set(ids)
        assert ids <= set(by_id), 'Unknown candidate selected'
        conflicts = sum(left in ids and right in ids for left, right in expected_edges)
        assert conflicts == 0, 'Selected contacts conflict'
        satellites = [by_id[slot]['satellite_id'] for slot in ids]
        assert len(satellites) == len(set(satellites)), 'Satellite selected more than once'
        return sum(milliweight[slot] for slot in ids) / 1000

    producer_summary = metrics(PRODUCER / 'final_summary.csv')
    final_summary = metrics(ANNEALER / 'reconstructed_summary.csv')
    classical = read_rows(PRODUCER / 'mwis_solution.csv')
    classical_objective = validate_selection([row['slot_id'] for row in classical])
    assert abs(classical_objective - float(producer_summary['Total feasible objective / lower bound'])) < TOL

    schedule = read_rows(ANNEALER / 'reconstructed_schedule.csv')
    selected = [row['slot_id'] for row in schedule]
    final_objective = validate_selection(selected)
    assert all(abs(float(row['weight']) - weight[row['slot_id']]) < TOL for row in schedule)
    assert set(selected) == set(json.loads((ANNEALER / 'reconstructed_selected_nodes.json').read_text(encoding='utf-8'))['selected_nodes'])
    assert abs(final_objective - float(final_summary['Total feasible objective'])) < TOL
    assert len(selected) == int(final_summary['Selected contacts'])
    upper = float(final_summary['Hybrid conservative objective upper bound'])
    gap = float(final_summary['Hybrid schedule absolute gap bound'])
    assert upper + TOL >= final_objective and abs(gap - max(0, upper - final_objective)) < TOL
    assert (final_summary['Status'] == 'hybrid_exact') == (abs(upper - final_objective) <= TOL)

    with gzip.open(PRODUCER / 'reconstruction_metadata.json.gz', 'rt', encoding='utf-8') as stream:
        metadata = json.load(stream)
    terminals = json.loads((PRODUCER / metadata['terminal_qubos_file']).read_text(encoding='utf-8'))
    samples = {row['terminal_id']: row for row in read_rows(ANNEALER / 'annealing_results.csv')}
    assert len(samples) == len(terminals)
    for index, terminal in enumerate(terminals):
        slot_map = terminal['index_map']
        assert terminal['encoding'] == 'one_hot'
        assert terminal['qubits'] == len(slot_map) <= CONFIG['max_qpu_terminal_nodes']
        terminal_edges = {pair(*edge) for edge in terminal['conflicts']}
        assert terminal_edges == {edge for edge in expected_edges if set(edge) <= set(slot_map)}
        assert terminal['couplers'] == len(terminal_edges) <= CONFIG['max_qpu_terminal_couplers']
        indices = {slot: bits[0] for slot, bits in slot_map.items()}
        assert set(indices.values()) == set(range(len(indices)))
        q = {(i, j): coefficient for i, j, coefficient in terminal['qubo']}
        for slot, bit in indices.items():
            assert abs(q[bit, bit] + CONFIG['qubo_reward_factor'] * weight[slot]) < TOL
        assert terminal['penalty'] > max(CONFIG['qubo_reward_factor'] * weight[slot] for slot in indices)
        for left, right in terminal_edges:
            i, j = sorted((indices[left], indices[right]))
            assert abs(q[i, j] - terminal['penalty']) < TOL
        row = samples[terminal['terminal_id']]
        assert int(row['num_reads']) == CONFIG['num_reads']
        assert int(row['seed']) == CONFIG['base_seed'] + index
        sample_ids = json.loads(row['selected_nodes'])
        assert set(sample_ids) <= set(slot_map)
        sample_feasible = not any(left in sample_ids and right in sample_ids for left, right in terminal_edges)
        assert sample_feasible == (row['is_independent'].lower() == 'true')
        assert abs(sum(weight[slot] for slot in sample_ids) - float(row['objective'])) < TOL
        active_bits = {indices[slot] for slot in sample_ids}
        energy = sum(value for (i, j), value in q.items() if i in active_bits and j in active_bits)
        assert abs(energy - float(row['energy'])) < TOL

    # Independent reduction proof: validate each saved witness using closed neighborhoods.
    import networkx as nx
    raw_graph = nx.Graph()
    raw_graph.add_nodes_from(by_id)
    raw_graph.add_edges_from(expected_edges)
    original_edges = tuple(expected_edges)
    def induced_edge_count(ids):
        live = set(ids)
        return sum(left in live and right in live for left, right in original_edges)
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
            assert induced_edge_count(input_ids) == node['input_edge_count']
            survivors = audit_removals(input_ids, records)
            assert all(not (set(raw_graph.neighbors(slot)) & survivors) for slot in node['forced_selected'])
            assert induced_edge_count(survivors - set(node['forced_selected'])) == node['qpu_edge_count']
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
    true_gap = 13058.865 - before_milp
    assert true_gap >= -TOL and final_objective <= 13058.865 + TOL
    if final_summary['Status'] == 'hybrid_exact':
        assert abs(final_objective - 13058.865) < TOL
    record = {
        'Annealed objective before MILP': before_milp,
        'Hybrid gap (%)': float(final_summary['Hybrid schedule relative gap bound (%)']),
        'MILP proven optimal': int(final_summary['MILP components proven optimal']),
        'MILP attempted': int(final_summary['MILP incomplete components attempted']),
        'Incomplete components': int(final_summary['Components with incomplete separator decomposition (all states)']),
    }
    report = {
        'run_name': RUN, 'validation_passed': True,
        'preserved_artifact_hashes_checked': len(preservation),
        'global_dominance_removed': len(removals), 'recursive_dominance_removed_occurrences': recursive_removed,
        'retained_solver_nodes': len(retained), 'retained_solver_edges': len(solver_edges),
        'contact_origins': origin_counts,
        'contact_origin_objectives': {origin: sum(milliweight[slot] for slot, kind in final_origins.items() if kind == origin) / 1000 for origin in origin_counts},
        'pre_milp_contact_origins': dict(Counter(pre_origins.values())),
        'pre_milp_absolute_gap_against_certified_optimum': true_gap,
        'pre_milp_relative_gap_against_certified_optimum_percent': 100 * true_gap / 13058.865,
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'candidate_slots': len(slots), 'satellites_with_slots': len({row['satellite_id'] for row in slots}),
        'antennas_with_slots': len({row['antenna_id'] for row in slots}),
        'conflict_edges': len(expected_edges), 'natural_components': int(float(producer_summary['Number of components (total)'])),
        'terminal_qubos': len(terminals), 'selected_contacts': len(selected), 'selected_conflicts': 0,
        'classical_baseline_objective': classical_objective,
        'annealed_objective_before_milp': record['Annealed objective before MILP'],
        'final_feasible_objective': final_objective, 'hybrid_upper_bound': upper,
        'absolute_gap': gap, 'relative_gap_percent': record['Hybrid gap (%)'],
        'status': final_summary['Status'], 'milp_components_proven_optimal': record['MILP proven optimal'],
        'milp_components_attempted': record['MILP attempted'], 'incomplete_components': record['Incomplete components'],
        'max_terminal_variables': max((terminal['qubits'] for terminal in terminals), default=0),
        'max_terminal_couplers': max((terminal['couplers'] for terminal in terminals), default=0),
        'checks': ['frozen input and contact constants', 'full conflict graph rebuilt from input',
                   'classical and final schedule feasibility', 'objectives and certification consistency',
                   'terminal QUBOs and penalty coefficients', 'annealing seeds, reads, decoded objectives and energies',
                   'global and recursive closed-neighborhood dominance witnesses', 'reduced graph and component consistency', 'independent pre-MILP reconstruction and final provenance', 'preservation of existing sources, config, inputs and MWIS runs'],
    }
    (LOG_DIR / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    execution = json.loads((LOG_DIR / 'execution_summary.json').read_text(encoding='utf-8'))
    assert execution['status'] == 'completed_pending_independent_validation'
    execution['status'] = 'completed_and_validated'
    execution['validation_report'] = 'validation_report.json'
    (LOG_DIR / 'execution_summary.json').write_text(json.dumps(execution, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
