"""Verify reduction safety independently by exhaustive small-instance optima."""
import ast
import json
import random
from itertools import combinations
from pathlib import Path
import networkx as nx

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'results/_config.json').is_file())
NOTEBOOK = ROOT / 'results/02.A_mwis_pipeline_dominance_enhanced.ipynb'
source = ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][8]['source'])
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'reduce_same_satellite_dominance')
namespace = {'nx': nx, 'Any': object}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
reduce = namespace['reduce_same_satellite_dominance']


def brute(graph):
    nodes = list(graph)
    best = 0
    for mask in range(1 << len(nodes)):
        selected = {v for i, v in enumerate(nodes) if mask & (1 << i)}
        if not any(u in selected and v in selected for u, v in graph.edges):
            best = max(best, sum(graph.nodes[v]['weight'] for v in selected))
    return best


def audit_witnesses(graph, records):
    live = set(graph)
    for pass_index in sorted({r['pass'] for r in records}):
        batch = [r for r in records if r['pass'] == pass_index]
        for row in batch:
            a, b = row['removed_slot'], row['witness_slot']
            assert a in live and b in live and a != b
            assert graph.nodes[a]['satellite_id'] == graph.nodes[b]['satellite_id']
            assert graph.has_edge(a, b)
            assert graph.nodes[b]['weight'] >= graph.nodes[a]['weight']
            # Closed-neighbourhood inclusion is an independent MWIS safety check.
            assert ((set(graph.neighbors(b)) | {b}) & live) <= ((set(graph.neighbors(a)) | {a}) & live)
        live.difference_update(r['removed_slot'] for r in batch)
    return live


def main():
    rng = random.Random(42)
    removed_total = 0
    for case in range(300):
        graph = nx.Graph()
        for i in range(rng.randint(1, 10)):
            graph.add_node(f'v{i:02d}', satellite_id=str(rng.randrange(4)), weight=rng.randrange(8))
        for a, b in combinations(graph, 2):
            if graph.nodes[a]['satellite_id'] == graph.nodes[b]['satellite_id'] or rng.random() < .3:
                graph.add_edge(a, b)
        before = (dict(graph.nodes(data='weight')), set(graph.edges))
        result = reduce(graph)
        assert set(result['graph']) == audit_witnesses(graph, result['removals'])
        assert brute(graph) == brute(result['graph']), f'Optimum changed in case {case}'
        assert before == (dict(graph.nodes(data='weight')), set(graph.edges)), 'Input graph mutated'
        assert not reduce(result['graph'])['removals'], 'Reduction did not reach fixed point'
        removed_total += len(result['removals'])
    graph = nx.Graph()
    graph.add_nodes_from([('a', {'satellite_id': 's', 'weight': 5}), ('b', {'satellite_id': 's', 'weight': 6}), ('c', {'satellite_id': 'other', 'weight': 100})])
    graph.add_edges_from([('a', 'b'), ('b', 'c')])
    assert not reduce(graph)['removals'], 'Higher weight alone incorrectly dominates'
    graph.nodes['b']['weight'] = 5
    graph.remove_edge('b', 'c')
    assert set(reduce(graph)['graph']) == {'a', 'c'}, 'Equivalent alternatives must keep smallest ID; unrelated satellite must survive'
    graph.remove_edge('a', 'b')
    try:
        reduce(graph)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing same-satellite clique was accepted')
    report = {'passed': True, 'random_exhaustive_instances': 300, 'removed_vertices_tested': removed_total,
              'checks': ['exact optimum preservation by brute force', 'closed-neighbourhood witness safety', 'fixed point', 'unchanged input graph', 'weight versus conflicts tradeoff', 'deterministic ties', 'no cross-satellite pruning', 'reject non-clique domains']}
    (Path(__file__).parent / 'dominance_test_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
