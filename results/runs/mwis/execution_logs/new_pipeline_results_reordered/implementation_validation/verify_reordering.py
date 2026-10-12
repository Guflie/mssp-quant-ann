"""Check budget boundaries and unchanged recursion before executing experiments."""
import ast
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

FOLDER = Path(__file__).resolve().parent


def namespace(filename, variables=30, couplers=200):
    nodes = [n for n in ast.parse((FOLDER / filename).read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef)]
    values = dict(nx=nx, time=time, Any=Any, combinations=combinations,
                  MAX_QPU_TERMINAL_NODES=variables, MAX_QPU_TERMINAL_COUPLERS=couplers,
                  MAX_SEPARATOR_SIZE=3, MAX_SEPARATOR_CANDIDATE_POOL=14,
                  MAX_SEPARATOR_STATES=10000, MAX_RECURSION_DEPTH=12,
                  MAX_DECOMPOSITION_NODES=10000)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), values)
    return values


def graph(vertices, edges):
    result = nx.path_graph(vertices)
    for left, right in combinations(range(vertices), 2):
        if result.number_of_edges() >= edges:
            break
        result.add_edge(left, right)
    assert result.number_of_edges() == edges
    nx.set_node_attributes(result, {n: float(n + 1) for n in result}, 'weight')
    return result


def build(values, instance, number=1, depth=0):
    context = dict(next_node_number=number, cache={}, nodes={}, cache_hits=0,
                   separator_state_count=0, profile={})
    return values['build_decomposition_node'](instance, 'test', context, depth), context


def main():
    old = namespace('decomposition_functions_before.py')
    new = namespace('decomposition_functions_after.py')
    checks = []
    for budget, number, depth in [('nodes', 10006, 0), ('depth', 1, 13), ('both', 10006, 13)]:
        for name, instance in [('28_variables_199_couplers', graph(28, 199)),
                               ('exact_limits', graph(30, 200)),
                               ('empty', nx.Graph()),
                               ('resolved_isolated_vertices', nx.empty_graph(5))]:
            nx.set_node_attributes(instance, 1.0, 'weight')
            before, _ = build(old, instance, number, depth)
            after, context = build(new, instance, number, depth)
            assert before['node_type'] == 'incomplete'
            assert after['node_type'] == 'terminal'
            assert not after['children'] and not after['branches']
            assert context['separator_state_count'] == 0 and len(context['nodes']) == 1
            assert set(after['forced_selected']) | set(after['graph_node_ids']) == set(instance)
            checks.append(f'{budget}: {name} accepted without recursion')
        for name, instance in [('too_many_variables', graph(31, 30)),
                               ('too_many_couplers', graph(21, 210))]:
            before, _ = build(old, instance, number, depth)
            after, context = build(new, instance, number, depth)
            assert before == after and after['node_type'] == 'incomplete'
            assert context['separator_state_count'] == 0
            checks.append(f'{budget}: {name} still blocked')
    stricter = namespace('decomposition_functions_after.py', 20, 100)
    rejected, _ = build(stricter, graph(28, 199), 10006)
    assert rejected['node_type'] == 'incomplete'
    checks.append('28/199 remains oversized for the 20/100 profile')

    disconnected = nx.disjoint_union(nx.cycle_graph(3), nx.path_graph(4))
    nx.set_node_attributes(disconnected, 1.0, 'weight')
    before, _ = build(old, disconnected)
    after, _ = build(new, disconnected)
    assert before == after and after['node_type'] == 'independent_sum'
    checks.append('ordinary independent splitting remains unchanged')
    for seed in range(20):
        instance = nx.gnp_random_graph(9, 0.15 + 0.025 * seed, seed=seed)
        nx.set_node_attributes(instance, {n: float(n + 1) for n in instance}, 'weight')
        a = namespace('decomposition_functions_before.py', 4, 5)
        b = namespace('decomposition_functions_after.py', 4, 5)
        a['MAX_RECURSION_DEPTH'] = b['MAX_RECURSION_DEPTH'] = 100
        before, ca = build(a, instance)
        after, cb = build(b, instance)
        assert before == after and ca['separator_state_count'] == cb['separator_state_count']
    checks.append('20 small graphs retain identical unexhausted decomposition trees')
    report = dict(validation_passed=True, boundary_checks=len(checks) - 1,
                  unchanged_recursive_graphs=20, checks=checks)
    (FOLDER / 'verification_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
