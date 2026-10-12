"""Prepare a controlled dominance-only MWIS baseline without changing the control."""
from pathlib import Path
import hashlib
import json
import nbformat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'results/_config.json').is_file())
OUT = ROOT / 'results/runs/mwis-dominance-enhanced'
LOG = OUT / 'execution_logs/baseline_s200_g50'
LOG.mkdir(parents=True, exist_ok=True)
CONTROL = ROOT / 'results/runs/mwis/execution_logs/new_pipeline_results_reordered/exp1_s200_g50'
preserved = {}
for base in [ROOT / 'results/runs/mwis', ROOT / 'data/experiment_inputs/exp_snapshot_2026-10-12']:
    for p in base.rglob('*'):
        if p.is_file():
            preserved[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['results/02.A_mwis_pipeline.ipynb', 'results/3_qubo_annealing.ipynb', 'results/_config.json', 'results/qubo_sweep_test.ipynb']:
    p = ROOT / name
    if p.exists():
        preserved[name] = hashlib.sha256(p.read_bytes()).hexdigest()
(LOG / 'preserved_artifacts_manifest.json').write_text(json.dumps(preserved, indent=2), encoding='utf-8')

DOMINANCE = '''def reduce_same_satellite_dominance(graph: nx.Graph) -> dict[str, Any]:
    """Remove dominated alternatives to a fixed point, preserving MWIS value.

    Same-satellite alternatives must form cliques. A contact a is removable
    when b has at least its weight and a superset of b's external conflicts.
    Equivalent alternatives retain the lexicographically smallest slot ID.
    Each pass uses a snapshot; a witness removed earlier in that pass is skipped.
    Removal records are local to this input graph, including recursive branches.
    """
    groups: dict[str, set[str]] = {}
    for v, data in graph.nodes(data=True):
        groups.setdefault(str(data["satellite_id"]), set()).add(v)
    for group in groups.values():
        if any(len(set(graph.neighbors(v)) & group) != len(group) - 1 for v in group):
            raise ValueError("Dominance requires same-satellite alternatives to form a clique.")
    live = set(graph.nodes)
    records = []
    passes = 0
    while True:
        external = {v: (set(graph.neighbors(v)) & live) - groups[str(graph.nodes[v]["satellite_id"])] for v in live}
        removed = set()
        for satellite in sorted(groups):
            domain = sorted(groups[satellite] & live)
            for a in domain:
                if a in removed:
                    continue
                for b in domain:
                    if a == b or b in removed:
                        continue
                    wa, wb = float(graph.nodes[a]["weight"]), float(graph.nodes[b]["weight"])
                    if wb >= wa and external[b] <= external[a] and (
                        wb > wa or external[b] < external[a] or b < a
                    ):
                        removed.add(a)
                        records.append({"pass": passes + 1, "removed_slot": a, "witness_slot": b,
                                        "satellite_id": satellite, "removed_weight": wa, "witness_weight": wb})
                        break
        if not removed:
            break
        live.difference_update(removed)
        passes += 1
    return {"graph": graph.subgraph(live), "removals": records, "passes": passes}


phase_started = time.perf_counter()
original_conflict_graph = conflict_graph
global_dominance = reduce_same_satellite_dominance(original_conflict_graph)
# Materialize the reduced root once; recursive reductions remain induced views.
conflict_graph = global_dominance["graph"].copy()
pd.DataFrame(global_dominance["removals"], columns=["pass", "removed_slot", "witness_slot", "satellite_id", "removed_weight", "witness_weight"]).to_csv(RUN_DIR / "global_dominance_removals.csv", index=False)
pd.DataFrame([{"slot_id": v, **d} for v, d in conflict_graph.nodes(data=True)]).to_csv(RUN_DIR / "conflict_graph_nodes.csv", index=False)
pd.DataFrame([{"source": u, "target": v, **d} for u, v, d in conflict_graph.edges(data=True)], columns=["source", "target", "reason", "reconfiguration_time", "satellite_id"]).to_csv(RUN_DIR / "conflict_graph_edges.csv", index=False)
dominance_summary = {
    "input_nodes": original_conflict_graph.number_of_nodes(), "input_edges": original_conflict_graph.number_of_edges(),
    "input_components": nx.number_connected_components(original_conflict_graph),
    "retained_nodes": conflict_graph.number_of_nodes(), "retained_edges": conflict_graph.number_of_edges(),
    "retained_components": nx.number_connected_components(conflict_graph),
    "removed_nodes": len(global_dominance["removals"]), "passes": global_dominance["passes"],
    "original_graph_nodes_file": "original_conflict_graph_nodes.csv",
    "original_graph_edges_file": "original_conflict_graph_edges.csv",
    "solver_graph_nodes_file": "conflict_graph_nodes.csv", "solver_graph_edges_file": "conflict_graph_edges.csv",
}
(RUN_DIR / "dominance_summary.json").write_text(json.dumps(dominance_summary, indent=2), encoding="utf-8")
record_phase_timing("global_dominance_and_export", phase_started, **dominance_summary)
print(dominance_summary)

'''

notebook = nbformat.read(ROOT / 'results/02.A_mwis_pipeline.ipynb', as_version=4)
notebook.cells[0].source = '# MWIS pipeline with exact dominance reduction\n\nControlled experimental variant of `02.A_mwis_pipeline.ipynb`. The conflict rules, separator algorithm, budgets, QUBO and classical terminal solvers are unchanged. Outputs go to `runs/mwis-dominance-enhanced`; existing runs remain the control.'
notebook.cells[2].source = notebook.cells[2].source.replace('"mwis" / "pipeline_results"', '"mwis-dominance-enhanced" / "pipeline_results"')
notebook.cells[6].source = notebook.cells[6].source.replace('"conflict_graph_nodes.csv"', '"original_conflict_graph_nodes.csv"').replace('"conflict_graph_edges.csv"', '"original_conflict_graph_edges.csv"')
notebook.cells[7].source = '''## Exact dominance reduction and independent components

Before splitting the graph, remove a contact only if another contact of the same satellite has at least its weight and no additional conflicts with other satellites. Because contacts of each satellite form a clique, replacing the removed contact by its witness preserves feasibility and cannot decrease the objective. Equivalent contacts retain the smallest slot identifier. Repeat until no removal is possible, then split the reduced graph into connected components. This reduction preserves the optimum but can change component boundaries.

The full input graph is saved in `original_conflict_graph_*.csv`. The downstream `conflict_graph_*.csv` tables describe the reduced solver graph, so reconstruction and MILP use the reduced components consistently. Dominated contacts are removed alternatives and are never added back to the schedule. Removal witnesses are saved for independent validation.'''
notebook.cells[8].source = DOMINANCE + notebook.cells[8].source
old = notebook.cells[10].source
start, end = old.index('def exact_preprocess('), old.index('def residual_components_after_separator(')
preprocess = '''def exact_preprocess(graph: nx.Graph) -> dict[str, Any]:
    """Repeat safe dominance in each branch, then fix isolated vertices."""
    dominance = reduce_same_satellite_dominance(graph)
    surviving = dominance["graph"]
    forced_selected = set(nx.isolates(surviving))
    reduced = surviving.subgraph(set(surviving.nodes) - forced_selected)
    return {"graph": reduced, "forced_selected": forced_selected,
            "forced_objective": sum(graph.nodes[v]["weight"] for v in forced_selected),
            "dominance_removals": dominance["removals"], "dominance_passes": dominance["passes"]}


'''
old = old[:start] + preprocess + old[end:]
old = old.replace('"forced_objective": preprocessing["forced_objective"],', '"forced_objective": preprocessing["forced_objective"],\n        "dominance_removals": preprocessing["dominance_removals"],\n        "dominance_input_nodes": sorted(graph.nodes) if preprocessing["dominance_removals"] else [],')
notebook.cells[10].source = old
notebook.cells[9].source += '\n\nAt every uncached recursive node, the same exact dominance rule is applied again before isolated vertices are fixed. Branch-local reductions are recorded in reconstruction metadata. Separator search and all budgets retain their original behavior.'
notebook.cells[16].source = notebook.cells[16].source.replace('if not is_independent(conflict_graph, solution["selected_nodes"]):', 'if not is_independent(original_conflict_graph, solution["selected_nodes"]):')
notebook.cells[16].source = notebook.cells[16].source.replace('"forced_objective": float(node.get("forced_objective", 0.0)),', '"forced_objective": float(node.get("forced_objective", 0.0)),\n        "dominance_removals": node.get("dominance_removals", []),\n        "dominance_input_nodes": node.get("dominance_input_nodes", []),')
notebook.cells[16].source = notebook.cells[16].source.replace('"schema_version": 1,', '"schema_version": 1,\n    "variant": "same-satellite dominance before components and during recursion",\n    "dominance_summary": dominance_summary,')
notebook.cells[18].source = notebook.cells[18].source.replace('"Natural connected components."', '"Connected components after exact dominance."').replace('"Conflict edges in the full graph."', '"Conflict edges in the reduced solver graph."')
notebook.cells[18].source = notebook.cells[18].source.replace('final_summary_rows = [', 'final_summary_rows = [\n    ("Original candidate nodes", original_conflict_graph.number_of_nodes(), "All input candidate contacts."),\n    ("Original conflict edges", original_conflict_graph.number_of_edges(), "All conflicts before dominance."),\n    ("Contacts removed by global dominance", dominance_summary["removed_nodes"], "Exact removal of dominated alternatives."),\n    ("Contacts retained after global dominance", conflict_graph.number_of_nodes(), "Solver graph nodes before component preprocessing."),')
for cell in notebook.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        compile(cell.source, '<variant-cell>', 'exec')
nbformat.write(notebook, ROOT / 'results/02.A_mwis_pipeline_dominance_enhanced.ipynb')

config = json.loads((CONTROL / 'config.json').read_text(encoding='utf-8'))
config['run_name_mwis'] = 'baseline_s200_g50'
config['qubo_input_run'] = 'runs/mwis-dominance-enhanced/pipeline_results/baseline_s200_g50'
config['qubo_output_run_name'] = 'baseline_s200_g50'
(LOG / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
executor = (CONTROL / 'execute_run.py').read_text(encoding='utf-8')
executor = executor.replace("'results/runs/mwis'", "'results/runs/mwis-dominance-enhanced'").replace("'02.A_mwis_pipeline.ipynb'", "'02.A_mwis_pipeline_dominance_enhanced.ipynb'")
needle = '        assert notebook.metadata.kernelspec.name == \'python313\''
routing = '''        # Only output routing changes in the execution copy of the shared annealer.
        # Its canonical source, sampling, reconstruction and MILP are untouched.
        if stage == 'annealing':
            old_route = 'RESULTS_DIR / "runs" / PRODUCER_FOLDER / "qubo_annealing" / OUTPUT_RUN_NAME'
            new_route = 'RESULTS_DIR / "runs" / "mwis-dominance-enhanced" / "qubo_annealing" / OUTPUT_RUN_NAME'
            matches = 0
            for cell in notebook.cells:
                matches += cell.source.count(old_route)
                cell.source = cell.source.replace(old_route, new_route)
            assert matches == 1
        execution_source_sha256 = hashlib.sha256('\\n'.join(c.source for c in notebook.cells).encode()).hexdigest()
'''
executor = executor.replace(needle, routing + needle)
executor = executor.replace("'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),", "'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),\n                                         'execution_source_sha256': execution_source_sha256,\n                                         'routing_only_patch': stage == 'annealing',")
(LOG / 'execute_run.py').write_text(executor, encoding='utf-8')
(OUT / 'README.md').write_text('''# Dominance-enhanced MWIS baseline

This is an isolated experiment using the frozen 200-satellite/50-gateway baseline, 30/200 logical limits, 500 simulated annealing reads and the existing decomposition/MILP budgets. The only algorithmic addition is exact same-satellite dominance before component splitting and during recursive preprocessing.

The producer is `results/02.A_mwis_pipeline_dominance_enhanced.ipynb`. The executor uses the unchanged shared `3_qubo_annealing.ipynb`, applying only an output-directory replacement to its execution copy. Per-run configuration is passed with `SWE2_CONFIG`; the global configuration is unchanged. Original and reduced graphs plus removal witnesses are saved separately. The original campaign is preserved as the control.
''', encoding='utf-8')
print('Prepared copied notebook, isolated configuration, executor and preservation manifest.')
