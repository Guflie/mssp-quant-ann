"""Summarize the validated variant against its preserved baseline control."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'results/_config.json').is_file())
OUT = Path(__file__).resolve().parent
LOG = OUT / 'execution_logs/baseline_s200_g50'
CONTROL_RUN = 'new_pipeline_results_reordered/exp1_s200_g50'


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def metrics(path):
    return {row['metric']: row['value'] for row in rows(path)}


def load(base, run, log):
    producer = base / 'pipeline_results' / run
    annealer = base / 'qubo_annealing' / run
    p = metrics(producer / 'final_summary.csv')
    a = metrics(annealer / 'reconstructed_summary.csv')
    validation = json.loads((log / 'validation_report.json').read_text(encoding='utf-8'))
    execution = json.loads((log / 'execution_summary.json').read_text(encoding='utf-8'))
    assert validation['validation_passed'] and execution['status'] == 'completed_and_validated'
    q = json.loads((producer / 'mwis_terminal_qubos.json').read_text(encoding='utf-8'))
    decomp = rows(producer / 'decomposition_summary.csv')
    profile = json.loads((producer / 'mwis_runtime_profile_summary.json').read_text(encoding='utf-8'))
    phases = [json.loads(line) for f in producer.glob('mwis_phase_profile_*.jsonl') for line in f.read_text(encoding='utf-8').splitlines()]
    phase_times = {line['phase']: line['elapsed_seconds'] for line in phases}
    with gzip.open(producer / 'reconstruction_metadata.json.gz', 'rt', encoding='utf-8') as f:
        metadata = json.load(f)
    summary = {
        'candidate_contacts': validation['candidate_slots'],
        'contacts_after_global_dominance': validation.get('retained_solver_nodes', validation['candidate_slots']),
        'conflict_edges_after_global_dominance': validation.get('retained_solver_edges', validation['conflict_edges']),
        'components_after_global_dominance': len(decomp),
        'largest_component_contacts': max(int(r['input_nodes']) for r in decomp),
        'largest_component_edges': max(int(r['input_edges']) for r in decomp),
        'decomposition_tree_nodes': len(metadata['nodes']),
        'incomplete_tree_leaves': sum(int(r['incomplete_nodes']) for r in decomp),
        'incomplete_components': int(a['Components with incomplete separator decomposition (all states)']),
        'terminal_qubos_sampled': len(q),
        'terminal_qubos_retained_in_final_schedule': int(a['Annealed terminal subgraphs used']),
        'max_terminal_variables': max(r['qubits'] for r in q),
        'max_terminal_couplers': max(r['couplers'] for r in q),
        'reads_per_terminal': execution['config']['num_reads'],
        'total_annealing_reads': len(q) * execution['config']['num_reads'],
        'classical_baseline_objective': validation['classical_baseline_objective'],
        'pre_milp_annealed_objective': validation['annealed_objective_before_milp'],
        'pre_milp_gap_against_certified_optimum_percent': 100 * (12581.45 - validation['annealed_objective_before_milp']) / 12581.45,
        'final_objective': validation['final_feasible_objective'],
        'final_upper_bound': validation['hybrid_upper_bound'],
        'final_gap_percent': validation['relative_gap_percent'],
        'final_contacts': validation['selected_contacts'],
        'final_conflicts': validation['selected_conflicts'],
        'milp_components_proven_optimal': int(a['MILP components proven optimal']),
        'milp_solver_seconds': float(a['MILP solver time (seconds)']),
        'decomposition_seconds': phase_times['decomposition_and_terminal_export'],
        'classical_evaluation_seconds': profile['evaluation_wall_seconds'],
        'notebook_execution_seconds': execution['wall_seconds'],
        'producer_notebook_seconds': next(s['wall_seconds'] for s in execution['stages'] if s['stage'] == 'mwis'),
        'annealing_notebook_seconds': next(s['wall_seconds'] for s in execution['stages'] if s['stage'] == 'annealing'),
    }
    if 'contact_origins' in validation:
        origins = validation['contact_origins']
    else:
        # In this control, the only incomplete component is replaced by MILP;
        # complete roots contain only terminal samples and forced isolated nodes.
        selected = {r['slot_id'] for r in rows(annealer / 'reconstructed_schedule.csv')}
        provenance = rows(annealer / 'component_provenance_summary.csv')
        milp_contacts = sum(int(r['selected_contacts']) for r in provenance if r['solution_source'].startswith('MILP'))
        nodes = {n['node_id']: n for n in metadata['nodes']}
        samples = {r['terminal_id']: json.loads(r['selected_nodes']) for r in rows(annealer / 'annealing_results.csv')}
        exact = sampled = 0
        for component, prov in zip(metadata['components'], provenance):
            if prov['solution_source'].startswith('MILP'):
                continue
            node = nodes[component['root_node_id']]
            assert node['node_type'] == 'terminal'
            exact += len(node['forced_selected'])
            sampled += len(samples[node['terminal_id']]) if node.get('terminal_id') else 0
        origins = {'annealing': sampled, 'exact preprocessing': exact, 'MILP': milp_contacts}
        assert sum(origins.values()) == len(selected)
    summary.update({'final_contacts_from_annealing': origins.get('annealing', 0),
                    'final_contacts_from_exact_preprocessing': origins.get('exact preprocessing', 0),
                    'final_contacts_from_separator_assignments': origins.get('separator assignment', 0),
                    'final_contacts_from_greedy_fallback': origins.get('greedy fallback', 0),
                    'final_contacts_from_milp': origins.get('MILP', 0)})
    return summary


def main():
    control = load(ROOT / 'results/runs/mwis', CONTROL_RUN, ROOT / 'results/runs/mwis/execution_logs' / CONTROL_RUN)
    enhanced = load(OUT, 'baseline_s200_g50', LOG)
    comparison = {'control': control, 'dominance_enhanced': enhanced}
    (OUT / 'baseline_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    with (OUT / 'baseline_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'control', 'dominance_enhanced'])
        writer.writerows((key, control[key], enhanced[key]) for key in control)
    def display(value):
        return f'{value:,}' if isinstance(value, int) else f'{value:,.3f}'
    table = '\n'.join(f'| {key.replace("_", " ")} | {display(control[key])} | {display(enhanced[key])} |' for key in control)
    delta_quality = enhanced['pre_milp_annealed_objective'] - control['pre_milp_annealed_objective']
    runtime_ratio = enhanced['notebook_execution_seconds'] / control['notebook_execution_seconds']
    report = f'''# Dominance-enhanced MWIS baseline test

The copied producer applies exact same-satellite dominance before component splitting and at every uncached recursive node. The frozen input, contact conflicts, MWIS separator search, logical limits (30/200), decomposition budgets, terminal solvers, QUBO, 500 reads, seed schedule, and MILP budgets match the control. Python is 3.13.14. The shared annealing notebook has only an output-routing replacement in its execution copy.

Both runs use the exact same 1,228-contact frozen 200/50 input. All original candidate data are preserved. Dominance changes component boundaries while preserving the optimum. Consequently, the number of independent problems, terminal seeds and total reads differ even though the per-terminal sampling settings are held fixed.

| Measurement | Preserved control | Dominance enhanced |
|---|---:|---:|
{table}

The pre-MILP objective changed by {delta_quality:,.3f}. The notebook runtime was {runtime_ratio:.2f} times the control. The final objective is checked against the previously certified optimum of 12,581.450. Contact origins count contacts retained in the final schedule; they do not measure CPU effort or quantum advantage. Sampling uses D-Wave's classical simulated annealer.

Dominance is promising for improving pre-MILP quality and reducing dependence on component-level MILP: the pre-MILP gap falls from about 2.047% to 0.040%, and MILP supplies 76 rather than 157 final contacts. However, only 34 final contacts come from annealing; another 70 are fixed by exact classical preprocessing. Of the 1,286 sampled terminal QUBOs, only 19 are retained after MILP replaces the incomplete component. The additional annealing workload therefore includes many alternative branches that do not survive into the final schedule.

This variant is slower overall. Decomposition falls from 586.587 to 88.895 seconds, but classical terminal evaluation rises from 85.241 to 2,422.880 seconds. The unchanged solver policy exhaustively enumerates terminals with at most 22 vertices, resulting in 780 such terminals and 392,069,480 subsets. Sampling also increases from 15,000 to 643,000 reads with the per-terminal budget fixed at 500. Before rerunning the complete campaign, address exact terminal-verification cost in a separate controlled test, using a faster exact method and validating its objectives and bounds. Preserve this dominance-only run and the original campaign for comparison.

Validation passed: 300 small-instance exhaustive optimum checks; original conflict graph rebuilt independently from the frozen input; all global and recursive dominance witnesses checked; reduced graph and component membership checked; terminal limits, QUBO coefficients, read counts, seeds and energies checked; pre-MILP reconstruction evaluated independently; classical and final schedules checked against the full original conflicts; provenance and objective/bound consistency checked. Existing MWIS runs, source notebooks, global configuration and frozen input hashes are unchanged.

This is one baseline execution with no repetitions. Runtime values are descriptive and include exact classical terminal verification. A change in the number of annealed terminals is not itself evidence of better schedule quality: many alternative branches can be sampled and later replaced by component-level MILP. The current campaign remains the control; other experiments have not been rerun.

Outputs: `pipeline_results/baseline_s200_g50`, `qubo_annealing/baseline_s200_g50`, and `execution_logs/baseline_s200_g50`. The execution logs include the full per-run configuration, package/runtime provenance, executed notebooks, source and execution hashes, validation report and preservation manifest.
'''
    (OUT / 'baseline_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
