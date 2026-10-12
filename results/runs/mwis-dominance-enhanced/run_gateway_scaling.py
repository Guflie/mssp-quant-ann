"""Run the isolated dominance variant on the frozen gateway-scaling inputs.

The completed 200/50 baseline is reused. Source notebooks, global configuration,
control runs and the baseline are read-only. Fresh 200/25 and 200/100 outputs stay
inside mwis-dominance-enhanced. Run --prepare once, then --run.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
ROOT = next(p for p in VARIANT_DIR.parents if (p / 'results/_config.json').is_file())
BASELINE = VARIANT_DIR / 'execution_logs/baseline_s200_g50'
CAMPAIGN = VARIANT_DIR / 'execution_logs/exp3_gateway_scaling'
OUTPUT = VARIANT_DIR / 'experiment_comparison/exp3_gateway_scaling'
SCENARIOS = [(25, 'exp3_s200_g25', 'exp3_s200_g25'),
             (50, 'baseline_s200_g50', 'exp1_s200_g50'),
             (100, 'exp3_s200_g100', 'exp3_s200_g100')]
VARYING = {'candidate_slot_csv', 'run_name_mwis', 'qubo_input_run', 'qubo_output_run_name'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    assert sys.version.split()[0] == '3.13.14'
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'mwis-improvements'
    baseline = read(BASELINE / 'execution_summary.json')
    assert baseline['status'] == 'completed_and_validated'
    assert read(BASELINE / 'validation_report.json')['validation_passed']
    assert read(VARIANT_DIR / 'dominance_test_report.json')['passed']
    for stage in baseline['stages']:
        assert sha(ROOT / 'results' / stage['notebook']) == stage['source_sha256'], stage['notebook']
    for package, version in baseline['packages'].items():
        assert importlib.metadata.version(package) == version, package
    for g, _, control in SCENARIOS:
        folder = ROOT / 'results/runs/mwis/execution_logs/new_pipeline_results_reordered' / control
        validation, execution = read(folder / 'validation_report.json'), read(folder / 'execution_summary.json')
        assert validation['validation_passed'] and execution['status'] == 'completed_and_validated'
        assert validation['status'] == 'hybrid_exact'
        assert validation['satellites_with_slots'] == 200 and validation['antennas_with_slots'] == g
        assert sha(ROOT / execution['config']['candidate_slot_csv']) == validation['input_sha256']
        assert execution['packages'] == baseline['packages']
        for key, value in baseline['config'].items():
            if key not in VARYING:
                assert execution['config'][key] == value, key
    return baseline


def prepare():
    baseline = preflight()
    assert not CAMPAIGN.exists() and not OUTPUT.exists(), 'Never overwrite a campaign'
    for _, run, _ in SCENARIOS:
        if run == 'baseline_s200_g50':
            continue
        for kind in ['execution_logs', 'pipeline_results', 'qubo_annealing']:
            assert not (VARIANT_DIR / kind / run).exists()
    preserved = {}
    for directory in [ROOT / 'results/runs/mwis', VARIANT_DIR, ROOT / 'data/experiment_inputs',
                      ROOT / 'results/figures/csp_mwis_reordered']:
        for path in directory.rglob('*'):
            if path.is_file():
                preserved[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in [ROOT / 'results/_config.json', ROOT / 'results/02.A_mwis_pipeline.ipynb',
                 ROOT / 'results/02.A_mwis_pipeline_dominance_enhanced.ipynb',
                 ROOT / 'results/3_qubo_annealing.ipynb', ROOT / 'results/5_csp_mwis_comparison_plots.py']:
        preserved[path.relative_to(ROOT).as_posix()] = sha(path)
    CAMPAIGN.mkdir(parents=True)
    write(CAMPAIGN / 'preserved_artifacts_manifest.json', preserved)
    executor = (BASELINE / 'execute_run.py').read_text(encoding='utf-8')
    validator = (BASELINE / 'validate_run.py').read_text(encoding='utf-8')
    original = "    raw_graph.add_edges_from(expected_edges)\n"
    faster = original + "    original_edges = tuple(expected_edges)\n    def induced_edge_count(ids):\n        live = set(ids)\n        return sum(left in live and right in live for left, right in original_edges)\n"
    assert validator.count(original) == 1
    validator = validator.replace(original, faster)
    for old, new in [
        ("reduced_graph.subgraph(input_ids).number_of_edges()", "induced_edge_count(input_ids)"),
        ("reduced_graph.subgraph(survivors - set(node['forced_selected'])).number_of_edges()", "induced_edge_count(survivors - set(node['forced_selected']))"),
    ]:
        assert validator.count(old) == 1
        validator = validator.replace(old, new)
    manifest = []
    for g, run, control in SCENARIOS:
        control_folder = ROOT / 'results/runs/mwis/execution_logs/new_pipeline_results_reordered' / control
        v, e = read(control_folder / 'validation_report.json'), read(control_folder / 'execution_summary.json')
        manifest.append({'satellites': 200, 'gateways': g, 'run': run, 'reused': g == 50,
                         'control': control, 'input_sha256': v['input_sha256'],
                         'candidate_slots': v['candidate_slots'], 'control_certified_optimum': v['final_feasible_objective']})
        if g == 50:
            continue
        folder = VARIANT_DIR / 'execution_logs' / run
        folder.mkdir(parents=True)
        config = dict(baseline['config'], candidate_slot_csv=e['config']['candidate_slot_csv'],
                      run_name_mwis=run, qubo_input_run=f'runs/mwis-dominance-enhanced/pipeline_results/{run}',
                      qubo_output_run_name=run)
        assert {k for k in config if config[k] != baseline['config'][k]} == VARYING
        adapted = validator
        for before, after in [
            ("'8c7f74c2e3b02e7c8158429b0fa9f21df7bbdc3445de913fdf563f8c3e560c8e'", repr(v['input_sha256'])),
            ('len(by_id) == len(slots) == 1228', f"len(by_id) == len(slots) == {v['candidate_slots']}"),
            ("len({row['station_id'] for row in slots}) == 50", f"len({{row['station_id'] for row in slots}}) == {g}"),
            ("len({row['antenna_id'] for row in slots}) == 50", f"len({{row['antenna_id'] for row in slots}}) == {g}"),
            ('12581.45', repr(v['final_feasible_objective'])),
            ('Frozen baseline input changed', 'Frozen gateway-scaling input changed'),
        ]:
            assert before in adapted, before
            adapted = adapted.replace(before, after)
        for source in [executor, adapted]:
            compile(source, '<gateway-scaling executor>', 'exec')
        write(folder / 'config.json', config)
        write(folder / 'input_manifest_entry.json', manifest[-1])
        write(folder / 'preserved_artifacts_manifest.json', preserved)
        (folder / 'execute_run.py').write_text(executor, encoding='utf-8')
        (folder / 'validate_run.py').write_text(adapted, encoding='utf-8')
    write(CAMPAIGN / 'scenarios.json', manifest)
    print(json.dumps({'prepared': manifest, 'preserved_files': len(preserved)}, indent=2), flush=True)


def check_preservation():
    preserved = read(CAMPAIGN / 'preserved_artifacts_manifest.json')
    changed = [p for p, digest in preserved.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    assert not changed, changed
    return len(preserved)


def run():
    preflight()
    state_path = CAMPAIGN / 'campaign.json'
    assert not state_path.exists(), 'Existing campaign must be inspected before restarting'
    started = time.perf_counter()
    state = {'status': 'starting', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'reused_baseline': 'baseline_s200_g50', 'completed_runs': []}
    def save(status, **fields):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write(state_path, state)
        print(json.dumps(state), flush=True)
    def call(script, destination):
        with destination.open('w', encoding='utf-8') as output:
            process = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(script)], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', bufsize=1)
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end='', flush=True)
            assert process.wait() == 0, f'{script} failed; inspect saved outputs'
    try:
        for g, name, _ in SCENARIOS:
            if g == 50:
                continue
            folder = VARIANT_DIR / 'execution_logs' / name
            save('running_notebooks', current_run=name)
            call(folder / 'execute_run.py', folder / 'console.log')
            save('validating', current_run=name)
            call(folder / 'validate_run.py', folder / 'validation_console.log')
            assert read(folder / 'execution_summary.json')['status'] == 'completed_and_validated'
            state['completed_runs'].append(name)
            save('case_completed', preserved_artifacts_checked=check_preservation())
        save('generating_comparison', current_run=None)
        report()
        save('completed_and_validated', wall_seconds=time.perf_counter() - started,
             preserved_artifacts_checked=check_preservation(), report=str(OUTPUT / 'gateway_scaling_report.md'))
    except BaseException:
        save('failed', error=traceback.format_exc())
        raise


def report():
    preflight()
    controls = list(csv.DictReader((ROOT / 'results/figures/csp_mwis_reordered/comparison_metrics.csv').open(encoding='utf-8')))
    rows = []
    for g, run_name, control in SCENARIOS:
        folder = VARIANT_DIR / 'execution_logs' / run_name
        v, e = read(folder / 'validation_report.json'), read(folder / 'execution_summary.json')
        assert v['validation_passed'] and e['status'] == 'completed_and_validated'
        original = next(r for r in controls if r['method'] == 'MWIS' and r['run'].endswith('/' + control))
        assert original['input_sha256'] == v['input_sha256']
        producer = {r['metric']: r['value'] for r in csv.DictReader((VARIANT_DIR / 'pipeline_results' / run_name / 'final_summary.csv').open(encoding='utf-8'))}
        parts = list(csv.DictReader((VARIANT_DIR / 'pipeline_results' / run_name / 'components_summary.csv').open(encoding='utf-8')))
        stages = {s['stage']: s for s in e['stages']}
        def cell(stage, index):
            return next(c['seconds'] for c in stages[stage]['cells'] if c['original_cell'] == index)
        for method in ['Current MWIS', 'Dominance MWIS']:
            enhanced = method == 'Dominance MWIS'
            rows.append({'satellites': 200, 'gateways': g, 'method': method,
                         'reused_baseline': enhanced and g == 50, 'input_sha256': v['input_sha256'],
                         'candidate_contacts': v['candidate_slots'],
                         'retained_contacts': v['retained_solver_nodes'] if enhanced else int(original['retained_graph_contacts']),
                         'global_dominance_removed': v['global_dominance_removed'] if enhanced else 0,
                         'components': v['natural_components'] if enhanced else int(original['components']),
                         'largest_component': max(int(p['node_count']) for p in parts) if enhanced else int(original['largest_component_contacts']),
                         'sampled_qubos': v['terminal_qubos'] if enhanced else int(original['terminal_qubos']),
                         'total_reads': v['terminal_qubos'] * e['config']['num_reads'] if enhanced else int(original['total_reads']),
                         'incomplete_leaves': int(float(producer['Incomplete decomposition nodes (all states)'])) if enhanced else int(original['incomplete_leaves']),
                         'incomplete_components': v['incomplete_components'] if enhanced else int(original['incomplete_components']),
                         'pre_milp_objective': v['annealed_objective_before_milp'] if enhanced else float(original['pre_milp_objective']),
                         'pre_milp_gap_percent': v['pre_milp_relative_gap_against_certified_optimum_percent'] if enhanced else float(original['pre_milp_gap_to_certified_optimum_percent']),
                         'final_objective': v['final_feasible_objective'] if enhanced else float(original['final_objective']),
                         'final_upper_bound': v['hybrid_upper_bound'] if enhanced else float(original['final_upper_bound']),
                         'certified_optimal': v['status'] == 'hybrid_exact' if enhanced else original['certified_optimal'] == 'True',
                         'final_contacts': v['selected_contacts'] if enhanced else int(original['final_selected_contacts']),
                         'final_annealed_contacts': v['contact_origins'].get('annealing', 0) if enhanced else int(original['final_contacts_annealing']),
                         'final_milp_contacts': v['contact_origins'].get('MILP', 0) if enhanced else int(original['final_contacts_milp']),
                         'pre_milp_annealed_contacts': v['pre_milp_contact_origins'].get('annealing', 0) if enhanced else int(original['pre_milp_contacts_annealing']),
                         'pre_milp_greedy_contacts': v['pre_milp_contact_origins'].get('greedy fallback', 0) if enhanced else int(original['pre_milp_contacts_greedy_fallback']),
                         'decomposition_seconds': cell('mwis', 11) if enhanced else float(original['time_decomposition']),
                         'classical_evaluation_seconds': cell('mwis', 16) if enhanced else float(original['time_classical_verification']),
                         'producer_notebook_seconds': stages['mwis']['wall_seconds'] if enhanced else float(original['producer_notebook_seconds']),
                         'annealing_notebook_seconds': stages['annealing']['wall_seconds'] if enhanced else float(original['annealing_notebook_seconds']),
                         'total_notebook_seconds': sum(s['wall_seconds'] for s in e['stages']) if enhanced else float(original['total_notebook_seconds'])})
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT / 'gateway_scaling_summary.json', rows)
    with (OUTPUT / 'gateway_scaling_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Dominance-enhanced MWIS: gateway scaling', '',
             'Fresh 200/25 and 200/100 executions are compared with the preserved reordered MWIS controls. '
             'The completed dominance 200/50 baseline is reused. All runs use the same frozen per-scenario input, '
             'Python 3.13.14, package versions, 30/200 logical limits, 500 reads per terminal, seed 42 and decomposition/MILP budgets. '
             'Only the dominance producer differs; annealer output routing is changed in memory. No control, baseline, source notebook or global configuration is modified.', '',
             '| Gateways | Method | Retained contacts | Largest component | QUBOs | Incomplete leaves | Pre-MILP gap (%) | Final objective | Annealed / MILP contacts | Notebook seconds |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['gateways']} | {r['method']} | {r['retained_contacts']} | {r['largest_component']} | {r['sampled_qubos']} | {r['incomplete_leaves']} | {r['pre_milp_gap_percent']:.6f} | {r['final_objective']:.3f} | {r['final_annealed_contacts']} / {r['final_milp_contacts']} | {r['total_notebook_seconds']:.3f} |")
    lines += ['', 'Pre-MILP gaps are retrospective distances to the control campaign\'s certified optimum. '
              'Final annealing provenance counts only contacts retained after branch selection and MILP replacement; '
              'QUBO/read counts include discarded alternative branches. These are single executions, not replicated timing estimates. '
              'Notebook runtime excludes independent validation and outer orchestration; detailed phase times and pre-MILP provenance are in the CSV. '
              'Classical exact evaluation of terminals is included and can dominate runtime. Sampling is classical simulated annealing.', '',
              'The active-resource inputs are selected independently per scenario. Changing the gateway count also changes '
              'the selected satellite identities. Comparisons of the variant and control use an identical input; '
              'cross-scenario scaling also reflects these composition differences.', '',
              'All existing experiment artifacts and figure/analysis outputs were checked unchanged by SHA-256. '
              'Each new case was independently checked for original conflicts, schedule feasibility, objective/bound consistency, '
              'QUBO coefficients, caps, reads and seeds, dominance witnesses, reconstruction and final provenance.']
    (OUTPUT / 'gateway_scaling_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    assert sys.argv[1:] in [['--prepare'], ['--run'], ['--report']]
    {'--prepare': prepare, '--run': run, '--report': report}[sys.argv[1]]()
