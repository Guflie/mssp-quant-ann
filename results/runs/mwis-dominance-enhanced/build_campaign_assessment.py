"""Compare seven validated dominance/control pairs; never execute experiments."""
import csv
import hashlib
import json
from pathlib import Path

VARIANT = Path(__file__).resolve().parent
ROOT = next(p for p in VARIANT.parents if (p / 'results/_config.json').is_file())
OUTPUT = VARIANT / 'experiment_comparison/campaign_assessment'
CASES = [
    ('s100_g50', 100, 50, 30, 200, 'exp2_s100_g50', 'exp2_s100_g50'),
    ('baseline', 200, 50, 30, 200, 'exp1_s200_g50', 'baseline_s200_g50'),
    ('s500_g50', 500, 50, 30, 200, 'exp2_s500_g50', 'exp2_s500_g50'),
    ('s200_g25', 200, 25, 30, 200, 'exp3_s200_g25', 'exp3_s200_g25'),
    ('s200_g100', 200, 100, 30, 200, 'exp3_s200_g100', 'exp3_s200_g100'),
    ('n20_c100', 200, 50, 20, 100, 'exp4_s200_g50_n20_c100', 'exp4_s200_g50_n20_c100'),
    ('n50_c350', 200, 50, 50, 350, 'exp4_s200_g50_n50_c350', 'exp4_s200_g50_n50_c350'),
]
SOURCES = {}


def read(path):
    payload = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def rows(path):
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def duration(seconds):
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}h{minutes:02d}m{seconds:02d}s' if hours else f'{minutes}m{seconds:02d}s'


def main():
    reference = read(VARIANT / 'execution_logs/baseline_s200_g50/execution_summary.json')
    controls = rows(ROOT / 'results/figures/csp_mwis_reordered/comparison_metrics.csv')
    records, pairs = [], []
    for key, sats, gateways, variables, couplers, control_name, variant_name in CASES:
        original = next(r for r in controls if r['method'] == 'MWIS' and r['scenario'] == key)
        csp = next(r for r in controls if r['method'] == 'CSP' and r['scenario'] == key)
        executions, validations = [], []
        for method, folder in [('Current MWIS', ROOT / 'results/runs/mwis/execution_logs/new_pipeline_results_reordered' / control_name),
                               ('Dominance MWIS', VARIANT / 'execution_logs' / variant_name)]:
            e, v = read(folder / 'execution_summary.json'), read(folder / 'validation_report.json')
            assert e['status'] == 'completed_and_validated' and v['validation_passed']
            assert v['selected_conflicts'] == 0
            assert (v['satellites_with_slots'], v['antennas_with_slots']) == (sats, gateways)
            assert (e['config']['max_qpu_terminal_nodes'], e['config']['max_qpu_terminal_couplers']) == (variables, couplers)
            assert e['packages'] == reference['packages']
            executions.append(e)
            validations.append(v)
        ce, de = executions
        cv, dv = validations
        varying = {'run_name_mwis', 'qubo_input_run', 'qubo_output_run_name'}
        assert {k: v for k, v in ce['config'].items() if k not in varying} == {k: v for k, v in de['config'].items() if k not in varying}
        assert cv['input_sha256'] == dv['input_sha256'] == original['input_sha256']
        source_input = ROOT / de['config']['candidate_slot_csv']
        SOURCES[source_input.relative_to(ROOT).as_posix()] = hashlib.sha256(source_input.read_bytes()).hexdigest()
        assert SOURCES[source_input.relative_to(ROOT).as_posix()] == dv['input_sha256']
        assert next(s['source_sha256'] for s in ce['stages'] if s['stage'] == 'annealing') == next(s['source_sha256'] for s in de['stages'] if s['stage'] == 'annealing')
        assert [(s['notebook'], s['source_sha256']) for s in de['stages']] == [(s['notebook'], s['source_sha256']) for s in reference['stages']]
        producer = VARIANT / 'pipeline_results' / variant_name
        summary = {r['metric']: r['value'] for r in rows(producer / 'final_summary.csv')}
        parts = rows(producer / 'components_summary.csv')
        stages = {s['stage']: s for s in de['stages']}
        def cell(stage, index):
            return next(c['seconds'] for c in stages[stage]['cells'] if c['original_cell'] == index)
        for enhanced in [False, True]:
            v = dv if enhanced else cv
            records.append({
                'scenario': key, 'method': 'Dominance MWIS' if enhanced else 'Current MWIS',
                'satellites': sats, 'gateways': gateways, 'variable_limit': variables, 'coupler_limit': couplers,
                'run': variant_name if enhanced else 'new_pipeline_results_reordered/' + control_name,
                'input_sha256': v['input_sha256'], 'candidate_contacts': v['candidate_slots'],
                'retained_contacts': dv['retained_solver_nodes'] if enhanced else int(original['retained_graph_contacts']),
                'largest_component': max(int(r['node_count']) for r in parts) if enhanced else int(original['largest_component_contacts']),
                'components': v['natural_components'], 'sampled_qubos': v['terminal_qubos'],
                'total_reads': v['terminal_qubos'] * de['config']['num_reads'],
                'incomplete_leaves': int(float(summary['Incomplete decomposition nodes (all states)'])) if enhanced else int(original['incomplete_leaves']),
                'pre_milp_objective': v['annealed_objective_before_milp'],
                'classical_reference_objective': v['classical_baseline_objective'],
                'sampling_objective_change': v['annealed_objective_before_milp'] - v['classical_baseline_objective'],
                'pre_milp_gap_percent': dv['pre_milp_relative_gap_against_certified_optimum_percent'] if enhanced else float(original['pre_milp_gap_to_certified_optimum_percent']),
                'final_objective': v['final_feasible_objective'], 'final_upper_bound': v['hybrid_upper_bound'],
                'certified_optimal': v['status'] == 'hybrid_exact', 'selected_contacts': v['selected_contacts'],
                'selected_conflicts': v['selected_conflicts'],
                'final_annealed_contacts': dv['contact_origins'].get('annealing', 0) if enhanced else int(original['final_contacts_annealing']),
                'final_milp_contacts': dv['contact_origins'].get('MILP', 0) if enhanced else int(original['final_contacts_milp']),
                'decomposition_seconds': cell('mwis', 11) if enhanced else float(original['time_decomposition']),
                'classical_evaluation_seconds': cell('mwis', 16) if enhanced else float(original['time_classical_verification']),
                'total_notebook_seconds': sum(s['wall_seconds'] for s in de['stages']) if enhanced else float(original['total_notebook_seconds']),
            })
        a, b = records[-2:]
        pairs.append({'scenario': key, 'input_sha256': dv['input_sha256'],
                      'equal_final_objective': abs(a['final_objective'] - b['final_objective']) < 1e-6,
                      'dominance_pre_milp_gap_improvement_pp': a['pre_milp_gap_percent'] - b['pre_milp_gap_percent'],
                      'dominance_runtime_ratio': b['total_notebook_seconds'] / a['total_notebook_seconds'],
                      'retained_annealing_contact_change': b['final_annealed_contacts'] - a['final_annealed_contacts'],
                      'dominance_csp_final_annealed_contacts_match': b['final_annealed_contacts'] == int(csp['final_contacts_annealing']),
                      'dominance_csp_final_milp_contacts_match': b['final_milp_contacts'] == int(csp['final_contacts_milp'])})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in [('all_seven_configurations', records), ('paired_comparison', pairs)]:
        with (OUTPUT / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        (OUTPUT / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    lines = ['# Dominance-enhanced MWIS: complete campaign assessment', '',
             'Seven distinct configurations compare the preserved reordered MWIS campaign with the isolated dominance variant. '
             'The 200/50, 30/200 baseline is counted once; its reuse across experiment groups is not an independent repetition. '
             'Within each pair, inputs, package versions, logical limits, 500 reads per terminal, seed and other configuration values match. '
             'The producer adds same-satellite dominance; the shared annealer changes only its output routing in memory.', '',
             'Values show current MWIS → dominance-enhanced MWIS.', '',
             '| Scenario | Pre-MILP gap (%) | Final annealed contacts | Final MILP contacts | Sampled QUBOs | Notebook time |',
             '|---|---:|---:|---:|---:|---|']
    for a, b in zip(records[::2], records[1::2]):
        label = f"{a['satellites']}/{a['gateways']}; {a['variable_limit']}/{a['coupler_limit']}"
        lines.append(f"| {label} | {a['pre_milp_gap_percent']:.3f} → {b['pre_milp_gap_percent']:.3f} | {a['final_annealed_contacts']} → {b['final_annealed_contacts']} | {a['final_milp_contacts']} → {b['final_milp_contacts']} | {a['sampled_qubos']} → {b['sampled_qubos']} | {duration(a['total_notebook_seconds'])} → {duration(b['total_notebook_seconds'])} |")
    faster = sum(p['dominance_runtime_ratio'] < 1 for p in pairs)
    more_annealed = sum(p['retained_annealing_contact_change'] > 0 for p in pairs)
    better_pre = sum(p['dominance_pre_milp_gap_improvement_pp'] > 1e-6 for p in pairs)
    equal = sum(p['equal_final_objective'] for p in pairs)
    dominance_records = records[1::2]
    matching_reference = sum(abs(r['sampling_objective_change']) < 1e-6 for r in dominance_records)
    deficit_cases = [r for r in dominance_records if r['sampling_objective_change'] < -1e-6]
    lines += ['', f'Dominance improves pre-MILP quality in {better_pre}/7 configurations, retains more final annealed contacts in {more_annealed}/7, '
              f'and is faster in {faster}/7 observed runs. Final objectives agree in {equal}/7 pairs. '
              f"All {sum(r['certified_optimal'] for r in records)}/14 records are certified optimal; all 14 have zero conflicts.", '',
              'The reduction is supported by the existing exhaustive tests and per-run independent dominance-witness checks. '
              'Its strongest practical benefits are reduced fallback dependence and better pre-MILP schedules, rather than a higher final objective. '
              'Its runtime effect depends on the graph and bounded decomposition tree. The 500/50 case is the clearest counterexample: '
              'only a small pre-MILP improvement, unchanged retained annealing contribution and approximately two hours of separator decomposition.', '',
              'The 200/50 baseline and terminal-profile timings also include exact classical evaluation of all alternative terminals. '
              'The unchanged implementation enumerates subsets through 22 vertices and uses a weighted-clique solver above that threshold, through 50 vertices. '
              'The traces show some smaller enumerated terminals taking substantially longer than larger clique-solved terminals. '
              'This is a verification implementation bottleneck; observed timing differences should not be attributed solely to dominance, QUBO size or sampling. '
              'No solver choice or algorithm was changed mid-campaign.', '',
              f'The sampled pre-MILP objective matches the classical tree reference in {matching_reference}/7 dominance configurations. '
              + ('For ' + ', '.join(r['scenario'] + f" it is {-r['sampling_objective_change']:.3f} lower" for r in deficit_cases)
                 + '. The subsequent MILP resolves the final gap. ' if deficit_cases else '')
              + 'No improvement over the exact classical terminal-reference objective was observed. '
              'This distinguishes a benefit of the reduction/decomposition pipeline from a benefit attributable to sampling alone.', '',
              'My recommendation is to retain both complete campaigns and keep dominance as a separately evaluated improvement for now. '
              'The mathematical reduction is useful and safe for this model. A blanket replacement with the current implementation would conceal substantial runtime regressions. '
              'Before adopting it as the default, test a faster exact terminal-evaluation strategy and a way to limit expensive separator exploration on large, weakly separated components. '
              'Those changes require validation and fresh timing measurements; they were not implemented in this campaign.', '',
              'Counts of sampled QUBOs include discarded alternative branches. Final provenance records contacts from sampled terminals that survive branch selection and MILP replacement. '
              'A MILP-replaced component is attributed to MILP as a whole, even when some contact identities also appeared in the pre-MILP incumbent. '
              'These are complementary measurements, not evidence that sampling alone caused an objective improvement. '
              'Sampling was classical simulated annealing; no quantum hardware or physical embedding was tested.', '',
              'If the dominance campaign is adopted for the report, regenerate the CSP/MWIS figures and revise the analysis explicitly around the new implementation. '
              f"The final annealed/MILP contact counts now both match CSP in {sum(p['dominance_csp_final_annealed_contacts_match'] and p['dominance_csp_final_milp_contacts_match'] for p in pairs)}/7 matched configurations. "
              'This changes the earlier provenance comparison and supports a more careful distinction between preprocessing and formulation effects. '
              'CSP has a different separator method, encoding, 1,000 reads per terminal and incomplete hardware provenance, so runtime comparisons remain descriptive.', '',
              'All comparisons use single saved executions. Pre-MILP gaps are retrospective distances to certified final optima; '
              'the looser bounds available before fallback are different metrics. Cross-scenario active-resource identities are not fully nested. '
              'Only terminal-size sensitivity uses the identical input across all profiles. Certificates concern the finite candidate-contact model, not unrestricted operational MMSP.', '',
              'Detailed source hashes and paired records are saved beside this assessment. No experiment, official figure or original analysis was modified.']
    (OUTPUT / 'adoption_assessment.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    source_script = Path(__file__).resolve()
    SOURCES[source_script.relative_to(ROOT).as_posix()] = hashlib.sha256(source_script.read_bytes()).hexdigest()
    (OUTPUT / 'source_manifest.json').write_text(json.dumps(SOURCES, indent=2) + '\n', encoding='utf-8')
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in SOURCES.items())
    print(json.dumps({'configurations': 7, 'records': 14, 'equal_final_objectives': equal,
                      'improved_pre_milp': better_pre, 'more_retained_annealing': more_annealed,
                      'faster': faster, 'source_hashes_checked': len(SOURCES),
                      'assessment': str(OUTPUT / 'adoption_assessment.md')}, indent=2))


if __name__ == '__main__':
    main()
