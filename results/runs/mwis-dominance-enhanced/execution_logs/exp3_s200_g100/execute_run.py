"""Execute this frozen MWIS reordered active-count baseline scenario and save notebooks, timings, and provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import threading
import psutil
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

LOG_DIR = Path(__file__).resolve().parent
ROOT = next(parent for parent in LOG_DIR.parents if (parent / 'results/_config.json').is_file())
CONFIG_PATH = LOG_DIR / 'config.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
PACKAGES = ['numpy', 'pandas', 'networkx', 'scipy', 'dwave-samplers', 'dimod',
            'nbclient', 'nbformat', 'ipykernel', 'jupyter-client']


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(event, **fields):
    record = {'utc': utc_now(), 'event': event, **fields}
    with (LOG_DIR / 'execution_events.jsonl').open('a', encoding='utf-8') as output:
        output.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


class Tee:
    """Keep captured cell outputs while making long-cell progress visible."""
    def __init__(self, captured, console):
        self.captured = captured
        self.console = console

    def write(self, text):
        self.captured.write(text)
        self.console.write(text)
        self.console.flush()
        return len(text)

    def flush(self):
        self.captured.flush()
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.captured, name)


def main():
    assert sys.version.split()[0] == '3.13.14', sys.version
    run = CONFIG['run_name_mwis']
    for folder in ['pipeline_results', 'qubo_annealing']:
        target = ROOT / 'results/runs/mwis-dominance-enhanced' / folder / run
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite an existing run: {target}')
    os.environ['SWE2_CONFIG'] = str(CONFIG_PATH)
    os.environ['PYTHONUTF8'] = '1'
    os.environ['IPYTHONDIR'] = str(LOG_DIR / 'ipython_runtime')
    started = time.perf_counter()
    provenance = {
        'run_name': run, 'started_utc': utc_now(), 'status': 'running',
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_modified': True, 'config': CONFIG,
        'execution_method': 'IPython in-process notebook code-cell execution',
        'startup_attempt': None,
        'input_sha256': hashlib.sha256((ROOT / CONFIG['candidate_slot_csv']).read_bytes()).hexdigest(),
        'packages': {name: importlib.metadata.version(name) for name in PACKAGES},
        'stages': [],
    }

    def save_provenance():
        (LOG_DIR / 'execution_summary.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')

    save_provenance()
    progress = {'stage': 'starting', 'original_cell': None}
    stop_heartbeat = threading.Event()
    process = psutil.Process(os.getpid())

    def heartbeat():
        while not stop_heartbeat.wait(20):
            try:
                cpu = process.cpu_times()
                log('heartbeat', **dict(progress),
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                    cpu_seconds=round(cpu.user + cpu.system, 3),
                    resident_mb=round(process.memory_info().rss / 1024 / 1024, 2))
            except Exception:
                pass  # Monitoring must never change notebook execution.

    threading.Thread(target=heartbeat, daemon=True).start()
    for stage, filename in [('mwis', '02.A_mwis_pipeline_dominance_enhanced.ipynb'), ('annealing', '3_qubo_annealing.ipynb')]:
        path = ROOT / 'results' / filename
        notebook = nbformat.read(path, as_version=4)
        # Only output routing changes in the execution copy of the shared annealer.
        # Its canonical source, sampling, reconstruction and MILP are untouched.
        if stage == 'annealing':
            old_route = 'RESULTS_DIR / "runs" / PRODUCER_FOLDER / "qubo_annealing" / OUTPUT_RUN_NAME'
            new_route = 'RESULTS_DIR / "runs" / "mwis-dominance-enhanced" / "qubo_annealing" / OUTPUT_RUN_NAME'
            matches = 0
            for cell in notebook.cells:
                matches += cell.source.count(old_route)
                cell.source = cell.source.replace(old_route, new_route)
            assert matches == 1
        execution_source_sha256 = hashlib.sha256('\n'.join(c.source for c in notebook.cells).encode()).hexdigest()
        assert notebook.metadata.kernelspec.name == 'python313'
        runtime_file = LOG_DIR / f'{stage}_kernel_environment.json'
        # Instrument only the execution copy; original notebook cells stay intact.
        runtime_source = (
            'import sys, json, platform, importlib.metadata\n'
            'from pathlib import Path\n'
            "assert sys.version.split()[0] == '3.13.14', sys.version\n"
            f'Path({str(runtime_file)!r}).write_text(json.dumps('
            "{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),"
            f"'packages':{{name:importlib.metadata.version(name) for name in {PACKAGES!r}}}}}, indent=2), encoding='utf-8')\n"
        )
        notebook.cells.insert(0, nbformat.v4.new_code_cell(runtime_source))
        timings = []
        cell_started = {}

        def on_start(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                progress.update(stage=stage, original_cell=cell_index - 1)
                cell_started[cell_index] = time.perf_counter()
                log('cell_start', stage=stage, executed_cell=cell_index,
                    original_cell=cell_index - 1, first_line=cell.source.splitlines()[0] if cell.source else '')

        def on_executed(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                duration = time.perf_counter() - cell_started[cell_index]
                timings.append({'executed_cell': cell_index, 'original_cell': cell_index - 1,
                                'seconds': duration})
                log('cell_complete', stage=stage, executed_cell=cell_index, seconds=round(duration, 3))
                # Cell streams were already mirrored live to the console.

        log('stage_start', stage=stage, notebook=filename)
        stage_started = time.perf_counter()
        status = 'completed'
        try:
            # Separate namespaces mirror the original separate notebook kernels.
            InteractiveShell.clear_instance()
            shell = InteractiveShell.instance(user_ns={}, config=Config({'HistoryManager': {'enabled': False}}))
            os.chdir(path.parent)
            execution_count = 0
            for cell_index, cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                execution_count += 1
                on_start(cell, cell_index)
                console_out, console_err = sys.stdout, sys.stderr
                with capture_output() as captured:
                    sys.stdout = Tee(sys.stdout, console_out)
                    sys.stderr = Tee(sys.stderr, console_err)
                    result = shell.run_cell(cell.source, store_history=False)
                cell.execution_count = execution_count
                cell.outputs = []
                for stream, text in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
                    if text:
                        cell.outputs.append(nbformat.v4.new_output('stream', name=stream, text=text))
                for rich_output in captured.outputs:
                    cell.outputs.append(nbformat.v4.new_output('display_data', data=rich_output.data, metadata=rich_output.metadata))
                on_executed(cell, cell_index)
                error = result.error_before_exec or result.error_in_exec
                if error:
                    raise error
        except Exception:
            status = 'failed'
            (LOG_DIR / f'{stage}_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise
        finally:
            InteractiveShell.clear_instance()
            os.chdir(ROOT)
            nbformat.write(notebook, LOG_DIR / f'{stage}_executed.ipynb')
            duration = time.perf_counter() - stage_started
            provenance['stages'].append({'stage': stage, 'notebook': filename, 'status': status,
                                         'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                         'execution_source_sha256': execution_source_sha256,
                                         'routing_only_patch': stage == 'annealing',
                                         'wall_seconds': duration, 'cells': timings})
            provenance['status'] = 'failed' if status == 'failed' else 'running'
            provenance['wall_seconds'] = time.perf_counter() - started
            save_provenance()
            log('stage_end', stage=stage, status=status, seconds=round(duration, 3))
    stop_heartbeat.set()
    provenance['finished_utc'] = utc_now()
    provenance['status'] = 'completed_pending_independent_validation'
    save_provenance()
    log('execution_complete', seconds=round(provenance['wall_seconds'], 3))


if __name__ == '__main__':
    main()
