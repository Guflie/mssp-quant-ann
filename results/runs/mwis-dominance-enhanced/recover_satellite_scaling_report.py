"""Recover reporting only, preserving the executed driver and all run artifacts.

The all-exact 100/50 producer exports its zero incomplete-leaf count as ``0.0``.
Read this numeric CSV count as a float before converting to an integer. The
original driver stays byte-identical to its recorded preservation manifest.
No notebook, sampling or solver is executed by this reporting recovery.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
driver = VARIANT_DIR / 'run_satellite_scaling.py'
source = driver.read_text(encoding='utf-8')
before = "int(producer['Incomplete decomposition nodes (all states)'])"
after = "int(float(producer['Incomplete decomposition nodes (all states)']))"
assert source.count(before) == 1
namespace = {'__file__': str(driver), '__name__': 'dominance_reporting_recovery'}
exec(compile(source.replace(before, after), str(driver), 'exec'), namespace)
namespace['report']()
preserved = namespace['check_preservation']()
campaign = namespace['CAMPAIGN'] / 'campaign.json'
state = json.loads(campaign.read_text(encoding='utf-8'))
assert state['status'] == 'failed' and state['completed_runs'] == ['exp2_s100_g50', 'exp2_s500_g50']
backup = campaign.with_name('campaign_before_reporting_recovery.json')
assert not backup.exists()
backup.write_bytes(campaign.read_bytes())
state.pop('error')
finished = datetime.now(timezone.utc)
state.update(status='completed_and_validated', updated_utc=finished.isoformat(),
             wall_seconds=(finished - datetime.fromisoformat(state['started_utc'])).total_seconds(),
             preserved_artifacts_checked=preserved,
             report=str(namespace['OUTPUT'] / 'satellite_scaling_report.md'),
             reporting_recovery={'reason': 'Numeric CSV count 0.0 requires int(float(value))',
                                 'executed_driver_preserved': True,
                                 'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
                                 'recovery_script': Path(__file__).name,
                                 'notebooks_or_solvers_rerun': False})
campaign.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': state['status'], 'preserved_files': preserved,
                  'report': state['report']}, indent=2))
