"""Resume B through completed CPU preparation and marker-gated evaluation."""
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare import COORD,OUT,ROOT,read,sha,load_common
from evaluate import Budget

HERE=Path(__file__).resolve().parent


def translation_running():
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args=(process/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(arg.decode(errors='replace')==str(HERE/'translate_cpu.py') for arg in args):
            return True
    return False


def main():
    common=load_common()
    legacy=Path.home()/'.venvs/pl-mmb-legacy-cu130/bin/python'
    def state(phase,**extra):
        value=dict(schema_version=1,study_id='data-ood-20261030',generation=1,owner='B',
            status='in_progress',phase=phase,pid=os.getpid(),manual_annotations_requested=0,
            contract_sha256=sha(COORD/'COORDINATION.md'),**extra)
        Budget(common,'pipeline-status').json(COORD/'B_STATUS.json',value)
    def execute(name,python=legacy):
        print('B_STAGE_START',name,flush=True)
        environment=dict(os.environ,REQUESTS_CA_BUNDLE=str(ROOT/'combined-ca.pem'),SSL_CERT_FILE=str(ROOT/'combined-ca.pem'))
        subprocess.run([str(python),str(HERE/name)],env=environment,check=True)
        print('B_STAGE_COMPLETE',name,flush=True)
    if (COORD/'B_RESULTS_READY.json').exists():
        common.verify_marker(COORD/'B_RESULTS_READY.json')
        print('B_ALREADY_COMPLETE')
        return
    state('translation_CPU',GPU_started=False)
    while not (OUT/'translation_summary.json').exists() or translation_running():
        if not translation_running() and not (OUT/'translation_summary.json').exists():
            raise RuntimeError('CPU translator stopped without final summary; preserve cache and inspect its error')
        time.sleep(10)
    if not (OUT/'flickr_translation_QA_summary.json').exists():
        execute('qa_flickr_cpu.py')
    state('final_PL_universe_audit_CPU',GPU_started=False)
    execute('audit_candidates.py')
    state('freezing_data_CPU',GPU_started=False)
    execute('freeze_cpu.py')
    state('evaluation_requires_all_markers_and_shared_GPU_lock',GPU_started=False)
    execute('evaluate.py',Path('/usr/bin/python3'))
    state('component_bootstrap_and_reporting_CPU',GPU_started=True,GPU_lock_released=True)
    execute('report_results.py')
    print('SESSION_B_COMPLETE',flush=True)


if __name__=='__main__':
    main()
