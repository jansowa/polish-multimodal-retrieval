"""Read-only end-to-end check of final B handoffs, tables and released budgets."""
import csv
from pathlib import Path
from prepare import ROOT,OUT,COORD,load_common,read,sha,verify_marker
from handoff import verify_A


def main():
    common=load_common()
    verify_A()
    frozen=verify_marker('B_EVAL_FROZEN')
    ready=common.verify_marker(COORD/'B_RESULTS_READY.json')
    if (ready['owner'],ready['phase'],ready['status'],ready['generation'])!=('B','B_RESULTS_READY','complete',1):
        raise ValueError('Invalid final B marker envelope')
    for name in ('A_FINALISTS_FROZEN','A_GPU_DONE','B_EVAL_FROZEN'):
        if ready[name+'_sha256']!=sha(COORD/(name+'.json')):
            raise ValueError('Final dependency marker changed: '+name)
    summary=read(OUT/'summary.json')
    if summary['status']!='complete' or len(summary['models'])!=9 or len(summary['references'])!=3:
        raise ValueError('Incomplete model results')
    expected={'metrics.csv':480,'seed_mean_SD.csv':120,'paired_CI.csv':440}
    for name,count in expected.items():
        with (OUT/name).open(encoding='utf-8',newline='') as stream:
            actual=sum(1 for _ in csv.DictReader(stream))
        if actual!=count:
            raise ValueError(f'{name}: expected {count}, actual {actual}')
    registry=read(COORD/'DISK_RESERVATIONS.json')
    if any(sum(record['remaining'].values()) or record['temporary_bytes'] for record in registry['reservations'].values() if record['owner']=='B'):
        raise ValueError('B still has unwritten disk reservations')
    active=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            arguments=(process/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(argument.decode(errors='replace')==str(Path(__file__).parent/name) for argument in arguments
               for name in ('run.py','evaluate.py','translate_cpu.py','report_results.py','vizwiz_assets.py')):
            active.append(process.name)
    if active:
        raise ValueError('B worker processes remain: '+repr(active))
    print('FINAL_VERIFICATION_PASSED',{'ready_artifacts':len(ready['artifacts']),'CSV_rows':expected,
        'datasets':{name:row['images'] for name,row in summary['dataset_decision']['counts'].items()},
        'B_committed_bytes':registry['committed']['B'],'B_reservations':0,'B_active_workers':0})


if __name__=='__main__':
    main()
