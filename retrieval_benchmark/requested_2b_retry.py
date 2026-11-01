"""Detached repair queue for the two explicitly requested 2B embedders only."""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
REPORT=ROOT/'reports/f_series/S09-LEADERBOARD-2B-REPAIR'
RUN=ROOT/'training/runs/f_series/leaderboard-2b-repair-20261101'
CACHE=Path.home()/'.cache/pl_mmb_requested_2b_20261101/hub'
os.environ.update(PL_MMB_REQUESTED_REPORT=str(REPORT),PL_MMB_REQUESTED_RUN=str(RUN),
                  PL_MMB_REQUESTED_CACHE=str(CACHE),PL_MMB_REQUESTED_STAGE='S09-LEADERBOARD-2B-REPAIR')
from retrieval_benchmark import requested_queue as q
from retrieval_benchmark.core import read,write,sha,public_check


def prepare():
    if q.REG.exists(): return read(q.REG)
    previous=ROOT/'reports/leaderboard_requested_20261101/registration.json'
    reg=read(previous)
    chosen=['Qwen/Qwen3-VL-Embedding-2B','Alibaba-NLP/gme-Qwen2-VL-2B-Instruct']
    reg['jobs']=[next(j for j in reg['jobs'] if j['repository']==repo) for repo in chosen]
    doc=public_check(read(q.HERE/'space/results.json'))
    for job in reg['jobs']:
        done=q.completed_cells(doc,job['model_id'])
        job['missing']=[cell for cell in job['missing'] if tuple(cell) not in done]
    reg.update(created=q.now(),stage=q.STAGE,previous_registration_sha256=sha(previous),
               authorization='Owner explicitly requested repair and evaluation of both 2B embedders; no 8B retry; serial cleanup and publication remain authorized.',
               baseline_results_sha256=sha(q.HERE/'space/results.json'),
               implementation={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),q.HERE/'requested_queue.py',
                   q.HERE/'requested_models.py',ROOT/'src/pl_mmb_benchmark/models.py',ROOT/'src/pl_mmb_benchmark/full_gallery.py']})
    write(q.REG,reg,immutable=True)
    q.verify(reg)
    print('REGISTERED_2B_ONLY',[(j['alias'],len(j['missing'])) for j in reg['jobs']])
    return reg


def probe(index):
    reg=read(q.REG); job=reg['jobs'][index]
    os.environ.update(HF_HUB_CACHE=str(CACHE),HUGGINGFACE_HUB_CACHE=str(CACHE),HF_HUB_DISABLE_XET='1',
                      REQUESTS_CA_BUNDLE=str(ROOT/'combined-ca.pem'),SSL_CERT_FILE=str(ROOT/'combined-ca.pem'))
    from experiments.data_ood_common_20261030.common import gpu_lock
    from retrieval_benchmark.requested_models import RequestedEncoder,smoke_encoder
    with gpu_lock():
        q.verify(reg)
        for disk in [Path.home(),Path('/mnt/c')]:
            if shutil.disk_usage(disk).free<4*2**30+job['weight_bytes']:
                raise RuntimeError('Insufficient disk reserve: '+str(disk))
        encoder=RequestedEncoder(job,CACHE)
        smoke_encoder(encoder,q.folder_for(job))
        q.event('ADAPTER_SMOKE_PASSED',model=job['alias'])


def run():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg=read(q.REG); failed=[]
        # GME first, then Qwen: retain the last probe's weights for the first full worker.
        for index in [1,0]:
            job=reg['jobs'][index]; folder=q.folder_for(job); folder.mkdir(parents=True,exist_ok=True)
            if not job['missing'] or (folder/'complete.json').exists(): continue
            if not (folder/'adapter-smoke.json').exists():
                write(RUN/'STATE.json',dict(status='adapter_smoke',model=job['alias'],time=q.now(),supervisor_pid=os.getpid()))
                with (folder/'probe.log').open('a') as stream:
                    process=subprocess.Popen([str(Path.home()/'.venvs'/job['interpreter']/'bin/python'),'-u',
                        str(Path(__file__).resolve()),'probe','--index',str(index)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    write(RUN/'PROBE.json',dict(model=job['alias'],pid=process.pid,time=q.now()))
                    code=process.wait()
                if code: failed.append(job['alias'])
                if index==1 or code: q.clean(job)
        if failed:
            write(RUN/'STATE.json',dict(status='smoke_failed',failures=failed,time=q.now()))
            raise RuntimeError('Adapter smoke failed; see probe.log')
        q.run()


def start():
    reg=read(q.REG); q.verify(reg); RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (RUN/'supervisor.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    command='~/.venvs/pl-mmb-benchmark/bin/python retrieval_benchmark/requested_2b_retry.py start'
    write(RUN/'JOB.json',dict(run_id=RUN.name,pid=process.pid,started=q.now(),start_command=command,resume_command=command,
        estimated_end='Several hours; refine from first completed gallery, no measured ETA yet.',
        registration_sha256=sha(q.REG),registration=str(q.REG.relative_to(ROOT)),output=reg['config']['output'],
        state=str((RUN/'STATE.json').relative_to(ROOT)),log=str((RUN/'supervisor.log').relative_to(ROOT)),
        models=[j['repository'] for j in reg['jobs']],simultaneous_models=1,smoke_before_full_runs=True,
        cleanup='Only registered private weight blobs, after model process exits; shared cache untouched.',git_target='origin/main'))
    print('DETACHED_2B_QUEUE_STARTED',process.pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','start','run','probe','status'])
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='start': start()
    elif args.command=='run': run()
    elif args.command=='probe': probe(args.index)
    else: print(read(RUN/'STATE.json') if (RUN/'STATE.json').exists() else dict(status='not_started'))
