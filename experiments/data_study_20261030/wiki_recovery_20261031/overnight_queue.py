"""Durable serial queue. Network -> audited freeze -> GPU -> CPU report.

Named overnight_queue to avoid shadowing Python's stdlib queue used by Torch.
"""
import subprocess
import sys
import time
import traceback
from config import *

STAGES=['order_downloads.py','download_compact.py','audit_generation.py','train_generation.py','finish.py']


def main():
    STORE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    with lock(STORE/'queue.lock',blocking=False):
        for name in STAGES:
            complete=STORE/'queue_state'/(name+'.done.json')
            if complete.exists():continue
            for attempt in range(1,4):
                status('queue_stage',stage=name,attempt=attempt,queued_stages=STAGES,worker_pid=__import__('os').getpid())
                log=STORE/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('ab',buffering=0) as output:
                    result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name(name))],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                if result.returncode==0:
                    atomic_json(complete,dict(status='complete',stage=name,attempt=attempt,code_sha256=sha256(Path(__file__).with_name(name)),completed_at=utc()),immutable=True)
                    break
                if attempt==3:
                    status('failed',stage=name,exit_code=result.returncode,log_path=str(log),GPU_lock_released=True)
                    (REPORT/'FAILED.md').write_text(f'# Kolejka zatrzymana bez pomijania bramki\n\nEtap: {name}; kod: {result.returncode}.\nLog: {log}\nPobrane pliki zachowane. Wznowienie: uruchom ponownie queue.py po usunięciu przyczyny.\n',encoding='utf-8')
                    reserve(remaining=0)
                    raise RuntimeError('Stage failed after 3 attempts: '+name)
                time.sleep(60)
        print('QUEUE_ALL_STAGES_COMPLETE',flush=True)


if __name__=='__main__':main()
