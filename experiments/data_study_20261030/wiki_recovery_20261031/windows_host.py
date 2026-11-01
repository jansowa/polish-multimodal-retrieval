"""Console-free Windows Task Scheduler host; no PowerShell/console signal dependency."""
import ctypes
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
LOG=Path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031/logs')
STATUS=REPO/'reports/data_study_20261030/wiki_recovery_20261031/STATUS.json'


def event(message):
    with (LOG/'windows_host.log').open('a',encoding='utf-8') as stream:
        stream.write(datetime.datetime.now().astimezone().isoformat()+' '+message+'\n');stream.flush()


def main():
    power=ctypes.windll.kernel32.SetThreadExecutionState
    power.argtypes=[ctypes.c_uint32];power.restype=ctypes.c_uint32
    if not power(0x80000001):raise ctypes.WinError()
    event('HOST_STARTED pid='+str(os.getpid())+' sleep_prevention=enabled console_free=true')
    try:
        for attempt in range(1,6):
            if STATUS.exists() and json.loads(STATUS.read_text(encoding='utf-8')).get('status')=='complete':
                event('COMPLETE');return
            with (LOG/'windows_foreground.out.log').open('ab',buffering=0) as out,(LOG/'windows_foreground.err.log').open('ab',buffering=0) as err:
                process=subprocess.Popen(['wsl.exe','-e','/home/jsowa/.venvs/pl-mmb-benchmark/bin/python','-u',
                    '/mnt/d/work/translate-vision-embedder-benchmark/pl-multimodal-benchmark-translator/pl-multimodal-benchmark-translator/experiments/data_study_20261030/wiki_recovery_20261031/foreground_owner.py'],
                    cwd=REPO,stdout=out,stderr=err,creationflags=0x08000000)
                event(f'WSL_STARTED pid={process.pid} attempt={attempt}')
                while process.poll() is None:
                    event(f'HEARTBEAT child_pid={process.pid}');time.sleep(30)
                event('WSL_EXIT code='+str(process.returncode))
                if process.returncode==0:return
            time.sleep(60)
        raise RuntimeError('Five host retries exhausted')
    finally:power(0x80000000);event('HOST_STOPPED sleep_prevention=released')


if __name__=='__main__':
    try:main()
    except Exception:
        event(traceback.format_exc());raise
