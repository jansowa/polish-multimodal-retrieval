"""Console-free independent audit/training task; leave acquisition running."""
import ctypes
import datetime
import os
from pathlib import Path
import subprocess
import traceback

REPO=Path(__file__).resolve().parents[3]
LOG=Path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031/logs')


def event(message):
    with (LOG/'early_pilot_host.log').open('a',encoding='utf-8') as stream:
        stream.write(datetime.datetime.now().astimezone().isoformat()+' '+message+'\n')


def main():
    power=ctypes.windll.kernel32.SetThreadExecutionState
    power.argtypes=[ctypes.c_uint32];power.restype=ctypes.c_uint32
    if not power(0x80000001):raise ctypes.WinError()
    try:
        event('START pid='+str(os.getpid()))
        with (LOG/'early_pilot.out.log').open('ab',buffering=0) as out,(LOG/'early_pilot.err.log').open('ab',buffering=0) as err:
            result=subprocess.run(['wsl.exe','-e','/home/jsowa/.venvs/pl-mmb-benchmark/bin/python','-u',
                '/mnt/d/work/translate-vision-embedder-benchmark/pl-multimodal-benchmark-translator/pl-multimodal-benchmark-translator/experiments/data_study_20261030/wiki_recovery_20261031/early_pilot.py'],
                cwd=REPO,stdout=out,stderr=err,creationflags=0x08000000)
        event('EXIT code='+str(result.returncode))
        if result.returncode:raise SystemExit(result.returncode)
    finally:power(0x80000000)


if __name__=='__main__':
    try:main()
    except Exception:event(traceback.format_exc());raise
