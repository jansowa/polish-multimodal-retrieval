"""Keep a WSL foreground login alive; adopt an already-running queue safely."""
import subprocess
import sys
import time
from config import *


def main():
    with lock(STORE/'foreground_owner.lock',blocking=False):
        failures=0
        while True:
            p=REPORT/'STATUS.json'
            if p.exists() and read(p).get('status')=='complete':return
            busy=False
            try:
                with lock(STORE/'queue.lock',blocking=False):pass
            except BlockingIOError:busy=True
            if busy:
                time.sleep(15);continue
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('overnight_queue.py'))],cwd=ROOT)
            if result.returncode==0:return
            failures+=1
            if failures>=3:raise RuntimeError('Foreground queue restart budget exhausted')
            time.sleep(60)


if __name__=='__main__':main()
