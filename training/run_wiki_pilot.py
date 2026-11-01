"""Run the authorized pilot after its immutable image acquisition finishes."""
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pl_mmb_training import wiki_pilot as pilot

if __name__=='__main__':
    pilot.initialize()
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_wiki_pilot.py','-v'],cwd=ROOT,check=True)
    complete=pilot.wd.FOLDER/'download_complete.json';last=0
    while not complete.exists():
        if time.monotonic()-last>=60:
            print('WIKI_PIPELINE_WAITING_FOR_VERIFIED_DOWNLOADS',flush=True);last=time.monotonic()
        time.sleep(5)
    for command in ('prepare_wiki_pilot.py','wiki_pilot_gallery.py','wiki_pilot.py','verify_wiki_pilot.py'):
        subprocess.run([sys.executable,str(ROOT/'training'/command)],cwd=ROOT,check=True)
