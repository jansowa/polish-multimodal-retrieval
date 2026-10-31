import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.fetch_wiki_pilot_images import WikiRateLimiter,HttpConfig


@unittest.skipUnless(os.name=='nt','Windows mandatory byte-lock regression')
class WikiHttpTests(unittest.TestCase):
    def test_waits_for_other_process_without_bypassing_shared_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            state=Path(folder)/'state.json';lock=state.with_suffix('.json.lock')
            lock.write_bytes(b'0')
            child_code=('import msvcrt,sys,time; f=open(sys.argv[1],"r+b"); '
                'msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1); '
                'print("LOCKED",flush=True); time.sleep(.8); '
                'f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1); f.close()')
            process=subprocess.Popen([sys.executable,'-c',child_code,str(lock)],stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(),'LOCKED')
                limiter=WikiRateLimiter(HttpConfig(),state)
                with limiter.shared_state():
                    limiter.state['regression_check']=True;limiter.save()
                self.assertTrue(state.exists())
                self.assertEqual(process.wait(timeout=5),0)
            finally:
                if process.poll() is None:process.kill();process.wait()
                process.stdout.close()


if __name__=='__main__':unittest.main()
