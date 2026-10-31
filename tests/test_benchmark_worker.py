import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from pl_mmb_benchmark.cli import run_worker


class WorkerRecoveryTests(unittest.TestCase):
    def test_transient_cuda_failure_recovers_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/'attempts'
            program = "from pathlib import Path; import sys; p=Path(sys.argv[1]); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); print('torch.OutOfMemoryError: CUDA out of memory' if n==1 else 'ok'); sys.exit(1 if n==1 else 0)"
            run_worker([sys.executable, '-c', program, str(marker)])
            self.assertEqual(marker.read_text(), '2')

    def test_non_cuda_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/'attempts'
            program = "from pathlib import Path; import sys; p=Path(sys.argv[1]); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); print('Invalid model configuration'); sys.exit(7)"
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                run_worker([sys.executable, '-c', program, str(marker)])
            self.assertEqual(caught.exception.returncode, 7)
            self.assertEqual(marker.read_text(), '1')
