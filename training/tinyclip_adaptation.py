"""Validation-only TinyCLIP adaptation study on the immutable FULL_CLEAN snapshot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training.tinyclip_adaptation import main

if __name__ == '__main__':
    main()
