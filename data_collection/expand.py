"""Create a larger acquisition run without discarding existing records."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pl_mmb_collection.config import load_config
from pl_mmb_collection.pipeline.expansion import expand

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(expand(args.from_output, load_config(args.config)), ensure_ascii=False, indent=2))
