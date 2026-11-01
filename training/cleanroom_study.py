"""Audit the existing guard, review near duplicates and run CLEAN-v1 ablations."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'near', 'train', 'report'))
    parser.add_argument('--output', type=Path, default=ROOT / 'training/runs/clean-v1')
    parser.add_argument('--variant', choices=('FULL_CLEAN', 'NO_MBEIR_MSCOCO', 'NON_COCO_CLEANROOM'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from pl_mmb_training.common import write_json
    from pl_mmb_training.cleanroom import audit, near_duplicates
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / f'{args.command}_status.json'
    write_json(status, dict(status='running', command=args.command))
    try:
        if args.command == 'audit':
            value = audit(args.output)
        elif args.command == 'near':
            value = near_duplicates(args.output)
        else:
            from pl_mmb_training.cleanroom_training import run, export
            value = run(args.output, args.variant) if args.command == 'train' else export(args.output)
        write_json(status, dict(status='complete', command=args.command))
        import json
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        write_json(status, dict(status='failed', command=args.command, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
