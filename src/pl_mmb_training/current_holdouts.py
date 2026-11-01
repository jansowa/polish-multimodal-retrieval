"""Current holdout registry; keep the G2-bound holdouts.py byte-for-byte frozen."""
import sys
from . import holdouts as legacy
from .holdouts import Components, GUARD, ROOT, digest, local_path, read_jsonl, canonical_path, extra_tokens

DEFAULT_REGISTRY=ROOT/'retrieval_benchmark/holdouts/registry-v4.json'


class Registry(legacy.Registry):
    def __init__(self,path=DEFAULT_REGISTRY):
        super().__init__(path)


def require_training_release(store,report,registry_path=DEFAULT_REGISTRY):
    return legacy.require_training_release(store,report,registry_path)


def verify_release_proof(proof,store,registry_path=DEFAULT_REGISTRY):
    return legacy.verify_release_proof(proof,store,registry_path)


if __name__=='__main__':
    if '--registry' not in sys.argv:sys.argv.extend(['--registry',str(DEFAULT_REGISTRY)])
    legacy.main()
