"""Finalize the pre-inference registration, preserving its initial preflight copy."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.core import read, write, sha
from retrieval_benchmark.requested_queue import REG, REPORT, RUN, verify

if __name__ == '__main__':
    if (RUN/'JOB.json').exists():
        raise RuntimeError('Never amend registration after launch')
    reg = read(REG)
    write(REPORT/'registration-preflight.json', reg, immutable=True)
    reg['implementation'] = {name:sha(ROOT/name) for name in reg['implementation']}
    reg['preflight_registration_sha256'] = sha(REPORT/'registration-preflight.json')
    reg['amendment'] = 'Before any inference: finalized adapters, private cleanup and no-repeat resumption tests.'
    # Existing adapters use these exact separately pinned paired vision components.
    from pl_mmb_benchmark.models import CLIP_IMAGE_REVISION
    from pl_mmb_benchmark.backends import MCLIP_IMAGE_REVISION
    for job in reg['jobs']:
        for spec in job['components']:
            if spec['repository'] == 'openai/clip-vit-base-patch32' and spec['revision'] != CLIP_IMAGE_REVISION:
                raise RuntimeError('Paired CLIP vision revision differs')
            if spec['repository'] == 'timm/vit_base_patch16_plus_clip_240.laion400m_e31' and spec['revision'] != MCLIP_IMAGE_REVISION:
                raise RuntimeError('Paired MCLIP vision revision differs')
    write(REG,reg)
    verify(reg)
    summary = dict(registration_sha256=sha(REG), models=len(reg['jobs']),
                   existing_cells=sum(len(j['existing_cells']) for j in reg['jobs']),
                   missing_cells=sum(len(j['missing']) for j in reg['jobs']),
                   protected=reg['protected'], same_cell_recomputation=False,
                   source='registration.json', tests='python -m unittest retrieval_benchmark.tests.test_requested_queue -v')
    write(REPORT/'summary.json',summary)
    print(summary)
