"""Preserve the current benchmark catalog and restore exact frozen G2 inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/benchmark_decontamination_20261031/frozen_compatibility'


def digest(value):return hashlib.sha256(value).hexdigest()


def main(restore=False):
    catalog=ROOT/'retrieval_benchmark/catalog.json'
    guard=ROOT/'src/pl_mmb_training/holdouts.py'
    reg=json.loads((ROOT/'retrieval_benchmark/holdouts/registry-v2.json').read_text(encoding='utf-8'))
    binding=json.loads((ROOT/'reports/data_study_20261030/wiki_recovery_20261031/holdout_registry_binding.json').read_text(encoding='utf-8'))
    expected=next(x['sha256'] for x in reg['inputs'] if x['key']=='benchmark_catalog')
    OUT.mkdir(parents=True,exist_ok=True)
    if not restore:
        current=catalog.read_bytes();text=current.decode('utf-8')
        oldline='    "xm3600_pl": {"title": "XM3600 PL · FINAL BLIND", "role": "native_final", "caption_origin": "native_human", "status": "final_blind", "source": "https://google.github.io/crossmodal-3600/", "notes": "Dane lokalne. Brak wyników; wymaga osobnej zamkniętej rejestracji wspólnej oceny. Ten runner nie otwiera bramki."},'
        old='\n'.join(oldline if line.strip().startswith('"xm3600_pl":') else line for line in text.split('\n')).replace('retrieval-pl-v2-xm3600','retrieval-pl-v1')
        candidates=[old.encode(),old.replace('\r\n','\n').replace('\n','\r\n').encode()]
        legacy=next((value for value in candidates if digest(value)==expected),None)
        if legacy is None:raise RuntimeError('Cannot reproduce frozen catalog exactly; no inputs changed')
        legacy_guard=guard.read_bytes().replace(b'holdouts/registry-v3.json',b'holdouts/registry-v2.json')
        if digest(legacy_guard)!=binding['guard_sha256']:raise RuntimeError('Cannot reproduce frozen guard exactly; no inputs changed')
        for path,payload in [(OUT/'catalog_current_before.json',current),(OUT/'guard_before.py',guard.read_bytes()),
                             (OUT/'catalog_legacy.json',legacy),(OUT/'guard_legacy.py',legacy_guard),
                             (ROOT/'retrieval_benchmark/catalog-current.json',current)]:
            if path.exists():
                if path.read_bytes()!=payload:raise RuntimeError('Existing backup differs: '+str(path))
            else:path.write_bytes(payload)
        print('EXACT_FROZEN_INPUTS_PREPARED')
    else:
        legacy=(OUT/'catalog_legacy.json').read_bytes();legacy_guard=(OUT/'guard_legacy.py').read_bytes()
        if digest(legacy)!=expected or digest(legacy_guard)!=binding['guard_sha256']:raise RuntimeError('Backup identity differs')
        if (ROOT/'retrieval_benchmark/catalog-current.json').read_bytes()!=(OUT/'catalog_current_before.json').read_bytes():raise RuntimeError('Active catalog differs')
        for target,payload in [(catalog,legacy),(guard,legacy_guard)]:
            temporary=target.with_name(target.name+'.frozen-restore.tmp')
            temporary.write_bytes(payload);temporary.replace(target)
        print('FROZEN_G2_INPUTS_RESTORED_WITH_CURRENT_CATALOG_PRESERVED')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true');args=parser.parse_args();main(args.restore)
