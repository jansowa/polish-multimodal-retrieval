"""Prepare exact original VN dev assets and explicit qrels for the common harness."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pyarrow.parquet as pq
from pl_mmb_training.series_20261023 import ROOT,OUT,VN,read,rows,freeze,file_sha,require_space,local_path
from pl_mmb_training.leakage import image_identity,normalized_text
from pl_mmb_benchmark.protocol import PREFIXES,without_instruction

ASSETS=local_path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/vn-dev-originals')


def check_space(expected_bytes=0):
    # The reserve applies to both volumes; the payload is written only to C.
    require_space([local_path('C:/'),local_path('D:/')])
    require_space([ASSETS],expected_bytes=expected_bytes)


def graph(records,dataset_id,manifest_sha):
    by_image={}
    for r in sorted(records,key=lambda r:r['id']):by_image.setdefault(r['target_id'],r)
    image_ids=sorted(by_image);texts=sorted({r['text_pl'] for r in records})
    ii={key:i for i,key in enumerate(image_ids)};ti={key:i for i,key in enumerate(texts)}
    components={r['text_pl']:r['component'] for r in records}
    return dict(status='frozen_image_verified',id=dataset_id,manifest_sha256=manifest_sha,
        images=[by_image[key]['image'] for key in image_ids],image_ids=image_ids,texts=texts,
        query_ids=[hashlib.sha256(t.encode()).hexdigest() for t in texts],
        edges=sorted({(ti[r['text_pl']],ii[r['target_id']]) for r in records}),
        image_components=[by_image[key]['component'] for key in image_ids],text_components=[components[t] for t in texts])


def vn_assets():
    folder=OUT/'vn_dev_protocol';folder.mkdir(exist_ok=True)
    if (folder/'retrieval_protocol.json').exists():return
    records=rows(VN/'vn_dev.jsonl');ASSETS.mkdir(parents=True,exist_ok=True)
    check_space(512*2**20)
    import PIL
    freeze(folder/'asset_acquisition_decision_v2_20261029.json',dict(status='registered_before_original_asset_copy',
        script_sha256=file_sha(__file__),frozen_VN_dev_sha256=file_sha(VN/'vn_dev.jsonl'),asset_root=str(ASSETS),
        pillow=PIL.__version__,supersedes='asset_acquisition_decision.json',
        reason='Use the original Linux Pillow decoder for exact frozen pixel SHA; preserve 4 GiB on both volumes and budget new payload only on target C',
        source='Only frozen shard,row,did,path mapping in VN dev and SHA-pinned extraction metadata; no SQLite/WAL'))
    selected=defaultdict(lambda:defaultdict(list))
    for r in records:selected[r['shard']][r['shard_row']].append(r)
    expected=read(VN/'extraction.json')['verified_shards']
    materialized=[]
    for shard,mapping in sorted(selected.items()):
        path=local_path(shard)
        if file_sha(path)!=expected[shard]:raise ValueError('Frozen VN shard changed')
        parquet=pq.ParquetFile(path);offset=0
        for group in range(parquet.num_row_groups):
            length=parquet.metadata.row_group(group).num_rows
            wanted=sorted(i for i in mapping if offset<=i<offset+length)
            if wanted:
                table=parquet.read_row_group(group,columns=['did','img_path','img'])
                for index in wanted:
                    source=table.slice(index-offset,1).to_pylist()[0];r=mapping[index][0]
                    if source['did']!=r['image_did'] or source['img_path']!=r['source_image']:raise ValueError('VN parent mapping changed')
                    payload=source['img']['bytes'];identity=image_identity(payload=payload)
                    if identity!=dict(sha256=r['sha256'],pixels_sha256=r['pixels_sha256']):raise ValueError('VN source bytes/pixels differ')
                    target=ASSETS/(r['sha256']+'.bin')
                    if not target.exists():
                        check_space(len(payload))
                        with target.open('xb') as output:output.write(payload)
                    if file_sha(target)!=r['sha256']:raise ValueError('VN copied asset differs')
                    linux=str(target) if str(target).startswith('/mnt/') else '/mnt/c'+str(target)[2:].replace('\\','/')
                    materialized.extend(dict(row,image=linux) for row in mapping[index])
                del table
            offset+=length
        print(f'VN_ORIGINAL_ASSETS {len(materialized)}/{len(records)}',flush=True)
    materialized.sort(key=lambda r:r['id'])
    sha=freeze(folder/'manifest.jsonl',materialized,jsonl=True)
    freeze(folder/'retrieval_protocol.json',graph(materialized,'VN_dev',sha))
    freeze(folder/'preparation.json',dict(status='frozen_image_verified',rows=len(materialized),
        images=len({r['target_id'] for r in materialized}),manifest_sha256=sha,
        original_asset_bytes=sum(p.stat().st_size for p in ASSETS.glob('*.bin')),
        labels='model-generated and translated; in-domain for VN-trained projectors'))


def xtd_styles():
    folder=OUT/'xtd10_v2';full=rows(folder/'full_1k.jsonl');clean=rows(folder/'no_overlap.jsonl')
    freeze(folder/'full_retrieval_protocol.json',graph(full,'xtd10_full_1k',file_sha(folder/'full_1k.jsonl')))
    freeze(folder/'clean_retrieval_protocol.json',graph(clean,'xtd10_no_overlap',file_sha(folder/'no_overlap.jsonl')))
    wanted={normalized_text(r['text_en']) for r in full};matches=defaultdict(list);inputs={};excluded=[]
    for run in sorted((ROOT/'outputs').glob('qwen*')):
        for path in sorted(run.rglob('translations.jsonl')):
            if any(token in str(path).casefold() for token in ('pilot','review-ab','visualnews')):continue
            inputs[str(path.relative_to(ROOT))]=file_sha(path)
            for row in rows(path):
                if not str(row.get('model','')).casefold().startswith('qwen'):continue
                source=row.get('source',{});translated=row.get('translated',{})
                for key,en in source.items():
                    pl=translated.get(key)
                    if not isinstance(en,str) or not isinstance(pl,str):continue
                    if key=='qry_text':
                        prefix=PREFIXES['mscoco_t2i'][0]
                        if not en.strip().startswith(prefix):continue
                        try:pl=without_instruction('mscoco_t2i',en,pl)
                        except ValueError:
                            excluded.append(dict(export=str(path.relative_to(ROOT)),translation_id=row['id'],
                                reason='unrecognized translated instruction; no guessed caption boundary'))
                            continue
                        en=en.strip()[len(prefix):].strip()
                    normalized=normalized_text(en)
                    if normalized in wanted:
                        matches[normalized].append(dict(translation_id=row['id'],model=row['model'],field=key,text_pl=pl,
                            export=str(path.relative_to(ROOT)),export_sha256=inputs[str(path.relative_to(ROOT))]))
    paired=[];ambiguous=[]
    for r in full:
        found=matches[normalized_text(r['text_en'])];variants={m['text_pl'] for m in found}
        if len(variants)==1:
            paired.append(dict(id=r['id'],image=r['image'],target_id=r['target_id'],component=r['component'],
                human_pl=r['text_pl'],cached_translation_pl=next(iter(variants)),text_en=r['text_en'],provenance=found))
        elif len(variants)>1:ambiguous.append(dict(id=r['id'],translation_variants=sorted(variants),provenance=found))
    sha=freeze(folder/'paired_cached_styles.jsonl',paired,jsonl=True)
    freeze(folder/'cached_style_audit.json',dict(status='complete',paired_images=len(paired),ambiguous=ambiguous,
        paired_manifest_sha256=sha,translation_export_SHA256=inputs,
        rule='Exact normalized EN after only the existing known MMEB instruction removal; conflicting cached PL variants excluded',
        excluded_translations=excluded,new_translations_generated=False,results_computed=False))
    if paired:
        human=[dict(r,text_pl=r['human_pl']) for r in paired];qwen=[dict(r,text_pl=r['cached_translation_pl']) for r in paired]
        freeze(folder/'paired_human_protocol.json',graph(human,'xtd10_paired_human',sha))
        freeze(folder/'paired_qwen_protocol.json',graph(qwen,'xtd10_paired_qwen',sha))
    print('XTD_CACHED_STYLE_PAIRS '+str(len(paired)),flush=True)


if __name__=='__main__':
    vn_assets();xtd_styles()
