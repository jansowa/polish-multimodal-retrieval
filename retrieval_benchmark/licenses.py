"""Fetch declared model licenses at the exact evaluated Hugging Face revisions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import ssl
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import HERE,identity,read,sha,write

NAMES={'mit':'MIT','apache-2.0':'Apache-2.0','cc-by-4.0':'CC BY 4.0',
       'cc-by-nc-4.0':'CC BY-NC 4.0','cc-by-nc-sa-4.0':'CC BY-NC-SA 4.0',
       'gemma':'Gemma','apple-amlr':'Apple AMLR (research only)'}


def fetch(item):
    repository,revision=item
    dest=HERE/'local/license_evidence'/(identity(item)+'.json')
    if dest.exists():value=read(dest)
    else:
        context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
        with urlopen('https://huggingface.co/api/models/'+repository+'/revision/'+revision,context=context,timeout=60) as response:
            value=json.load(response)
        if value['sha']!=revision:raise ValueError('License revision mismatch: '+repository)
        write(dest,value,immutable=True)
    card=value.get('cardData') or {}
    license_id=card.get('license')
    if license_id is None:
        license_id=next((t[len('license:'):] for t in value.get('tags',[]) if t.startswith('license:')),None)
    if isinstance(license_id,list):display=' / '.join(NAMES.get(v,v) for v in license_id)
    else:display=NAMES.get(license_id,license_id)
    if license_id=='other':display=card.get('license_name') or 'Inna — patrz karta modelu'
    info=dict(id=license_id,display=display or 'Brak deklaracji',
        source='https://huggingface.co/'+repository+'/blob/'+revision+'/README.md',
        revision=revision,declared=bool(license_id),evidence_sha256=sha(dest))
    if card.get('license_link'):info['license_link']=card['license_link']
    return item,info


def main():
    document=read(HERE/'space/results.json')
    sources={}
    for mid,model in document['models'].items():
        if mid.startswith('hf:'):sources[mid]=(model['repository'],model['revision'])
    components={}
    for mid,model in document['models'].items():
        if not mid.startswith('checkpoint:'):continue
        vision=model['parameters']['vision'];revision=(model.get('encoder_revisions') or {}).get('vision')
        if vision==38587392:repo='wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M';revision=revision or '07a4b0bc751cb64fecd2b661c048c1dd98d69444'
        elif vision==35815232:repo='apple/MobileCLIP2-S2'
        elif vision==92930304:repo='google/siglip2-base-patch16-256'
        else:raise ValueError('Unknown project backbone: '+model['label'])
        if not revision:raise ValueError('Missing project vision revision')
        components[mid]=dict(vision=(repo,revision),text=('OPI-PIB/PolDense-32M',
            (model.get('encoder_revisions') or {}).get('text','ca70494a25dad90d980db7c735658dc7bf88ed25')))
    requested=set(sources.values())|{v for item in components.values() for v in item.values()}
    with ThreadPoolExecutor(max_workers=4) as pool:licenses=dict(pool.map(fetch,sorted(requested)))
    models={mid:licenses[item] for mid,item in sources.items()}
    for mid,values in components.items():
        parts={role:dict(repository=item[0],**licenses[item]) for role,item in values.items()}
        models[mid]=dict(id=None,declared=False,display='V: '+parts['vision']['display']+'; T: '+parts['text']['display']+'; head: brak deklaracji',
            kind='local_checkpoint_components',components=parts)
    write(HERE/'local/model_licenses.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),models=models))
    for mid,info in models.items():print(document['models'][mid]['label'],info['display'],flush=True)


if __name__=='__main__':main()
