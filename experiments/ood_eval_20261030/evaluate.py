"""Gated common-harness OOD evaluation. All CUDA is inside A's shared flock."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='2'
os.environ['HF_HUB_DISABLE_XET']='1'
import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import subprocess
import sys

from prepare import ROOT,OUT,COORD,read,sha,verify_marker,load_common,committed_bytes,local_path
from acquire import ASSETS
from handoff import verify_A


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def gates():
    common=load_common()
    finalists,done=verify_A()
    frozen=verify_marker('B_EVAL_FROZEN')
    common.verify_marker(COORD/'B_EVAL_FROZEN.json')
    if frozen['A_UNIVERSE_FROZEN_sha256']!=sha(COORD/'A_UNIVERSE_FROZEN.json'):
        raise ValueError('Evaluation data belongs to another universe')
    if frozen['A_FINALISTS_FROZEN_sha256']!=sha(COORD/'A_FINALISTS_FROZEN.json'):
        raise ValueError('Evaluation registration belongs to another finalist list')
    registration=read(OUT/'execution_registration.json')
    if registration['status']!='closed_ready_for_execution' or registration['models']!=finalists['models']:
        raise ValueError('Execution model list is not the frozen list')
    if sha(ROOT/'src/pl_mmb_benchmark/full_gallery.py')!=registration['shared_scorer_sha256']:
        raise ValueError('Shared scorer changed')
    if sha(ROOT/'src/pl_mmb_benchmark/projector_adapter.py')!=registration['adapter_sha256']:
        raise ValueError('Shared adapter changed')
    protocols={a['role'].split(':',1)[1]:local_path(a['path']) for a in frozen['artifacts'] if a['role'].startswith('protocol:')}
    return common,registration,protocols


def owned_native_bytes():
    seen=set();total=0
    for plan in OUT.glob('reference_weight_plan_*.json'):
        for entry in read(plan)['files']:
            path=Path(entry['snapshot_path'])
            if entry['preexisting'] or not path.is_file():
                continue
            resolved=path.resolve()
            if resolved not in seen:
                seen.add(resolved);total+=resolved.stat().st_size
    return total


class Budget:
    def __init__(self,common,key):
        self.common,self.key=common,key
        self.base=committed_bytes()+owned_native_bytes()
        self.added=0

    def reserve(self,size=0,drive='C',temporary=0):
        self.common.reserve('B',self.key,{drive:size},committed_bytes=self.base+self.added,temporary_bytes=temporary)

    def json(self,path,value):
        size=len(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True).encode())+1
        old=path.stat().st_size if path.exists() else 0
        self.reserve(2*size,'D' if str(path).startswith('/mnt/d/') else 'C',size)
        self.common.atomic_json(path,value)
        self.added+=max(0,path.stat().st_size-old)
        self.reserve()

    def tensor(self,path,value,identity):
        import torch
        path.parent.mkdir(parents=True,exist_ok=True)
        self.reserve(2*(value.numel()*value.element_size()+1024**2),temporary=value.numel()*value.element_size()+1024**2)
        if path.exists():
            raise ValueError('Refusing to replace frozen feature cache')
        temporary=path.with_suffix('.tmp')
        with temporary.open('xb') as stream:
            torch.save(value,stream)
        temporary.rename(path)
        self.added+=path.stat().st_size
        self.json(path.with_suffix('.json'),dict(identity=identity,sha256=sha(path),bytes=path.stat().st_size))


class Originals:
    def __init__(self,protocols):
        self.records={}
        for path in protocols.values():
            for row in read(path.parent/'manifest.json'):
                self.records[row['image']]=row
        self.flickr=None

    @contextmanager
    def paths(self,values):
        descriptors=[];paths=[]
        try:
            soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft<len(values)+128:
                resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,len(values)+256),hard))
            for value in values:
                record=self.records[value]
                if value.startswith('parquet-original://'):
                    if self.flickr is None:
                        import pyarrow.parquet as pq
                        self.flickr=pq.read_table(ASSETS/'flickr30k/test.parquet',columns=['image']).column('image').to_pylist()
                    raw=self.flickr[int(value.rsplit('/',1)[1])]['bytes']
                    if hashlib.sha256(raw).hexdigest()!=record['sha256']:
                        raise ValueError('Embedded original image changed')
                    fd=os.memfd_create('B-original',flags=0)
                    descriptors.append(fd)
                    with os.fdopen(os.dup(fd),'wb') as stream:
                        stream.write(raw)
                    paths.append('/proc/self/fd/'+str(fd))
                else:
                    path=local_path(value)
                    if sha(path)!=record['sha256']:
                        raise ValueError('Original image changed')
                    paths.append(str(path))
            yield paths
        finally:
            for descriptor in descriptors:
                os.close(descriptor)


def features_cached(path,identity):
    if not path.exists():
        return None
    import torch
    metadata=read(path.with_suffix('.json'))
    if metadata['identity']!=identity or metadata['sha256']!=sha(path):
        raise ValueError('Feature cache identity/SHA mismatch')
    value=torch.load(path,weights_only=True,map_location='cpu')
    if value.ndim!=2 or not torch.isfinite(value).all() or (value.norm(dim=1)==0).any():
        raise ValueError('Invalid feature cache')
    return value


def token_report(tokenizer,texts,limit,prefix=''):
    lengths=[]
    for start in range(0,len(texts),128):
        lengths.extend(len(ids) for ids in tokenizer([prefix+t for t in texts[start:start+128]],
            truncation=False,padding=False,add_special_tokens=True)['input_ids'])
    return dict(tokens=lengths,maximum=max(lengths),mean=sum(lengths)/len(lengths),
                context_limit=limit,truncated=sum(n>limit for n in lengths),
                truncation_fraction=sum(n>limit for n in lengths)/len(lengths),prefix=prefix)


def our_worker(common,registration,protocol_paths,budget):
    import torch
    from PIL import Image
    from pl_mmb_benchmark.projector_adapter import ProjectorEncoder
    from pl_mmb_benchmark.full_gallery import evaluate_features
    from pl_mmb_benchmark.efficiency import parameter_counts
    models=registration['models']
    encoder=ProjectorEncoder(local_path(models[0]['path']),batch_size=64,device='cuda',local_files_only=True)
    originals=Originals(protocol_paths)
    counts=parameter_counts(encoder)
    if counts['vision']!=38587392 or counts['text']!=31883136 or counts['head']!=196992:
        raise ValueError('Encoder parameter counts differ')
    for name,path in protocol_paths.items():
        protocol=read(path)
        common_identity=dict(protocol_sha256=sha(path),encoder_components={k:v for k,v in encoder.components.items() if k!='projector_sha256'},
            precision=encoder.precision,batch_size=64,torch=torch.__version__,
            transformers=version('transformers'),Pillow=version('Pillow'),
            implementation_sha256=sha(Path(__file__)))
        native={}
        for modality,values in [('images',protocol['images']),('texts',protocol['texts'])]:
            identity=dict(common_identity,modality=modality)
            cache=ASSETS/'features/base'/(digest(identity)+'.pt')
            value=features_cached(cache,identity)
            if value is None:
                pieces=[]
                for start in range(0,len(values),64):
                    batch=values[start:start+64]
                    with torch.inference_mode():
                        if modality=='images':
                            with originals.paths(batch) as paths:
                                opened=[]
                                try:
                                    for image_path in paths:
                                        with Image.open(image_path) as image:
                                            opened.append(image.convert('RGB'))
                                    pixels=encoder.preprocess(opened).to('cuda')
                                    with encoder._amp():
                                        output=encoder.vision_forward(pixels).float()
                                finally:
                                    for image in opened:
                                        image.close()
                        else:
                            with encoder._amp():
                                output=encoder.text.encode(['[query]: '+text for text in batch],batch_size=len(batch),
                                    convert_to_tensor=True,normalize_embeddings=False,show_progress_bar=False,prompt='')
                        pieces.append(output.float().cpu())
                value=torch.cat(pieces)
                budget.tensor(cache,value,identity)
            native[modality]=value
        # Verify the reused native image path against the unchanged adapter with
        # identical batch shape before computing any retrieval score.
        with originals.paths(protocol['images'][:64]) as paths:
            direct=encoder.encode(paths,images=True)
        with torch.inference_mode():
            reused=encoder.projector(native['images'][:64].to('cuda')).float().cpu()
        if not torch.equal(direct,reused):
            raise ValueError('Base feature/projector parity differs for identical extraction batch')
        budget.json(OUT/('adapter_parity_'+name+'.json'),dict(status='passed',images=64,max_abs=0.0,
            protocol_sha256=sha(path),adapter_sha256=registration['adapter_sha256']))
        tokenization=token_report(encoder.text.tokenizer,protocol['texts'],128,'[query]: ')
        for model in models:
            saved=torch.load(local_path(model['path']),weights_only=True,map_location='cpu')
            encoder.projector.load_state_dict(saved.get('projector',saved.get('state_dict')))
            with torch.inference_mode():
                vectors=torch.cat([encoder.projector(batch.to('cuda')).float().cpu()
                    for batch in native['images'].split(64)])
            for mode in registration['prompt_modes']:
                target=ASSETS/'results'/model['id']/name/(mode+'.json')
                if target.exists():
                    continue
                result=evaluate_features(vectors.numpy(),native['texts'].numpy(),protocol['image_ids'],protocol['query_ids'],
                    protocol['edges'],protocol['image_components'],protocol['text_components'],name,protocol['manifest_sha256'],mode)
                result.update(model=model['id'],variant=model['variant'],seed=model['seed'],representative=model['representative'],
                    trained_on_Wiki=model['trained_on_Wiki'],source_OOD=True,checkpoint_sha256=model['sha256'],
                    registration_sha256=sha(OUT/'execution_registration.json'),precision=encoder.precision,
                    parameters=counts,dimension=vectors.shape[1],tokenization=tokenization,
                    feature_cache_policy='base vision/text once for all nine heads; intrinsic [query]: in both modes')
                budget.json(target,result)
                print('OUR_OOD_RESULT_SAVED',model['id'],name,mode,flush=True)
    del encoder
    gc.collect();torch.cuda.empty_cache()


class ReferenceFeatures:
    def __init__(self,encoder,originals,budget,alias):
        self.encoder,self.originals,self.budget,self.alias=encoder,originals,budget,alias

    def encode(self,values,images=False):
        import torch
        identity=dict(model=self.encoder.repo,revision=self.encoder.revision,components=self.encoder.components,
            precision=self.encoder.precision,text_max_length=self.encoder.text_max_length,
            values=values,images=images,torch=torch.__version__,transformers=version('transformers'),
            Pillow=version('Pillow'),batch_size=self.encoder.batch_size)
        path=ASSETS/'features/references'/(digest(identity)+'.pt')
        value=features_cached(path,identity)
        if value is not None:
            self.dimension=value.shape[1]
            return value
        self.encoder._cache_path=lambda values,modality:path
        self.encoder._save_cache=lambda path,value:self.budget.tensor(path,value,identity)
        if images:
            with self.originals.paths(values) as paths:
                result=self.encoder.encode(paths,images=True)
        else:
            result=self.encoder.encode(values)
        self.dimension=result.shape[1]
        return result


def reference_worker(common,registration,protocol_paths,budget,alias):
    import torch
    from pl_mmb_benchmark.models import Encoder,model_batch_size,check_cache
    from pl_mmb_benchmark.projector_adapter import AuthorPromptEncoder
    from pl_mmb_benchmark.full_gallery import evaluate_encoder
    from pl_mmb_benchmark.efficiency import parameter_counts
    item=next(r for r in registration['reference_models'] if r['alias']==alias)
    completion=OUT/('reference_complete_'+alias+'.json')
    if completion.exists():
        prior=read(completion)
        for artifact in prior['results']:
            if sha(local_path(artifact['path']))!=artifact['sha256']:
                raise ValueError('Completed reference result changed')
        return
    check_cache()
    from huggingface_hub.constants import HF_HUB_CACHE
    hub=Path(HF_HUB_CACHE).expanduser().resolve()
    if str(hub).startswith('/mnt/'):
        raise ValueError('Reference cache must remain on the native Linux filesystem')
    repo=hub/('models--'+item['model'].replace('/','--'))
    snapshot=repo/'snapshots'/item['revision']
    plan_path=OUT/('reference_weight_plan_'+alias+'.json')
    if plan_path.exists():
        plan=read(plan_path)
    else:
        plan=dict(model=item['model'],revision=item['revision'],files=[dict(snapshot_path=str(snapshot/f['name']),
            bytes=f['bytes'],preexisting=(snapshot/f['name']).is_file()) for f in item['weight_storage']['files']])
        budget.json(plan_path,plan)
    missing=sum(f['bytes'] for f in plan['files'] if not Path(f['snapshot_path']).is_file())
    budget.reserve(missing+64*1024**2,temporary=32*1024**2)
    encoder=Encoder(item['model'],model_batch_size(item['model'],64))
    if encoder.revision!=item['revision'] or encoder.precision not in item['historical_precision_policies']:
        raise ValueError('Reference revision or historical precision differs')
    for entry in plan['files']:
        blob=Path(entry['snapshot_path']).resolve(strict=True)
        if blob.stat().st_size!=entry['bytes'] or len(blob.name)!=64 or sha(blob)!=blob.name:
            raise ValueError('Reference weight file SHA/size does not match immutable HF blob identity')
        entry['sha256']=blob.name
    budget.base=committed_bytes()+owned_native_bytes();budget.added=0
    budget.json(plan_path,plan)
    budget.base=committed_bytes()+owned_native_bytes();budget.added=0;budget.reserve()
    counts=parameter_counts(encoder)
    cached=ReferenceFeatures(encoder,Originals(protocol_paths),budget,alias)
    results=[]
    tokenizer=getattr(encoder.model,'tokenizer',None) or getattr(getattr(encoder,'processor',None),'tokenizer',None)
    if tokenizer is None:
        raise ValueError('Cannot report actual native tokenizer truncation')
    for name,path in protocol_paths.items():
        protocol=read(path)
        for mode in registration['prompt_modes']:
            wrapped=AuthorPromptEncoder(cached,alias,mode)
            result=evaluate_encoder(wrapped,dict(protocol,prompt_mode=mode),registration)
            prefix='task: search result | query: ' if alias=='embeddinggemma-2' and mode=='author_recommended' else ''
            tokenization=token_report(tokenizer,protocol['texts'],encoder.text_max_length,prefix)
            if alias=='embeddinggemma-2' and mode=='author_recommended':
                tokenization=dict(query=tokenization,document=token_report(tokenizer,protocol['texts'],encoder.text_max_length,'title: none | text: '))
            result.update(model=alias,revision=encoder.revision,precision=encoder.precision,parameters=counts,
                dimension=cached.dimension,
                tokenization=tokenization,pretraining_exposure='unestablished',
                registration_sha256=sha(OUT/'execution_registration.json'))
            target=ASSETS/'results'/alias/name/(mode+'.json')
            budget.json(target,result)
            results.append(common.artifact('result',target))
            print('REFERENCE_OOD_RESULT_SAVED',alias,name,mode,flush=True)
    budget.json(completion,dict(status='complete',results=results,registration_sha256=sha(OUT/'execution_registration.json')))
    del wrapped,cached,encoder
    gc.collect();torch.cuda.empty_cache()
    # Remove only newly acquired, exact SHA-verified weight blobs, after all
    # datasets/modes/results and features are saved. Existing weights stay intact.
    cleanup=[]
    for entry in plan['files']:
        link=Path(entry['snapshot_path'])
        if entry['preexisting'] or not link.is_file():
            continue
        blob=link.resolve()
        if not blob.is_relative_to(repo.resolve()) or blob.stat().st_size!=entry['bytes']:
            raise ValueError('Reference cleanup scope/size mismatch')
        value=sha(blob)
        if len(blob.name)!=64 or value!=blob.name:
            raise ValueError('Reference cleanup requires a SHA256-addressed weight blob')
        foreign=[p for p in (repo/'snapshots').rglob('*') if p.is_symlink() and p.resolve()==blob and p!=link]
        if foreign:
            cleanup.append(dict(path=str(blob),status='retained_other_snapshot_reference'))
            continue
        cleanup.append(dict(path=str(blob),sha256=value,bytes=entry['bytes'],status='removed_own_new_weight'))
        blob.unlink()
    budget.base=committed_bytes()+owned_native_bytes();budget.added=0
    budget.json(OUT/('reference_cleanup_'+alias+'.json'),dict(files=cleanup,datasets_and_features_saved=True,
        Xet_untouched=True,protected_primary_models_untouched=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    common,registration,protocols=gates()
    code_path=OUT/'execution_code_registration.json'
    code=dict(status='before_first_score',runner_sha256=sha(Path(__file__)),
        registration_sha256=sha(OUT/'execution_registration.json'),B_EVAL_FROZEN_sha256=sha(COORD/'B_EVAL_FROZEN.json'),
        model_order=['ours']+[m['alias'] for m in registration['reference_models']],
        CUDA_policy='all operations inside shared common.gpu_lock; verified A finalists/done and B frozen',
        base_cache='one FP32 native vision/text extraction shared by all 9 projectors; identical-batch adapter parity first')
    if code_path.exists():
        if read(code_path)!=code:
            raise ValueError('Execution code registration changed; retain evidence and investigate')
    else:
        Budget(common,'execution-code').json(code_path,code)
    if args.worker:
        # Recheck all handoffs before entering the only shared lock; no torch was
        # imported before this point, including CUDA availability probes.
        with common.gpu_lock(blocking=False):
            import torch
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            budget=Budget(common,'evaluation')
            if args.worker=='ours':
                our_worker(common,registration,protocols,budget)
            else:
                reference_worker(common,registration,protocols,budget,args.worker)
            budget.reserve()
        return
    legacy=Path.home()/'.venvs/pl-mmb-legacy-cu130/bin/python'
    modern=Path.home()/'.venvs/pl-mmb-modern/bin/python'
    for worker in code['model_order']:
        executable=modern if worker=='embeddinggemma-2' else legacy
        if not executable.is_file():
            raise ValueError('Missing registered existing interpreter: '+str(executable))
        environment=dict(os.environ,REQUESTS_CA_BUNDLE=str(ROOT/'combined-ca.pem'),SSL_CERT_FILE=str(ROOT/'combined-ca.pem'))
        subprocess.run([str(executable),str(Path(__file__)), '--worker',worker],env=environment,check=True)
    print('B_MODEL_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
