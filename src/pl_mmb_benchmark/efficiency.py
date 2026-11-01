"""Common uncached CPU/GPU encoding measurements on predecoded inputs."""
import gc
import time
from contextlib import nullcontext


def move(encoder,device):
    import torch
    target=torch.device(device)
    if hasattr(encoder,'projector'):
        encoder.device=target
        for module in (encoder.vision,encoder.text,encoder.projector):module.float().to(target).eval()
    else:
        encoder.model.float().to(target).eval()
        if hasattr(encoder,'vision_model'):encoder.vision_model.float().to(target).eval()
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def raw_encode(encoder,values,images,device):
    """Native encoder without on-disk embedding caching or model-specific prompts."""
    import torch
    target=torch.device(device)
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if target.type=='cuda' else nullcontext()
    with torch.inference_mode():
        if hasattr(encoder,'projector'):
            if images:
                pixels=encoder.preprocess(values).to(target)
                with amp():native=encoder.vision_forward(pixels).float()
                result=encoder.projector(native)
            else:
                with amp():
                    result=encoder.text.encode(['[query]: '+v for v in values],batch_size=len(values),
                        convert_to_tensor=True,normalize_embeddings=False,show_progress_bar=False,prompt='')
        else:
            with amp():
                if encoder.mobile or encoder.pe or encoder.open_clip:
                    inputs=torch.stack([encoder.processor(im) for im in values]).to(target) if images else encoder.tokenizer(values).to(target)
                    result=encoder.model.encode_image(inputs,normalize=False) if images else encoder.model.encode_text(inputs,normalize=False)
                elif encoder.tips:
                    result=encoder.model.encode_image(torch.stack([encoder.processor(im) for im in values]).to(target)).cls_token[:,0,:] if images else encoder.model.encode_text(values)
                elif encoder.bge:
                    inputs=encoder.processor(images=values,return_tensors='pt') if images else encoder.processor(text=values,padding=True,
                        truncation=True,max_length=encoder.text_max_length,return_tensors='pt')
                    inputs={k:v.to(target) for k,v in inputs.items()}
                    result=encoder.model.get_image_features(**inputs) if images else encoder.model.get_text_features(**inputs)
                elif encoder.mclip:
                    if images:result=encoder.vision_model(torch.stack([encoder.processor(im) for im in values]).to(target))
                    else:
                        inputs=encoder.tokenizer(values,padding=True,truncation=True,max_length=encoder.text_max_length,return_tensors='pt')
                        result=encoder.model({k:v.to(target) for k,v in inputs.items()})
                elif encoder.sentence_clip:
                    if images:
                        inputs={k:v.to(target) for k,v in encoder.processor(images=values,return_tensors='pt').items()}
                        result=encoder.vision_model(**inputs).image_embeds
                    else:
                        result=encoder.model.encode(values,batch_size=len(values),convert_to_tensor=True,normalize_embeddings=False,show_progress_bar=False)
                elif encoder.native_multimodal:
                    result=encoder.model.encode(values,batch_size=len(values),convert_to_tensor=True,normalize_embeddings=False,prompt='',show_progress_bar=False)
                else:
                    padding='max_length' if encoder.model.config.model_type in ('siglip','siglip2') else True
                    inputs=encoder.processor(images=values,return_tensors='pt') if images else encoder.processor(text=values,padding=padding,
                        truncation=True,max_length=encoder.text_max_length,return_tensors='pt')
                    inputs={k:v.to(target) for k,v in inputs.items()}
                    result=encoder.model.get_image_features(**inputs) if images else encoder.model.get_text_features(**inputs)
        if not torch.isfinite(result).all() or (result.float().norm(dim=1)==0).any():raise ValueError('Invalid efficiency-probe embeddings')
        return result.float().cpu()


def parameter_counts(encoder):
    if hasattr(encoder,'projector'):
        return dict(text=sum(p.numel() for p in encoder.text.parameters()),vision=sum(p.numel() for p in encoder.vision.parameters()),
            head=sum(p.numel() for p in encoder.projector.parameters()),shared=0)
    counts=dict(text=0,vision=0,head=0,shared=0)
    seen=set()
    modules=[('model',encoder.model)]+([('vision',encoder.vision_model)] if hasattr(encoder,'vision_model') else [])
    for prefix,module in modules:
        for name,p in module.named_parameters():
            if id(p) in seen:continue
            seen.add(id(p));label=(prefix+'.'+name).casefold()
            kind='vision' if any(token in label for token in ('vision','visual','image_encoder','image_model')) else 'text'
            if name in ('logit_scale','logit_bias'):kind='shared'
            counts[kind]+=p.numel()
    return counts


def measure_efficiency(encoder,images,texts,warmup=10,repetitions=100):
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    decoded=[]
    for path in images[:64]:
        with Image.open(path) as image:decoded.append(image.convert('RGB'))
    if len(decoded)!=64 or len(texts)<64:raise ValueError('A registered batch of 64 real inputs is required')
    result=dict(parameters=parameter_counts(encoder),warmup_batches=warmup,measured_batches=repetitions,
        input_policy='same frozen VN dev inputs; predecoded RGB; tokenization/preprocessing/host transfers included; no disk/cache/download time',
        cpu_threads=2,device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',measurements={})
    try:
        for device,batch in [('cuda',64),('cpu',1)]:
            move(encoder,device)
            for modality,values,is_image in [('query',texts[:batch],False),('image',decoded[:batch],True)]:
                name=device+'_'+modality
                samples=[]
                try:
                    for index in range(warmup+repetitions):
                        if device=='cuda':torch.cuda.synchronize()
                        started=time.perf_counter()
                        encoded=raw_encode(encoder,values,is_image,device)
                        if device=='cuda':torch.cuda.synchronize()
                        elapsed=(time.perf_counter()-started)*1000
                        if index>=warmup:samples.append(elapsed)
                    result['dimension']=encoded.shape[1]
                    result['measurements'][name]=dict(status='measured',batch_size=batch,
                        precision='fp32-weights-bf16-autocast' if device=='cuda' else 'fp32',
                        median_batch_ms=float(np.median(samples)),p95_batch_ms=float(np.quantile(samples,.95)),
                        median_ms_per_item=float(np.median(samples)/batch))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    result['measurements'][name]=dict(status='fixed_batch64_does_not_fit',batch_size=batch,
                        no_smaller_batch_substituted=True)
    finally:
        for image in decoded:image.close()
    return result
