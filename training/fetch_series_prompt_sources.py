"""Pin reference model cards for author-prompt audit; never downloads weights."""
import concurrent.futures
import json
import ssl
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import OUT,ROOT,freeze,read,file_sha


def main():
    registration=read(OUT/'reference_eval_registration.json')
    folder=OUT/'prompt_sources';folder.mkdir(exist_ok=True)
    context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    def one(item):
        path=folder/(item['alias']+'.md')
        url=f"https://huggingface.co/{item['model']}/raw/{item['revision']}/README.md"
        if not path.exists():
            with urllib.request.urlopen(url,context=context,timeout=45) as response:raw=response.read(2_000_000)
            with path.open('xb') as output:output.write(raw)
        lines=path.read_text(encoding='utf-8').splitlines()
        excerpts=[dict(line=i+1,text=line[:500]) for i,line in enumerate(lines)
            if any(key in line.casefold() for key in ('prompt','instruction','prefix','searchquery','searchdoc','is_causal'))]
        return dict(alias=item['alias'],model=item['model'],revision=item['revision'],url=url,
                    sha256=file_sha(path),prompt_mentions=excerpts)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(one,item):item for item in registration['reference_models']}
        for future in concurrent.futures.as_completed(futures):
            item=futures[future]
            try:result=future.result()
            except Exception as error:result=dict(alias=item['alias'],error=type(error).__name__+': '+str(error))
            results.append(result)
    results.sort(key=lambda r:r['alias'])
    freeze(folder/'sources.json',results)
    print(json.dumps(results,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
