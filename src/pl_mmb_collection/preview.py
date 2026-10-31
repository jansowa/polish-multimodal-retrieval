"""Portable, static HTML review artifact; no server or model dependencies."""
from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.parse import quote, urlparse

from .storage.checkpoint import CheckpointStore


def write_preview(store: CheckpointStore, path: Path, sample_size: int = 12, seed: int = 42) -> Path:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    # Reservoir sampling keeps memory bounded for large datasets.
    rng = random.Random(seed)
    sample = []
    total = 0
    for total, record in enumerate(store.records(), 1):
        if len(sample) < sample_size:
            sample.append(record)
        else:
            index = rng.randrange(total)
            if index < sample_size:
                sample[index] = record
    rng.shuffle(sample)
    rows = []
    for record in sample:
        row = record.to_dict()
        if record.source == "wikimedia":
            filename = record.image_filename.removeprefix("File:")
            row["preview_url"] = "https://commons.wikimedia.org/wiki/Special:Redirect/file/" + quote(filename, safe="") + "?width=960"
        else:
            row["preview_url"] = record.image_url
        for key in ("preview_url", "image_original_url", "source_page_url", "license_url"):
            value = row.get(key)
            if value and urlparse(value).scheme not in {"http", "https"}:
                row[key] = None
        rows.append(row)
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = TEMPLATE.replace("__DATA__", data).replace("__TOTAL__", str(total)).replace("__SEED__", str(seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(document, encoding="utf-8")
    tmp.replace(path)
    return path


TEMPLATE = r'''<!doctype html>
<html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polska kultura · podgląd par obraz–tekst</title>
<style>
:root{color-scheme:light;--ink:#182622;--muted:#58665e;--paper:#f3f2eb;--line:#d7ddd3;--accent:#24634d}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}
header{max-width:1540px;margin:auto;padding:42px 32px 24px}h1{font:600 clamp(26px,4vw,42px)/1.15 Georgia,serif;margin:8px 0 15px}
small,.muted{color:var(--muted)}header p{max-width:900px}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px;align-items:center}
button,input,select{font:inherit;border:1px solid var(--line);border-radius:7px;padding:9px 12px;background:#fff;color:var(--ink)}
button{cursor:pointer;background:var(--accent);color:white;border-color:var(--accent)}input{min-width:240px;flex:1}
main{max-width:1540px;margin:auto;padding:0 32px 48px;display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:22px}
article{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:white;box-shadow:0 3px 12px #18262208}
.picture{display:block;background:#e5e8df;height:290px;position:relative}.picture img{width:100%;height:100%;object-fit:contain}
.body{padding:21px}.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}h2{font-size:19px;margin:7px 0 15px}
.caption{border-left:3px solid #b8cdbd;padding-left:12px;margin:15px 0}.caption p{margin:4px 0 0;white-space:pre-wrap;overflow-wrap:anywhere}
.tag{font-size:11px;display:inline-block;border-radius:4px;padding:2px 6px;background:#edf2ec;color:#24583e;margin-left:6px}
.weak{background:#fff1d5;color:#7b5211}.links{display:flex;gap:15px;flex-wrap:wrap;margin-top:16px;font-size:13px}a{color:var(--accent)}
summary{cursor:pointer;font-size:13px;color:var(--muted)}details{margin-top:16px}pre{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:350px;overflow:auto}
.source-description{font-size:13px;max-height:200px;overflow:auto}.empty{padding:30px;grid-column:1/-1}
@media(max-width:550px){header{padding:25px 18px}main{padding:0 18px 35px;grid-template-columns:1fr}.picture{height:260px}}
</style>
<header><div class="eyebrow">Wikimedia Commons × Wikidata</div><h1>Polska kultura. Obraz i tekst.</h1>
<p>Losowa próbka z <strong>__TOTAL__</strong> zebranych rekordów, seed <strong>__SEED__</strong>.
Teksty pochodzą ze źródeł. Ocena <b>D</b> oznacza kontekst encji lub kategorii; może ogólnie opisywać temat, a nie zawartość zdjęcia.</p>
<small>A — caption konkretnego obrazu · B — opis/tytuł obrazu · C — structured metadata · D — kontekst encji/kategorii.<br>
Obrazy ładują się z Wikimedia po otwarciu strony; wymagają internetu. Kliknięcie zdjęcia otwiera plik źródłowy.</small>
<nav><input id="search" placeholder="Szukaj encji, opisu lub pliku…" aria-label="Szukaj"><select id="quality" aria-label="Jakość tekstu"><option value="">Każda jakość</option><option>A</option><option>B</option><option>C</option><option>D</option></select><button id="shuffle">Przetasuj próbkę</button><span id="count" class="muted"></span></nav></header>
<main id="cards"></main><script type="application/json" id="records">__DATA__</script>
<script>
const records=JSON.parse(document.querySelector('#records').textContent);
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sources={structured_caption:'caption obrazu',image_description:'opis obrazu',image_title:'tytuł pliku',depicts:'depicts',entity_description:'opis encji',entity_title:'nazwa encji',category_description:'opis kategorii',collection_description:'opis kolekcji'};
function caption(r,lang){const text=r['caption_'+lang];if(!text)return `<div class="caption"><small>${lang.toUpperCase()}: brak tekstu w tym języku</small></div>`;
const q=r.caption_quality[lang],source=r.caption_source[lang];return `<div class="caption"><small>${lang.toUpperCase()} · ${escape(sources[source]||source)} <span class="tag ${q==='D'?'weak':''}">${escape(q)}</span></small><p>${escape(text)}</p></div>`}
function render(){const query=document.querySelector('#search').value.toLocaleLowerCase(),quality=document.querySelector('#quality').value;
const selected=records.filter(r=>(!query||JSON.stringify(r).toLocaleLowerCase().includes(query))&&(!quality||Object.values(r.caption_quality).includes(quality)));
document.querySelector('#count').textContent=`${selected.length} / ${records.length}`;
document.querySelector('#cards').innerHTML=selected.length?selected.map(r=>`<article>
<a class="picture" href="${escape(r.source_page_url)}" target="_blank" rel="noopener noreferrer"><img src="${escape(r.preview_url)}" data-fallback="${escape(r.image_original_url)}" loading="lazy" referrerpolicy="no-referrer" alt="${escape(r.image_filename)}"></a>
<div class="body"><div class="eyebrow">${escape(r.wikidata_qid||'Kategoria Commons')} · ${r.width} × ${r.height}</div><h2>${escape(r.entity_label_pl||r.entity_label_en||r.image_filename)}</h2>
${caption(r,'pl')}${caption(r,'en')}
<small>${escape(r.creator||'Autor: brak w metadanych')}<br>${escape(r.license||'Licencja: brak')}</small>
<div class="links"><a href="${escape(r.source_page_url)}" target="_blank" rel="noopener noreferrer">Strona pliku ↗</a>${r.license_url?`<a href="${escape(r.license_url)}" target="_blank" rel="noopener noreferrer">Licencja ↗</a>`:''}</div>
${r.description?`<details><summary>Pełny opis źródłowy</summary><p class="source-description">${escape(r.description)}</p></details>`:''}
<details><summary>Provenance i metadane</summary><pre>${escape(JSON.stringify(r,null,2))}</pre></details></div></article>`).join(''):'<p class="empty">Brak par spełniających filtr.</p>';
document.querySelectorAll('img').forEach(img=>img.onerror=()=>{if(img.dataset.fallback){img.src=img.dataset.fallback;delete img.dataset.fallback}else{img.alt='Nie udało się załadować obrazu — otwórz stronę pliku.'}})}
document.querySelector('#search').oninput=render;document.querySelector('#quality').onchange=render;
document.querySelector('#shuffle').onclick=()=>{for(let i=records.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[records[i],records[j]]=[records[j],records[i]]}render()};render();
</script></html>'''
