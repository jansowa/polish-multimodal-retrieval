/* A shared interface for a live read-only API and a portable Sites snapshot. */
'use strict';
const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => new Intl.NumberFormat('pl-PL').format(value || 0);
const sourceNames = {structured_caption:'caption obrazu',image_description:'opis obrazu',image_title:'tytuł pliku',depicts:'depicts',entity_description:'opis encji',entity_title:'nazwa encji',category_description:'opis kategorii',collection_description:'opis kolekcji',category_title:'tytuł kategorii'};
const defaults = {dataset:'',page:1,size:12,q:'',language:'any',quality:'',license:'',sort:'newest',seed:'42'};
let state = {...defaults}, catalog, provider, current = {items:[],page:1,pages:1,total:0,size:12};
let requestVersion = 0, lastFocusedCard, detailIndex = -1, detailVersion = 0;

function readURL() {
  const params = new URLSearchParams(location.search), next = {...defaults};
  for (const key of Object.keys(next)) if (params.has(key)) next[key] = params.get(key);
  next.page = Math.max(1,parseInt(next.page,10)||1);
  next.size = [12,24,48,96].includes(Number(next.size)) ? Number(next.size) : 12;
  if (!['any','pl','en','both'].includes(next.language)) next.language='any';
  if (!['','A','B','C','D'].includes(next.quality)) next.quality='';
  if (!['newest','oldest','entity','resolution','random'].includes(next.sort)) next.sort='newest';
  next.q=next.q.slice(0,200);next.license=next.license.slice(0,200);next.seed=next.seed.slice(0,100);
  return next;
}
function syncURL(replace=false) {
  const params = new URLSearchParams();
  for (const [key,value] of Object.entries(state)) if (String(value)!==String(defaults[key])) params.set(key,value);
  const url = location.pathname+(params.size?'?'+params:'');
  history[replace?'replaceState':'pushState'](null,'',url);
}
async function getJSON(url) {
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok){let message;try{message=(await response.json()).error}catch{}throw new Error(message||'Nie udało się odczytać zbioru ('+response.status+').')}
  return response.json();
}
class LiveProvider {
  mode='live';
  catalog(){return getJSON('api/catalog')}
  records(query){return getJSON('api/records?'+new URLSearchParams(query))}
  record(dataset,id){return getJSON('api/record?'+new URLSearchParams({dataset,id}))}
}
class SnapshotProvider {
  mode='snapshot';indexes=new Map();hashes=new Map();
  catalog(){return getJSON('site-data/catalog.json')}
  async index(dataset){if(!this.indexes.has(dataset))this.indexes.set(dataset,await getJSON('site-data/'+encodeURIComponent(dataset)+'/index.json'));return this.indexes.get(dataset)}
  async records(query){
    const index=await this.index(query.dataset), term=query.q.toLocaleLowerCase();
    let rows=index.filter(row=>{
      if(term&&!row.search_text.includes(term))return false;
      if(query.language==='pl'&&!row.caption_pl||query.language==='en'&&!row.caption_en||query.language==='both'&&(!row.caption_pl||!row.caption_en))return false;
      const langs=['pl','en'].includes(query.language)?[query.language]:['pl','en'];
      if(query.quality&&!langs.some(lang=>row.caption_quality[lang]===query.quality))return false;
      return !query.license||row.license===query.license;
    });
    let hashes;
    if(query.sort==='random'){
      const cacheKey=query.dataset+':'+query.seed;
      if(!this.hashes.has(cacheKey)){
        const results=await Promise.all(index.map(async row=>{const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(query.seed+':'+row.id));return [row.id,Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('')]}));
        this.hashes.set(cacheKey,new Map(results));
      }hashes=this.hashes.get(cacheKey);
    }
    const compare=(a,b)=>a<b?-1:a>b?1:0;
    rows.sort((a,b)=>{
      let delta=0;
      if(query.sort==='newest'||query.sort==='oldest')delta=compare(a.retrieved_at||'',b.retrieved_at||'')*(query.sort==='newest'?-1:1);
      if(query.sort==='entity')delta=compare((a.entity_label_pl||a.entity_label_en||'').toLocaleLowerCase(),(b.entity_label_pl||b.entity_label_en||'').toLocaleLowerCase());
      if(query.sort==='resolution')delta=(b.width||0)*(b.height||0)-(a.width||0)*(a.height||0);
      if(query.sort==='random')delta=compare(hashes.get(a.id),hashes.get(b.id));
      return delta||compare(a.id,b.id);
    });
    const total=rows.length,pages=Math.max(1,Math.ceil(total/query.size)),page=Math.min(query.page,pages);
    return {items:rows.slice((page-1)*query.size,page*query.size),total,pages,page,size:query.size};
  }
  async record(dataset,id){const row=(await this.index(dataset)).find(row=>row.id===id);if(!row)throw new Error('Nie znaleziono rekordu.');return getJSON('site-data/'+encodeURIComponent(dataset)+'/records/'+row.detail_key+'.json')}
}
function formatDate(value){try{return new Intl.DateTimeFormat('pl-PL',{dateStyle:'medium',timeStyle:'short',timeZone:'Europe/Warsaw'}).format(new Date(value))}catch{return 'brak daty'}}
function notice(text,error=false){const element=$('#notice');element.hidden=!text;element.textContent=text;element.classList.toggle('error',error)}
function renderCatalog(){
  $('#dataset').innerHTML=catalog.datasets.map(d=>`<option value="${esc(d.id)}">${esc(d.label)} · ${number(d.count)}</option>`).join('');
  $('#dataset').value=state.dataset;
  const dataset=catalog.datasets.find(d=>d.id===state.dataset);if(!dataset)return;
  $('#license').innerHTML='<option value="">Wszystkie licencje</option>'+dataset.licenses.map(license=>`<option>${esc(license)}</option>`).join('');
  $('#license').value=state.license;
  $('#metric-total').textContent=number(dataset.count);$('#metric-direct').textContent=number(dataset.direct_count);$('#metric-entities').textContent=number(dataset.entities);
  $('#phase').textContent=provider.mode==='snapshot'?(dataset.count?'Migawka kolekcji':dataset.phase):dataset.phase;
  $('#updated').textContent=(provider.mode==='snapshot'?'Stan z ':'Odczyt: ')+formatDate(catalog.generated_at);
  $('#mode').textContent=provider.mode==='live'?'Dane na bieżąco':'Migawka zbioru';
  $('#about-mode').textContent=provider.mode==='live'?'Ten widok czyta aktualne wyniki z komputera, na którym działa scraper. Dane dopisywane przez scraper pojawiają się po odświeżeniu.':'Ta prywatna strona pokazuje migawkę zbioru z '+formatDate(catalog.generated_at)+'. Nowe wyniki scrapera wymagają ponownej synchronizacji strony. Widok lokalny pokazuje je na bieżąco.';
}
function syncControls(){for(const id of ['dataset','language','quality','license','sort','size','q'])$('#'+id).value=state[id];$('#reroll').hidden=state.sort!=='random'}
function captionMarkup(row,lang,full=false){
  const text=row['caption_'+lang],quality=row.caption_quality?.[lang]||'',source=row.caption_source?.[lang];
  return `<div class="${full?'detail-caption':'card-caption'}"><div class="caption-label">${lang.toUpperCase()} ${quality?`<span class="grade ${quality.toLowerCase()}">${esc(quality)}</span>`:''} <span>${esc(sourceNames[source]||source||'brak opisu')}</span></div><p class="${text?'':'no-caption'}">${esc(text||'Brak tekstu w tym języku.')}</p>${full&&quality==='D'?'<div class="detail-warning">To kontekst encji lub kategorii. Może nie opisywać tego, co widać na zdjęciu.</div>':''}</div>`;
}
function cardLanguages(row){if(state.language==='both')return ['pl','en'];if(['pl','en'].includes(state.language))return [state.language];if(state.quality){const matched=['pl','en'].find(lang=>row.caption_quality[lang]===state.quality);if(matched)return [matched]}return [row.caption_pl?'pl':'en']}
function bindImages(root){root.querySelectorAll('img[data-fallback]').forEach(img=>{img.onerror=()=>{if(img.dataset.fallback){img.src=img.dataset.fallback;delete img.dataset.fallback}else{img.hidden=true;const note=document.createElement('span');note.className='image-failed';note.textContent='Obraz jest chwilowo niedostępny. Otwórz stronę źródłową w szczegółach.';img.parentElement.append(note)}}})}
function renderCards(){
  const gallery=$('#gallery');gallery.setAttribute('aria-busy','false');
  if(!current.items.length){const dataset=catalog.datasets.find(d=>d.id===state.dataset),empty=dataset?.count===0;gallery.innerHTML=`<div class="empty-state"><p class="overline">${empty?'KOLEKCJA W PRZYGOTOWANIU':'SZUKAJ DALEJ'}</p><h2>${empty?'Obrazy jeszcze się zbierają.':'Nie znaleziono takich obrazów.'}</h2><p>${empty?'Ten przebieg jest na etapie odkrywania tematów. Możesz już przeglądać ukończony pilot lub odświeżyć dane później.':'Zmień zapytanie lub usuń część filtrów. Wyszukiwanie obejmuje opisy, kategorie, nazwy i identyfikatory.'}</p><button class="primary" id="empty-action">${empty?'Otwórz gotowy zbiór':'Wyczyść filtry'}</button></div>`;$('#empty-action').onclick=()=>empty?change({dataset:catalog.default_dataset}):reset();return}
  gallery.innerHTML=current.items.map((row,index)=>`<article class="image-card"><button class="image-button" data-open="${index}" aria-label="Zobacz szczegóły: ${esc(row.entity_label_pl||row.image_filename)}"><img src="${esc(row.thumbnail_url||row.image_original_url)}" data-fallback="${esc(row.image_original_url||'')}" loading="lazy" decoding="async" referrerpolicy="no-referrer" alt="${esc(row.image_filename)}"><span class="image-label">${number(row.width)} × ${number(row.height)}</span></button><div class="card-content"><div class="card-kicker"><span>WIKIMEDIA COMMONS</span><span>${esc(row.wikidata_qid||'KATEGORIA')}</span></div><button class="card-title" data-open="${index}">${esc(row.entity_label_pl||row.entity_label_en||row.image_filename.replace(/^File:/,''))}</button>${cardLanguages(row).map(lang=>captionMarkup(row,lang)).join('')}<div class="card-bottom"><span title="${esc(row.creator||'')}">${esc(row.creator||'Autor niepodany')}</span>${row.license_url?`<a href="${esc(row.license_url)}" target="_blank" rel="noopener noreferrer">${esc(row.license)} ↗</a>`:`<span>${esc(row.license||'Brak licencji')}</span>`}</div></div></article>`).join('');
  gallery.querySelectorAll('[data-open]').forEach(button=>button.onclick=()=>{lastFocusedCard=button;openDetail(Number(button.dataset.open))});bindImages(gallery);
}
function pageWindow(page,pages){const values=new Set([1,pages]);for(let i=Math.max(1,page-2);i<=Math.min(pages,page+2);i++)values.add(i);return [...values].sort((a,b)=>a-b)}
function renderPagination(){
  const {page,pages,total,size}=current,first=total?(page-1)*size+1:0,last=Math.min(page*size,total);
  $('#result-count').innerHTML=`<strong>${number(total)}</strong> ${total===1?'obraz':'obrazów'}${state.q?' dla „'+esc(state.q)+'”':''}`;
  $('#page-summary').textContent=`${number(first)}–${number(last)} z ${number(total)} · strona ${number(page)} / ${number(pages)}`;
  let buttons=`<button data-page="1" ${page===1?'disabled':''} aria-label="Pierwsza strona" title="Pierwsza strona">«</button><button data-page="${page-1}" ${page===1?'disabled':''} aria-label="Poprzednia strona" title="Poprzednia strona">←</button>`,previous=0;
  for(const value of pageWindow(page,pages)){if(previous&&value>previous+1)buttons+='<span class="ellipsis" aria-hidden="true">…</span>';buttons+=`<button data-page="${value}" ${value===page?'aria-current="page"':''} aria-label="Strona ${value}">${number(value)}</button>`;previous=value}
  buttons+=`<button data-page="${page+1}" ${page===pages?'disabled':''} aria-label="Następna strona" title="Następna strona">→</button><button data-page="${pages}" ${page===pages?'disabled':''} aria-label="Ostatnia strona" title="Ostatnia strona">»</button>`;
  $('#page-buttons').innerHTML=buttons;$('#page-buttons').querySelectorAll('button').forEach(button=>button.onclick=()=>go(Number(button.dataset.page)));
  $('#jump').value=page;$('#jump').max=pages;
}
async function loadRecords({scroll=false}={}){
  const version=++requestVersion;$('#gallery').setAttribute('aria-busy','true');$('#result-count').textContent='Wczytywanie obrazów…';
  if(!current.items.length)$('#gallery').innerHTML=Array.from({length:6},()=>'<div class="skeleton" aria-hidden="true"></div>').join('');
  try{const result=await provider.records({...state});if(version!==requestVersion)return;current=result;
    if(state.page!==result.page){state.page=result.page;syncURL(true)}renderCards();renderPagination();notice('');
    if(scroll)$('#results').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
  }catch(error){if(version!==requestVersion)return;$('#gallery').setAttribute('aria-busy','false');$('#result-count').textContent='Nie udało się wczytać danych';if(!current.items.length)$('#gallery').innerHTML='';notice(error.message+' Użyj przycisku odświeżania.',true)}
}
function change(patch,{page=false}={}){state={...state,...patch,page:page?patch.page:1};if(patch.dataset)state.license='';renderCatalog();syncControls();syncURL();return loadRecords({scroll:page})}
function go(page){if(page<1||page>current.pages||page===current.page)return;return change({page},{page:true})}
function reset(){state={...defaults,dataset:state.dataset};renderCatalog();syncControls();syncURL();loadRecords()}
async function refresh(){const button=$('#refresh');button.disabled=true;try{catalog=await provider.catalog();renderCatalog();await loadRecords()}catch(error){notice(error.message,true)}finally{button.disabled=false}}
function link(url,label){try{if(!['https:','http:'].includes(new URL(url).protocol))return esc(label)}catch{return esc(label)}return `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>`}
async function openDetail(index){
  const version=++detailVersion;detailIndex=index;const row=current.items[index],dialog=$('#detail');if(!row)return;
  $('#detail-body').innerHTML='<div class="about-content"><p>Wczytywanie pełnych metadanych…</p></div>';if(!dialog.open)dialog.showModal();
  try{const record=await provider.record(state.dataset,row.id);if(version!==detailVersion)return;
    const fields=[['Autor',esc(record.creator||'Brak w źródle')],['Attribution',esc(record.attribution||'Brak w źródle')],['Licencja',link(record.license_url,record.license||'Brak')],['Data',esc(record.date||'Brak w źródle')],['Lokalizacja',esc(record.location||'Brak w źródle')],['Encja',record.wikidata_qid?link('https://www.wikidata.org/wiki/'+encodeURIComponent(record.wikidata_qid),record.wikidata_qid):'Kategoria Commons'],['Źródło',link(record.source_page_url,'Strona pliku')],['Pozyskano',esc(formatDate(record.retrieved_at))]];
    $('#detail-body').innerHTML=`<div class="detail-layout"><div class="detail-picture"><img src="${esc(row.thumbnail_url||row.image_original_url)}" data-fallback="${esc(row.image_original_url||'')}" alt="${esc(record.image_filename)}" referrerpolicy="no-referrer"><div><span>${number(record.width)} × ${number(record.height)} · ${esc(record.mime_type)}</span>${link(record.image_original_url,'Oryginalny obraz')}</div></div><div class="detail-info"><p class="overline">WIKIMEDIA COMMONS · ${esc(record.id)}</p><h2 id="detail-title">${esc(record.entity_label_pl||record.entity_label_en||record.image_filename)}</h2><p class="detail-filename">${esc(record.image_filename)}</p>${captionMarkup(record,'pl',true)}${captionMarkup(record,'en',true)}<dl>${fields.map(([name,value])=>`<dt>${name}</dt><dd>${value}</dd>`).join('')}</dl>${record.description?`<details><summary>Pełny opis źródłowy</summary><p class="text-description">${esc(record.description)}</p></details>`:''}<details><summary>Kategorie (${number(record.categories?.length)}) i depicts</summary><ul class="category-chips">${(record.categories||[]).map(c=>`<li>${esc(c.replace(/^Category:/,''))}</li>`).join('')}</ul><p class="detail-filename">Depicts: ${(record.depicts||[]).map(id=>link('https://www.wikidata.org/wiki/'+encodeURIComponent(id),id)).join(', ')||'brak w źródle'}</p></details><details><summary>Pełny rekord i provenance</summary><pre>${esc(JSON.stringify(record,null,2))}</pre></details><div class="detail-actions"><button class="plain" id="download-record">Pobierz rekord JSON ↓</button><div><button class="icon-button" id="previous-detail" ${index===0?'disabled':''} aria-label="Poprzedni obraz">←</button><button class="icon-button" id="next-detail" ${index===current.items.length-1?'disabled':''} aria-label="Następny obraz">→</button></div></div></div></div>`;
    $('#previous-detail').onclick=()=>openDetail(index-1);$('#next-detail').onclick=()=>openDetail(index+1);$('#download-record').onclick=()=>downloadJSON(record);bindImages($('#detail-body'));
  }catch(error){$('#detail-body').innerHTML=`<div class="about-content"><h2 id="detail-title">Nie udało się otworzyć rekordu.</h2><p>${esc(error.message)}</p></div>`}
}
function downloadJSON(record){const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=record.id.replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function bindEvents(){
  for(const key of ['dataset','language','quality','license','sort','size'])$('#'+key).onchange=event=>change({[key]:key==='size'?Number(event.target.value):event.target.value});
  $('#search-form').onsubmit=event=>{event.preventDefault();change({q:$('#q').value.trim()})};$('#q').addEventListener('search',()=>{if(!$('#q').value)change({q:''})});
  $('#jump-form').onsubmit=event=>{event.preventDefault();go(Math.min(current.pages,Math.max(1,Number($('#jump').value)||1)))};
  $('#reset').onclick=reset;$('#refresh').onclick=refresh;$('#reroll').onclick=()=>change({seed:String(Date.now())});
  $('#close-detail').onclick=()=>$('#detail').close();$('#detail').addEventListener('close',()=>{detailVersion++;lastFocusedCard?.focus()});
  $('#about').onclick=()=>$('#about-dialog').showModal();$('#close-about').onclick=()=>$('#about-dialog').close();
  for(const dialog of document.querySelectorAll('dialog'))dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close()}});
  addEventListener('popstate',()=>{state=readURL();if(!catalog.datasets.some(d=>d.id===state.dataset))state.dataset=catalog.default_dataset;renderCatalog();syncControls();loadRecords()});
  addEventListener('keydown',event=>{if(event.altKey||event.ctrlKey||event.metaKey||['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName)||document.querySelector('dialog[open]'))return;if(event.key==='ArrowRight')go(current.page+1);if(event.key==='ArrowLeft')go(current.page-1)});
}
async function boot(){
  try{const config=await getJSON('app-config.json');provider=config.mode==='live'?new LiveProvider():new SnapshotProvider();catalog=await provider.catalog();state=readURL();if(!catalog.datasets.length)throw new Error('Brak dostępnych zbiorów. Uruchom najpierw zbieranie danych.');if(!catalog.datasets.some(d=>d.id===state.dataset))state.dataset=catalog.default_dataset;renderCatalog();syncControls();syncURL(true);bindEvents();await loadRecords();
    if(provider.mode==='live')setInterval(async()=>{if(document.hidden)return;try{const updated=await provider.catalog(),before=catalog.datasets.find(d=>d.id===state.dataset)?.count,after=updated.datasets.find(d=>d.id===state.dataset)?.count;catalog=updated;renderCatalog();if(after>before)notice('Pojawiły się nowe obrazy. Odśwież kolekcję przyciskiem ↻, kiedy zechcesz je zobaczyć.')}catch{}},30000);
  }catch(error){$('#mode').textContent='Zbiór niedostępny';$('#gallery').setAttribute('aria-busy','false');notice(error.message+' Spróbuj ponownie za chwilę.',true);$('#refresh').onclick=()=>location.reload()}
}
boot();
