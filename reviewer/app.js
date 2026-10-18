"use strict";
const $ = id => document.getElementById(id);
const labels = {unreviewed:"Nieocenione",ok:"Poprawne",fix:"Do poprawy",unclear:"Niejasne źródło"};
const state = {run:"", records:[], id:null, dirty:false, revision:0, saving:null, timer:null, generation:0, offset:0, total:0, allCount:0, reviewedCount:0};
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Nie udało się pobrać danych");
  return data;
}
function error(message="") { $("error").textContent=message; $("error").hidden=!message; }
const current = () => state.records.find(r=>r.id===state.id);
function caption(text, language) {
  text=text.replace(/<\|image_\d+\|>\s*/g, "");
  return (language==="en" ? text.replace(/^Find me an everyday image that matches the given caption:\s*/, "")
    : text.replace(/^Znajd[zź][^\n]*?(?:opisu|opisowi):\s*/, "")).trim();
}
function filtered() { return state.records; }
function updateStats() {
  $("stats").replaceChildren();
  const strong=document.createElement("strong");strong.textContent=`${state.reviewedCount} / ${state.allCount}`;
  $("stats").append(strong,document.createElement("br"),document.createTextNode("przykładów ocenionych"));
}
function renderFields(node, r, language) {
  node.replaceChildren();
  const fields=r.fields || [{key:"qry_text",source:r.source,translated:r.translated}];
  for(const field of fields) {
    const section=document.createElement("span");section.className="text-field";
    if(fields.length>1){const label=document.createElement("small");label.textContent=field.label;section.append(label);}
    const text=document.createElement("span");text.textContent=field.key==="qry_text" ? caption(field[language],language==="source"?"en":"pl") : field[language];
    section.append(text);node.append(section);
  }
}
function updateRating() {
  const r=current(); if(!r)return;
  document.querySelectorAll("[data-status]").forEach(button=> {
    if(button.tagName==="BUTTON")button.setAttribute("aria-pressed",String(button.dataset.status===r.review.status));
  });
  $("record-status").textContent=labels[r.review.status];$("record-status").dataset.status=r.review.status;
  updateStats();
}
function render() {
  const rows=filtered();
  if (!rows.some(r=>r.id===state.id))state.id=rows[0]?.id || null;
  const r=current();
  $("workspace").hidden=!r;$("empty").hidden=!!r;
  $("empty-message").textContent=state.allCount?"Zmień wyszukiwanie lub filtr oceny.":"Ten przebieg nie ma jeszcze ukończonych przykładów. Dane odświeżają się automatycznie.";
  $("position").textContent=`${r ? state.offset+rows.findIndex(x=>x.id===r.id)+1 : 0} / ${state.total}`;
  ["previous","next","random","save-next"].forEach(id=>$(id).disabled=!r);
  updateStats(); if(!r)return;
  $("record-id").textContent=r.id;renderFields($("source"),r,"source");renderFields($("translated"),r,"translated");
  $("note").value=r.review.note;$("correction").value=r.review.correction;
  $("correction-details").open=!!r.review.correction;
  $("save-state").textContent=r.review.updated_at?"Zapisano na dysku":"Bez zapisanych zmian";
  const gallery=$("image-gallery");gallery.replaceChildren();gallery.hidden=!r.image;
  $("image-message").hidden=!!r.image;$("full-image").hidden=!r.image;
  if(r.image) {
    const imageLabels={"query_image[0]":"Obraz wejściowy",groundtruth_target_image:"Obraz docelowy"};
    for(const item of r.images) {
      const url="/api/image?"+new URLSearchParams({run:state.run,id:r.id,slot:item.slot});
      const figure=document.createElement("figure");const image=document.createElement("img");
      const label=imageLabels[item.label] || item.label;
      const title=document.createElement("figcaption");title.textContent=label;
      const link=document.createElement("a");link.href=url;link.target="_blank";link.rel="noopener";
      image.alt=label;image.src=url;image.onerror=()=>{image.hidden=true;title.textContent=label+" — nie udało się wczytać obrazu";};
      link.append(image);figure.append(title,link);gallery.append(figure);
      if(item.slot===r.images[0].slot)$("full-image").href=url;
    }
  } else {
    $("image-message").textContent="Brak lokalnego obrazu dla tego przykładu. Możesz ocenić teksty; obrazy ARO, SugarCrepe i M-BEIR nie zostały pobrane do tej przeglądarki.";
  }
  $("flags").hidden=!r.flags.length;$("flags-title").textContent=`Uwagi modelu (${r.flags.length})`;
  $("flags-list").replaceChildren();for(const flag of r.flags){const li=document.createElement("li");li.textContent=flag.reason;$("flags-list").append(li);}
  updateRating();
  localStorage.setItem("pl-review-run",state.run);localStorage.setItem("pl-review-id:"+state.run,state.id);
}
function modified() {
  const r=current();if(!r)return;
  r.review.note=$("note").value;r.review.correction=$("correction").value;
  state.dirty=true;state.revision++;$("save-state").textContent="Zapisywanie…";
  clearTimeout(state.timer);state.timer=setTimeout(()=>flush(),500);
}
async function flush() {
  clearTimeout(state.timer);
  if(state.saving) {const success=await state.saving;if(!success)return false;if(state.dirty)return flush();return true;}
  if(!state.dirty)return true;
  const r=current();if(!r)return true;
  const revision=state.revision;
  const payload={run:state.run,id:r.id,text_hash:r.text_hash,...r.review};
  state.dirty=false;
  state.saving=(async()=> {
    try {
      const saved=await api("/api/annotation",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
      if(revision===state.revision){r.review=saved;$("save-state").textContent="Zapisano na dysku";}
      error();return true;
    } catch(e) {state.dirty=true;$("save-state").textContent="Błąd zapisu";error(e.message);return false;}
  })();
  const success=await state.saving;state.saving=null;return success;
}
async function navigate(direction) {
  if(!await flush() || !state.total)return;
  const index=state.records.findIndex(r=>r.id===state.id);
  const filter=$("filter").value;
  if(filter in labels && current() && current().review.status!==filter) {
    const remaining=state.total-1;
    const target=remaining ? (state.offset+Math.max(0,index)+(direction<0?-1:0)+remaining)%remaining : 0;
    if(await load(state.run,Math.floor(target/100)*100)) {
      state.id=state.records[target-state.offset]?.id || state.records[0]?.id;render();
    }
    return;
  }
  const target=(state.offset+Math.max(0,index)+direction+state.total)%state.total;
  if(target<state.offset || target>=state.offset+state.records.length) {
    if(!await load(state.run,Math.floor(target/100)*100))return;
  }
  state.id=state.records[target-state.offset]?.id || state.records[0]?.id;render();
}
async function load(run=state.run,offset=0,random=false) {
  if(!await flush())return false;
  const generation=++state.generation;
  try {
    const params=new URLSearchParams({run,offset,limit:100,search:$("search").value,filter:$("filter").value,random:random?"1":"0"});
    const page=await api("/api/page?"+params);
    if(generation!==state.generation || state.dirty || state.saving)return false;
    if(run!==state.run){state.id=localStorage.getItem("pl-review-id:"+run);}
    state.run=run;state.records=page.records;state.offset=page.offset;state.total=page.total;
    state.allCount=page.all_count;state.reviewedCount=page.reviewed_count;
    $("run").value=run;error();render();return true;
  }catch(e){error(e.message);return false;}
}
async function refreshRuns(initial=false) {
  try {
    const runs=await api("/api/runs");$("run").replaceChildren();
    for(const pilot of [false,true]) {
      const group=document.createElement("optgroup");group.label=pilot?"Pilotaże":"Przebiegi pełne i eksperymenty";
      for(const run of runs.filter(r=>r.pilot===pilot)){const option=document.createElement("option");option.value=run.id;option.textContent=`${run.label} · ${run.count}`;group.append(option);}
      if(group.children.length)$("run").append(group);
    }
    const remembered=state.run || localStorage.getItem("pl-review-run");
    const selected=runs.find(r=>r.id===remembered) || runs.find(r=>r.count>0) || runs[0];
    if(!selected){$("empty").hidden=false;$("empty-message").textContent="Nie znaleziono checkpointów tłumaczeń w skonfigurowanych katalogach.";return;}
    $("run").value=selected.id;
    if(initial)await load(selected.id);
  }catch(e){error(e.message);}
}
$("run").onchange=()=>load($("run").value);
let searchTimer;
for(const id of ["search","filter"])$(id).addEventListener(id==="search"?"input":"change",()=> {
  clearTimeout(searchTimer);searchTimer=setTimeout(()=>load(state.run),id==="search"?300:0);
});
$("refresh").onclick=async()=>{await refreshRuns();await load(state.run,state.offset);};
$("previous").onclick=()=>navigate(-1);$("next").onclick=()=>navigate(1);$("save-next").onclick=()=>navigate(1);
$("random").onclick=async()=>{if(await load(state.run,0,true)){state.id=state.records[0]?.id;render();}};
document.querySelectorAll("button[data-status]").forEach(button=>button.onclick=()=>{const r=current();if(r){state.reviewedCount+=(r.review.status==="unreviewed"?1:0);r.review.status=button.dataset.status;modified();updateRating();}});
$("clear-rating").onclick=()=>{const r=current();if(r){state.reviewedCount-=(r.review.status!=="unreviewed"?1:0);r.review.status="unreviewed";modified();updateRating();}};
$("note").oninput=modified;$("correction").oninput=modified;
document.querySelectorAll("[data-copy]").forEach(button=>button.onclick=async()=> {
  const r=current();if(!r)return;try{await navigator.clipboard.writeText(caption(r[button.dataset.copy],button.dataset.copy==="source"?"en":"pl"));const old=button.textContent;button.textContent="Skopiowano";setTimeout(()=>button.textContent=old,1200);}catch(e){error("Nie udało się skopiować tekstu.");}
});
$("export").onclick=async()=> {
  if(!state.run||!await flush())return;
  try {const rows=await api("/api/export?run="+encodeURIComponent(state.run));
    const url=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:"application/json"}));
    const a=document.createElement("a");a.href=url;a.download="oceny-"+state.run.split("/").slice(-3).join("-")+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){error(e.message);}
};
document.addEventListener("keydown",e=> {
  if(e.target.closest("input,textarea,select")||e.ctrlKey||e.metaKey||e.altKey)return;
  if(e.key==="ArrowLeft"){e.preventDefault();navigate(-1);}if(e.key==="ArrowRight"){e.preventDefault();navigate(1);}
  const status={"1":"ok","2":"fix","3":"unclear"}[e.key];if(status)document.querySelector(`button[data-status="${status}"]`).click();
});
window.addEventListener("beforeunload",e=>{if(state.dirty||state.saving){e.preventDefault();e.returnValue="";}});
setInterval(async()=> {if(!state.dirty&&!state.saving){await refreshRuns();if(state.run)await load(state.run,state.offset);}},30000);
refreshRuns(true);
