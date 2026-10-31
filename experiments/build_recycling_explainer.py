"""Build one self-contained Polish explainer from an actual saved recycling example."""
import base64
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/visualnews_pipeline/recycling_explained'
TARGET=ROOT/'reports/recykling_visualnews.html'

def main():
    example=json.loads((SOURCE/'example.json').read_text(encoding='utf-8'))
    outputs=example['outputs'];before=outputs['judge'];after=outputs['recheck']
    assert example['route']=='ready_recycled'
    assert before['visual_grounding'] in [1,2] and not before['image_caption_mismatch']
    assert after['visual_grounding']>=3 and not after['image_caption_mismatch']
    assert example['final_en']==outputs['recycle']['recycled_caption']
    assert example['final_pl']==outputs['translate']['text']
    esc=html.escape
    original=esc(example['caption'])
    for phrase in sorted(before['unsupported_information'],key=len,reverse=True):
        original=original.replace(esc(phrase),'<mark class="context">'+esc(phrase)+'</mark>')
    original=original.replace('performs','<mark class="visible">performs</mark>')
    image='data:image/jpeg;base64,'+base64.b64encode((SOURCE/'example.jpg').read_bytes()).decode()
    raw=json.dumps(outputs,ensure_ascii=False,indent=2)
    downloadable='data:application/json;base64,'+base64.b64encode(json.dumps(example,ensure_ascii=False,indent=2).encode()).decode()
    page='''<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Recykling podpisów VisualNews, krok po kroku na rzeczywistym przykładzie.">
<title>Jak działa recykling podpisu · VisualNews</title>
<style>
:root{--ink:#192d35;--muted:#586971;--paper:#f6f5f0;--line:#dce3df;--green:#15694b;--mint:#e5f4eb;--red:#8c3830;--rose:#fbe7e1;--blue:#225b88;--sky:#e9f1fa}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1100px;margin:0 auto;padding:42px 28px 48px}a{color:var(--blue);text-underline-offset:3px}header{margin-bottom:30px}.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;font-weight:750;color:var(--green)}h1{font-size:clamp(32px,5vw,52px);line-height:1.12;letter-spacing:-.04em;margin:13px 0 18px;max-width:800px}h2{font-size:25px;line-height:1.25;letter-spacing:-.02em;margin:0 0 14px}h3{font-size:18px;margin:0 0 6px}p{margin:0 0 14px}.lead{max-width:810px;font-size:20px;color:var(--muted)}.route{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-top:24px;font-size:13px}.route a{background:white;border:1px solid var(--line);border-radius:99px;padding:7px 13px;text-decoration:none;color:var(--ink)}.route span{color:var(--muted)}.hero{display:grid;grid-template-columns:.95fr 1.05fr;gap:24px;align-items:stretch;margin-bottom:30px}.photo{margin:0;background:#e9e4db;border-radius:18px;overflow:hidden;display:flex;flex-direction:column}.photo img{display:block;width:100%;height:330px;object-fit:contain;background:#29221f}.photo figcaption{font-size:13px;color:var(--muted);padding:12px 17px}.card{background:white;border:1px solid var(--line);border-radius:18px;padding:25px}.label{font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:750;margin:0 0 12px}blockquote{margin:0;font-size:19px;line-height:1.65}mark{padding:2px 3px;border-radius:4px;color:inherit;box-decoration-break:clone;-webkit-box-decoration-break:clone}.context{background:var(--rose);color:var(--red)}.visible{background:var(--mint);color:var(--green)}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin-top:18px;color:var(--muted)}.legend span{display:flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}.dot.red{background:#d58c7d}.dot.green{background:#60a784}.explanation{background:#eaf0ed;border-left:4px solid #7a9e8a;padding:16px 20px;border-radius:0 10px 10px 0;margin:0 0 32px}.explanation p{margin:0}.steps{display:grid;gap:18px}.step{display:grid;grid-template-columns:54px 1fr;gap:17px;background:white;border:1px solid var(--line);border-radius:18px;padding:25px;scroll-margin-top:18px}.number{display:grid;place-items:center;width:46px;height:46px;border-radius:14px;background:var(--sky);color:var(--blue);font-size:20px;font-weight:800}.step.final{border-color:#9fc6ae;background:#f7fcf8}.step.final .number{background:var(--mint);color:var(--green)}.input{display:flex;align-items:center;gap:7px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:12px 0 16px}.chip{padding:4px 9px;background:#f1f4f5;border-radius:6px;border:1px solid #e3e9e9;color:#3d5660}.scores{display:flex;gap:11px;flex-wrap:wrap;margin:12px 0}.score{background:#f3f6f5;border-radius:10px;padding:9px 13px;min-width:135px}.score small{display:block;font-size:11px;letter-spacing:.02em;color:var(--muted)}.score strong{font-size:23px}.score.good{background:var(--mint);color:var(--green)}.score.caution{background:#fff1dc;color:#8c5a16}.notes{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:17px}.notes>div{padding:15px;border-radius:10px}.remove{background:#fcf1ed}.keep{background:#eef7f0}ul{margin:8px 0 0;padding-left:21px}li{margin:3px 0}.en{background:#f4f7fb;border-left:3px solid #92aec7;padding:17px 19px;border-radius:0 9px 9px 0;margin:14px 0}.pl{background:#eaf5ed;border-left:3px solid #61a77d;padding:19px;border-radius:0 9px 9px 0;margin:14px 0}.result{display:inline-block;font-size:13px;font-weight:750;background:var(--green);color:white;border-radius:99px;padding:6px 12px}.small{font-size:14px;color:var(--muted)}.small:last-child{margin-bottom:0}.rules{margin-top:26px}details{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin-top:15px}summary{cursor:pointer;font-weight:700}details[open] summary{margin-bottom:15px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:12px}pre{overflow:auto;white-space:pre-wrap;word-break:break-word;font:12px/1.6 ui-monospace,Consolas,monospace;background:#f3f5f6;padding:16px;border-radius:8px;max-height:520px}.foot{margin:24px 0 0;font-size:12px;color:var(--muted)}.download{display:inline-block;margin-top:4px;font-size:14px}
@media(max-width:760px){main{padding:26px 17px}.hero{grid-template-columns:1fr}.photo img{height:290px}.step{grid-template-columns:1fr;padding:20px}.number{width:37px;height:37px;font-size:17px}.notes{grid-template-columns:1fr}.card{padding:20px}blockquote{font-size:17px}.lead{font-size:18px}.route{gap:5px}.route a{padding:6px 9px}.route span{display:none}h2{font-size:23px}}
@media print{body{background:white;font-size:11pt}main{max-width:none;padding:0}.route,.download{display:none}h1{font-size:30pt}.photo img{height:230px}.step,.hero,.explanation{break-inside:avoid}.step{padding:14px}.number{width:34px;height:34px}.small{font-size:10pt}details{break-inside:avoid}}
</style></head><body><main>
<header><div class="eyebrow">VisualNews · prawdziwy przykład z pipeline’u</div>
<h1>Jak działa recykling podpisu?</h1>
<p class="lead">Z opisu wiadomości robimy opis tego, co widać na zdjęciu. Dzięki temu para obraz–tekst może zostać użyta do treningu wyszukiwania obrazów.</p>
<nav class="route" aria-label="Etapy procesu"><a href="#ocena">1. Ocena</a><span>→</span><a href="#recykling">2. Recykling</a><span>→</span><a href="#kontrola">3. Kontrola</a><span>→</span><a href="#polski">4. Polski</a></nav></header>
<section class="hero" aria-label="Obraz i oryginalny podpis">
<figure class="photo"><img src="@@IMAGE@@" alt="Kobieta z długimi, kręconymi włosami trzyma mikrofon przy ustach. Ma czarny strój, a w tle widać rozmyte światła."><figcaption>Ten sam obraz trafia do każdego etapu. Nie zmieniamy zdjęcia.</figcaption></figure>
<div class="card"><p class="label">Oryginalny podpis EN · autentyczne wejście</p><blockquote lang="en">@@ORIGINAL@@</blockquote>
<div class="legend"><span><i class="dot red" aria-hidden="true"></i>Kontekst nieweryfikowalny na zdjęciu</span><span><i class="dot green" aria-hidden="true"></i>Widoczna czynność</span></div>
<p class="small" style="margin-top:18px">Widać występ z mikrofonem. W tym przykładzie model nie znajduje na zdjęciu potwierdzenia nazwiska, nazwy imprezy ani miasta.</p></div></section>
<aside class="explanation"><p><strong>Co odzyskujemy?</strong> Informację o występie i widocznych cechach osoby. Usuwamy nazwisko, nazwę wydarzenia i lokalizację. Nowy opis może być dłuższy: obraz pozwala doprecyzować widoczne włosy, strój czy mikrofon.</p></aside>
<section class="steps" aria-label="Cztery osobne etapy">
<article class="step" id="ocena"><div class="number" aria-hidden="true">1</div><div><h2>Najpierw oceniamy oryginał</h2><p>Judge sprawdza, jaka część podpisu ma oparcie w samym obrazie.</p>
<div class="input">Wejście: <span class="chip">obraz</span> + <span class="chip">oryginalny podpis EN</span></div>
<div class="scores"><div class="score caution"><small>Grounding · wynik modelu</small><strong>@@BEFORE@@ / 4</strong></div><div class="score"><small>Potrzeba kontekstu zewnętrznego</small><strong>@@EXTERNAL_BEFORE@@ / 3</strong></div><div class="score good"><small>Sprzeczność z obrazem?</small><strong>Nie</strong></div></div>
<p class="small"><strong>Decyzja:</strong> grounding 1–2 bez mismatch uruchamia recykling. Nieweryfikowalna nazwa miasta nie oznacza automatycznie sprzeczności.</p></div></article>
<article class="step" id="recykling"><div class="number" aria-hidden="true">2</div><div><h2>Osobny prompt tworzy opis wizualny</h2><p>Model dostaje komplet informacji z pierwszego etapu i ponownie patrzy na obraz.</p>
<div class="input">Wejście: <span class="chip">ten sam obraz</span> + <span class="chip">oryginalny podpis EN</span> + <span class="chip">pełny wynik judge z etapu 1</span></div>
<p class="label">Rzeczywisty wynik recyklingu EN</p><blockquote class="en" lang="en">@@RECYCLED@@</blockquote>
<div class="notes"><div class="remove"><h3>Usunięto kontekst</h3><ul><li>Nazwisko: „Nicki Minaj”.</li><li>Wydarzenie: „Hot 1079 Birthday Bash”.</li><li>Miasto: „Atlanta”.</li></ul></div><div class="keep"><h3>Zachowano i doprecyzowano obraz</h3><ul><li>Osobę występującą z mikrofonem.</li><li>Długie, ciemne, kręcone włosy.</li><li>Czarny strój, pasek i światła w tle.</li></ul></div></div>
<p class="small" style="margin-top:15px">Polecenie wymaga użytecznego opisu do wyszukiwania obrazu. Gdy nie ma czego odzyskać, model może zwrócić „nieużyteczny” zamiast wymuszać nowy podpis.</p></div></article>
<article class="step" id="kontrola"><div class="number" aria-hidden="true">3</div><div><h2>Nowy podpis przechodzi ponowną ocenę</h2><p>Oczyszczony tekst wraca do judge w kolejnym, oddzielnym zapytaniu.</p>
<div class="input">Wejście: <span class="chip">ten sam obraz</span> + <span class="chip">podpis EN po recyklingu</span></div>
<div class="scores"><div class="score good"><small>Grounding · wynik modelu</small><strong>@@AFTER@@ / 4</strong></div><div class="score good"><small>Potrzeba kontekstu zewnętrznego</small><strong>@@EXTERNAL_AFTER@@ / 3</strong></div><div class="score good"><small>Sprzeczność z obrazem?</small><strong>Nie</strong></div></div>
<p class="small"><strong>Warunek przejścia:</strong> grounding 3–4 i brak mismatch. Ten przykład spełnił oba warunki.</p></div></article>
<article class="step final" id="polski"><div class="number" aria-hidden="true">4</div><div><h2>Dopiero zatwierdzony opis tłumaczymy na polski</h2><p>Tłumaczenie jest osobnym etapem. Dostaje już oczyszczony tekst angielski oraz obraz; reasoning jest włączony.</p>
<div class="input">Wejście: <span class="chip">zatwierdzony podpis EN</span> + <span class="chip">ten sam obraz</span></div>
<p class="label">Rzeczywiste tłumaczenie PL · bez redakcyjnych zmian</p><blockquote class="pl" lang="pl">@@POLISH@@</blockquote>
<span class="result">Gotowa para treningowa: obraz + polski opis</span>
<p class="small" style="margin-top:15px">Jeśli dokładnie ten końcowy tekst EN ma już polskie tłumaczenie w checkpointach, pipeline wykorzystuje je ponownie. W pokazanym przykładzie powstało nowe tłumaczenie.</p></div></article>
</section>
<section class="rules" aria-label="Dodatkowe informacje"><details><summary>Co dzieje się z pozostałymi przykładami?</summary>
<table><thead><tr><th>Wynik pierwszej oceny</th><th>Dalszy krok</th></tr></thead><tbody>
<tr><td>Grounding 3–4, bez mismatch</td><td>Oryginalny EN trafia do tłumaczenia lub korzysta z istniejącego PL.</td></tr>
<tr><td>Grounding 1–2, bez mismatch</td><td>Recykling → ponowna ocena → tłumaczenie, jeśli kontrola została zaliczona.</td></tr>
<tr><td>Grounding 0</td><td>Odrzut z gotowego zbioru; źródłowy rekord pozostaje zachowany.</td></tr>
<tr><td>Mismatch lub nieudany recykling</td><td>Przegląd / wyłączenie z gotowych par treningowych.</td></tr></tbody></table></details>
<details><summary>Pokaż dokładne zapisane odpowiedzi modelu</summary><p class="small">Poniższy JSON to zapis etapów tego przykładu. Komentarze i kolory powyżej są objaśnieniem redakcyjnym. Oceny i confidence pochodzą od modelu; nie stanowią gwarancji poprawności.</p><pre>@@RAW@@</pre><a class="download" href="@@DOWNLOAD@@" download="przyklad_recyklingu_visualnews.json">Pobierz pełny przykład jako JSON</a></details></section>
<footer class="foot">Przykład: <strong>@@ID@@</strong> · VisualNews / Washington Post · zapis testu integracyjnego tego samego pipeline’u.<br>Obraz jest osadzony w tym pliku — HTML działa samodzielnie, bez internetu. Oryginalne teksty i odpowiedzi pozostają zapisane osobno.</footer>
</main></body></html>'''
    replacements={'@@IMAGE@@':image,'@@ORIGINAL@@':original,'@@RECYCLED@@':esc(example['final_en']),
      '@@POLISH@@':esc(example['final_pl']),'@@BEFORE@@':str(before['visual_grounding']),
      '@@AFTER@@':str(after['visual_grounding']),'@@EXTERNAL_BEFORE@@':str(before['external_context_dependency']),
      '@@EXTERNAL_AFTER@@':str(after['external_context_dependency']),'@@RAW@@':esc(raw),
      '@@DOWNLOAD@@':downloadable,'@@ID@@':esc(example['id'])}
    for key,value in replacements.items():page=page.replace(key,value)
    assert '@@' not in page
    TARGET.write_text(page,encoding='utf-8',newline='\n')
    print(json.dumps(dict(path=str(TARGET),bytes=TARGET.stat().st_size,example=example['id'],self_contained=True),ensure_ascii=False))

if __name__=='__main__':main()
