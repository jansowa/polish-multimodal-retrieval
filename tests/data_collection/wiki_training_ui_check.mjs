// Exercise the running training gallery in a real headless Chrome via DevTools.
import {writeFileSync} from 'node:fs';
const origin='http://127.0.0.1:8767';
const target=await (await fetch('http://127.0.0.1:9336/json/new?about:blank',{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true})});
let sequence=0;const pending=new Map(),errors=[];
socket.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const call=pending.get(m.id);if(call){pending.delete(m.id);m.error?call.reject(new Error(JSON.stringify(m.error))):call.resolve(m.result)}}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text)});
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const wait=condition=>evaluate(`new Promise((resolve,reject)=>{const start=Date.now(),timer=setInterval(()=>{if(${condition}){clearInterval(timer);resolve(true)}else if(Date.now()-start>20000){clearInterval(timer);reject(new Error('Timed out: '+${JSON.stringify(condition)}))}},100)})`);
const assert=(condition,message)=>{if(!condition)throw new Error(message)};
const idle="document.querySelector('#gallery').getAttribute('aria-busy')==='false'";
const report={origin,checks:[]};
try{
    await send('Runtime.enable');await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:origin});
    await wait(`${idle}&&document.querySelectorAll('.card').length===12`);
    await evaluate("document.querySelector('#auto').checked=false");
    report.checks.push('12 real ready pairs displayed');
    const first=await evaluate("[...document.querySelectorAll('.meta span')].map(e=>e.textContent)");
    await evaluate("document.querySelector('#next').click()");
    await wait(`${idle}&&document.querySelector('#page').textContent.startsWith('Strona 2')`);
    const second=await evaluate("[...document.querySelectorAll('.meta span')].map(e=>e.textContent)");
    assert(!first.some(id=>second.includes(id)),'Pagination duplicated a record');report.checks.push('pagination');
    await evaluate("document.querySelector('#grade').value='A';document.querySelector('#route').value='recycled';document.querySelector('#filters').requestSubmit()");
    await wait(`${idle}&&document.querySelectorAll('.card').length>0&&[...document.querySelectorAll('.tags')].every(e=>e.textContent.includes('Klasa A')&&e.textContent.includes('Po recyklingu'))`);
    report.checks.push('source-grade A and recycling filters');
    await evaluate("document.querySelector('.photo').click()");
    await wait("document.querySelector('#detail').open&&!!document.querySelector('#download')");
    assert(await evaluate("document.querySelector('.final').textContent.trim().length>0&&document.querySelectorAll('.source-text').length>=2&&document.querySelectorAll('.scores span').length===2"),'Missing final/source/recycled texts or independent scores');
    report.checks.push('detail: final Polish, original, recycled text and both scores');
    assert(await evaluate("[...document.querySelectorAll('details summary')].some(e=>e.textContent.includes('zachowane i usunięte'))"),'Missing recycle provenance');
    await evaluate("document.querySelector('#close').click();document.querySelector('#reset').click()");
    await wait(`${idle}&&document.querySelectorAll('.card').length===12`);
    // Allow thumbnails time to load; unavailable external images have a source link fallback.
    await evaluate("new Promise(resolve=>setTimeout(resolve,2000))");
    report.loaded_images=await evaluate("[...document.querySelectorAll('.photo img')].filter(img=>img.complete&&img.naturalWidth>0).length");
    const desktop=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    writeFileSync('reports/wiki_training_browser/desktop.png',Buffer.from(desktop.data,'base64'));
    await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Horizontal overflow on mobile');
    const mobile=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    writeFileSync('reports/wiki_training_browser/mobile.png',Buffer.from(mobile.data,'base64'));
    report.checks.push('desktop and mobile rendering');
    assert(errors.length===0,'JavaScript runtime errors: '+errors.join(', '));
    report.checks.push('no JavaScript exceptions');report.status='passed';
    writeFileSync('reports/wiki_training_browser/ui_verification.json',JSON.stringify(report,null,2));
    console.log(JSON.stringify(report));
}finally{
    await send('Browser.close').catch(()=>{});socket.close();
}
