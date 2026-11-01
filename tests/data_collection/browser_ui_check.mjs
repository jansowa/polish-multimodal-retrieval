// Exercises the real page in an installed Chrome via its local DevTools endpoint.
// No browser automation packages or production code hooks are required.
import {writeFileSync} from 'node:fs';

const origin=process.argv[2]||'http://127.0.0.1:8766';
const name=process.argv[3]||'live';
const target=await (await fetch('http://127.0.0.1:9335/json/new?'+encodeURIComponent('about:blank'),{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true})});
let sequence=0;const pending=new Map();const errors=[];
socket.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const call=pending.get(message.id);if(call){pending.delete(message.id);message.error?call.reject(new Error(JSON.stringify(message.error))):call.resolve(message.result)}}if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails.text)});
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const result=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value};
const wait=condition=>evaluate(`new Promise((resolve,reject)=>{const start=Date.now();const timer=setInterval(()=>{if(${condition}){clearInterval(timer);resolve(true)}else if(Date.now()-start>12000){clearInterval(timer);reject(new Error('Timed out: '+${JSON.stringify(condition)}))}},80)})`);
const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
const select=(selector,value)=>evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(selector)});el.value=${JSON.stringify(value)};el.dispatchEvent(new Event('change',{bubbles:true}))})()`);
const idle="document.querySelector('#gallery').getAttribute('aria-busy')==='false'";
const count="document.querySelectorAll('.image-card').length";
const assert=(value,message)=>{if(!value)throw new Error(message)};
const report={origin,checks:[]};
try{
  await send('Runtime.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:origin+'/?dataset=pilot&size=12'});
  await wait(`${idle}&&${count}===12`);report.checks.push('initial page: 12 cards');
  const first=await evaluate("[...document.querySelectorAll('.image-card img')].map(img=>img.alt)");
  await click('#page-buttons [data-page="2"]');await wait(`${idle}&&${count}===8`);
  const second=await evaluate("[...document.querySelectorAll('.image-card img')].map(img=>img.alt)");
  assert(!first.some(id=>second.includes(id)),'Pagination repeated an image');
  assert((await evaluate("new URL(location.href).searchParams.get('page')"))==='2','Page missing from URL');report.checks.push('page 2: 8 different cards; URL persisted');
  await select('#size','24');await wait(`${idle}&&${count}===20`);report.checks.push('page size: all 20 records');
  await select('#language','pl');await wait(idle);await select('#quality','A');
  await wait(`${idle}&&${count}>0&&[...document.querySelectorAll('.card-caption .grade')].every(el=>el.textContent==='A')`);report.checks.push('Polish caption quality filter');
  await click('#reset');await wait(`${idle}&&${count}===12`);
  await select('#license','CC0');await wait(`${idle}&&${count}===3`);report.checks.push('license filter: 3 CC0 images');
  await click('#reset');await wait(`${idle}&&${count}===12`);
  await evaluate("document.querySelector('#q').value='Kraków';document.querySelector('#search-form').requestSubmit()");
  await wait(`${idle}&&${count}>0&&${count}<12`);report.checks.push('Unicode search');
  await click('#reset');await wait(`${idle}&&${count}===12`);
  await click('.image-button');await wait("document.querySelector('#detail').open&&!!document.querySelector('#detail-title')&&!!document.querySelector('#download-record')");
  assert((await evaluate("document.querySelectorAll('.detail-caption').length"))===2,'Missing bilingual details');report.checks.push('detail dialog: bilingual captions, provenance and download');
  await click('#close-detail');
  await select('#sort','random');await wait(`${idle}&&${count}===12`);report.checks.push('seeded random ordering');
  await select('#dataset','production');await wait(`${idle}&&!!document.querySelector('.empty-state')`);report.checks.push('empty production dataset has a usable state');
  await select('#dataset','pilot');await wait(`${idle}&&${count}===12`);await select('#sort','newest');await wait(idle);
  await evaluate("document.fonts.ready");
  const desktop=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  writeFileSync('outputs/collection_browser/'+name+'-desktop.png',Buffer.from(desktop.data,'base64'));
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate('new Promise(resolve=>setTimeout(resolve,300))');
  assert(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'),'Mobile layout overflows');
  const mobile=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  writeFileSync('outputs/collection_browser/'+name+'-mobile.png',Buffer.from(mobile.data,'base64'));report.checks.push('mobile: no horizontal overflow');
  assert(!errors.length,'Browser runtime errors: '+errors.join(', '));report.passed=true;
}catch(error){report.passed=false;report.error=error.message;process.exitCode=1}
finally{writeFileSync('outputs/collection_browser/'+name+'-qa.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));socket.close();await fetch('http://127.0.0.1:9335/json/close/'+target.id)}
