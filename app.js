import {Input,ALL_FORMATS,BlobSource,AudioBufferSink,VideoSampleSink,Output,Mp4OutputFormat,BufferTarget,CanvasSource,AudioBufferSource} from "https://cdn.jsdelivr.net/npm/mediabunny@1.42.0/+esm";
const $=id=>document.getElementById(id),source=$("source"),sample=$("sample"),sctx=sample.getContext("2d",{willReadFrequently:true}),canvas=$("renderCanvas"),ctx=canvas.getContext("2d");
const speakerSample=document.createElement("canvas"),speakerCtx=speakerSample.getContext("2d",{willReadFrequently:true});
speakerSample.width=320;speakerSample.height=180;
const MOBILE=/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)||navigator.maxTouchPoints>1||matchMedia("(pointer:coarse)").matches;
source.muted=true;source.playsInline=true;source.setAttribute("playsinline","");source.setAttribute("webkit-playsinline","");
let file=null,url=null,sourceUrl=null,duration=0,features=[],clips=[],editIndex=-1,detector=null,detectorTried=false,transcriber=null,loadingTranscriber=null,lastFace={x:0,y:0};
const faceToggle=$("faceTracking")?.closest(".toggle");
if(faceToggle){
 const title=faceToggle.querySelector("b"),note=faceToggle.querySelector(".small");
 if(title)title.textContent="Multi-face active speaker tracking";
 if(note)note.textContent="Recognizes multiple faces and follows the person who is speaking."
}
if($("layout")?.options?.[0])$("layout").options[0].textContent="Auto Reframe · active speaker";
const esc=s=>String(s??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function wait(el,name,timeout=MOBILE?20000:7000){return new Promise((res,rej)=>{let done=false;const end=(err)=>{if(done)return;done=true;clearTimeout(t);el.removeEventListener(name,ok);el.removeEventListener("error",bad);err?rej(err):res()},ok=()=>end(),bad=()=>end(new Error("Video could not be read")),t=setTimeout(()=>end(new Error("Timed out")),timeout);el.addEventListener(name,ok,{once:true});el.addEventListener("error",bad,{once:true})})}
async function meta(el=source){if(el.readyState>=1&&Number.isFinite(el.duration))return;await wait(el,"loadedmetadata")};async function seek(t,el=source){return new Promise(r=>{let x=false;const done=()=>{if(x)return;x=true;el.removeEventListener("seeked",done);r()};el.addEventListener("seeked",done,{once:true});el.currentTime=Math.max(0,Math.min(t,(el.duration||duration)-.03));setTimeout(done,1200)})};function fmt(t){t=Math.max(0,+t||0);return `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,"0")}`}
$("file").onchange=async()=>{file=$("file").files?.[0]||null;clips=[];$("resultsCard").classList.add("hidden");if(sourceUrl&&sourceUrl!==url)URL.revokeObjectURL(sourceUrl);if(url)URL.revokeObjectURL(url);url=null;sourceUrl=null;if(!file)return;url=URL.createObjectURL(file);sourceUrl=url;source.src=sourceUrl;$("fileInfo").textContent=`${file.name} · reading…`;try{await meta();duration=source.duration;$("fileInfo").textContent=`${file.name} · ${(file.size/1048576).toFixed(1)} MB · ${fmt(duration)}`;$("startSec").value="0.00";$("endSec").value=(duration/60).toFixed(2);$("startSec").max=(duration/60).toFixed(2);$("endSec").max=(duration/60).toFixed(2);$("rangeBox").classList.remove("hidden");$("settings").classList.remove("hidden");syncRange()}catch(e){$("fileInfo").textContent=e.message}};
function syncRange(){let s=Math.max(0,(+$("startSec").value||0)*60),e=Math.min(duration,(+$("endSec").value||duration/60)*60);if(e<s+.5)e=Math.min(duration,s+.5);if(s>e-.5)s=Math.max(0,e-.5);$("startSec").value=(s/60).toFixed(2);$("endSec").value=(e/60).toFixed(2);$("startRange").value=duration?s/duration*100:0;$("endRange").value=duration?e/duration*100:100;$("rangeText").textContent=`Analyze ${fmt(s)} – ${fmt(e)} (${fmt(e-s)})`};$("startRange").oninput=()=>{$("startSec").value=(duration*+$("startRange").value/100/60).toFixed(2);syncRange()};$("endRange").oninput=()=>{$("endSec").value=(duration*+$("endRange").value/100/60).toFixed(2);syncRange()};$("startSec").onchange=syncRange;$("endSec").onchange=syncRange;
async function initDetector(){if(!$("faceTracking").checked)return null;if(detector)return detector;if(detectorTried)return null;detectorTried=true;try{const {FilesetResolver,FaceDetector}=await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm");const vision=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");detector=await FaceDetector.createFromOptions(vision,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",delegate:"CPU"},runningMode:"IMAGE",minDetectionConfidence:.38,minSuppressionThreshold:.25});return detector}catch(e){console.warn(e);return null}}
function statsPixels(prev){const d=sctx.getImageData(0,0,200,112).data,g=new Uint8Array(22400);let motion=0,edge=0,lum=0;for(let i=0,p=0;i<d.length;i+=4,p++){const v=(d[i]*77+d[i+1]*150+d[i+2]*29)>>8;g[p]=v;lum+=v}if(prev)for(let i=0;i<g.length;i+=3)motion+=Math.abs(g[i]-prev[i]);for(let y=1;y<111;y+=2)for(let x=1;x<199;x+=2){const p=y*200+x;edge+=Math.abs(g[p]-g[p-1])+Math.abs(g[p]-g[p-200])}return {g,motion:prev?motion/(Math.ceil(g.length/3)*255):0,edge:edge/(100*56*2*255),lum:lum/g.length/255}}
function stats(prev){sctx.drawImage(source,0,0,200,112);return statsPixels(prev)}
async function analyzeDecoded(){
 const s=(+$("startSec").value)*60,e=(+$("endSec").value)*60,span=e-s,n=Math.min(60,Math.max(24,Math.ceil(span/90))),det=await initDetector();
 const input=new Input({formats:ALL_FORMATS,source:new BlobSource(file)});
 features=[];let prev=null,i=0;
 try{
  const track=await input.getPrimaryVideoTrack();
  if(!track)throw new Error("No video track found");
  const sink=new VideoSampleSink(track,{hardwareAcceleration:"prefer-hardware"});
  const timestamps=Array.from({length:n},(_,k)=>s+(span-.05)*(k/(n-1)));
  for await(const frameSample of sink.samplesAtTimestamps(timestamps)){
   const t=timestamps[i];
   if(!frameSample){features.push({t,score:0,face:0});i++;continue}
   try{
    frameSample.draw(sctx,0,0,200,112);
    const f=statsPixels(prev);prev=f.g;
    let face=0;
    if(det&&i%2===0){try{const r=det.detect(sample);if(r?.detections?.length){const b=r.detections.reduce((a,b)=>b.boundingBox.width*b.boundingBox.height>a.boundingBox.width*a.boundingBox.height?b:a);face=Math.min(1,(b.boundingBox.width*b.boundingBox.height)/(200*112)*8+.3)}}catch{}}
    const scene=Math.min(1,f.motion*2.4),score=f.motion*.4+scene*.25+f.edge*.2+face*.12+Math.min(1,Math.abs(f.lum-.5)+.2)*.03;
    features.push({t,score,face})
   }finally{frameSample.close()}
   i++;$("bar").style.width=`${Math.round(5+78*i/n)}%`;$("status").textContent=`Analyzing ${i}/${n} frames…`;
   if(i%4===0)await sleep(0)
  }
 }finally{input.dispose()}
}
async function analyze(){if(MOBILE)return analyzeDecoded();const s=(+$("startSec").value)*60,e=(+$("endSec").value)*60,span=e-s,n=Math.min(260,Math.max(24,Math.ceil(span))),det=await initDetector();features=[];let prev=null;for(let i=0;i<n;i++){const t=s+(span-.05)*(i/(n-1));await seek(t);const f=stats(prev);prev=f.g;let face=0;if(det&&i%2===0){try{const r=det.detect(source);if(r?.detections?.length){const b=r.detections.reduce((a,b)=>b.boundingBox.width*b.boundingBox.height>a.boundingBox.width*a.boundingBox.height?b:a);face=Math.min(1,(b.boundingBox.width*b.boundingBox.height)/(source.videoWidth*source.videoHeight)*8+.3)}}catch{}}const scene=Math.min(1,f.motion*2.4),score=f.motion*.4+scene*.25+f.edge*.2+face*.12+Math.min(1,Math.abs(f.lum-.5)+.2)*.03;features.push({t,score,face});if(i%3===0){$("bar").style.width=`${Math.round(5+78*(i+1)/n)}%`;$("status").textContent=`Analyzing ${i+1}/${n} frames…`}if(i%6===0)await sleep(0)}}
function choose(){const rs=(+$("startSec").value)*60,re=(+$("endSec").value)*60,len=+$("clipLength").value,count=+$("clipCount").value;if(re-rs<=len)return[{start:rs,end:re,score:90,words:null,transcript:""}];const wins=[],step=Math.max(2,len/6);for(let st=rs;st<=re-len+.001;st+=step){const en=st+len,a=features.filter(x=>x.t>=st&&x.t<en);if(!a.length)continue;const vals=a.map(x=>x.score),avg=vals.reduce((x,y)=>x+y,0)/vals.length,peak=Math.max(...vals),early=a.filter(x=>x.t<st+len/3),hook=early.length?Math.max(...early.map(x=>x.score)):0,faces=a.reduce((x,y)=>x+y.face,0)/a.length;wins.push({start:st,end:en,raw:avg*.48+peak*.31+hook*.15+faces*.06,words:null,transcript:""})}wins.sort((a,b)=>b.raw-a.raw);const out=[];for(const w of wins){if(out.some(c=>Math.max(0,Math.min(c.end,w.end)-Math.max(c.start,w.start))>len*.32))continue;out.push(w);if(out.length>=count)break}for(const w of wins){if(out.length>=count)break;if(!out.includes(w))out.push(w)}const lo=Math.min(...out.map(x=>x.raw)),hi=Math.max(...out.map(x=>x.raw)),sp=Math.max(.0001,hi-lo);out.forEach(x=>x.score=Math.round(70+29*(x.raw-lo)/sp));return out.sort((a,b)=>a.start-b.start)}
$("generate").onclick=async()=>{if(!file)return;$("generate").disabled=true;$("status").className="small";$("status").textContent="Starting…";$("bar").style.width="2%";try{await analyze();clips=choose();renderCards();$("bar").style.width="100%";$("status").textContent=`Generated ${clips.length} clips.`;$("resultsCard").classList.remove("hidden");$("resultsCard").scrollIntoView({behavior:"smooth"})}catch(e){$("status").className="small error";$("status").textContent=e.message||e}finally{$("generate").disabled=false}};
function pausePreviews(){document.querySelectorAll(".clipPreview").forEach(v=>{try{v.pause()}catch{}})}
function suspendPreviews(){
 if(!MOBILE)return;
 document.querySelectorAll(".clipPreview").forEach(v=>{try{v.pause();if(v.src){v.dataset.previewSrc=v.src;v.removeAttribute("src");v.load()}}catch{}})
}
function restorePreviews(){
 if(!MOBILE)return;
 document.querySelectorAll(".clipPreview").forEach(v=>{if(v.dataset.previewSrc&&!v.src){v.src=v.dataset.previewSrc;delete v.dataset.previewSrc;v.load()}})
}
function renderCards(){$("results").innerHTML=clips.map((c,i)=>`<article class="clip"><div class="row"><div><b>Clip ${i+1}</b><div class="small">${fmt(c.start)} – ${fmt(c.end)} · ${(c.end-c.start).toFixed(1)}s</div></div><div class="score">${c.score}/100</div></div><video class="clipPreview" controls playsinline webkit-playsinline preload="metadata" src="${url}#t=${c.start},${c.end}"></video><div style="margin-top:7px"><span class="badge">${esc($("aspect").selectedOptions[0].text)}</span><span class="badge">${esc($("layout").selectedOptions[0].text)}</span>${c.words?.length?'<span class="badge">Captions ✓</span>':''}</div><div class="row" style="margin-top:10px"><button class="secondary" onclick="window.openEdit(${i})">Edit clip</button><button class="green" onclick="window.quickExport(${i},this)">Create & Download</button></div><div id="cardStatus${i}" class="small"></div></article>`).join("")}
window.openEdit=async i=>{pausePreviews();editIndex=i;const c=clips[i];$("editTitle").textContent=`Edit Clip ${i+1}`;$("editStart").value=(c.start/60).toFixed(2);$("editEnd").value=(c.end/60).toFixed(2);$("transcript").textContent=c.transcript||"Captions not prepared yet.";$("exportStatus").className="small";$("exportStatus").textContent="Ready to export.";$("exportBar").style.width="0%";const v=$("editVideo");v.src=url;await meta(v).catch(()=>{});v.currentTime=c.start;$("modal").classList.remove("hidden")};$("close").onclick=()=>{const v=$("editVideo");v.pause();if(MOBILE){v.removeAttribute("src");v.load()}$("modal").classList.add("hidden")};$("applyTrim").onclick=()=>{if(editIndex<0)return;const c=clips[editIndex],s=(+$("editStart").value)*60,e=(+$("editEnd").value)*60;c.start=Math.max(0,Math.min(duration-.2,s));c.end=Math.max(c.start+.2,Math.min(duration,e));$("editStart").value=(c.start/60).toFixed(2);$("editEnd").value=(c.end/60).toFixed(2);c.words=null;c.transcript="";c.captionSkipped=false;$("transcript").textContent="Trim changed. Prepare captions again if needed.";renderCards()};
async function audio16k(start,end,cb){const input=new Input({formats:ALL_FORMATS,source:new BlobSource(file)});try{const track=await input.getPrimaryAudioTrack();if(!track)throw new Error("No audio track found");const sink=new AudioBufferSink(track),parts=[];let total=0,sr=0;for await(const {buffer,timestamp} of sink.buffers(start,end)){sr=buffer.sampleRate;const mono=new Float32Array(buffer.length);for(let ch=0;ch<buffer.numberOfChannels;ch++){const d=buffer.getChannelData(ch);for(let i=0;i<mono.length;i++)mono[i]+=d[i]/buffer.numberOfChannels}parts.push(mono);total+=mono.length;cb(`Extracting audio… ${Math.round(Math.max(0,Math.min(1,(timestamp-start)/(end-start)))*100)}%`)}if(!total)throw new Error("Audio could not be decoded");const all=new Float32Array(total);let o=0;for(const p of parts){all.set(p,o);o+=p.length}if(sr===16000)return all;const ratio=sr/16000,n=Math.floor(all.length/ratio),out=new Float32Array(n);for(let i=0;i<n;i++){const x=i*ratio,j=Math.floor(x),f=x-j;out[i]=(all[j]||0)*(1-f)+(all[Math.min(all.length-1,j+1)]||0)*f}return out}finally{input.dispose()}}
async function getWhisper(cb){
 if(transcriber)return transcriber;
 if(loadingTranscriber)return await loadingTranscriber;
 loadingTranscriber=(async()=>{
  cb("Starting free Whisper AI model…");
  const {env,pipeline}=await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm");
  env.allowLocalModels=false;
  try{env.useBrowserCache=true}catch{}
  if(MOBILE){try{env.backends.onnx.wasm.numThreads=1}catch{}}
  let last=-1;
  const progress_callback=p=>{
   try{
    let pct=null;
    if(Number.isFinite(p?.progress))pct=Math.round(p.progress);
    else if(Number.isFinite(p?.loaded)&&Number.isFinite(p?.total)&&p.total>0)pct=Math.round(p.loaded/p.total*100);
    if(pct!==null){pct=Math.max(0,Math.min(100,pct));if(pct!==last){last=pct;cb(`Downloading Whisper AI model… ${pct}%`)}}
    else if(p?.status==="ready")cb("Whisper AI model ready.");
    else if(p?.status==="initiate")cb(`Starting ${p.file||"AI model file"}…`);
   }catch{}
  };
  const model="onnx-community/whisper-tiny";
  const cpuOpts={dtype:"q4",progress_callback};
  if("gpu" in navigator&&!MOBILE){
   try{return await pipeline("automatic-speech-recognition",model,{dtype:"q4",device:"webgpu",progress_callback})}
   catch(e){console.warn("WebGPU Whisper failed; using browser CPU",e)}
  }
  try{return await pipeline("automatic-speech-recognition",model,cpuOpts)}
  catch(e){
   if(!MOBILE)throw e;
   console.warn("Mobile Whisper download failed; retrying without browser cache",e);
   cb("Mobile browser: retrying AI caption download…");
   try{env.useBrowserCache=false}catch{}
   await sleep(800);
   return await pipeline("automatic-speech-recognition",model,cpuOpts)
  }
 })();
 try{transcriber=await loadingTranscriber;cb("Whisper AI model ready.");return transcriber}
 finally{loadingTranscriber=null}
}
async function captionsFor(i,cb){
 const c=clips[i];
 if(c.words)return c.words;
 if(!$('captions').checked){c.words=[];return[]}

 const audio=await audio16k(c.start,c.end,cb),pipe=await getWhisper(cb);
 cb('Transcribing with Whisper AI…');

 const code=$('captionLanguage')?.value||'en';
 const languageNames={
  en:'english',es:'spanish',de:'german',fr:'french',pt:'portuguese',
  it:'italian',ur:'urdu',hi:'hindi'
 };

 // Use segment timestamps. The lightweight Q4 browser model does not expose
 // cross-attention tensors required by true Whisper word timestamps.
 const opts={
  chunk_length_s:24,
  stride_length_s:4,
  return_timestamps:true,
  task:'transcribe'
 };
 if(code!=='auto'&&languageNames[code])opts.language=languageNames[code];

 const r=await pipe(audio,opts);
 c.transcript=(r.text||'').replace(/\uFFFD/g,'').replace(/\s+/g,' ').trim();

 function cleanWord(value){
  let s=String(value||'').replace(/\uFFFD/g,'').trim();
  if(code==='en')s=s.replace(/[^A-Za-z0-9'’.,!?&%$€£:+\-]/g,'');
  return s;
 }

 const out=[];
 for(const seg of (r.chunks||[])){
  const text=String(seg.text||'').replace(/\uFFFD/g,'').trim();
  if(!text)continue;
  const words=text.split(/\s+/).map(cleanWord).filter(Boolean);
  if(!words.length)continue;
  let a=Number(seg.timestamp?.[0]),b=Number(seg.timestamp?.[1]);
  if(!Number.isFinite(a))a=0;
  if(!Number.isFinite(b)||b<=a)b=a+Math.max(.7,words.length*.32);
  const segStart=c.start+a,segEnd=Math.min(c.end,c.start+b),span=Math.max(.18,segEnd-segStart),step=span/words.length;
  words.forEach((text,n)=>out.push({text,start:segStart+n*step,end:Math.min(segEnd,segStart+(n+1)*step)}));
 }

 if(!out.length&&c.transcript){
  const words=c.transcript.split(/\s+/).map(cleanWord).filter(Boolean),span=Math.max(.2,c.end-c.start),step=span/Math.max(1,words.length);
  words.forEach((text,n)=>out.push({text,start:c.start+n*step,end:Math.min(c.end,c.start+(n+1)*step)}));
 }

 c.words=out.filter(x=>x.text&&x.end>x.start);
 return c.words
}
$("prepareCaptions").onclick=async()=>{if(editIndex<0)return;const b=$("prepareCaptions");b.disabled=true;try{await captionsFor(editIndex,m=>$("transcript").textContent=m);$("transcript").textContent=clips[editIndex].transcript||"No speech detected";renderCards()}catch(e){$("transcript").textContent="Captions failed: "+(e.message||e)}finally{b.disabled=false}};
function overlay(){
 const v=$("editVideo"),c=clips[editIndex];
 if(!c?.words?.length)return $("captionOverlay").innerHTML="";
 const t=v.currentTime;
 let k=c.words.findIndex(w=>t>=w.start&&t<=w.end);
 if(k<0)k=c.words.findIndex(w=>w.start>t);
 if(k<0)k=c.words.length-1;

 // Show only a compact 4-word phrase so captions never run across the whole screen.
 const groupStart=Math.floor(k/4)*4;
 const list=c.words.slice(groupStart,groupStart+4);
 $("captionOverlay").innerHTML=list.map((w,j)=>
  `<span class="${groupStart+j===k?'active':''}">${esc(w.text.trim())}</span>`
 ).join(" ")
}
$("editVideo").addEventListener("timeupdate",overlay);
function size(){const a=$("aspect").value;return a==="landscape"?[1280,720]:a==="square"?[1080,1080]:[720,1280]};function drawCover(v,cx,cy){const W=canvas.width,H=canvas.height,sw=v.videoWidth,sh=v.videoHeight,tar=W/H,src=sw/sh;let sx=0,sy=0,cw=sw,ch=sh;if(src>tar){cw=sh*tar;sx=Math.max(0,Math.min(sw-cw,cx-cw/2))}else{ch=sw/tar;sy=Math.max(0,Math.min(sh-ch,cy-ch/2))}ctx.drawImage(v,sx,sy,cw,ch,0,0,W,H)}function drawFit(v){const W=canvas.width,H=canvas.height,sw=v.videoWidth,sh=v.videoHeight;ctx.save();ctx.filter="blur(28px) brightness(.62)";let s=Math.max(W/sw,H/sh);ctx.drawImage(v,(W-sw*s)/2,(H-sh*s)/2,sw*s,sh*s);ctx.restore();s=Math.min(W/sw,H/sh);ctx.drawImage(v,(W-sw*s)/2,(H-sh*s)/2,sw*s,sh*s)}
function drawSampleCover(frameSample,cx,cy){const W=canvas.width,H=canvas.height,sw=frameSample.displayWidth,sh=frameSample.displayHeight,tar=W/H,src=sw/sh;let sx=0,sy=0,cw=sw,ch=sh;if(src>tar){cw=sh*tar;sx=Math.max(0,Math.min(sw-cw,cx-cw/2))}else{ch=sw/tar;sy=Math.max(0,Math.min(sh-ch,cy-ch/2))}frameSample.draw(ctx,sx,sy,cw,ch,0,0,W,H)}
function drawSampleFit(frameSample){const W=canvas.width,H=canvas.height,sw=frameSample.displayWidth,sh=frameSample.displayHeight;ctx.save();ctx.filter="blur(28px) brightness(.62)";let s=Math.max(W/sw,H/sh);frameSample.draw(ctx,(W-sw*s)/2,(H-sh*s)/2,sw*s,sh*s);ctx.restore();s=Math.min(W/sw,H/sh);frameSample.draw(ctx,(W-sw*s)/2,(H-sh*s)/2,sw*s,sh*s)}
function drawText(c,t){
 if(!$("captions").checked||!c.words?.length)return;
 let k=c.words.findIndex(w=>t>=w.start&&t<=w.end);
 if(k<0)k=c.words.findIndex(w=>w.start>t);
 if(k<0)k=c.words.length-1;

 const groupStart=Math.floor(k/4)*4;
 const list=c.words.slice(groupStart,groupStart+4);
 const W=canvas.width,H=canvas.height,scale=W/720;
 let font=Math.round(+$("captionSize").value*scale);
 const pos=$("captionPos").value;
 const style=$("captionStyle").value;
 const maxWidth=W*.78;
 const gap=10*scale;

 ctx.save();
 ctx.textAlign="center";
 ctx.textBaseline="middle";
 ctx.font=`900 ${font}px Arial, sans-serif`;
 ctx.lineJoin="round";
 ctx.lineWidth=(style==="bold"?10:style==="clean"?5:8)*scale;

 // Reduce font only if even the short phrase is too wide.
 let phraseWidth=list.reduce((sum,w)=>sum+ctx.measureText(w.text.trim()).width,0)+gap*Math.max(0,list.length-1);
 if(phraseWidth>maxWidth){
  font=Math.max(Math.round(34*scale),Math.round(font*maxWidth/phraseWidth));
  ctx.font=`900 ${font}px Arial, sans-serif`;
 }

 // Wrap into up to 2 centered lines.
 const lines=[[]];
 for(const w of list){
  const line=lines[lines.length-1];
  const trial=[...line,w];
  const width=trial.reduce((sum,x)=>sum+ctx.measureText(x.text.trim()).width,0)+gap*Math.max(0,trial.length-1);
  if(width>maxWidth&&line.length&&lines.length<2)lines.push([w]);
  else line.push(w);
 }

 const lineH=font*1.15;
 const centerY=pos==="top"?H*.18:pos==="middle"?H*.52:H*.82;
 let y=centerY-(lines.length-1)*lineH/2;

 for(const line of lines){
  const widths=line.map(w=>ctx.measureText(w.text.trim()).width);
  const total=widths.reduce((a,b)=>a+b,0)+gap*Math.max(0,line.length-1);
  let x=(W-total)/2;
  for(let j=0;j<line.length;j++){
   const w=line[j],txt=w.text.trim(),ww=widths[j];
   const globalIndex=c.words.indexOf(w);
   ctx.strokeStyle="rgba(0,0,0,.94)";
   ctx.strokeText(txt,x+ww/2,y);
   ctx.fillStyle=globalIndex===k?$("activeColor").value:$("textColor").value;
   ctx.fillText(txt,x+ww/2,y);
   x+=ww+gap;
  }
  y+=lineH;
 }
 ctx.restore()
}
function mime(){return ["video/mp4;codecs=avc1.42E01E,mp4a.40.2","video/mp4","video/webm;codecs=vp9,opus","video/webm;codecs=vp8,opus","video/webm"].find(t=>window.MediaRecorder&&MediaRecorder.isTypeSupported(t))||""}

const mediaWidth=media=>media.videoWidth||media.width||1,mediaHeight=media=>media.videoHeight||media.height||1;
function normalizedFaceBox(d,video=source){
 const b=d?.boundingBox;
 if(!b)return null;
 const vw=mediaWidth(video),vh=mediaHeight(video);
 const x=Math.max(0,b.originX),y=Math.max(0,b.originY);
 const w=Math.max(1,Math.min(vw-x,b.width));
 const h=Math.max(1,Math.min(vh-y,b.height));
 const mouth=d.keypoints?.[3];
 return{x,y,w,h,cx:x+w/2,cy:y+h/2,mouthX:mouth?mouth.x*vw:x+w/2,mouthY:mouth?mouth.y*vh:y+h*.74,confidence:d.categories?.[0]?.score??.5}
}

function boxIou(a,b){
 const x1=Math.max(a.x,b.x),y1=Math.max(a.y,b.y);
 const x2=Math.min(a.x+a.w,b.x+b.w),y2=Math.min(a.y+a.h,b.y+b.h);
 const intersection=Math.max(0,x2-x1)*Math.max(0,y2-y1);
 return intersection/Math.max(1,a.w*a.h+b.w*b.h-intersection)
}

function mouthSignature(frame,box){
 // Lower-middle face region contains the lips and jaw. An 8×4 luminance
 // signature is enough to measure speaking motion without a heavy landmark model.
 const cols=8,rows=4,out=new Float32Array(cols*rows);
 const sx=speakerSample.width/Math.max(1,source.videoWidth),sy=speakerSample.height/Math.max(1,source.videoHeight);
 const left=(box.mouthX-box.w*.34)*sx,top=(box.mouthY-box.h*.16)*sy;
 const width=box.w*.68*sx,height=box.h*.34*sy;
 for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
  const x0=Math.max(0,Math.floor(left+width*col/cols));
  const x1=Math.min(speakerSample.width,Math.ceil(left+width*(col+1)/cols));
  const y0=Math.max(0,Math.floor(top+height*row/rows));
  const y1=Math.min(speakerSample.height,Math.ceil(top+height*(row+1)/rows));
  let sum=0,count=0;
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){
   const p=(y*speakerSample.width+x)*4;
   sum+=(frame.data[p]*77+frame.data[p+1]*150+frame.data[p+2]*29)>>8;
   count++
  }
  out[row*cols+col]=count?sum/count:0
 }
 return out
}

function signatureChange(a,b){
 if(!a||!b||a.length!==b.length)return 0;
 let total=0;
 for(let i=0;i<a.length;i++)total+=Math.abs(a[i]-b[i]);
 return Math.min(1,total/(a.length*24))
}

async function buildFacePath(c,det,cb){
 if(!det)return null;
 const span=Math.max(.2,c.end-c.start);
 const sampleCount=Math.min(300,Math.max(18,Math.ceil(span/.35)));
 const step=span/Math.max(1,sampleCount-1);
 const points=[],tracks=new Map();
 let previousVisible=[],nextTrackId=1,activeId=null,challengerId=null,challengerFrames=0,smooth=null,pixelSampling=!MOBILE;

 for(let n=0;n<sampleCount;n++){
  const t=Math.min(c.end-.03,c.start+n*step);
  await seek(t);
  let detections=[];
  try{detections=det.detect(source)?.detections||[]}catch{}

  let frame=null;
  if(pixelSampling){
   try{
    speakerCtx.clearRect(0,0,speakerSample.width,speakerSample.height);
    speakerCtx.drawImage(source,0,0,speakerSample.width,speakerSample.height);
    frame=speakerCtx.getImageData(0,0,speakerSample.width,speakerSample.height)
   }catch(e){
    // Some Samsung Chrome versions incorrectly mark a local video canvas as
    // cross-origin after repeated seeks. Continue with landmark motion instead.
    console.warn("Pixel speaker sampling unavailable; using face landmarks",e);
    pixelSampling=false;speakerSample.width=320;speakerSample.height=180
   }
  }
  const faces=detections.map(d=>normalizedFaceBox(d,source)).filter(Boolean).sort((a,b)=>b.w*b.h-a.w*a.h);
  const usedTracks=new Set(),visible=[];

  for(const box of faces){
   let match=null,bestCost=Infinity;
   for(const tr of previousVisible){
    if(usedTracks.has(tr.id))continue;
    const distance=Math.hypot((box.cx-tr.box.cx)/Math.max(1,source.videoWidth),(box.cy-tr.box.cy)/Math.max(1,source.videoHeight));
    const overlap=boxIou(box,tr.box);
    const sizeCost=Math.abs(Math.log((box.w*box.h)/Math.max(1,tr.box.w*tr.box.h)));
    const cost=distance*2.8+(1-overlap)*.28+sizeCost*.12;
    if((overlap>.03||distance<.17)&&cost<bestCost){match=tr;bestCost=cost}
   }

   const id=match?.id??nextTrackId++;
   const old=tracks.get(id)||{id,activity:0,box:null,signature:null};
   const signature=frame?mouthSignature(frame,box):null;
   const mouth=frame?signatureChange(signature,old.signature):0;
   const movement=old.box?Math.min(1,Math.hypot(box.cx-old.box.cx,box.cy-old.box.cy)/Math.max(20,old.box.w)*.8):0;
   const landmarkMotion=old.box?Math.min(1,Math.hypot(
    (box.mouthX-box.x)/Math.max(1,box.w)-(old.box.mouthX-old.box.x)/Math.max(1,old.box.w),
    (box.mouthY-box.y)/Math.max(1,box.h)-(old.box.mouthY-old.box.y)/Math.max(1,old.box.h)
   )*8):0;
   const mouthScore=frame?Math.min(1,mouth*4.8):landmarkMotion;
   const instantActivity=frame?mouthScore*.82+movement*.18:mouthScore*.68+movement*.32;
   const activity=old.activity*.48+instantActivity*.52;
   const area=Math.min(1,Math.sqrt((box.w*box.h)/Math.max(1,source.videoWidth*source.videoHeight)/.11));
   const center=Math.max(0,1-Math.abs(box.cx/source.videoWidth-.5)*1.35);
   const score=mouthScore*.52+activity*.25+movement*.10+area*.07+center*.025+box.confidence*.035;
   const tr={id,box,signature,activity,mouth:mouthScore,movement,score,lastSeen:n};
   tracks.set(id,tr);usedTracks.add(id);visible.push(tr)
  }

  let switched=false;
  const best=visible.reduce((a,b)=>!a||b.score>a.score?b:a,null);
  const current=visible.find(x=>x.id===activeId)||null;
  if(!current){
   if(best){activeId=best.id;switched=true}
   challengerId=null;challengerFrames=0
  }else if(best&&best.id!==activeId){
   const clearlySpeaking=best.mouth>current.mouth+.08||best.activity>current.activity+.10;
   const clearlyStronger=best.score>current.score+.10;
   if(clearlySpeaking&&clearlyStronger){
    if(challengerId===best.id)challengerFrames++;
    else{challengerId=best.id;challengerFrames=1}
    if(challengerFrames>=2){activeId=best.id;switched=true;challengerId=null;challengerFrames=0}
   }else{challengerId=null;challengerFrames=0}
  }else{challengerId=null;challengerFrames=0}

  const chosen=visible.find(x=>x.id===activeId)||best;
  if(chosen){
   const target={x:chosen.box.cx,y:chosen.box.cy};
   if(!smooth)smooth={...target};
   else{
    const jump=Math.abs(target.x-smooth.x)/Math.max(1,source.videoWidth);
    const alpha=switched?.72:jump>.18?.52:.30;
    smooth.x=smooth.x*(1-alpha)+target.x*alpha;
    smooth.y=smooth.y*(1-alpha*.68)+target.y*(alpha*.68)
   }
  }else if(!smooth)smooth={x:source.videoWidth/2,y:source.videoHeight/2};

  points.push({t,x:smooth.x,y:smooth.y,activeId,faceCount:visible.length});
  previousVisible=visible;
  if(n%5===0)cb(`Preparing face reframe · active speaker… ${Math.round((n+1)/sampleCount*100)}%`);
  if(n%5===0)await sleep(0)
 }
 return points
}

function faceAt(path,t,video=source){
 if(!path?.length)return{x:video.videoWidth/2,y:video.videoHeight/2};
 if(t<=path[0].t)return{x:path[0].x,y:path[0].y};
 if(t>=path[path.length-1].t)return{x:path[path.length-1].x,y:path[path.length-1].y};
 let lo=0,hi=path.length-1;
 while(hi-lo>1){
  const m=(lo+hi)>>1;
  if(path[m].t<=t)lo=m;else hi=m;
 }
 const a=path[lo],b=path[hi],u=(t-a.t)/Math.max(.001,b.t-a.t);
 return{x:a.x+(b.x-a.x)*u,y:a.y+(b.y-a.y)*u}
}

function createLiveSpeakerTracker(video,det){
 const vw=mediaWidth(video),vh=mediaHeight(video);
 let previous=[],nextId=1,activeId=null,challengerId=null,challengerFrames=0;
 let smooth={x:vw/2,y:vh/2};
 return()=>{
  let detections=[];
  try{detections=det.detect(video)?.detections||[]}catch{}
  const boxes=detections.map(d=>normalizedFaceBox(d,video)).filter(Boolean).sort((a,b)=>b.w*b.h-a.w*a.h);
  const used=new Set(),visible=[];
  for(const box of boxes){
   let match=null,bestCost=Infinity;
   for(const old of previous){
    if(used.has(old.id))continue;
    const distance=Math.hypot((box.cx-old.box.cx)/vw,(box.cy-old.box.cy)/vh);
    const overlap=boxIou(box,old.box),cost=distance*2.7+(1-overlap)*.3;
    if((overlap>.02||distance<.2)&&cost<bestCost){match=old;bestCost=cost}
   }
   const id=match?.id??nextId++;
   const movement=match?Math.min(1,Math.hypot(box.cx-match.box.cx,box.cy-match.box.cy)/Math.max(20,match.box.w)):0;
   const mouthMotion=match?Math.min(1,Math.hypot(
    (box.mouthX-box.x)/Math.max(1,box.w)-(match.box.mouthX-match.box.x)/Math.max(1,match.box.w),
    (box.mouthY-box.y)/Math.max(1,box.h)-(match.box.mouthY-match.box.y)/Math.max(1,match.box.h)
   )*9):0;
   const activity=(match?.activity||0)*.45+(mouthMotion*.72+movement*.28)*.55;
   const area=Math.min(1,Math.sqrt((box.w*box.h)/Math.max(1,vw*vh)/.11));
   const center=Math.max(0,1-Math.abs(box.cx/vw-.5)*1.35);
   const score=mouthMotion*.50+activity*.28+movement*.10+area*.07+center*.025+box.confidence*.025;
   const track={id,box,activity,mouth:mouthMotion,score};used.add(id);visible.push(track)
  }
  const best=visible.reduce((a,b)=>!a||b.score>a.score?b:a,null);
  const current=visible.find(x=>x.id===activeId)||null;
  let switched=false;
  if(!current){if(best){activeId=best.id;switched=true}challengerId=null;challengerFrames=0}
  else if(best&&best.id!==activeId&&best.score>current.score+.08){
   if(challengerId===best.id)challengerFrames++;else{challengerId=best.id;challengerFrames=1}
   if(challengerFrames>=2){activeId=best.id;switched=true;challengerId=null;challengerFrames=0}
  }else{challengerId=null;challengerFrames=0}
  const chosen=visible.find(x=>x.id===activeId)||best;
  if(chosen){
   const alpha=switched?.7:.32;
   smooth.x=smooth.x*(1-alpha)+chosen.box.cx*alpha;
   smooth.y=smooth.y*(1-alpha*.68)+chosen.box.cy*(alpha*.68)
  }
  previous=visible;
  return smooth
 }
}

async function prepareMobileSource(t){
 suspendPreviews();
 const editVideo=$("editVideo");
 if(MOBILE&&editVideo){try{editVideo.pause();editVideo.removeAttribute("src");editVideo.load()}catch{}}
 source.pause();source.muted=true;source.playsInline=true;
 if(source.error||source.readyState<1){
  if(sourceUrl&&sourceUrl!==url)URL.revokeObjectURL(sourceUrl);
  source.removeAttribute("src");source.load();await sleep(80);
  sourceUrl=URL.createObjectURL(file);source.src=sourceUrl;source.load();await meta(source)
 }
 await seek(t)
}

async function exportClipDecoded(i,cb){
 const c=clips[i],[W,H]=size();
 canvas.width=W;canvas.height=H;
 if(!window.VideoEncoder)throw new Error("Smooth MP4 export needs an updated Chrome or Edge browser");
 suspendPreviews();source.pause();
 const editVideo=$("editVideo");
 if(editVideo){try{editVideo.pause();editVideo.removeAttribute("src");editVideo.load()}catch{}}

 cb("Opening video frames directly…");
 const mediaInput=new Input({formats:ALL_FORMATS,source:new BlobSource(file)});
 const target=new BufferTarget(),output=new Output({format:new Mp4OutputFormat(),target});
 const videoOut=new CanvasSource(canvas,{codec:"avc",bitrate:4200000,hardwareAcceleration:"prefer-hardware",keyFrameInterval:2});
 output.addVideoTrack(videoOut,{frameRate:30});
 let audioOut=null,started=false,videoClosed=false,audioClosed=false;
 try{
  const videoTrack=await mediaInput.getPrimaryVideoTrack();
  if(!videoTrack)throw new Error("No video track found");
  const inputAudioTrack=await mediaInput.getPrimaryAudioTrack();
  if(inputAudioTrack){audioOut=new AudioBufferSource({codec:"aac",bitrate:160000});output.addAudioTrack(audioOut)}
  const videoSink=new VideoSampleSink(videoTrack,{hardwareAcceleration:"prefer-hardware"});
  const det=$("layout").value==="auto"?await initDetector():null;
  await output.start();started=true;

  cb("Rendering directly… 0%");
  const fps=30,frameDuration=1/fps,total=Math.max(frameDuration,c.end-c.start);
  let frameNumber=0,nextFrameTime=0,lastDetection=-Infinity,liveTracker=null,liveFace=null,decodedFrames=0;
  for await(const frameSample of videoSink.samples(c.start,c.end)){
   try{
    const t=Math.max(c.start,Math.min(c.end,frameSample.timestamp)),relative=t-c.start;
    if(det&&t-lastDetection>=.45){
     if(!liveTracker){
      speakerSample.width=320;speakerSample.height=Math.max(1,Math.round(320*frameSample.displayHeight/frameSample.displayWidth));
      liveTracker=createLiveSpeakerTracker(speakerSample,det);
      liveFace={x:speakerSample.width/2,y:speakerSample.height/2}
     }
     speakerCtx.clearRect(0,0,speakerSample.width,speakerSample.height);
     frameSample.draw(speakerCtx,0,0,speakerSample.width,speakerSample.height);
     liveFace=liveTracker();lastDetection=t
    }
    const face=liveFace?{x:liveFace.x/speakerSample.width*frameSample.displayWidth,y:liveFace.y/speakerSample.height*frameSample.displayHeight}:{x:frameSample.displayWidth/2,y:frameSample.displayHeight/2};
    ctx.fillStyle="#000";ctx.fillRect(0,0,W,H);
    if($("layout").value==="fit")drawSampleFit(frameSample);else drawSampleCover(frameSample,face.x,face.y);
    drawText(c,t);
    while(nextFrameTime<=relative+frameDuration/2&&nextFrameTime<total){
     await videoOut.add(nextFrameTime,Math.min(frameDuration,total-nextFrameTime),{keyFrame:frameNumber%(fps*2)===0});
     frameNumber++;nextFrameTime=frameNumber/fps
    }
    decodedFrames++;
    if(decodedFrames%8===0){cb(`Rendering directly… ${Math.round(Math.max(0,Math.min(1,relative/total))*100)}%`);await sleep(0)}
   }finally{frameSample.close()}
  }
  if(!decodedFrames)throw new Error("The selected clip has no readable video frames");
  while(nextFrameTime<total){await videoOut.add(nextFrameTime,Math.min(frameDuration,total-nextFrameTime),{keyFrame:frameNumber%(fps*2)===0});frameNumber++;nextFrameTime=frameNumber/fps}
  videoOut.close();videoClosed=true;

  if(audioOut&&inputAudioTrack){
   cb("Adding audio…");
   try{
    const audioSink=new AudioBufferSink(inputAudioTrack);
    for await(const {buffer,timestamp} of audioSink.buffers(c.start,c.end)){
     await audioOut.add(buffer);
     if(Math.round(timestamp)%5===0)await sleep(0)
    }
   }catch(e){console.warn("Mobile audio decode unavailable",e)}
   audioOut.close();audioClosed=true
  }
  cb("Finalizing smooth MP4…");
  await output.finalize();
  return{blob:new Blob([target.buffer],{type:"video/mp4"}),ext:"mp4"}
 }catch(e){
  if(!videoClosed){try{videoOut.close()}catch{}}
  if(audioOut&&!audioClosed){try{audioOut.close()}catch{}}
  if(started)await output.cancel().catch(()=>{});
  throw e
 }finally{mediaInput.dispose()}
}

async function exportClip(i,cb){
 const c=clips[i];

 // Export has 3 clear visible phases:
 // 1) Whisper transcription, 2) face reframe preparation, 3) rendering.
 if($("captions").checked&&!c.words){
  cb("Transcribing with Whisper AI…");
  try{await captionsFor(i,m=>cb(m||"Transcribing with Whisper AI…"));c.captionSkipped=false}
  catch(e){
   if(!MOBILE)throw e;
   console.warn("Mobile caption model unavailable; continuing without new captions",e);
   c.words=[];c.captionSkipped=true;c.transcript="This phone could not download the AI caption model. The clip was exported without new captions.";
   cb("Caption download unavailable · continuing export…")
  }
 }

 if(MOBILE)return exportClipDecoded(i,cb);

 if(MOBILE){
  suspendPreviews();
  const editVideo=$("editVideo");
  if(editVideo){try{editVideo.pause();editVideo.removeAttribute("src");editVideo.load()}catch{}}
 }else await prepareMobileSource(c.start);

 const [W,H]=size();
 canvas.width=W;canvas.height=H;

 cb("Preparing face reframe…");
 const det=$("layout").value==="auto"?await initDetector():null;
 const facePath=det&&!MOBILE?await buildFacePath(c,det,m=>cb(m.startsWith("Preparing face reframe")?m:"Preparing face reframe…")):null;
 if(det&&MOBILE)cb("Active speaker tracking ready · rendering in one pass…");

 cb("Rendering… 0%");

 if(!window.VideoEncoder)
  throw new Error("Smooth MP4 export needs an updated Chrome or Edge browser");

 const playback=source;
 playback.pause();playback.muted=true;
 await seek(c.start,playback);

 // Mediabunny writes explicit, monotonic 30 FPS timestamps. This avoids the
 // duplicate MP4 timestamps produced by Chrome's MediaRecorder implementation.
 const target=new BufferTarget();
 const output=new Output({format:new Mp4OutputFormat(),target});
 const videoOut=new CanvasSource(canvas,{
  codec:"avc",
  bitrate:5500000,
  hardwareAcceleration:"prefer-hardware",
  keyFrameInterval:2
 });
 output.addVideoTrack(videoOut,{frameRate:30});

 let audioOut=null,audioInput=null,inputAudioTrack=null;
 try{
  audioInput=new Input({formats:ALL_FORMATS,source:new BlobSource(file)});
  inputAudioTrack=await audioInput.getPrimaryAudioTrack();
  if(inputAudioTrack){
   audioOut=new AudioBufferSource({
    codec:"aac",
    bitrate:192000
   });
   output.addAudioTrack(audioOut)
  }
 }catch(e){
  console.warn("Audio extraction unavailable",e);
  if(audioInput){audioInput.dispose();audioInput=null}
 }

 await output.start();

 // Decode only this clip's audio. AudioBufferSource appends the first buffer at
 // exactly 0 seconds, preventing tiny negative timestamps from live capture.
 const audioJob=audioOut&&inputAudioTrack?(async()=>{
  const sink=new AudioBufferSink(inputAudioTrack);
  for await(const {buffer} of sink.buffers(c.start,c.end))await audioOut.add(buffer);
  audioOut.close()
 })():Promise.resolve();

 try{await playback.play()}
 catch(e){
  throw new Error(MOBILE?"Samsung could not start the original video. Select the video again and retry.":(e.message||e))
 }

 const fps=30,frameDuration=1/fps,total=Math.max(frameDuration,c.end-c.start);
 const liveTracker=det&&MOBILE?createLiveSpeakerTracker(playback,det):null;
 let liveFace={x:playback.videoWidth/2,y:playback.videoHeight/2},lastDetection=-Infinity;
 let frameNumber=0,nextFrameTime=0,finished=false,renderError=null,lastVideoTime=playback.currentTime,lastAdvanceAt=performance.now();
 try{
  await new Promise((resolve,reject)=>{
   const tick=async()=>{
    if(finished)return;
    try{
     const t=Math.min(c.end,playback.currentTime);
     const relative=Math.max(0,t-c.start);
     if(t>lastVideoTime+.002){lastVideoTime=t;lastAdvanceAt=performance.now()}
     else if(performance.now()-lastAdvanceAt>8000)throw new Error("Samsung stopped video playback. Please keep this page open while exporting.");

     if(liveTracker&&t-lastDetection>=.45){liveFace=liveTracker();lastDetection=t}

     // Add every required output frame once, using exact sequential timestamps.
     while(nextFrameTime<=relative+frameDuration/2&&nextFrameTime<total){
      const face=liveTracker?liveFace:faceAt(facePath,t,playback);
      ctx.fillStyle="#000";
      ctx.fillRect(0,0,W,H);
      if($("layout").value==="fit")drawFit(playback);
      else drawCover(playback,face.x,face.y);
      drawText(c,t);

      await videoOut.add(nextFrameTime,Math.min(frameDuration,total-nextFrameTime),{
       keyFrame:frameNumber%(fps*2)===0
      });
      frameNumber++;
      nextFrameTime=frameNumber/fps
     }

     cb(`Rendering… ${Math.round(Math.max(0,Math.min(1,relative/total))*100)}%`);
     if(t>=c.end||playback.ended){finished=true;resolve();return}
     requestAnimationFrame(tick)
    }catch(e){finished=true;reject(e)}
   };
   requestAnimationFrame(tick)
  });
 }catch(e){renderError=e}
 finally{
  playback.pause();
  videoOut.close()
 }

 try{await audioJob}
 finally{if(audioInput)audioInput.dispose()}
 if(renderError){await output.cancel().catch(()=>{});throw renderError}

 cb("Finalizing smooth MP4…");
 await output.finalize();
 const blob=new Blob([target.buffer],{type:"video/mp4"});
 return{blob,ext:"mp4"}
}
function save(blob,name){const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),30000)}
function exportUi(message){
 const s=$("exportStatus"),bar=$("exportBar");
 s.className="small";
 s.textContent=message;
 let pct=0;
 if(message.startsWith("Transcribing with Whisper AI"))pct=28;
 else if(message.startsWith("Preparing face reframe")){
  const m=message.match(/(\d+)%/);
  pct=36+(m?Math.round(+m[1]*.24):4);
 }else if(message.startsWith("Rendering")){
  const m=message.match(/(\d+)%/);
  pct=60+(m?Math.round(+m[1]*.4):0);
 }else if(message.startsWith("Finalizing smooth MP4"))pct=99;
 bar.style.width=`${Math.min(100,pct)}%`;
}
window.quickExport=async(i,b)=>{const s=$(`cardStatus${i}`);b.disabled=true;try{const r=await exportClip(i,m=>s.textContent=m);save(r.blob,`ClipNova-Clip-${i+1}.${r.ext}`);s.className="small ok";s.textContent=clips[i].captionSkipped?`Done · ${(r.blob.size/1048576).toFixed(1)} MB · without new captions`:`Done · ${(r.blob.size/1048576).toFixed(1)} MB`}catch(e){s.className="small error";s.textContent=e.message||e}finally{restorePreviews();b.disabled=false}};
$("exportOne").onclick=async()=>{const b=$("exportOne");b.disabled=true;const old=b.textContent;b.textContent="Exporting…";$("exportBar").style.width="0%";try{const r=await exportClip(editIndex,exportUi);save(r.blob,`ClipNova-Clip-${editIndex+1}.${r.ext}`);$("exportBar").style.width="100%";$("exportStatus").className="small ok";$("exportStatus").textContent=clips[editIndex].captionSkipped?"Done · without new captions":"Done";}catch(e){$("exportStatus").className="small error";$("exportStatus").textContent=e.message||e}finally{restorePreviews();b.disabled=false;b.textContent=old}};
$("exportAll").onclick=async()=>{const b=$("exportAll");b.disabled=true;try{for(let i=0;i<clips.length;i++){const s=$(`cardStatus${i}`),r=await exportClip(i,m=>s.textContent=`Batch: ${m}`);save(r.blob,`ClipNova-Clip-${i+1}.${r.ext}`);s.textContent=clips[i].captionSkipped?"Exported · without new captions":"Exported";await sleep(500)}}finally{restorePreviews();b.disabled=false}};
