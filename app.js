import {Input,ALL_FORMATS,BlobSource,AudioBufferSink} from "https://cdn.jsdelivr.net/npm/mediabunny@1.42.0/+esm";
const $=id=>document.getElementById(id),source=$("source"),sample=$("sample"),sctx=sample.getContext("2d",{willReadFrequently:true}),canvas=$("renderCanvas"),ctx=canvas.getContext("2d");
let file=null,url=null,duration=0,features=[],clips=[],editIndex=-1,detector=null,detectorTried=false,transcriber=null,loadingTranscriber=null,lastFace={x:0,y:0};
const esc=s=>String(s??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function wait(el,name,timeout=7000){return new Promise((res,rej)=>{let done=false;const end=(err)=>{if(done)return;done=true;clearTimeout(t);el.removeEventListener(name,ok);el.removeEventListener("error",bad);err?rej(err):res()},ok=()=>end(),bad=()=>end(new Error("Video could not be read")),t=setTimeout(()=>end(new Error("Timed out")),timeout);el.addEventListener(name,ok,{once:true});el.addEventListener("error",bad,{once:true})})}
async function meta(el=source){if(el.readyState>=1&&Number.isFinite(el.duration))return;await wait(el,"loadedmetadata")};async function seek(t,el=source){return new Promise(r=>{let x=false;const done=()=>{if(x)return;x=true;el.removeEventListener("seeked",done);r()};el.addEventListener("seeked",done,{once:true});el.currentTime=Math.max(0,Math.min(t,(el.duration||duration)-.03));setTimeout(done,1200)})};function fmt(t){t=Math.max(0,+t||0);return `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,"0")}`}
$("file").onchange=async()=>{file=$("file").files?.[0]||null;clips=[];$("resultsCard").classList.add("hidden");if(url)URL.revokeObjectURL(url);if(!file)return;url=URL.createObjectURL(file);source.src=url;$("fileInfo").textContent=`${file.name} · reading…`;try{await meta();duration=source.duration;$("fileInfo").textContent=`${file.name} · ${(file.size/1048576).toFixed(1)} MB · ${fmt(duration)}`;$("startSec").value="0.00";$("endSec").value=(duration/60).toFixed(2);$("startSec").max=(duration/60).toFixed(2);$("endSec").max=(duration/60).toFixed(2);$("rangeBox").classList.remove("hidden");$("settings").classList.remove("hidden");syncRange()}catch(e){$("fileInfo").textContent=e.message}};
function syncRange(){let s=Math.max(0,(+$("startSec").value||0)*60),e=Math.min(duration,(+$("endSec").value||duration/60)*60);if(e<s+.5)e=Math.min(duration,s+.5);if(s>e-.5)s=Math.max(0,e-.5);$("startSec").value=(s/60).toFixed(2);$("endSec").value=(e/60).toFixed(2);$("startRange").value=duration?s/duration*100:0;$("endRange").value=duration?e/duration*100:100;$("rangeText").textContent=`Analyze ${fmt(s)} – ${fmt(e)} (${fmt(e-s)})`};$("startRange").oninput=()=>{$("startSec").value=(duration*+$("startRange").value/100/60).toFixed(2);syncRange()};$("endRange").oninput=()=>{$("endSec").value=(duration*+$("endRange").value/100/60).toFixed(2);syncRange()};$("startSec").onchange=syncRange;$("endSec").onchange=syncRange;
async function initDetector(){if(!$("faceTracking").checked)return null;if(detector)return detector;if(detectorTried)return null;detectorTried=true;try{const {FilesetResolver,FaceDetector}=await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm");const vision=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");detector=await FaceDetector.createFromOptions(vision,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",delegate:"CPU"},runningMode:"VIDEO",minDetectionConfidence:.5,minSuppressionThreshold:.3});return detector}catch(e){console.warn(e);return null}}
function stats(prev){sctx.drawImage(source,0,0,200,112);const d=sctx.getImageData(0,0,200,112).data,g=new Uint8Array(22400);let motion=0,edge=0,lum=0;for(let i=0,p=0;i<d.length;i+=4,p++){const v=(d[i]*77+d[i+1]*150+d[i+2]*29)>>8;g[p]=v;lum+=v}if(prev)for(let i=0;i<g.length;i+=3)motion+=Math.abs(g[i]-prev[i]);for(let y=1;y<111;y+=2)for(let x=1;x<199;x+=2){const p=y*200+x;edge+=Math.abs(g[p]-g[p-1])+Math.abs(g[p]-g[p-200])}return {g,motion:prev?motion/(Math.ceil(g.length/3)*255):0,edge:edge/(100*56*2*255),lum:lum/g.length/255}}
async function analyze(){const s=+$("startSec").value*60,e=+$("endSec").value*60,span=e-s,n=Math.min(260,Math.max(24,Math.ceil(span))),det=await initDetector();features=[];let prev=null;for(let i=0;i<n;i++){const t=s+(span-.05)*(i/(n-1));await seek(t);const f=stats(prev);prev=f.g;let face=0;if(det&&i%2===0){try{const r=det.detectForVideo(source,Math.round(t*1000));if(r?.detections?.length){const b=r.detections.reduce((a,b)=>b.boundingBox.width*b.boundingBox.height>a.boundingBox.width*a.boundingBox.height?b:a);face=Math.min(1,(b.boundingBox.width*b.boundingBox.height)/(source.videoWidth*source.videoHeight)*8+.3)}}catch{}}const scene=Math.min(1,f.motion*2.4),score=f.motion*.4+scene*.25+f.edge*.2+face*.12+Math.min(1,Math.abs(f.lum-.5)+.2)*.03;features.push({t,score,face});if(i%3===0){$("bar").style.width=`${Math.round(5+78*(i+1)/n)}%`;$("status").textContent=`Analyzing ${i+1}/${n} frames…`}if(i%6===0)await sleep(0)}}
function choose(){const rs=+$("startSec").value*60,re=+$("endSec").value*60,len=+$("clipLength").value,count=+$("clipCount").value;if(re-rs<=len)return[{start:rs,end:re,score:90,words:null,transcript:""}];const wins=[],step=Math.max(2,len/6);for(let st=rs;st<=re-len+.001;st+=step){const en=st+len,a=features.filter(x=>x.t>=st&&x.t<en);if(!a.length)continue;const vals=a.map(x=>x.score),avg=vals.reduce((x,y)=>x+y,0)/vals.length,peak=Math.max(...vals),early=a.filter(x=>x.t<st+len/3),hook=early.length?Math.max(...early.map(x=>x.score)):0,faces=a.reduce((x,y)=>x+y.face,0)/a.length;wins.push({start:st,end:en,raw:avg*.48+peak*.31+hook*.15+faces*.06,words:null,transcript:""})}wins.sort((a,b)=>b.raw-a.raw);const out=[];for(const w of wins){if(out.some(c=>Math.max(0,Math.min(c.end,w.end)-Math.max(c.start,w.start))>len*.32))continue;out.push(w);if(out.length>=count)break}for(const w of wins){if(out.length>=count)break;if(!out.includes(w))out.push(w)}const lo=Math.min(...out.map(x=>x.raw)),hi=Math.max(...out.map(x=>x.raw)),sp=Math.max(.0001,hi-lo);out.forEach(x=>x.score=Math.round(70+29*(x.raw-lo)/sp));return out.sort((a,b)=>a.start-b.start)}
$("generate").onclick=async()=>{if(!file)return;$("generate").disabled=true;$("status").className="small";$("status").textContent="Starting…";$("bar").style.width="2%";try{await analyze();clips=choose();renderCards();$("bar").style.width="100%";$("status").textContent=`Generated ${clips.length} clips.`;$("resultsCard").classList.remove("hidden");$("resultsCard").scrollIntoView({behavior:"smooth"})}catch(e){$("status").className="small error";$("status").textContent=e.message||e}finally{$("generate").disabled=false}};
function renderCards(){$("results").innerHTML=clips.map((c,i)=>`<article class="clip"><div class="row"><div><b>Clip ${i+1}</b><div class="small">${fmt(c.start)} – ${fmt(c.end)} · ${fmt(c.end-c.start)}</div></div><div class="score">${c.score}/100</div></div><video controls playsinline preload="metadata" src="${url}#t=${c.start},${c.end}"></video><div style="margin-top:7px"><span class="badge">${esc($("aspect").selectedOptions[0].text)}</span><span class="badge">${esc($("layout").selectedOptions[0].text)}</span>${c.words?'<span class="badge">Captions ✓</span>':''}</div><div class="row" style="margin-top:10px"><button class="secondary" onclick="window.openEdit(${i})">Edit clip</button><button class="green" onclick="window.quickExport(${i},this)">Create & Download</button></div><div id="cardStatus${i}" class="small"></div></article>`).join("")}
window.openEdit=async i=>{editIndex=i;const c=clips[i];$("editTitle").textContent=`Edit Clip ${i+1}`;$("editStart").value=(c.start/60).toFixed(2);$("editEnd").value=(c.end/60).toFixed(2);$("transcript").textContent=c.transcript||"Captions not prepared yet.";const v=$("editVideo");v.src=url;await meta(v).catch(()=>{});v.currentTime=c.start;$("modal").classList.remove("hidden")};$("close").onclick=()=>{$("editVideo").pause();$("modal").classList.add("hidden")};$("applyTrim").onclick=()=>{if(editIndex<0)return;const c=clips[editIndex];c.start=Math.max(0,Math.min(duration-.2,(+$("editStart").value)*60));c.end=Math.max(c.start+.2,Math.min(duration,(+$("editEnd").value)*60));c.words=null;c.transcript="";c._facePath=null;c._faceKey="";$("transcript").textContent="Trim changed. Prepare captions again if needed.";renderCards()};
async function audio16k(start,end,cb){const input=new Input({formats:ALL_FORMATS,source:new BlobSource(file)});try{const track=await input.getPrimaryAudioTrack();if(!track)throw new Error("No audio track found");const sink=new AudioBufferSink(track),parts=[];let total=0,sr=0;for await(const {buffer,timestamp} of sink.buffers(start,end)){sr=buffer.sampleRate;const mono=new Float32Array(buffer.length);for(let ch=0;ch<buffer.numberOfChannels;ch++){const d=buffer.getChannelData(ch);for(let i=0;i<mono.length;i++)mono[i]+=d[i]/buffer.numberOfChannels}parts.push(mono);total+=mono.length;cb(`Extracting audio… ${Math.round(Math.max(0,Math.min(1,(timestamp-start)/(end-start)))*100)}%`)}if(!total)throw new Error("Audio could not be decoded");const all=new Float32Array(total);let o=0;for(const p of parts){all.set(p,o);o+=p.length}if(sr===16000)return all;const ratio=sr/16000,n=Math.floor(all.length/ratio),out=new Float32Array(n);for(let i=0;i<n;i++){const x=i*ratio,j=Math.floor(x),f=x-j;out[i]=(all[j]||0)*(1-f)+(all[Math.min(all.length-1,j+1)]||0)*f}return out}finally{input.dispose()}}
async function getWhisper(cb){
 if(transcriber)return transcriber;
 if(loadingTranscriber)return await loadingTranscriber;
 loadingTranscriber=(async()=>{
  cb("Starting free Whisper AI model…");
  const {env,pipeline}=await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm");
  env.allowLocalModels=false;
  try{env.useBrowserCache=true}catch{}
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
  if("gpu" in navigator){
   try{return await pipeline("automatic-speech-recognition",model,{dtype:"q4",device:"webgpu",progress_callback})}
   catch(e){console.warn("WebGPU Whisper failed; using browser CPU",e)}
  }
  return await pipeline("automatic-speech-recognition",model,cpuOpts);
 })();
 try{transcriber=await loadingTranscriber;cb("Whisper AI model ready.");return transcriber}
 finally{loadingTranscriber=null}
}
async function captionsFor(i,cb){
 const c=clips[i];
 if(c.words)return c.words;
 if(!$('captions').checked){c.words=[];return[]}

 const audio=await audio16k(c.start,c.end,cb),pipe=await getWhisper(cb);
 cb('AI captions…');

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

async function buildFacePath(c,det,cb){
 if(!det)return null;
 const span=Math.max(.2,c.end-c.start);
 const sampleCount=Math.min(60,Math.max(10,Math.ceil(span/1.25)));
 const step=span/Math.max(1,sampleCount-1);
 const points=[];
 let smooth=null;

 for(let n=0;n<sampleCount;n++){
  const t=Math.min(c.end-.03,c.start+n*step);
  await seek(t);
  let target=null;
  try{
   const r=det.detectForVideo(source,Math.round(t*1000));
   if(r?.detections?.length){
    // Largest face is usually the active foreground speaker in interview/podcast footage.
    const d=r.detections.reduce((a,b)=>
     b.boundingBox.width*b.boundingBox.height>a.boundingBox.width*a.boundingBox.height?b:a
    );
    target={
     x:d.boundingBox.originX+d.boundingBox.width/2,
     y:d.boundingBox.originY+d.boundingBox.height/2
    };
   }
  }catch{}

  if(target){
   if(!smooth)smooth={...target};
   else{
    const jump=Math.abs(target.x-smooth.x)/Math.max(1,source.videoWidth);
    // Fast response on speaker/shot changes, gentle response during normal movement.
    const alpha=jump>.18?.62:.28;
    smooth.x=smooth.x*(1-alpha)+target.x*alpha;
    smooth.y=smooth.y*(1-alpha*.65)+target.y*(alpha*.65);
   }
  }else if(!smooth){
   smooth={x:source.videoWidth/2,y:source.videoHeight/2};
  }
  points.push({t,x:smooth.x,y:smooth.y});

  if(n%5===0)cb(`Preparing face reframe… ${Math.round((n+1)/sampleCount*100)}%`);
  if(n%6===0)await sleep(0);
 }
 return points
}

function faceAt(path,t){
 if(!path?.length)return{x:source.videoWidth/2,y:source.videoHeight/2};
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

async function exportClip(i,cb){
 const c=clips[i];
 cb("Preparing captions + face reframe…");

 const faceEnabled=$("layout").value==="auto"&&$("faceTracking").checked;
 const faceKey=`${c.start.toFixed(3)}:${c.end.toFixed(3)}:${source.videoWidth}x${source.videoHeight}`;

 const captionPromise=($("captions").checked&&!c.words)
  ? captionsFor(i,m=>cb(`Captions: ${m}`))
  : Promise.resolve(c.words||[]);

 const facePromise=faceEnabled
  ? (async()=>{
      if(c._facePath&&c._faceKey===faceKey)return c._facePath;
      const det=await initDetector();
      if(!det)return null;
      const path=await buildFacePath(c,det,m=>cb(`Face reframe: ${m.replace(/^Preparing face reframe…\s*/,"")}`));
      c._facePath=path;c._faceKey=faceKey;
      return path
    })()
  : Promise.resolve(null);

 const [,facePath]=await Promise.all([captionPromise,facePromise]);

 const [W,H]=size();
 canvas.width=W;canvas.height=H;

 await seek(c.start);
 source.muted=false;

 if(!canvas.captureStream||!window.MediaRecorder)
  throw new Error("Local export is not supported by this browser");

 const cs=canvas.captureStream(24);
 let ats=[];
 try{
  const vs=source.captureStream?source.captureStream():(source.mozCaptureStream?source.mozCaptureStream():null);
  if(vs)ats=vs.getAudioTracks()
 }catch{}

 const stream=new MediaStream([...cs.getVideoTracks(),...ats]);
 const mt=mime();
 const rec=new MediaRecorder(stream,mt?{mimeType:mt,videoBitsPerSecond:4200000}:undefined);
 const parts=[];
 rec.ondataavailable=e=>{if(e.data?.size)parts.push(e.data)};
 const stopped=new Promise((r,j)=>{rec.onstop=r;rec.onerror=e=>j(e.error||new Error("Recorder failed"))});

 rec.start(500);
 await source.play();

 await new Promise(done=>{
  const tick=()=>{
   const t=source.currentTime;
   const face=faceAt(facePath,t);

   ctx.fillStyle="#000";
   ctx.fillRect(0,0,W,H);

   if($("layout").value==="fit")drawFit(source);
   else drawCover(source,face.x,face.y);

   drawText(c,t);

   cb(`Rendering… ${Math.round(Math.max(0,Math.min(1,(t-c.start)/(c.end-c.start)))*100)}%`);

   if(t>=c.end||source.ended)return done();
   requestAnimationFrame(tick)
  };
  tick()
 });

 source.pause();
 rec.stop();
 await stopped;

 const blob=new Blob(parts,{type:rec.mimeType||"video/webm"});
 const ext=(blob.type||"").includes("mp4")?"mp4":"webm";
 return{blob,ext}
}
function save(blob,name){const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),30000)}window.quickExport=async(i,b)=>{const s=$(`cardStatus${i}`);b.disabled=true;try{const r=await exportClip(i,m=>s.textContent=m);save(r.blob,`ClipNova-Clip-${i+1}.${r.ext}`);s.className="small ok";s.textContent=`Done · ${(r.blob.size/1048576).toFixed(1)} MB`}catch(e){s.className="small error";s.textContent=e.message||e}finally{b.disabled=false}};$("exportOne").onclick=async()=>{const b=$("exportOne");b.disabled=true;try{const r=await exportClip(editIndex,m=>$("exportStatus").textContent=m);save(r.blob,`ClipNova-Clip-${editIndex+1}.${r.ext}`);$("exportStatus").className="small ok";$("exportStatus").textContent="Done"}catch(e){$("exportStatus").className="small error";$("exportStatus").textContent=e.message||e}finally{b.disabled=false}};$("exportAll").onclick=async()=>{const b=$("exportAll");b.disabled=true;try{for(let i=0;i<clips.length;i++){const s=$(`cardStatus${i}`),r=await exportClip(i,m=>s.textContent=`Batch: ${m}`);save(r.blob,`ClipNova-Clip-${i+1}.${r.ext}`);s.textContent="Exported";await sleep(500)}}finally{b.disabled=false}};
