const $ = id => document.getElementById(id);
const catalog = [
 {id:'onnx-community/Qwen2.5-0.5B-Instruct',name:'Qwen2.5 · 0.5B',modes:['text'],dtype:'q4',note:'Instruction model · Q4 · multilingual text decisions.'},
 {id:'onnx-community/Qwen3-0.6B-ONNX',name:'Qwen3 · 0.6B',modes:['text'],dtype:'q4',note:'Instruction model · Q4 · allow extra memory and load time.'},
 {id:'Qwen/Qwen3-0.6B-GGUF',name:'Qwen3 · 0.6B · GGUF',modes:['text'],backend:'llamacpp',modelUrl:'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/1208e45d782fe18602c5eaf10e5758d5b0f24c03/Qwen3-0.6B-Q4_K_M.gguf',note:'Q4_K_M · 484 MB · llama.cpp WASM. Model files load directly from Hugging Face.'},
 {id:'Xenova/clip-vit-base-patch32',name:'CLIP · ViT-B/32',modes:['image','video'],dtype:'q8',note:'Image–text similarity · Q8 · video uses four sampled frames, not motion reasoning.'},
 {id:'Xenova/clip-vit-base-patch16',name:'CLIP · ViT-B/16',modes:['image','video'],dtype:'q8',note:'Image–text similarity · Q8 · finer image patches, more compute.'},
 {id:'HuggingFaceTB/SmolVLM-256M-Instruct',name:'SmolVLM · 256M',modes:['image'],dtype:'q4',family:'vlm',note:'Vision-language model · Q4 decoder / FP32 vision · candidate-token scoring.'},
 {id:'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',name:'SmolVLM2 · 256M',modes:['image','video'],dtype:'q4',family:'vlm',note:'Vision-language model · Q4 decoder / FP32 vision · video scores four frames independently.'},
 {id:'Xenova/clap-htsat-unfused',name:'CLAP · HTSAT',modes:['audio'],dtype:'q8',note:'Audio–text similarity · Q8 · classify sounds, not transcribe speech.'},
];
const presets = {
 text:[['Support inbox','My package arrived broken. Please send a replacement.','billing\nshipping\ntechnical\nother'],['Sentiment','The food was delicious and the service was excellent.','positive\nnegative\nneutral'],['Language','¿Puedes ayudarme con mi pedido?','English\nSpanish\nFrench\nGerman'],['Urgency','Production is down. All customers are getting errors.','urgent\nnormal\nlow']],
 image:[['Animal recognition','','a tiger\na horse\na dog\na cat'],['Scene classification','','a beach\na city street\na forest\na kitchen']],
 audio:[['Sound recognition','','a dog barking\na vacuum cleaner\nmusic\nrunning water'],['Sound environment','','people talking\ntraffic\nbirds singing\nrain']],
 video:[['Scene across frames','','a person indoors\na street\na forest\na beach'],['Objects across frames','','an animal\na vehicle\na person\nfood']],
};
let mode='text', worker=null, ready=false, busy=false, lifecycle=0, nextId=0, pending=new Map(), result=null, runs=0, file=null, previewURL=null, started=0, timer=null;
const model = () => catalog.find(m => m.id === $('demo-model').value);
const status = (message,kind='') => { $('demo-status').textContent=message; $('demo-status').dataset.kind=kind; };
function controls() {
 $('demo-load').disabled=busy||ready; $('demo-run').disabled=busy||!ready;
 $('demo-unload').disabled=busy||!worker; $('demo-model').disabled=busy||ready; $('demo-device').disabled=busy||ready;$('demo-cyclic').disabled=busy||ready;
 $('demo-runtime').disabled=busy||ready;
 $('demo-cancel').hidden=!busy; $('demo-preset').disabled=busy; $('demo-file').disabled=busy; $('example-media').disabled=busy;
 $('demo-state').disabled=busy; $('demo-options').disabled=busy;
 document.querySelectorAll('[data-mode]').forEach(b=>b.disabled=busy||ready);
 $('demo-reset').disabled=busy;
}
function stop() {
 worker?.terminate();worker=null;ready=false;busy=false;clearInterval(timer);
 for(const {reject} of pending.values())reject(new Error('Cancelled.'));pending.clear();
 controls();
}
function call(op,args) { return new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});worker.postMessage({id,op,args});}); }
function progress(p) {
 if(p.status==='progress') {
  $('download-progress').value=p.progress??0;
  $('download-file').textContent=p.file||'Model files';
  $('download-percent').textContent=Number.isFinite(p.total)&&p.total>0?`${((p.loaded||0)/1e6).toFixed(1)} / ${(p.total/1e6).toFixed(1)} MB · ${Math.round(p.progress||0)}%`: `${Math.round(p.progress||0)}%`;
 } else if(p.status==='done') { $('download-file').textContent=`${p.file||'File'} ready`; }
 else if(p.status==='initiate') { $('download-file').textContent=p.file||'Fetching files…';$('download-progress').removeAttribute('value');$('download-percent').textContent=''; }
}
function clearResult(){result=null;$('demo-result').hidden=true;$('result-empty').hidden=false;$('run-time').textContent='—';}
function resetMedia(){file=null;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=null;$('media-preview').replaceChildren();$('demo-file').value='';}
function preset(){const p=presets[mode][Number($('demo-preset').value)];$('demo-state').value=p[1];$('demo-options').value=p[2];if(model()?.family==='vlm')$('demo-state').value='Which description best matches the image?';clearResult();}
function configure() {
 if(mode!=='text'&&$('demo-runtime').value==='llamacpp')$('demo-runtime').value='transformers';
 $('demo-runtime').querySelector('option[value="llamacpp"]').disabled=mode!=='text';
 const backend=$('demo-runtime').value;
 $('demo-model').replaceChildren(...catalog.filter(m=>m.modes.includes(mode)&&(backend==='llamacpp'?m.backend==='llamacpp':m.backend!=='llamacpp')).map(m=>new Option(m.name,m.id)));
 $('demo-preset').replaceChildren(...presets[mode].map((p,i)=>new Option(p[0],i)));
 const text=mode==='text';$('demo-cyclic').closest('label').hidden=!text;$('demo-state').hidden=!text;$('state-label').hidden=!text;$('media-panel').hidden=text;
 $('demo-file').accept=mode+'/*';$('upload-title').textContent=`Choose ${mode==='image'?'an image':mode==='audio'?'an audio clip':'a video'}`;
 $('example-media').hidden=mode==='video';$('example-media').textContent=mode==='audio'?'Use dog barking sample ↗':'Use tiger photo ↗';
 $('mode-note').textContent=text?(backend==='llamacpp'?'llama.cpp scores a constrained answer token in one inference pass per decision. CPU runs in WASM; GPU uses WebGPU when selected.':'The model scores the candidate answers without generating a response.'):mode==='video'?'Four evenly spaced frames are averaged. Audio and motion are not analyzed. Maximum clip length: 60 seconds.':mode==='audio'?'Audio is mixed to mono at 48 kHz. Maximum length: 10 seconds.':'Compare an image with the descriptions above. This model does not follow arbitrary instructions.';
 resetMedia();preset();describe();controls();
}
function describe(){
 $('model-description').textContent=model().note;$('model-link').href='https://huggingface.co/'+model().id;
 const vlm=model().family==='vlm';$('demo-state').hidden=mode!=='text'&&!vlm;$('state-label').hidden=mode!=='text'&&!vlm;
 $('state-label').textContent=mode==='text'?'Message':'Question';
 if(vlm&&!$('demo-state').value)$('demo-state').value='Which description best matches the image?';
 if(mode==='image')$('mode-note').textContent=vlm?'The vision-language model scores letter codes using its own tokenizer. Small models can make mistakes.':'Compare an image with the descriptions above. This model does not follow arbitrary instructions.';
}
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-selected',String(x===b)));configure();status('Choose a model to begin.');}));
$('demo-model').addEventListener('change',describe);$('demo-preset').addEventListener('change',preset);
$('demo-runtime').addEventListener('change',()=>{configure();status('Choose a model to begin.');});
$('demo-load').addEventListener('click',async()=>{
 if($('demo-device').value==='webgpu'&&!navigator.gpu){status('WebGPU is unavailable. Select CPU to continue.','error');return;}
 const turn=++lifecycle;busy=true;controls();$('download-panel').hidden=false;$('download-progress').removeAttribute('value');$('download-file').textContent='Preparing runtime…';$('download-percent').textContent='';started=performance.now();
 timer=setInterval(()=>{$('load-time').textContent=((performance.now()-started)/1000).toFixed(1)+' s';},100);
 status('Downloading and initializing. Files already cached by your browser are reused.');
 const workerURL=new URL('./demo-worker.js',import.meta.url);workerURL.search=new URL(import.meta.url).search;
 worker=new Worker(workerURL,{type:'module'});
 worker.onmessage=({data})=>{if(data.progress){progress(data.progress);return;}const p=pending.get(data.id);if(!p)return;pending.delete(data.id);data.error?p.reject(new Error(data.error)):p.resolve(data.value);};
 worker.onerror=e=>{for(const p of pending.values())p.reject(new Error(e.message||'Worker failed'));pending.clear();};
 try{await call('load',{model:model().id,modelUrl:model().modelUrl,modelLabel:model().name,backend:model().backend||'transformers',modality:mode,dtype:model().dtype,family:model().family,device:$('demo-device').value,cyclicOrderRobustness:$('demo-cyclic').checked});if(turn!==lifecycle)return;ready=true;$('download-progress').value=100;$('download-file').textContent='Model ready';$('download-percent').textContent='';status('Ready. Edit the input and evaluate.','ready');}
 catch(e){if(turn===lifecycle){stop();status(e.message,'error');$('download-panel').hidden=true;}}
 finally{if(turn===lifecycle){clearInterval(timer);busy=false;controls();}}
});
$('demo-cancel').addEventListener('click',()=>{lifecycle++;stop();$('download-panel').hidden=true;status('Cancelled. You can load the model again.');});
$('demo-unload').addEventListener('click',()=>{lifecycle++;stop();$('download-panel').hidden=true;clearResult();status('Model unloaded. Choose another model or input type.');});
$('demo-reset').addEventListener('click',()=>{preset();resetMedia();document.querySelector('.action-menu').open=false;});
$('demo-copy').addEventListener('click',async()=>{if(!result){status('Run an evaluation before copying.');return;}try{await navigator.clipboard.writeText(JSON.stringify(result,null,2));status('Result JSON copied.');}catch{status('Clipboard unavailable. Select the JSON output to copy it.','error');}document.querySelector('.action-menu').open=false;});
document.querySelectorAll('[data-output]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-output]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));$('result-json').hidden=b.dataset.output!=='json';$('result-scores').hidden=b.dataset.output!=='scores';}));
function acceptFile(blob){if(blob.size>24*1024*1024)throw new Error('Please choose a file smaller than 24 MB.');if(blob.type&&!blob.type.startsWith(mode+'/'))throw new Error(`Choose a ${mode} file.`);resetMedia();file=blob;previewURL=URL.createObjectURL(blob);const el=document.createElement(mode==='image'?'img':mode);el.src=previewURL;if(mode==='image')el.alt='Selected image';else el.controls=true;$('media-preview').append(el);clearResult();status('Media selected. Load the model if needed, then evaluate.');}
$('demo-file').addEventListener('change',()=>{const selected=$('demo-file').files[0];if(selected)try{acceptFile(selected);}catch(e){status(e.message,'error');}});
const upload=document.querySelector('.upload-area');upload.addEventListener('dragover',e=>e.preventDefault());upload.addEventListener('drop',e=>{e.preventDefault();if(busy)return;try{if(e.dataTransfer.files[0])acceptFile(e.dataTransfer.files[0]);}catch(err){status(err.message,'error');}});
$('example-media').addEventListener('click',async()=>{busy=true;controls();try{status('Fetching sample media…');const name=mode==='audio'?'dog_barking.wav':'tiger.jpg';const response=await fetch('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/'+name);if(!response.ok)throw new Error('Sample could not be downloaded. Upload your own file.');acceptFile(await response.blob());}catch(e){status(e.message,'error');}finally{busy=false;controls();}});
const dataURL=blob=>new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('Could not read media'));reader.readAsDataURL(blob);});
async function prepareMedia(){
 const encoded=await dataURL(file);let media;
 if(mode==='image')media={frames:[encoded]};
 else if(mode==='audio'){
  const context=new AudioContext({sampleRate:48000});
  try{const buffer=await context.decodeAudioData(await file.arrayBuffer());if(buffer.duration>10)throw new Error('Choose an audio clip of 10 seconds or less.');const audio=new Float32Array(buffer.length);for(let c=0;c<buffer.numberOfChannels;c++){const channel=buffer.getChannelData(c);for(let i=0;i<audio.length;i++)audio[i]+=channel[i]/buffer.numberOfChannels;}media={audio};}finally{await context.close();}
 }else{
  const video=document.createElement('video');video.muted=true;video.preload='auto';
  const wait=event=>new Promise((resolve,reject)=>{const timeout=setTimeout(()=>finish(new Error('Video decoding timed out. Try a different file.')),15000);const finish=err=>{clearTimeout(timeout);video.removeEventListener(event,ok);video.removeEventListener('error',fail);err?reject(err):resolve();};const ok=()=>finish();const fail=()=>finish(new Error('This browser cannot decode the video.'));video.addEventListener(event,ok,{once:true});video.addEventListener('error',fail,{once:true});});
  try{const loaded=wait('loadeddata');video.src=previewURL;await loaded;if(!Number.isFinite(video.duration)||video.duration>60)throw new Error('Choose a video of 60 seconds or less.');const canvas=document.createElement('canvas');const scale=Math.min(1,640/video.videoWidth);canvas.width=Math.max(1,Math.round(video.videoWidth*scale));canvas.height=Math.max(1,Math.round(video.videoHeight*scale));const ctx=canvas.getContext('2d');const frames=[];for(let i=0;i<4;i++){const sought=wait('seeked');video.currentTime=video.duration*(i+.5)/4;await sought;ctx.drawImage(video,0,0,canvas.width,canvas.height);frames.push(canvas.toDataURL('image/jpeg',.9));}media={frames};}finally{video.removeAttribute('src');video.load();}
 }
 return {media,state:{contract:'turbojev.media.v1',modality:mode,base64:encoded.split(',')[1]}};
}
function render(value){result=value;$('result-empty').hidden=true;$('demo-result').hidden=false;const answer=mode==='text'?value:value.answers?.decision; if(!answer)throw new Error('Runtime returned no decision. See the JSON output.');$('demo-choice').textContent=answer.choice??answer.value??'Decision';const probabilities=answer.probabilities??{};$('demo-probabilities').replaceChildren(...Object.entries(probabilities).sort((a,b)=>b[1]-a[1]).map(([label,p])=>{const row=document.createElement('div');row.className='demo-probability';const legend=document.createElement('div');legend.className='demo-probability-label';const name=document.createElement('span');name.textContent=label;const number=document.createElement('span');number.textContent=(p*100).toFixed(1)+'%';legend.append(name,number);const track=document.createElement('div');track.className='demo-track';const fill=document.createElement('i');fill.style.width=Math.max(0,Math.min(100,p*100))+'%';track.append(fill);row.append(legend,track);return row;}));$('result-json').firstElementChild.textContent=JSON.stringify(value,null,2);}
$('demo-run').addEventListener('click',async()=>{
 const options=$('demo-options').value.split('\n').map(x=>x.trim()).filter(Boolean),state=$('demo-state').value.trim();
 if(options.length<2||options.length>20||new Set(options).size!==options.length){status('Enter 2–20 different answers, one per line.','error');return;}
 if(mode==='text'&&!state){status('Enter a message first.','error');return;}if(mode!=='text'&&!file){status('Select or upload media first.','error');return;}
 busy=true;controls();const start=performance.now();timer=setInterval(()=>{$('run-time').textContent=((performance.now()-start)/1000).toFixed(1)+' s';},100);status('Evaluating on your device…');
 try{let value;if(mode==='text')value=await call('classify',{state,options});else{const prepared=await prepareMedia();value=await call('evaluate',{media:prepared.media,request:{state:prepared.state,questions:{decision:{type:'choice',instructions:model().family==='vlm'?state:'Select the best matching description.',criteria:Object.fromEntries(options.map(x=>[x,x]))}}}});}render(value);runs++;$('run-count').textContent=runs;status('Evaluation complete. Timing includes preprocessing and inference.','ready');}
 catch(e){status(e.message,'error');}
 finally{clearInterval(timer);$('run-time').textContent=((performance.now()-start)/1000).toFixed(2)+' s';busy=false;controls();}
});
configure();
