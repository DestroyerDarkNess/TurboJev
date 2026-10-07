import {TurboJev} from '../bindings/node/dist/index.js';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const [modelArg,nativeArg='artifacts/npm-native/node-win32-x64/turbojev.node',output='artifacts/real-option-order.json']=process.argv.slice(2);
if(!modelArg)throw new Error('Usage: node scripts/validate-real-option-order.mjs MODEL_GGUF [NATIVE_ADDON] [OUTPUT_JSON]');
const model=resolve(modelArg);
const nativePath=resolve(nativeArg);
const version=JSON.parse(await readFile(new URL('../bindings/node/package.json',import.meta.url),'utf8')).version;
const criteria={billing:'Payments, charges and refunds',shipping:'Packages, delivery and replacements',technical:'App errors, crashes and sign-in problems',other:'Unrelated requests'};
const orders=[Object.keys(criteria),Object.keys(criteria).reverse(),['technical','billing','other','shipping']];
const cases=[['My package arrived broken. Please send a replacement.','shipping'],['I was charged twice. Please refund the duplicate payment.','billing'],['The app crashes whenever I try to sign in.','technical']];
const sha=async p=>createHash('sha256').update(await readFile(p)).digest('hex');
const report={version,model,model_sha256:await sha(model),native_sha256:await sha(nativePath),device:'cpu',thinking:false,n_threads:4,results:[],timing_note:'Other work may overlap; these are observed latencies, not an isolated speed benchmark.'};
for(const policy of ['none','balanced']){
 const engine=await TurboJev.load({model,nativePath,device:'cpu',nThreads:4,nCtx:8192,execution:{orderRobustness:policy}});
 try{for(const [state,expected] of cases){const runs=[];
   for(const order of orders){const request={state,questions:{team:{type:'choice',instructions:'Which support team should handle this customer request?',criteria:Object.fromEntries(order.map(k=>[k,criteria[k]]))}}};
    const start=performance.now();const response=await engine.evaluate(request);
    runs.push({order,seconds:(performance.now()-start)/1000,response});
   }
   report.results.push({policy,state,expected,runs});
   console.log(JSON.stringify({policy,expected,choices:runs.map(r=>r.response.answers.team.choice),seconds:runs.map(r=>r.seconds)}));
 }}finally{await engine.close();}
}
await writeFile(output,JSON.stringify(report,null,2)+'\n');
