import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
const require = createRequire(new URL('../samples/TurboJev-js-native/package.json', import.meta.url));
const { build } = require('esbuild');
const root = fileURLToPath(new URL('.', import.meta.url));
await build({ entryPoints: [root + 'src/runtime.ts'], outfile: root + 'vendor/studio-runtime.js',
  bundle: true, format: 'esm', platform: 'browser', target: 'es2022', minify: true,
  nodePaths: [fileURLToPath(new URL('../samples/TurboJev-js-native/node_modules', import.meta.url))],
  define: { 'process.env.NODE_ENV': '"production"' } });
const digest = async file => createHash('sha256').update(await readFile(root + file)).digest('hex').slice(0,12);
const manifest = { runtime: await digest('vendor/studio-runtime.js'), wasm: await digest('vendor/wasm/turbojev_wasm_bg.wasm') };
await writeFile(root + 'vendor/manifest.json', JSON.stringify(manifest));
const version = createHash('sha256').update(JSON.stringify(manifest) + await digest('demo.js') + await digest('demo-worker.js') + await digest('app.js') + await digest('styles.css')).digest('hex').slice(0,12);
const landing = (await readFile(root + 'index.html','utf8')).replace(/(src="(?:demo|app)\.js|href="styles\.css)(?:\?v=[a-z0-9]+)?/g, `$1?v=${version}`);
await writeFile(root + 'index.html',landing);
const source = await readFile(root + 'src/docs-content.html', 'utf8');
const pages = [ ['quickstart','Quickstart'], ['architecture','Architecture'], ['api','Core API'],
  ['rust','Rust'], ['node','Node / TypeScript'], ['python','Python'], ['c','C ABI'], ['dotnet','.NET / C#'],
  ['browser','Browser / WASM'], ['mobile','Android + Swift'], ['multimodal','Multimodal'], ['examples','Examples'], ['publishing','Publishing'] ];
const esc = s => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const search = [];
await mkdir(root + 'docs', { recursive: true });
for (const [i,[slug,title]] of pages.entries()) {
  let content = source.match(new RegExp(`<section id="${slug}"[\\s\\S]*?</section>`))?.[0]
    ?? source.match(new RegExp(`<article id="${slug}"[\\s\\S]*?</article>`))?.[0];
  if (!content) throw new Error('Missing docs: ' + slug);
  content = content.replace(/<h[23]>(.*?)<\/h[23]>/, `<h1>${title}</h1>`);
  content = content.replace(/<div class="code-block">([\s\S]*?)<pre><code>/g, (match,toolbar) => {
    const label = toolbar.toLowerCase();
    const lang = label.includes('.py') ? 'python' : label.includes('.toml') ? 'toml' : label.includes('.rs') ? 'rust' : label.includes('.cs') ? 'csharp' : label.includes('kotlin') ? 'kotlin' : label.includes('swift') ? 'swift' : label.includes('canonical') || label.includes('evidence') ? 'json' : label.includes('.ts') ? 'typescript' : label.includes('.js') ? 'javascript' : label.includes('.c') ? 'c' : label.includes('powershell') || label.includes('release menu') ? 'powershell' : 'bash';
    return `<div class="code-block">${toolbar}<pre class="language-${lang}"><code class="language-${lang}">`;
  });
  const plain = content.replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim();
  search.push({title,slug,text:plain});
  const nav = pages.map(([s,t],j) => `${j===0?'<p>Get started</p>':j===3?'<p>Languages</p>':j===10?'<p>Guides</p>':''}<a ${s===slug?'aria-current="page"':''} href="${s}.html">${t}</a>`).join('');
  const prev = pages[i-1], next = pages[i+1];
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${title} — TurboJev</title><meta name="description" content="${esc(title)} documentation for TurboJev."><link rel="icon" href="../favicon.svg"><link rel="stylesheet" href="../styles.css"></head><body class="docs-page"><a class="skip-link" href="#article">Skip to content</a><header class="site-header"><a class="brand" href="../index.html">turbo<span>jev</span><b aria-hidden="true">↗</b></a><span class="header-divider">/</span><a href="quickstart.html">Documentation</a><button class="search-trigger" type="button">Search documentation <kbd>Ctrl K</kbd></button><a class="github-link" href="https://github.com/DestroyerDarkNess/TurboJev">GitHub ↗</a><button class="menu-toggle" aria-label="Toggle documentation navigation" aria-expanded="false">☰</button></header><div class="docs-layout"><aside class="docs-sidebar"><nav class="docs-nav" aria-label="Documentation">${nav}</nav></aside><main id="article" class="docs-content"><div class="breadcrumbs">Docs <span>/</span> ${title}</div>${content}<nav class="page-turn" aria-label="Adjacent pages">${prev?`<a href="${prev[0]}.html"><small>Previous</small>← ${prev[1]}</a>`:'<span></span>'}${next?`<a href="${next[0]}.html"><small>Next</small>${next[1]} →</a>`:''}</nav><div class="article-footer"><a href="https://github.com/DestroyerDarkNess/TurboJev/tree/main/website/src">Edit this page ↗</a><a href="../index.html#playground">Open playground →</a></div></main><aside class="on-this-page"><p>On this page</p><nav id="page-outline"></nav></aside></div><dialog id="search-dialog"><form method="dialog"><label for="docs-search">Search documentation</label><button aria-label="Close search">Esc</button></form><input type="search" id="docs-search" placeholder="Search APIs, languages, examples…" autocomplete="off"><div id="search-results" aria-live="polite"></div></dialog><script src="../vendor/prism/prism.js" defer></script><script src="../app.js" defer></script></body></html>`;
  await writeFile(root + `docs/${slug}.html`,html.replaceAll('../app.js"',`../app.js?v=${version}"`).replaceAll('../styles.css"',`../styles.css?v=${version}"`));
}
await writeFile(root + 'docs/search.json',JSON.stringify(search));
await writeFile(root + 'docs.html',`<!doctype html><html lang="en"><head><meta charset="utf-8"><title>TurboJev documentation</title><meta name="viewport" content="width=device-width,initial-scale=1"><script>const pages=${JSON.stringify(pages.map(p=>p[0]))};const slug=location.hash.slice(1);location.replace('docs/'+(pages.includes(slug)?slug:'quickstart')+'.html');</script></head><body><a href="docs/quickstart.html">Open documentation</a></body></html>`);
console.log('Built browser runtime and 13 documentation pages.');
