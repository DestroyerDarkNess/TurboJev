import { readFile, writeFile, mkdir, copyFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
const require = createRequire(
  new URL("../samples/TurboJev-js-native/package.json", import.meta.url),
);
const { build } = require("esbuild");
const prettier = require("prettier");
const formatting = JSON.parse(
  await readFile(new URL("../.prettierrc.json", import.meta.url), "utf8"),
);
const root = fileURLToPath(new URL(".", import.meta.url));
async function writeHtml(file, html) {
  const formatted = await prettier.format(html, { ...formatting, parser: "html" });
  for (let attempt = 0; ; attempt++) {
    try {
      await writeFile(root + file, formatted);
      return;
    } catch (error) {
      // Windows file watchers may briefly hold a generated page during a rebuild.
      // Retry transient sharing failures, but preserve persistent build errors.
      const transient = ["UNKNOWN", "EBUSY", "EPERM", "EACCES"].includes(error.code);
      if (process.platform !== "win32" || !transient || attempt >= 7) {
        throw error;
      }
      await delay(100 * (attempt + 1));
    }
  }
}
await build({
  entryPoints: [fileURLToPath(new URL("../bindings/web/src/index.ts", import.meta.url))],
  outfile: root + "vendor/web/dist/index.js",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
});
await copyFile(
  fileURLToPath(new URL("../bindings/web/worker.js", import.meta.url)),
  root + "vendor/web/worker.js",
);
await build({
  entryPoints: [root + "src/runtime.ts"],
  outfile: root + "vendor/studio-runtime.js",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
  nodePaths: [
    fileURLToPath(new URL("../samples/TurboJev-js-native/node_modules", import.meta.url)),
  ],
  define: { "process.env.NODE_ENV": '"production"' },
});
await build({
  entryPoints: [
    fileURLToPath(new URL("../bindings/web-runtimes/llamacpp-wasm/src/index.js", import.meta.url)),
  ],
  outfile: root + "vendor/llamacpp-runtime.js",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
});
await copyFile(
  fileURLToPath(new URL("../tests/fixtures/multimodal-audio/on.wav", import.meta.url)),
  root + "vendor/audio-on.wav",
);
const digest = async (file) =>
  createHash("sha256")
    .update(await readFile(root + file))
    .digest("hex")
    .slice(0, 12);
const manifest = {
  runtime: await digest("vendor/studio-runtime.js"),
  llamacpp: await digest("vendor/llamacpp-runtime.js"),
  wasm: await digest("vendor/wasm/turbojev_wasm_bg.wasm"),
};
await writeFile(root + "vendor/manifest.json", JSON.stringify(manifest));
const version = createHash("sha256")
  .update(
    JSON.stringify(manifest) +
      (await digest("demo.js")) +
      (await digest("demo-worker.js")) +
      (await digest("app.js")) +
      (await digest("styles.css")),
  )
  .digest("hex")
  .slice(0, 12);
const landing = (await readFile(root + "index.html", "utf8")).replace(
  /(src="(?:demo|app)\.js|href="styles\.css)(?:\?v=[a-z0-9]+)?/g,
  `$1?v=${version}`,
);
await writeHtml("index.html", landing);
const source =
  (await readFile(root + "src/docs-content.html", "utf8")) +
  "\n" +
  (await readFile(root + "src/cookbook-content.html", "utf8"));
const pages = [
  ["quickstart", "Quickstart"],
  ["architecture", "Architecture"],
  ["api", "Core API"],
  ["fields", "Typed fields"],
  ["cookbook", "Cookbook"],
  ["rust", "Rust"],
  ["node", "Node / TypeScript"],
  ["python", "Python"],
  ["c", "C ABI"],
  ["dotnet", ".NET / C#"],
  ["browser", "Browser / WASM"],
  ["mobile", "Android + Swift"],
  ["ordering", "Option ordering"],
  ["decisions", "Decision design"],
  ["multimodal", "Multimodal"],
  ["examples", "Examples"],
  ["publishing", "Publishing"],
];
const esc = (s) => s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;");

function codeLanguage(toolbar) {
  const label = toolbar.toLowerCase();
  const languages = [
    ["python", [".py"]],
    ["toml", [".toml"]],
    ["rust", [".rs"]],
    ["csharp", [".cs"]],
    ["kotlin", ["kotlin"]],
    ["swift", ["swift"]],
    ["json", ["canonical", "evidence", "json"]],
    ["typescript", [".ts"]],
    ["javascript", [".js"]],
    ["c", [".c"]],
    ["powershell", ["powershell", "release menu"]],
  ];
  const match = languages.find(([, labels]) => labels.some((text) => label.includes(text)));
  return match?.[0] ?? "bash";
}

function navigation(pages, currentSlug) {
  const groups = new Map([
    [0, "Get started"],
    [5, "Languages"],
    [12, "Guides"],
  ]);
  return pages
    .map(([slug, title], index) => {
      const group = groups.has(index) ? `<p>${groups.get(index)}</p>` : "";
      const current = slug === currentSlug ? 'aria-current="page"' : "";
      return `${group}<a ${current} href="${slug}.html">${title}</a>`;
    })
    .join("\n");
}

const search = [];
await mkdir(root + "docs", { recursive: true });
for (const [i, [slug, title]] of pages.entries()) {
  let content =
    source.match(new RegExp(`<section\\b[^>]*\\bid="${slug}"[^>]*>[\\s\\S]*?</section>`))?.[0] ??
    source.match(new RegExp(`<article\\b[^>]*\\bid="${slug}"[^>]*>[\\s\\S]*?</article>`))?.[0];
  if (!content) throw new Error("Missing docs: " + slug);
  content = content.replace(/<h[23]>[\s\S]*?<\/h[23]>/, `<h1>${title}</h1>`);
  content = content.replace(
    /<div class="code-block">([\s\S]*?)<pre>\s*<code>/g,
    (match, toolbar) => {
      const lang = codeLanguage(toolbar);
      return `<div class="code-block">${toolbar}<pre class="language-${lang}"><code class="language-${lang}">`;
    },
  );
  const plain = content
    .replace(/<[^>]*>/g, " ")
    .replace(
      /&(?:amp|lt|gt|quot|#x27|#39);/g,
      (entity) =>
        ({ "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#x27;": "'", "&#39;": "'" })[
          entity
        ],
    )
    .replace(/\s+/g, " ")
    .trim();
  search.push({ title, slug, text: plain });
  const nav = navigation(pages, slug);
  const prev = pages[i - 1],
    next = pages[i + 1];
  const previousLink = prev
    ? `<a href="${prev[0]}.html"><small>Previous</small>← ${prev[1]}</a>`
    : "<span></span>";
  const nextLink = next ? `<a href="${next[0]}.html"><small>Next</small>${next[1]} →</a>` : "";
  const html = `
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>${title} — TurboJev</title>
    <meta name="description" content="${esc(title)} documentation for TurboJev.">
    <link rel="icon" href="../favicon.svg">
    <link rel="stylesheet" href="../styles.css">
  </head>
  <body class="docs-page">
    <a class="skip-link" href="#article">Skip to content</a>
    <header class="site-header">
      <a class="brand" href="../index.html">
        turbo<span>jev</span><b aria-hidden="true">↗</b>
      </a>
      <span class="header-divider">/</span>
      <a href="quickstart.html">Documentation</a>
      <button class="search-trigger" type="button">
        Search documentation <kbd>Ctrl K</kbd>
      </button>
      <a class="github-link" href="https://github.com/DestroyerDarkNess/TurboJev">GitHub ↗</a>
      <button class="menu-toggle" aria-label="Toggle documentation navigation" aria-expanded="false">☰</button>
    </header>
    <div class="docs-layout">
      <aside class="docs-sidebar">
        <nav class="docs-nav" aria-label="Documentation">${nav}</nav>
      </aside>
      <main id="article" class="docs-content">
        <div class="breadcrumbs">Docs <span>/</span> ${title}</div>
        ${content}
        <nav class="page-turn" aria-label="Adjacent pages">${previousLink}${nextLink}</nav>
        <div class="article-footer">
          <a href="https://github.com/DestroyerDarkNess/TurboJev/tree/main/website/src">Edit this page ↗</a>
          <a href="../index.html#playground">Open playground →</a>
        </div>
      </main>
      <aside class="on-this-page">
        <p>On this page</p>
        <nav id="page-outline"></nav>
      </aside>
    </div>
    <dialog id="search-dialog">
      <form method="dialog">
        <label for="docs-search">Search documentation</label>
        <button aria-label="Close search">Esc</button>
      </form>
      <input type="search" id="docs-search" placeholder="Search APIs, languages, examples…" autocomplete="off">
      <div id="search-results" aria-live="polite"></div>
    </dialog>
    <script src="../vendor/prism/prism.js" defer></script>
    <script src="../app.js" defer></script>
  </body>
</html>
`;
  await writeHtml(
    `docs/${slug}.html`,
    html
      .replaceAll('../app.js"', `../app.js?v=${version}"`)
      .replaceAll('../styles.css"', `../styles.css?v=${version}"`),
  );
}
await writeFile(root + "docs/search.json", JSON.stringify(search, null, 2) + "\n");
const legacyPage = `
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>TurboJev documentation</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <script>
      const pages = ${JSON.stringify(pages.map(([slug]) => slug))};
      const slug = location.hash.slice(1);
      const page = pages.includes(slug) ? slug : "quickstart";
      location.replace("docs/" + page + ".html");
    </script>
  </head>
  <body>
    <a href="docs/quickstart.html">Open documentation</a>
  </body>
</html>
`;
await writeHtml("docs.html", legacyPage);
console.log(`Built browser runtime and ${pages.length} documentation pages.`);
