# TurboJev website

The website is a dependency-free static landing page and documentation site.
It can be hosted by GitHub Pages, Netlify, a CDN, or any ordinary static HTTP
server.

## Preview locally

From the repository root:

```powershell
python -m http.server 4173 --directory website
```

Open <http://localhost:4173/>. The landing page is `index.html`; the complete
language documentation is `docs.html`.

The site intentionally has no build step. `styles.css` contains the visual
system and `app.js` adds mobile navigation,
copy buttons, and documentation filtering.

For GitHub Pages, set the repository Pages source to the `website/` folder (or
copy these files to the Pages artifact). The included `.nojekyll` file keeps
the static paths untouched.
