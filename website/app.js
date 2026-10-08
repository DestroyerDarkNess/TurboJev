(() => {
  const toggle = document.querySelector(".menu-toggle");
  const mobileMenu = document.querySelector(".docs-sidebar,.main-nav");
  function closeMenu() {
    toggle?.setAttribute("aria-expanded", "false");
    mobileMenu?.classList.remove("mobile-open");
  }
  toggle?.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    mobileMenu?.classList.toggle("mobile-open", open);
  });
  mobileMenu?.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenu();
  });
  document.querySelectorAll(".copy-button").forEach((button) =>
    button.addEventListener("click", async () => {
      const code = button.closest(".code-block,.install-line")?.querySelector("code");
      if (!code) return;
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
        setTimeout(() => (button.textContent = "Copy"), 1500);
      } catch {
        button.textContent = "Select code";
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
    }),
  );
  document.addEventListener("click", (event) => {
    document.querySelectorAll("details[open]").forEach((menu) => {
      if (!menu.contains(event.target)) menu.open = false;
    });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape")
      document.querySelectorAll("details[open]").forEach((menu) => (menu.open = false));
  });
  const article = document.querySelector(".docs-content");
  if (!article) return;
  const outline = document.getElementById("page-outline");
  const headings = [...article.querySelectorAll("h2,h3,.code-toolbar>span")];
  headings.forEach((heading, i) => {
    heading.id = "section-" + i;
    const a = document.createElement("a");
    a.href = "#" + heading.id;
    a.textContent = heading.textContent;
    outline.append(a);
  });
  if (!headings.length) {
    const a = document.createElement("a");
    a.href = "#article";
    a.textContent = article.querySelector("h1").textContent;
    outline.append(a);
  }
  const dialog = document.getElementById("search-dialog"),
    input = document.getElementById("docs-search"),
    results = document.getElementById("search-results");
  let index;
  async function search() {
    results.replaceChildren();
    try {
      index ??= await fetch("search.json").then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      });
      const query = input.value.trim().toLowerCase();
      const terms = query.split(/\s+/).filter(Boolean);
      const matches = index
        .filter((page) =>
          terms.every((term) => (page.title + " " + page.text).toLowerCase().includes(term)),
        )
        .sort(
          (a, b) =>
            Number(b.title.toLowerCase().includes(query)) -
            Number(a.title.toLowerCase().includes(query)),
        );
      if (!matches.length) {
        const p = document.createElement("p");
        p.textContent = "No results. Try a language name, generate, ordering, or media.";
        results.append(p);
      }
      for (const page of matches) {
        const a = document.createElement("a");
        a.href = page.slug + ".html";
        const title = document.createElement("strong");
        title.textContent = page.title;
        const p = document.createElement("p");
        const start = query ? Math.max(0, page.text.toLowerCase().indexOf(terms[0]) - 45) : 0;
        p.textContent = (start ? "…" : "") + page.text.slice(start, start + 150) + "…";
        a.append(title, p);
        results.append(a);
      }
    } catch {
      results.textContent = "Search could not load. Browse the navigation or try again.";
    }
  }
  function openSearch() {
    dialog.showModal();
    input.focus();
    search();
  }
  document.querySelector(".search-trigger").addEventListener("click", openSearch);
  input.addEventListener("input", search);
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      results.querySelector("a")?.focus();
    }
    if (event.key === "Enter") {
      event.preventDefault();
      results.querySelector("a")?.click();
    }
  });
  results.addEventListener("keydown", (event) => {
    const links = [...results.querySelectorAll("a")],
      i = links.indexOf(document.activeElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      links[Math.min(i + 1, links.length - 1)]?.focus();
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      i <= 0 ? input.focus() : links[i - 1]?.focus();
    }
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (
        event.clientX < r.left ||
        event.clientX > r.right ||
        event.clientY < r.top ||
        event.clientY > r.bottom
      )
        dialog.close();
    }
  });
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      dialog.open ? dialog.close() : openSearch();
    }
  });
})();
