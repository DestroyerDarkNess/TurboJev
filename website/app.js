(() => {
  const revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries, current) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          current.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    revealItems.forEach((item) => observer.observe(item));
  } else {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  }

  document.querySelectorAll(".copy-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const block = button.closest(".code-block, .install-line");
      const code = block?.querySelector("code");
      if (!code) return;
      try {
        await navigator.clipboard.writeText(code.textContent || "");
        const original = button.textContent;
        button.textContent = "Copied";
        button.classList.add("copied");
        window.setTimeout(() => {
          button.textContent = original;
          button.classList.remove("copied");
        }, 1600);
      } catch {
        button.textContent = "Select";
      }
    });
  });

  const menuToggle = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".main-nav");
  if (menuToggle && nav) {
    menuToggle.addEventListener("click", () => {
      const open = menuToggle.getAttribute("aria-expanded") === "true";
      menuToggle.setAttribute("aria-expanded", String(!open));
      nav.classList.toggle("mobile-open", !open);
    });
    nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
      menuToggle.setAttribute("aria-expanded", "false");
      nav.classList.remove("mobile-open");
    }));
  }

  const search = document.querySelector("#docs-search");
  if (search) {
    const searchable = [...document.querySelectorAll(".searchable")];
    search.addEventListener("input", () => {
      const needle = search.value.trim().toLowerCase();
      searchable.forEach((section) => {
        section.hidden = Boolean(needle) && !section.textContent.toLowerCase().includes(needle);
      });
    });
  }

  const sections = document.querySelectorAll(".docs-content [id]");
  const links = document.querySelectorAll(".docs-nav a");
  if ("IntersectionObserver" in window && sections.length && links.length) {
    const activeObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === "#" + entry.target.id));
      });
    }, { rootMargin: "-16% 0px -70% 0px" });
    sections.forEach((section) => activeObserver.observe(section));
  }
})();
