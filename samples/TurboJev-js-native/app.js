const $ = id => document.getElementById(id);
let engine, busy = false;
function status(message, error = false) { $("status").textContent = message; $("status").classList.toggle("error", error); }
function controls() {
  $("load").disabled = busy || !!engine; $("release").disabled = busy || !engine;
  $("run").disabled = busy || !engine; $("model").disabled = busy || !!engine; $("device").disabled = busy || !!engine;
  $("badge").textContent = busy ? "En proceso…" : engine ? "Modelo listo" : "Sin modelo";
  $("badge").classList.toggle("ready", !!engine && !busy);
}
$("load").onclick = async () => {
  busy = true; controls(); status("Descargando e inicializando el modelo. La primera carga puede tardar unos minutos…");
  try {
    if ($("device").value === "webgpu" && !navigator.gpu) throw new Error("WebGPU no está disponible. Selecciona CPU.");
    const {TurboJevWeb} = await import(/* @vite-ignore */ new URL("/vendor/web/dist/index.js", location.href).href);
    engine = await TurboJevWeb.load({model: $("model").value.trim(), runtimeModuleUrl: "/vendor/runtime.js", wasmModuleUrl: "/vendor/wasm/turbojev_wasm.js", runtimeOptions: {device: $("device").value, dtype: "q4"}});
    status("Modelo cargado. Ya puedes evaluar mensajes localmente.");
  } catch (error) { status(`No se pudo cargar: ${error.message}. Comprueba que ejecutaste samples/build.py web.`, true); }
  finally { busy = false; controls(); }
};
$("release").onclick = async () => {
  busy = true; controls();
  try { await engine.close(); status("Modelo liberado."); }
  catch (error) { status(error.message, true); }
  finally { engine = null; busy = false; controls(); }
};
$("run").onclick = async () => {
  const options = $("options").value.split("\n").map(x => x.trim()).filter(Boolean);
  if (!$("state").value.trim() || options.length < 2 || new Set(options).size !== options.length) return status("Escribe un mensaje y al menos dos opciones distintas.", true);
  busy = true; controls(); status("Evaluando el mensaje…"); const start = performance.now();
  try {
    const answer = await engine.classify($("state").value, options);
    $("latency").textContent = `${Math.round(performance.now() - start)} ms`;
    $("winner").textContent = answer.choice; $("json").textContent = JSON.stringify(answer, null, 2); $("bars").replaceChildren();
    for (const [label, probability] of Object.entries(answer.probabilities).sort((a,b) => b[1]-a[1])) {
      const row = document.createElement("div"); row.className = "bar-label";
      const name = document.createElement("span"), value = document.createElement("span");
      name.textContent = label; value.textContent = `${(probability * 100).toFixed(1)}%`; row.append(name,value);
      const track = document.createElement("div"), fill = document.createElement("div"); track.className="track";fill.className="fill";fill.style.width=`${Math.max(0,Math.min(100,probability*100))}%`;track.append(fill);$("bars").append(row,track);
    }
    $("empty").hidden = true; $("result").hidden = false; status("Decisión completada.");
  } catch (error) { status(error.message, true); }
  finally { busy = false; controls(); }
};
document.querySelectorAll("[data-text]").forEach(button => button.onclick = () => { $("state").value = button.dataset.text; });
