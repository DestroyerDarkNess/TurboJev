const $ = id => document.getElementById(id);
let engine, busy = false;
function status(message, error = false) { $("status").textContent = message; $("status").classList.toggle("error", error); }
function controls() {
  $("load").disabled = busy || !!engine; $("release").disabled = busy || !engine;
  $("run").disabled = busy || !engine; $("model").disabled = busy || !!engine; $("runtime").disabled = busy || !!engine; $("device").disabled = busy || !!engine;
  $("badge").textContent = busy ? "En proceso…" : engine ? "Modelo listo" : "Sin modelo";
  $("badge").classList.toggle("ready", !!engine && !busy);
}
const modelDefaults = {
  transformers: "onnx-community/Qwen3-0.6B-ONNX",
  llamacpp: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/1208e45d782fe18602c5eaf10e5758d5b0f24c03/Qwen3-0.6B-Q4_K_M.gguf",
};
$("runtime").onchange = () => {
  const llama = $("runtime").value === "llamacpp";
  $("model-label").textContent = llama ? "URL directa del modelo GGUF" : "Modelo ONNX (ID de Hugging Face)";
  $("model").value = modelDefaults[llama ? "llamacpp" : "transformers"];
  $("model-hint").textContent = llama
    ? "llama.cpp WASM descarga un GGUF directamente. CPU usa WebAssembly; GPU solicita WebGPU."
    : "Transformers.js carga el modelo ONNX desde Hugging Face. El navegador puede reutilizar la caché.";
};
$("load").onclick = async () => {
  busy = true; controls(); status("Descargando e inicializando el modelo. La primera carga puede tardar unos minutos…");
  try {
    if ($("device").value === "webgpu" && !navigator.gpu) throw new Error("WebGPU no está disponible. Selecciona CPU.");
    const {TurboJevWeb} = await import(/* @vite-ignore */ new URL("/vendor/web/dist/index.js", location.href).href);
    const llama = $("runtime").value === "llamacpp";
    const model = $("model").value.trim();
    engine = await TurboJevWeb.load({
      model: llama ? "Qwen/Qwen3-0.6B-GGUF" : model,
      runtimeModuleUrl: llama ? "https://cdn.jsdelivr.net/npm/@turbojev/runtime-llamacpp-wasm@0.28.6/+esm" : "/vendor/runtime.js",
      wasmModuleUrl: "/vendor/wasm/turbojev_wasm.js",
      runtimeOptions: {device: $("device").value, dtype: "q4", ...(llama ? {modality: "text", modelUrl: model, modelLabel: "Qwen3 0.6B Q4_K_M"} : {})},
    });
    status(`Modelo cargado con ${llama ? "llama.cpp WASM" : "Transformers.js"}. Ya puedes evaluar mensajes localmente.`);
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
  const question = $("question").value.trim();
  if (!$("state").value.trim() || !question || options.length < 2 || new Set(options).size !== options.length) return status("Escribe el mensaje, una pregunta y al menos dos opciones distintas.", true);
  busy = true; controls(); status("Evaluando el mensaje…"); const start = performance.now();
  try {
    const criteria = Object.fromEntries(options.map((option, index) => [`option_${index + 1}`, option]));
    const response = await engine.evaluate({
      state: $("state").value,
      questions: { route: { type: "choice", instructions: question, criteria } },
    });
    const result = response.answers.route;
    const answer = {
      choice: criteria[result.choice] || result.choice,
      probabilities: Object.fromEntries(Object.entries(result.probabilities).map(([key, value]) => [criteria[key] || key, value])),
    };
    $("latency").textContent = `${Math.round(performance.now() - start)} ms`;
    $("winner").textContent = answer.choice; $("json").textContent = JSON.stringify(response, null, 2); $("bars").replaceChildren();
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
