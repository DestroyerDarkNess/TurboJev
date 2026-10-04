import { reserveMediaWorkers } from './media-worker-pool.js';

const WLLAMA_VERSION = '3.8.1';
const WLLAMA_ENTRY = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm/index.js`;
const WLLAMA_WASM = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm/wasm/wllama.wasm`;
const WLLAMA_WORKER = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/src/wasm/wllama.js`;

export async function createRuntime(dependencies = {}) {
  let engine;
  let model = '';
  let modalities = ['text'];
  let debug = false;
  const loadWllama = dependencies.loadWllama || (() => import(WLLAMA_ENTRY));
  const loadWorkerCode = dependencies.loadWorkerCode || (async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load Wllama worker: HTTP ${response.status}`);
    return response.text();
  });

  return {
    async initialize(options) {
      if (options.thinking === true) {
        throw new Error('THINKING_UNSUPPORTED: this adapter supports direct decision scores only; use the native GGUF text runtime for thinking.');
      }
      debug = options.debug === true;
      if (!options.modelUrl) {
        throw new Error('Choose a GGUF model with a direct download URL.');
      }
      if (options.modality === 'audio' && globalThis.crossOriginIsolated === false) {
        throw new Error('GGUF audio requires cross-origin isolation. Serve the page with COOP: same-origin and COEP: require-corp to enable WebAssembly workers.');
      }
      const inferenceThreads = Number.isInteger(options.nThreads) && options.nThreads > 0
        ? options.nThreads : Math.max(1, Math.floor((globalThis.navigator?.hardwareConcurrency || 2) / 2));

      const { Wllama, LoggerWithoutDebug } = await loadWllama();
      engine = new Wllama({ default: WLLAMA_WASM }, {
        logger: debug ? console : LoggerWithoutDebug,
        suppressNativeLog: !debug,
        parallelDownloads: 4,
      });
      if (options.mmprojUrl && globalThis.crossOriginIsolated !== false) {
        const getResources = engine.getWorkerResources.bind(engine);
        const resources = getResources();
        const source = resources.jsPath?.code || await loadWorkerCode(
          typeof resources.jsPath === 'string' ? resources.jsPath : WLLAMA_WORKER,
        );
        const worker = { code: reserveMediaWorkers(source, inferenceThreads) };
        engine.getWorkerResources = () => ({ ...getResources(), jsPath: worker });
      }
      model = options.modelLabel || options.modelUrl;

      const modelSource = options.mmprojUrl
        ? { url: options.modelUrl, mmprojUrl: options.mmprojUrl }
        : options.modelUrl;
      const loadOptions = {
        n_ctx: 2048,
        n_batch: 512,
        n_gpu_layers: options.device === 'webgpu' ? 999 : 0,
        n_threads: inferenceThreads,
        cache_prompt: false,
        jinja: options.jinja ?? !options.mmprojUrl,
        prefill_assistant: true,
        warmup: options.warmup ?? options.modality !== 'audio',
        progressCallback: ({ loaded = 0, total = 0 }) => {
          options.progressCallback?.({
            status: 'progress',
            file: 'GGUF model',
            loaded,
            total,
            progress: total > 0 ? Math.min(100, loaded / total * 100) : 0,
          });
        },
      };
      if (typeof options.chatTemplate === 'string' && options.chatTemplate.length > 0) {
        loadOptions.chat_template = options.chatTemplate;
      }
      await engine.loadModelFromUrl(modelSource, loadOptions);

      const supportsImage = engine.supportInputModality('image');
      const supportsAudio = engine.supportInputModality('audio');
      modalities = ['text'];
      if (supportsImage) modalities.push('image', 'video');
      if (supportsAudio) modalities.push('audio');

      const requested = options.modality;
      const requestedSupported = requested === undefined || requested === 'text'
        || (requested === 'image' && supportsImage)
        || (requested === 'audio' && supportsAudio)
        || (requested === 'video' && supportsImage);
      if (!requestedSupported) {
        await engine.exit();
        engine = undefined;
        throw new Error(`UNSUPPORTED_MODALITY: the loaded GGUF/projector does not support ${requested} input.`);
      }

      if (requested !== 'audio') {
        const warmup = await engine.createChatCompletion({
          messages: [{ role: 'user', content: 'Reply with one word: ready.' }],
          max_tokens: 1,
          temperature: 0,
          cache_prompt: false,
          chat_template_kwargs: { enable_thinking: false },
        });
        if (!warmup?.choices?.length) throw new Error('llama.cpp WASM loaded the model but warmup returned no completion.');
      }

      return {
        name: 'llama.cpp-wasm',
        model,
        codecProfile: 'llamacpp-wasm-json-prefill-v2',
        scoringMode: 'decisions',
        modalities,
        capabilities: {
          nextTokenLogits: false,
          sequenceScores: false,
          sharedPrefix: false,
          parallelSuffixes: false,
          selectiveOutputHead: false,
          localInference: true,
        },
      };
    },

    async scoreDecisionBatch(batchJson) {
      if (!engine) throw new Error('Load a GGUF model first.');
      const batch = JSON.parse(batchJson);
      debugLog(debug, `decision batch: ${batch.tasks?.length || 0} task(s), modality=${batch.state?.modality || 'text'}`);
      const scores = [];
      let inputTokens = 0;

      for (const task of batch.tasks) {
        const candidates = task.candidates || [];
        if (candidates.length < 2 || candidates.length > 20) {
          throw new Error(`llama.cpp WASM supports 2–20 choices per decision (received ${candidates.length}).`);
        }
        const labels = Array.from({ length: candidates.length }, (_, i) => String.fromCharCode(65 + i));
        const isBestMatch = String(task.intent).toLowerCase() === 'bestmatch';
        const criterion = task.instruction || (isBestMatch ? 'Choose the option that best matches the state.' : 'Choose the correct option.');
        const optionsText = candidates.map((candidate, i) => {
          const meaning = candidate.description && candidate.description !== candidate.key
            ? `${candidate.key}: ${candidate.description}`
            : candidate.description || candidate.key;
          return `${labels[i]}. ${meaning}`;
        }).join('\n');
        const media = batch.state?.contract === 'turbojev.media.v1' ? batch.state : null;
        const systemPrompt = 'Make the requested decision from the supplied state. Follow the requested output format exactly.';
        const userContent = media
          ? multimodalContent(media, criterion, optionsText, labels)
          : `State:\n${typeof batch.state === 'string' ? batch.state : JSON.stringify(batch.state)}\n\nQuestion:\n${criterion}\n\nAllowed options:\n${optionsText}\n\nAnswer as {"answer": "<option letter>"}.`;
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent },
          { role: 'assistant', content: '{"answer": "' },
        ];
        const grammar = `root ::= ${labels.map(label => `"${label}"`).join(' | ')}`;
        if (media) {
          const needsImage = media.modality === 'image' || media.modality === 'video';
          if (needsImage && !engine.supportInputModality('image')) {
            throw new Error(`UNSUPPORTED_MODALITY: loaded llama.cpp WASM model does not support ${media.modality} input.`);
          }
          if (media.modality === 'audio' && !engine.supportInputModality('audio')) {
            throw new Error('UNSUPPORTED_MODALITY: loaded llama.cpp WASM model does not support audio input.');
          }
          if (media.modality === 'audio' && globalThis.crossOriginIsolated === false) {
            throw new Error('GGUF audio requires cross-origin isolation and WebAssembly workers.');
          }
        }
        let response;
        try {
          debugLog(debug, `requesting model scores for task ${task.id}`);
          response = await engine.createChatCompletion({
            messages,
            prefill_assistant: true,
            max_tokens: 1,
            temperature: 1,
            top_k: 0,
            top_p: 1,
            logprobs: true,
            top_logprobs: 20,
            grammar,
            cache_prompt: false,
            chat_template_kwargs: { enable_thinking: false },
          });
          debugLog(debug, `model returned scores for task ${task.id}`);
        } catch (error) {
          const message = error?.message || String(error);
          if (media && /Failed to format input|Failed to tokenize prompt/i.test(message)) {
            throw new Error(`MODEL_PROMPT_FORMAT_UNSUPPORTED: Wllama could not tokenize this model's ${media.modality} input. Use a GGUF and projector verified with the selected Wllama version.`);
          }
          throw error;
        }

        inputTokens += response.usage?.prompt_tokens || 0;
        const alternatives = response.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs || [];
        const logits = labels.map(label => {
          const byte = label.charCodeAt(0);
          const match = alternatives.find(item => item.token === label || (item.bytes?.length === 1 && item.bytes[0] === byte));
          return Number(match?.logprob);
        });
        if (logits.some(value => !Number.isFinite(value))) {
          throw new Error(`llama.cpp did not return a score for every allowed answer (${labels.join(', ')}). Try another GGUF model or runtime version.`);
        }
        scores.push({ id: task.id, logits });
      }

      return JSON.stringify({ model, scores, metrics: { input_tokens: inputTokens } });
    },

    async close() {
      const loaded = engine;
      engine = undefined;
      if (loaded) await loaded.exit();
    },
  };
}

function multimodalContent(media, criterion, optionsText, labels) {
  const context = `State:\n${media.text || ''}\n\nEvidence:\n`;
  const content = [];
  if (media.modality === 'audio') {
    // Wllama's own chat client places the media block before the user text.
    // Some audio projectors only recognize the marker in that position.
    content.push({ type: 'audio', data: decodeBase64(media.base64) });
    content.push({ type: 'text', text: context });
  } else {
    content.push({ type: 'text', text: context });
  }
  if (media.modality === 'image') {
    content.push({ type: 'image', data: decodeBase64(media.base64) });
  } else if (media.modality === 'video') {
    for (const frame of media.frames || []) {
      content.push({ type: 'text', text: `[frame at ${frame.timestamp_ms} ms]\n` });
      content.push({ type: 'image', data: decodeBase64(frame.base64) });
    }
  } else if (media.modality !== 'audio') {
    throw new Error(`UNSUPPORTED_MODALITY: unknown media modality ${media.modality}.`);
  }
  content.push({
    type: 'text',
    text: `\nQuestion:\n${criterion}\n\nAllowed options:\n${optionsText}\n\nAnswer as {"answer": "<option letter>"}.`,
  });
  return content;
}

function decodeBase64(value) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error('INVALID_MEDIA: media payload is empty.');
  }
  let binary;
  try {
    binary = atob(value);
  } catch {
    throw new Error('INVALID_MEDIA: media payload is not valid base64.');
  }
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function debugLog(enabled, message) {
  if (enabled) console.debug(`[turbojev llama.cpp wasm] ${message}`);
}
