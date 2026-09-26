const WLLAMA_VERSION = '3.6.1';
const WLLAMA_ENTRY = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm/index.js`;
const WLLAMA_WASM = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm/wasm/wllama.wasm`;

export async function createRuntime() {
  let engine;
  let model = '';

  return {
    async initialize(options) {
      if (options.modality !== 'text') {
        throw new Error('llama.cpp WASM currently supports text GGUF models only.');
      }
      if (!options.modelUrl) {
        throw new Error('Choose a GGUF model with a direct download URL.');
      }

      const { Wllama, LoggerWithoutDebug } = await import(WLLAMA_ENTRY);
      engine = new Wllama({ default: WLLAMA_WASM }, {
        logger: LoggerWithoutDebug,
        suppressNativeLog: true,
        parallelDownloads: 4,
      });
      model = options.modelLabel || options.modelUrl;

      await engine.loadModelFromUrl(options.modelUrl, {
        n_ctx: 2048,
        n_batch: 512,
        n_gpu_layers: options.device === 'webgpu' ? 999 : 0,
        cache_prompt: false,
        progressCallback: ({ loaded = 0, total = 0 }) => {
          options.progressCallback?.({
            status: 'progress',
            file: 'GGUF model',
            loaded,
            total,
            progress: total > 0 ? Math.min(100, loaded / total * 100) : 0,
          });
        },
      });

      const warmup = await engine.createChatCompletion({
        messages: [{ role: 'user', content: 'Reply with one word: ready.' }],
        max_tokens: 1,
        temperature: 0,
        cache_prompt: false,
        chat_template_kwargs: { enable_thinking: false },
      });
      if (!warmup?.choices?.length) throw new Error('llama.cpp WASM loaded the model but warmup returned no completion.');

      return {
        name: 'llama.cpp-wasm',
        model,
        codecProfile: 'llamacpp-wasm-constrained-choice-v1',
        scoringMode: 'decisions',
        modalities: ['text'],
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
        const optionsText = candidates.map((candidate, i) => `${labels[i]}. ${candidate.description || candidate.key}`).join('\n');
        const messages = [
          { role: 'system', content: 'Make the requested decision from the supplied state. Follow the output format exactly.' },
          { role: 'user', content: `State:\n${typeof batch.state === 'string' ? batch.state : JSON.stringify(batch.state)}\n\nQuestion:\n${criterion}\n\nAllowed options:\n${optionsText}\n\nReply with exactly one option letter: ${labels.join(', ')}.` },
        ];
        const grammar = `root ::= ${labels.map(label => `"${label}"`).join(' | ')}`;
        const response = await engine.createChatCompletion({
          messages,
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
