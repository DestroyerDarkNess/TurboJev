import { env, pipeline, RawImage, AutoProcessor, AutoModelForVision2Seq } from '@huggingface/transformers';
import { createRuntime as createTextRuntime } from '../../bindings/web-runtimes/transformersjs/src/index';

env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';
env.backends.onnx.wasm.numThreads = 1;

// The site decodes browser media; this adapter returns raw model logits to Rust.
export async function createRuntime() {
  let runtime: any, classifier: any, options: any, media: any, processor: any, visionModel: any;
  const dispose = (value: any, seen = new Set<any>()) => {
    if (!value || typeof value !== 'object' || seen.has(value)) return;
    seen.add(value);
    if (typeof value.dispose === 'function') value.dispose();
    else for (const child of Object.values(value)) dispose(child, seen);
  };
  return {
    async initialize(input: any) {
      options = input;
      if (input.modality === 'text') {
        runtime = await createTextRuntime();
        return runtime.initialize(input);
      }
      if (input.family === 'vlm') {
        processor = await AutoProcessor.from_pretrained(input.model, { progress_callback: input.progressCallback });
        visionModel = await AutoModelForVision2Seq.from_pretrained(input.model, {
          device: input.device, dtype: { embed_tokens: 'q4', vision_encoder: 'fp32', decoder_model_merged: 'q4' }, progress_callback: input.progressCallback,
        });
      } else classifier = await pipeline(input.modality === 'audio' ? 'zero-shot-audio-classification' : 'zero-shot-image-classification', input.model, {
        device: input.device, dtype: input.dtype, progress_callback: input.progressCallback,
      });
      return { name: 'transformersjs-media', model: input.model, codecProfile: 'media-logits-v1', scoringMode: 'decisions',
        modalities: input.modality === 'audio' ? ['audio'] : ['image', 'video'],
        capabilities: { nextTokenLogits: false, sequenceScores: false, sharedPrefix: false,
          parallelSuffixes: false, selectiveOutputHead: false, localInference: true } };
    },
    setMedia(value: any) { media = value; },
    encodePrompt(value: string) { return runtime.encodePrompt(value); },
    encodePromptWithAnswer(value: string, answer: string) { return runtime.encodePromptWithAnswer(value, answer); },
    decodeTokens(value: string) { return runtime.decodeTokens(value); },
    scoreRuntimeBatch(batch: string, plan: string) { return runtime.scoreRuntimeBatch(batch, plan); },
    async scoreDecisionBatch(json: string) {
      const batch = JSON.parse(json);
      if (!media) throw new Error('Select media before evaluating.');
      const scores = [];
      for (const task of batch.tasks) {
        const labels = task.candidates.map((c: any) => c.description || c.key);
        if (visionModel) {
          // Codes are labels only: token IDs are discovered with this model's tokenizer.
          const codes = labels.map((_: string, i: number) => String.fromCharCode(65 + i));
          const question = `${task.instruction || 'Select the best description.'}\n${labels.map((s: string, i: number) => `${codes[i]}: ${s}`).join('\n')}\nAnswer with only the letter.`;
          const totals = labels.map(() => 0);
          for (const frame of media.frames) {
            const prompt = processor.apply_chat_template([{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: question }] }], { add_generation_prompt: true });
            const prefix = Array.from(processor.tokenizer(prompt, { add_special_tokens: false }).input_ids.data, Number);
            const ids = codes.map((code: string) => {
              const encoded = Array.from(processor.tokenizer(prompt + code, { add_special_tokens: false }).input_ids.data, Number);
              if (encoded.length !== prefix.length + 1 || prefix.some((id, i) => encoded[i] !== id)) throw new Error('This model cannot encode these answer codes as single tokens.');
              return encoded.at(-1)!;
            });
            // The example prioritizes interactive latency over patch-level detail.
            const inputs = await processor(prompt, await RawImage.read(frame), { do_image_splitting: false });
            let output;
            try {
              output = await visionModel.forward(inputs);
              const logits = output.logits, vocab = logits.dims.at(-1), base = logits.data.length - vocab;
              ids.forEach((id: number, i: number) => { totals[i] += Number(logits.data[base + id]) / media.frames.length; });
            } finally { dispose(inputs); dispose(output); }
          }
          scores.push({ id: task.id, logits: totals });
          continue;
        }
        const texts = labels.map((label: string) => options.modality === 'audio' ? `This is a sound of ${label}.` : `This is a photo of ${label}.`);
        const tokens = classifier.tokenizer(texts, { padding: true, truncation: true });
        const totals = labels.map(() => 0);
        const frames = options.modality === 'audio' ? [media.audio] : media.frames;
        try {
          for (const frame of frames) {
            const inputs = options.modality === 'audio'
              ? await classifier.processor(frame)
              : await classifier.processor(await RawImage.read(frame));
            let output;
            try {
              output = await classifier.model({ ...tokens, ...inputs });
              const logits = output[options.modality === 'audio' ? 'logits_per_audio' : 'logits_per_image'];
              labels.forEach((_: string, i: number) => { totals[i] += Number(logits.data[i]) / frames.length; });
            } finally {
              for (const value of Object.values(inputs) as any[]) value?.dispose?.();
              for (const value of Object.values(output ?? {}) as any[]) value?.dispose?.();
            }
          }
        } finally { for (const value of Object.values(tokens) as any[]) value?.dispose?.(); }
        scores.push({ id: task.id, logits: totals });
      }
      return JSON.stringify({ model: options.model, scores, metrics: {} });
    },
    async close() { media = null; await runtime?.close(); await classifier?.dispose(); await visionModel?.dispose(); },
  };
}
