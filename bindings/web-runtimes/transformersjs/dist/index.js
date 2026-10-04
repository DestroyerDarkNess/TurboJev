export async function createRuntime() {
    let model, tokenizer, Tensor, options;
    return {
        async initialize(input) { if (input.thinking)
            throw new Error("THINKING_UNSUPPORTED: this adapter supports direct logits only; use the native GGUF text runtime for thinking"); options = input; const t = await import("@huggingface/transformers"); Tensor = t.Tensor; tokenizer = await t.AutoTokenizer.from_pretrained(input.model, { progress_callback: input.progressCallback }); model = await t.AutoModelForCausalLM.from_pretrained(input.model, { device: input.device ?? "webgpu", dtype: input.dtype ?? "q4", progress_callback: input.progressCallback }); return { name: "transformersjs", model: input.model, codecProfile: "transformersjs-json-prefill-v3", capabilities: { nextTokenLogits: true, sequenceScores: false, sharedPrefix: false, parallelSuffixes: false, selectiveOutputHead: false, localInference: true } }; },
        encodePrompt(messagesJson) { const m = JSON.parse(messagesJson); const messages = [...(m.system ? [{ role: "system", content: m.system }] : []), { role: "user", content: m.user }]; const text = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true, enable_thinking: false }) + (m.assistant_prefix ?? ""); const enc = tokenizer(text, { add_special_tokens: false }); return JSON.stringify(Array.from(enc.input_ids.data, Number)); },
        encodePromptWithAnswer(messagesJson, answer) { const m = JSON.parse(messagesJson); const messages = [...(m.system ? [{ role: "system", content: m.system }] : []), { role: "user", content: m.user }]; const text = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true, enable_thinking: false }) + (m.assistant_prefix ?? "") + answer; const enc = tokenizer(text, { add_special_tokens: false }); return JSON.stringify(Array.from(enc.input_ids.data, Number)); },
        decodeTokens(tokenIdsJson) { return tokenizer.decode(JSON.parse(tokenIdsJson), { skip_special_tokens: false, clean_up_tokenization_spaces: false }); },
        async scoreRuntimeBatch(batchJson, _planJson) { const batch = JSON.parse(batchJson); const scores = []; let inputTokens = 0; for (const task of batch.tasks) {
            const ids = task.promptTokens;
            inputTokens += ids.length;
            const input_ids = new Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
            const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]);
            let output;
            try {
                const one = new Tensor("int64", BigInt64Array.from([1n]), []);
                try {
                    output = await model.forward({ input_ids, attention_mask, num_logits_to_keep: one });
                }
                catch {
                    output = await model.forward({ input_ids, attention_mask });
                }
                const logits = output.logits;
                const vocab = logits.dims.at(-1);
                const rows = logits.dims.at(-2);
                const base = (rows - 1) * vocab;
                scores.push({ id: task.id, logits: task.candidates.map((c) => Number(logits.data[base + c.tokenIds[0]])) });
                logits.dispose?.();
            }
            finally {
                input_ids.dispose?.();
                attention_mask.dispose?.();
            }
        } return JSON.stringify({ model: options.model, scores, metrics: { input_tokens: inputTokens } }); },
        async close() { await model?.dispose?.(); model = undefined; tokenizer = undefined; }
    };
}
