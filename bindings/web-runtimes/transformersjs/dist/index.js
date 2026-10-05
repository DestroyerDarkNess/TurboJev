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
        async inferField(taskJson) {
            const task = JSON.parse(taskJson);
            if (task.thinking)
                throw new Error("THINKING_UNSUPPORTED: this Transformers.js adapter supports direct scoring and bounded values only");
            if (task.values)
                throw new Error("Finite fields must use the Rust token-scoring path");
            if (task.state?.contract === "turbojev.media.v1")
                throw new Error("UNSUPPORTED_MODALITY: use a multimodal runtime");
            const state = typeof task.state === "string" ? task.state : JSON.stringify(task.state);
            const messages = [{ role: "system", content: "Answer the requested field from the supplied evidence. Return only the requested JSON object." }, { role: "user", content: `State:\n${state}\n\nQuestion:\n${task.instructions}\nReturn exactly one JSON object with key value. Value schema: ${JSON.stringify(task.output_schema)}` }];
            const text = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true, enable_thinking: false });
            const inputs = tokenizer(text, { add_special_tokens: false });
            let output;
            try {
                output = await model.generate({ ...inputs, max_new_tokens: task.max_tokens, do_sample: false });
                const sequences = output.sequences ?? output;
                const ids = Array.from(sequences.data, Number);
                const answerIds = ids.slice(inputs.input_ids.data.length);
                // This adapter validates bounded generation after decoding. GGUF runtimes
                // additionally constrain decoding with grammar. Invalid output is an error.
                const decoded = tokenizer.decode(answerIds, { skip_special_tokens: true });
                const answer = JSON.parse(decoded);
                if (Object.keys(answer).length !== 1 || !Object.hasOwn(answer, "value"))
                    throw new Error("Invalid generated field object");
                return JSON.stringify({ value: answer.value, logits: [], thinking: "", input_tokens: inputs.input_ids.data.length, generated_tokens: answerIds.length });
            }
            finally {
                inputs.input_ids.dispose?.();
                inputs.attention_mask?.dispose?.();
                (output?.sequences ?? output)?.dispose?.();
            }
        },
        async close() { await model?.dispose?.(); model = undefined; tokenizer = undefined; }
    };
}
