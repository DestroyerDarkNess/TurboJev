export interface TransformersRuntimeOptions {
  model: string;
  device?: string;
  dtype?: string;
  thinking?: boolean;
  progressCallback?: (p: unknown) => void;
}
export async function createRuntime(dependencies: { loadTransformers?: () => Promise<any> } = {}) {
  let model: any, tokenizer: any, Tensor: any, options: TransformersRuntimeOptions;
  return {
    async initialize(input: TransformersRuntimeOptions) {
      if (input.thinking)
        throw new Error(
          "THINKING_UNSUPPORTED: this adapter supports direct logits only; use the native GGUF text runtime for thinking",
        );
      options = input;
      const t = await (dependencies.loadTransformers?.() ?? import("@huggingface/transformers"));
      Tensor = t.Tensor;
      tokenizer = await t.AutoTokenizer.from_pretrained(input.model, {
        progress_callback: input.progressCallback,
      });
      model = await t.AutoModelForCausalLM.from_pretrained(input.model, {
        device: input.device ?? "wasm",
        dtype: input.dtype ?? "q4",
        progress_callback: input.progressCallback,
      });
      return {
        name: "transformersjs",
        model: input.model,
        codecProfile: "transformersjs-json-prefill-v3",
        capabilities: {
          nextTokenLogits: true,
          sequenceScores: true,
          sharedPrefix: false,
          parallelSuffixes: false,
          selectiveOutputHead: false,
          localInference: true,
        },
      };
    },
    encodePrompt(messagesJson: string) {
      const m = JSON.parse(messagesJson);
      const messages = [
        ...(m.system ? [{ role: "system", content: m.system }] : []),
        { role: "user", content: m.user },
      ];
      const text =
        tokenizer.apply_chat_template(messages, {
          tokenize: false,
          add_generation_prompt: true,
          enable_thinking: false,
        }) + (m.assistant_prefix ?? "");
      const enc = tokenizer(text, { add_special_tokens: false });
      return JSON.stringify(Array.from(enc.input_ids.data, Number));
    },
    encodePromptWithAnswer(messagesJson: string, answer: string) {
      const m = JSON.parse(messagesJson);
      const messages = [
        ...(m.system ? [{ role: "system", content: m.system }] : []),
        { role: "user", content: m.user },
      ];
      const text =
        tokenizer.apply_chat_template(messages, {
          tokenize: false,
          add_generation_prompt: true,
          enable_thinking: false,
        }) +
        (m.assistant_prefix ?? "") +
        answer;
      const enc = tokenizer(text, { add_special_tokens: false });
      return JSON.stringify(Array.from(enc.input_ids.data, Number));
    },
    decodeTokens(tokenIdsJson: string) {
      return tokenizer.decode(JSON.parse(tokenIdsJson), {
        skip_special_tokens: false,
        clean_up_tokenization_spaces: false,
      });
    },
    async scoreRuntimeBatch(batchJson: string, _planJson: string) {
      const batch = JSON.parse(batchJson);
      if (batch.state?.contract === "turbojev.media.v1") {
        throw new Error("UNSUPPORTED_MODALITY: use a multimodal runtime");
      }
      let inputTokens = 0;
      async function lastLogits(ids: number[]): Promise<number[]> {
        inputTokens += ids.length;
        const input_ids = new Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
        const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [
          1,
          ids.length,
        ]);
        const keep = new Tensor("int64", BigInt64Array.from([1n]), []);
        let output: any;
        try {
          try {
            output = await model.forward({ input_ids, attention_mask, num_logits_to_keep: keep });
          } catch {
            output = await model.forward({ input_ids, attention_mask });
          }
          const logits = output.logits;
          const width = logits.dims.at(-1);
          const base = (logits.dims.at(-2) - 1) * width;
          return Array.from(logits.data.slice(base, base + width), Number);
        } finally {
          input_ids.dispose?.();
          attention_mask.dispose?.();
          keep.dispose?.();
          for (const tensor of Object.values(output ?? {}) as any[]) tensor.dispose?.();
        }
      }
      const scores = [];
      for (const task of batch.tasks) {
        const prompt = task.promptTokens as number[];
        const initial = await lastLogits(prompt);
        const singleToken = task.candidates.every(
          (candidate: any) => candidate.tokenIds.length === 1,
        );
        const logits: number[] = [];
        for (const candidate of task.candidates) {
          if (!candidate.tokenIds.length) throw new Error("Empty candidate sequence");
          if (singleToken) {
            const value = initial[candidate.tokenIds[0]];
            if (!Number.isFinite(value)) throw new Error("Invalid candidate token or logit");
            logits.push(value);
            continue;
          }
          const ids = [...prompt];
          let row = initial;
          let score = 0;
          for (let offset = 0; offset < candidate.tokenIds.length; offset++) {
            if (offset) row = await lastLogits(ids);
            const token = candidate.tokenIds[offset];
            if (!Number.isInteger(token) || token < 0 || token >= row.length) {
              throw new Error("Candidate token is outside the model vocabulary");
            }
            const peak = row.reduce((maximum, value) => Math.max(maximum, value), -Infinity);
            const mass = row.reduce((sum, value) => sum + Math.exp(value - peak), 0);
            score += row[token] - peak - Math.log(mass);
            ids.push(token);
          }
          if (!Number.isFinite(score)) throw new Error("Non-finite candidate sequence score");
          logits.push(score);
        }
        scores.push({ id: task.id, logits });
      }
      return JSON.stringify({
        model: options.model,
        scores,
        metrics: { input_tokens: inputTokens },
      });
    },
    async inferField(taskJson: string) {
      const task = JSON.parse(taskJson);
      if (task.thinking)
        throw new Error(
          "THINKING_UNSUPPORTED: this Transformers.js adapter supports direct scoring and bounded values only",
        );
      if (task.values) throw new Error("Finite fields must use the Rust token-scoring path");
      if (task.state?.contract === "turbojev.media.v1")
        throw new Error("UNSUPPORTED_MODALITY: use a multimodal runtime");
      if (task.temperature > 0)
        throw new Error("SAMPLING_UNSUPPORTED: seeded open-value sampling requires a GGUF runtime");
      const state = typeof task.state === "string" ? task.state : JSON.stringify(task.state);
      const messages = [
        {
          role: "system",
          content:
            "Answer the requested field from the supplied evidence. Return only the requested JSON object.",
        },
        {
          role: "user",
          content: `State:\n${state}\n\nQuestion:\n${task.instructions}\nReturn exactly one JSON object with key value. Value schema: ${JSON.stringify(task.output_schema)}`,
        },
      ];
      const text = tokenizer.apply_chat_template(messages, {
        tokenize: false,
        add_generation_prompt: true,
        enable_thinking: false,
      });
      const inputs = tokenizer(text, { add_special_tokens: false });
      let output: any;
      try {
        output = await model.generate({
          ...inputs,
          max_new_tokens: task.max_tokens,
          do_sample: false,
        });
        const sequences = output.sequences ?? output;
        const ids = Array.from(sequences.data, Number) as number[];
        const answerIds = ids.slice(inputs.input_ids.data.length);
        // This adapter validates bounded generation after decoding. GGUF runtimes
        // additionally constrain decoding with grammar. Invalid output is an error.
        const decoded = tokenizer.decode(answerIds, { skip_special_tokens: true });
        const answer = JSON.parse(decoded);
        if (Object.keys(answer).length !== 1 || !Object.hasOwn(answer, "value"))
          throw new Error("Invalid generated field object");
        return JSON.stringify({
          value: answer.value,
          logits: [],
          thinking: "",
          input_tokens: inputs.input_ids.data.length,
          generated_tokens: answerIds.length,
        });
      } finally {
        inputs.input_ids.dispose?.();
        inputs.attention_mask?.dispose?.();
        (output?.sequences ?? output)?.dispose?.();
      }
    },
    async close() {
      await model?.dispose?.();
      model = undefined;
      tokenizer = undefined;
    },
  };
}
