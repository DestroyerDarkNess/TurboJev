# Model reasoning and the Snake audit

Date: 2026-10-04. These are source-built Windows CPU measurements, not measurements of the currently published packages.

## What was identified

The earlier adapter scored option letters at an unconstrained assistant boundary. The JSON answer prefill corrects that scoring boundary. This does not turn an ordinary language model into a trained planning policy.

The Snake physics were checked separately: each response selects one of four actual directions, that exact direction is applied, and a wall, body collision or forbidden reversal ends the game. No fallback move, action filtering or host path solver is used. The integrity tests check these properties; they do not measure model quality.

All five simpler tasks passed all four rotations with the models below. The original Snake wall state remained difficult:

| Model / quantization | Simpler decisions | Snake wall, four orders |
| --- | ---: | ---: |
| Qwen3-0.6B Q4_K_M | 20/20 | 0/4 |
| Qwen3-0.6B Q8_0 | 20/20 | 0/4 |
| Qwen3.5-0.8B Q4_K_M | 20/20 | 1/4 |
| Qwen3.5-2B Q4_K_M | 20/20 | 1/4 |
| LFM2-350M Q4_K_M | 20/20 | 1/4 |

A literal ASCII board improved Qwen3.5-2B's wall result to 2/4, but its seed-0 game still ate no food and hit a wall after six decisions. It is not a qualified navigation solution. A separate observation experiment exposed relative food position and adjacent-cell contents; Qwen0.6B still failed 0/4 and the game. That experimental preprocessing was removed from the demo. Its complete requests and negative results remain in the report, explicitly as an assisted perception experiment rather than raw-coordinate reasoning.

This evidence qualifies neither these models nor these prompts for reliable autonomous navigation. It also does not establish that all remaining failures are solely model capacity: input representation, instruction following and positional sensitivity interact. Replacing quantization or increasing parameter count alone did not solve this scenario.

## Why Valen behaves differently

[Valen-Preview-0923's official model card](https://huggingface.co/Valen-Team/Valen-Preview-0923) describes a Sokoban-trained checkpoint built on Qwen3.5-2B. It contains a trained decision head, language-model LoRA, visual merger and final four visual blocks. Training starts with a general decision head and continues on 30,000 Sokoban records for three epochs. It is not unchanged Qwen3.5-2B with a different API wrapper.

Its own reported results are 87.60% on single-step Sokoban questions and 38/100 complete games on a model-outcome-conditioned set. These are different tasks and evaluation conditions from this Snake audit, not a controlled TurboJev comparison. We reviewed Valen's code and card; we did not run its checkpoint here.

The local implementation makes the distinction concrete:

- `other/Valen/valen/modeling/qwen/heads.py`: learned projections combine candidate and decision hidden states; this replaces the ordinary next-token answer head.
- `other/Valen/evaluation/sokoban/dataset.py`: an offline solver supplies optimal-action training targets and candidate orders are shuffled.
- `other/Valen/evaluation/sokoban/evaluate.py`: `ValenPolicy.choose` receives pixels and rules, without reference actions or a legal-action mask. A separate privileged reference policy is excluded from model performance.

Copying an untrained scoring layer would not copy these learned weights or results. An ordinary GGUF decision engine continues to depend on its selected model and prompt. A specialized checkpoint can improve the policy, but the existing Valen checkpoint is not a drop-in GGUF.

## Optional native thinking

`thinking` is a boolean load option, default `false`. With `true`, the native text adapter uses the model's real Jinja reasoning template, generates reasoning until its declared closing delimiter, then verifies and scores candidate codes at the final JSON-answer prefill. The selected answer still comes from candidate logits. No host-generated thought or chosen direction is inserted.

`maxThinkingTokens` defaults to 512. An unfinished budget raises `THINKING_BUDGET_EXHAUSTED`; missing reasoning support raises `THINKING_UNSUPPORTED`. Context capacity includes the prompt, thoughts and answer prefill. Increase `nCtx` or reduce `nSeqMax` when necessary. The host caps thinking sessions at four sequences, and disables shared-prefix/parallel/fused scoring. Direct scoring retains its existing defaults.

The result's `usage.generated_tokens` reports actual generated reasoning, including totals across questions and cyclic evaluations. Direct native scoring remains zero. Native shim ABI 9 is required; rebuild its dependents together.

Python:

```python
with TurboJev.load(model, thinking=True, max_thinking_tokens=512,
                  n_ctx=4096, n_seq_max=4) as engine:
    response = engine.evaluate(request)
```

Node / TypeScript:

```ts
const engine = await TurboJev.load({model, thinking: true, maxThinkingTokens: 512,
                                  nCtx: 4096, nSeqMax: 4});
const response = await engine.evaluate(request);
```

C# / .NET:

```csharp
await using var engine = await TurboJevEngine.LoadAsync(new {
    model, thinking = true, maxThinkingTokens = 512, nCtx = 4096, nSeqMax = 4
});
var response = await engine.EvaluateJsonAsync(System.Text.Json.JsonSerializer.Serialize(request));
```

C, Android and Swift use the same load JSON fields. Rust callers can use `FfiLlamaCpp::open(model, config)?.with_thinking(true, 512)` with appropriate context/sequence settings. CPU/GPU selection is unchanged.

These APIs are introduced in 0.29.2. Worker-based scorers and native media currently reject this option. The browser adapters also reject `thinking: true` explicitly; they continue to support direct scores. Browser thinking is not implemented or qualified here.

Real verification: Qwen0.6B completed a color decision after 123 generated tokens. Qwen3.5-2B failed to finish the Snake thought within 512 tokens, returning a budget error instead of an invented action. Thinking is an optional inference mode, not a correctness guarantee. Timing reports are diagnostic; runs overlapped compilation/inference and must not be advertised as a controlled speed comparison.

## Evidence and reproduction

- [Qwen3.5-0.8B](validation/decision-order-Qwen3.5-0.8B.json)
- [Qwen3.5-2B](validation/decision-order-Qwen3.5-2B.json)
- [LFM2-350M](validation/decision-order-LFM2-350M.json)
- [ASCII board and full-game trace](validation/snake-grid-qwen35-2b.json)
- [Failed observation experiment](validation/snake-observations-qwen06.json)
- [Completed model thinking with the final native build](validation/decision-thinking-qwen-color-final.json)
- [Snake thinking budget error](validation/decision-thinking-qwen35-2b.json)

`scripts/validate_decision_order.py` takes `--thinking`, `--max-thinking-tokens`, `--case`, `--rotations` and native library/model paths. `scripts/validate_snake_decisions.py` uses the demo's actual serializer and game physics; `--demo` selects the checked-out demo file, `--formats coordinates grid` compares representations and `--game-turns 25` runs a complete trajectory. It isolates the source-built native library from the installed package's older DLL, and labels that inference origin in its report. The normal demo continues to import its registry package.
