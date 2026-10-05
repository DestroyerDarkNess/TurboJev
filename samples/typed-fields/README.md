# Typed-field workflow

Build the current native samples, then run the same request in each binding:

```powershell
python samples/build.py native
python samples/run.py python --generate --model C:/models/Qwen3-0.6B-Q4_K_M.gguf
python samples/run.py ts --generate --model C:/models/Qwen3-0.6B-Q4_K_M.gguf
python samples/run.py csharp --generate --model C:/models/Qwen3-0.6B-Q4_K_M.gguf
python samples/run.py rust --generate --model C:/models/Qwen3-0.6B-Q4_K_M.gguf
```

The request extracts a number, evaluates a boolean, returns a nullable reference and generates a conditional reminder using the earlier total. All answers come from the model. There are no host-computed extraction answers. The request is in `request.json`; edit its instructions and field options to explore dependencies, conditions and nullable values. Open text/numeric fields generate actual tokens, so expect different latency from finite decisions. The examples explicitly use CPU and a full single-sequence context.
