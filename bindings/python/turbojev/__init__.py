from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

# Python 3.8+ does not search a package directory for a native extension's DLL
# dependencies unless it is registered explicitly. Wheels place these DLLs next
# to _native.pyd; keep this handle alive for the lifetime of the imported package.
_DLL_DIRECTORY = os.add_dll_directory(str(Path(__file__).resolve().parent)) if os.name == "nt" else None

try:
    from ._native import NativeEngine
except Exception as exc:  # pragma: no cover - wheel/build diagnostic
    NativeEngine = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

class TurboJevError(RuntimeError):
    pass

@dataclass(frozen=True)
class LoadOptions:
    model: str
    mmproj: str | None = None
    n_ctx: int = 4096
    n_batch: int = 512
    n_seq_max: int = 8
    n_gpu_layers: int = 0
    n_threads: int | None = None
    n_threads_batch: int | None = None
    device: Literal["cpu", "gpu"] | None = None
    worker: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None
    thinking: bool = False
    max_thinking_tokens: int = 512

    def to_wire(self) -> dict[str, Any]:
        options = {
            "model": self.model,
            "thinking": self.thinking,
            "maxThinkingTokens": self.max_thinking_tokens,
            "nCtx": self.n_ctx,
            "nBatch": self.n_batch,
            "nSeqMax": self.n_seq_max,
            "nGpuLayers": self.n_gpu_layers,
            "worker": self.worker,
        }
        if self.mmproj is not None:
            options["mmproj"] = self.mmproj
        if self.n_threads is not None:
            options["nThreads"] = self.n_threads
        if self.n_threads_batch is not None:
            options["nThreadsBatch"] = self.n_threads_batch
        if self.device is not None:
            options["device"] = self.device
        if self.execution is not None:
            options["execution"] = self.execution
        return options

class TurboJev:
    def __init__(self, native: Any): self._native = native

    @classmethod
    def load(cls, model: str | LoadOptions, **kwargs: Any) -> "TurboJev":
        if NativeEngine is None:
            raise TurboJevError(f"TurboJev native extension is unavailable: {_IMPORT_ERROR}")
        options = model if isinstance(model, LoadOptions) else LoadOptions(model=model, **kwargs)
        try:
            return cls(NativeEngine.load_json(json.dumps(options.to_wire(), ensure_ascii=False)))
        except Exception as exc: raise TurboJevError(str(exc)) from exc

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(self._call("evaluate_json", json.dumps(request, ensure_ascii=False)))

    def generate(self, request: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Generate JSON-typed fields with enums, dependencies, and conditions.

        Pass a request mapping or use context=..., questions=... / schema=... .
        The native Rust core owns all validation and workflow semantics.
        """
        if request is not None and kwargs:
            raise TypeError("pass a request mapping or keyword arguments, not both")
        return json.loads(self._call("generate_json", json.dumps(request if request is not None else kwargs, ensure_ascii=False, allow_nan=False)))

    def evaluate_compatible(self, profile: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(self._call("evaluate_compatible_json", profile, json.dumps(request, ensure_ascii=False)))

    def info(self) -> dict[str, Any]: return json.loads(self._call("info_json"))

    def close(self) -> None:
        if self._native is not None:
            self._native.close(); self._native = None

    def _call(self, name: str, *args: Any) -> str:
        if self._native is None: raise TurboJevError("engine is closed")
        try: return getattr(self._native, name)(*args)
        except Exception as exc: raise TurboJevError(str(exc)) from exc

    def __enter__(self) -> "TurboJev": return self
    def __exit__(self, exc_type, exc, tb) -> None: self.close()

__all__ = ["TurboJev", "TurboJevError", "LoadOptions"]
