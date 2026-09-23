from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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
    n_ctx: int = 4096
    n_batch: int = 512
    n_seq_max: int = 8
    n_gpu_layers: int = -1
    worker: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "nCtx": self.n_ctx,
            "nBatch": self.n_batch,
            "nSeqMax": self.n_seq_max,
            "nGpuLayers": self.n_gpu_layers,
            "worker": self.worker,
            "execution": self.execution,
        }

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

    def evaluate_compatible(self, profile: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(self._call("evaluate_compatible_json", profile, json.dumps(request, ensure_ascii=False)))

    def classify(self, state: str, options: Iterable[str]) -> dict[str, Any]:
        return json.loads(self._call("classify_json", state, json.dumps(list(options), ensure_ascii=False)))

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
