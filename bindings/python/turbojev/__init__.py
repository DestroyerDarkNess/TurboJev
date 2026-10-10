from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .schema import (
    ArrayField,
    Choice,
    Decision,
    Dice,
    Generation,
    MAX_ARRAY_ITEMS,
    MAX_ENUM_CHOICES,
    SchemaError,
    Usage,
    candidate_softmax,
    compile_json_schema,
    example_schema,
    run_schema,
)
from .benchmark import benchmark_prefix_cache
from .images import encode_image

# Python 3.8+ does not search a package directory for a native extension's DLL
# dependencies unless it is registered explicitly. Wheels place these DLLs next
# to _native.pyd; keep this handle alive for the lifetime of the imported package.
_DLL_DIRECTORY = (
    os.add_dll_directory(str(Path(__file__).resolve().parent)) if os.name == "nt" else None
)

try:
    from ._native import NativeEngine
except Exception as exc:  # pragma: no cover - wheel/build diagnostic
    NativeEngine = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class TurboJevError(RuntimeError):
    def __init__(self, message: str):
        text, separator, usage_json = message.partition("\nTURBOJEV_USAGE: ")
        super().__init__(text)
        self.usage = Usage(**json.loads(usage_json)) if separator else None


class GenerationCancelled(TurboJevError):
    """Generation was cancelled cooperatively between model operations."""


class GenerationTimeout(TurboJevError, TimeoutError):
    """The deadline expired. An in-flight model operation must finish first."""


class GenerationControl:
    """Thread-safe cancellation and a timeout in seconds, measured from construction."""

    def __init__(self, timeout: float | None = None):
        from ._native import NativeGenerationControl

        self._native = NativeGenerationControl(timeout)

    def cancel(self) -> None:
        self._native.cancel()


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
    def __init__(self, native: Any):
        self._native = native

    @classmethod
    def load(cls, model: str | LoadOptions, **kwargs: Any) -> "TurboJev":
        if NativeEngine is None:
            raise TurboJevError(f"TurboJev native extension is unavailable: {_IMPORT_ERROR}")
        options = model if isinstance(model, LoadOptions) else LoadOptions(model=model, **kwargs)
        try:
            return cls(NativeEngine.load_json(json.dumps(options.to_wire(), ensure_ascii=False)))
        except Exception as exc:
            message = str(exc)
            if message.startswith("GENERATION_CANCELLED:"):
                raise GenerationCancelled(message) from exc
            if message.startswith("GENERATION_TIMEOUT:"):
                raise GenerationTimeout(message) from exc
            raise TurboJevError(message) from exc

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(self._call("evaluate_json", json.dumps(request, ensure_ascii=False)))

    def generate(
        self,
        request: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        control: GenerationControl | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate JSON-typed fields with enums, dependencies, and conditions.

        Pass a request mapping or use context=..., questions=... / schema=... .
        The native Rust core owns all validation and workflow semantics.
        """
        if request is not None and kwargs:
            raise TypeError("pass a request mapping or keyword arguments, not both")
        payload = dict(request if request is not None else kwargs)
        if "images" in payload:
            payload["images"] = [encode_image(image) for image in payload["images"]]
        wire = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if timeout is not None and control is not None:
            raise TypeError("pass timeout or control, not both")
        if timeout is not None or control is not None:
            active = control if control is not None else GenerationControl(timeout)
            response = json.loads(self._call("generate_json_controlled", wire, active._native))
        else:
            response = json.loads(self._call("generate_json", wire))
        if payload.get("print_final_prompt"):
            for prompt in response.get("prompts", ()):
                print(prompt)
        return response

    def compile_schema(self, schema: Mapping[str, Any]) -> list[Decision | ArrayField]:
        from .schema import compile_json_schema

        return compile_json_schema(schema)

    def generate_typed(self, request: Mapping[str, Any] | None = None, **kwargs: Any) -> Generation:
        from .schema import Generation

        return Generation.from_response(self.generate(request, **kwargs))

    def evaluate_compatible(self, profile: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(
            self._call("evaluate_compatible_json", profile, json.dumps(request, ensure_ascii=False))
        )

    def info(self) -> dict[str, Any]:
        return json.loads(self._call("info_json"))

    def close(self) -> None:
        if self._native is not None:
            self._native.close()
            self._native = None

    def _call(self, name: str, *args: Any) -> str:
        if self._native is None:
            raise TurboJevError("engine is closed")
        try:
            return getattr(self._native, name)(*args)
        except Exception as exc:
            message = str(exc)
            if message.startswith("GENERATION_CANCELLED:"):
                raise GenerationCancelled(message) from exc
            if message.startswith("GENERATION_TIMEOUT:"):
                raise GenerationTimeout(message) from exc
            raise TurboJevError(message) from exc

    def __enter__(self) -> "TurboJev":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "TurboJev",
    "TurboJevError",
    "LoadOptions",
    "GenerationControl",
    "GenerationCancelled",
    "GenerationTimeout",
]


__all__ += [
    "ArrayField",
    "Choice",
    "Decision",
    "Dice",
    "Generation",
    "MAX_ARRAY_ITEMS",
    "MAX_ENUM_CHOICES",
    "SchemaError",
    "Usage",
    "candidate_softmax",
    "compile_json_schema",
    "example_schema",
    "run_schema",
]


__all__ += ["benchmark_prefix_cache", "encode_image"]
