"""Benchmarks report observed timing and negotiated execution, without inventing cache hits."""

from copy import deepcopy
import statistics
import time
from typing import Any, Mapping


def benchmark_prefix_cache(
    model: str,
    request: Mapping[str, Any],
    *,
    repeats: int = 3,
    load_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare independent reference and shared-prefix evaluation of one multi-question batch.

    Uses separate loaded sessions and a warm-up in each. Load time is excluded.
    Requires a runtime that declares and uses shared-prefix reuse. This does not measure
    a server's cross-request radix cache or claim its private cached-token counters.
    """
    from . import TurboJev, TurboJevError

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if len(request.get("questions", {})) < 2:
        raise ValueError("Prefix comparison needs at least two questions")
    result = {}
    answers = {}
    for name, policy in (("reference", "reference"), ("shared_prefix", "deterministic-reuse")):
        options = deepcopy(dict(load_options or {}))
        options["execution"] = {**options.get("execution", {}), "policy": policy}
        with TurboJev.load(model, **options) as client:
            warm = client.evaluate(request)
            execution = warm["_turbojev"]["execution"]
            if name == "shared_prefix" and execution["prefix_reuse"] != "shared":
                raise TurboJevError("Runtime did not use shared-prefix reuse")
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                response = client.evaluate(request)
                times.append(time.perf_counter() - started)
            answers[name] = response["answers"]
            result[name] = {
                "seconds": times,
                "mean_seconds": statistics.fmean(times),
                "execution": execution,
                "answers": response["answers"],
                "input_tokens": response["usage"].get("input_tokens"),
                "cached_tokens": None,
            }
    result["answers_equal"] = answers["reference"] == answers["shared_prefix"]
    # Identical selected labels alone do not qualify an optimization: changed
    # probabilities can change confidence gates and downstream decisions.
    differences = []

    def same_values(reference: Any, shared: Any) -> bool:
        if isinstance(reference, dict) and isinstance(shared, dict):
            if reference.keys() != shared.keys():
                return False
            matches = [same_values(reference[key], shared[key]) for key in reference]
            return all(matches)
        if isinstance(reference, list) and isinstance(shared, list):
            if len(reference) != len(shared):
                return False
            matches = [same_values(left, right) for left, right in zip(reference, shared)]
            return all(matches)
        if type(reference) in (int, float) and type(shared) in (int, float):
            differences.append(abs(reference - shared))
            return True
        return type(reference) is type(shared) and reference == shared

    categorical_equal = same_values(answers["reference"], answers["shared_prefix"])
    maximum_difference = max(differences, default=0.0)
    result["parity"] = {
        "categorical_values_equal": categorical_equal,
        "max_absolute_numeric_difference": maximum_difference,
        "absolute_tolerance": 1e-5,
        "qualified": categorical_equal and maximum_difference <= 1e-5,
    }
    shared = result["shared_prefix"]["mean_seconds"]
    result["speedup"] = result["reference"]["mean_seconds"] / shared if shared else None
    return result
