#!/usr/bin/env python3
"""Offline checks for v0.25 server/runtime fused execution and deployment contracts."""
from __future__ import annotations
import json
import py_compile
from pathlib import Path
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
FIX = ROOT / "tests" / "server"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(schema: str, document: dict):
    errors = list(Draft202012Validator(load(SCHEMAS / schema)).iter_errors(document))
    if errors:
        raise AssertionError("; ".join(error.message for error in errors))


def plan_case(case: dict) -> tuple[str, int]:
    caps = case["capabilities"]
    remaining = case["remainingItems"]
    quantum = case["fairnessQuantum"]
    host_max = case["hostMaxFusedItems"]
    maximum_turn = min(remaining, quantum)
    if case["policy"] == "reference-serial" or not caps["fusedBatch"]:
        return "serial", maximum_turn
    backend_max = caps["maxFusedItems"] if caps["maxFusedItems"] is not None else 2**63 - 1
    fused = min(maximum_turn, host_max, backend_max)
    memory_budget = case.get("hostFusedMemoryBudgetBytes")
    bytes_per_item = caps.get("estimatedFusedItemBytes")
    if memory_budget is not None and bytes_per_item is not None:
        fused = min(fused, memory_budget // bytes_per_item)
    if fused < 2:
        return "serial", 1
    return "fused", fused


def main():
    config = load(FIX / "config-fixture.json")
    reverse_proxy = load(FIX / "reverse-proxy-config.json")
    batch = load(FIX / "batch-canonical.json")
    routes = load(FIX / "routes.json")
    execution = load(FIX / "execution-fixture.json")
    native_caps = load(FIX / "native-execution-capabilities.json")
    metrics = load(FIX / "metrics-fixture.json")
    lifecycle = load(FIX / "lifecycle-fixture.json")
    load_plan = load(FIX / "load-plan.json")
    load_report = load(FIX / "load-report-sample.json")
    benchmark_bundle = load(FIX / "benchmark-bundle.json")
    llama_multi_root = load(FIX / "llamacpp-multi-root.json")

    validate("turbojev.server-config.schema.json", config)
    validate("turbojev.server-config.schema.json", reverse_proxy)
    validate("turbojev.server-batch.schema.json", batch)
    validate("turbojev.server-execution.schema.json", execution)
    validate("turbojev.native-execution-capabilities.schema.json", native_caps)
    validate("turbojev.server-metrics.schema.json", metrics)
    validate("turbojev.server-lifecycle.schema.json", lifecycle)
    validate("turbojev.server-load-plan.schema.json", load_plan)
    validate("turbojev.server-load-report.schema.json", load_report)
    validate("turbojev.server-benchmark-bundle.schema.json", benchmark_bundle)
    validate("turbojev.llamacpp-multi-root.schema.json", llama_multi_root)

    assert config["bind"].startswith("127.")
    assert 0 < len(batch["requests"]) <= config["limits"]["maxBatchItems"]
    assert config["transport"]["fairnessQuantum"] >= 1
    assert config["transport"]["maxFusedItems"] >= 1
    assert config["transport"]["fusedMemoryBudgetBytes"] >= 1
    assert config["transport"]["drainTimeoutMs"] >= 1
    assert config["deployment"]["mode"] in {"loopback", "reverse-proxy"}
    assert reverse_proxy["deployment"] == {"mode": "reverse-proxy", "requireForwardedHttps": True}
    assert reverse_proxy["bind"].startswith("127.")

    assert routes["GET /healthz"] == "health"
    assert routes["GET /readyz"] == "ready"
    assert routes["POST /v1/openrouter/evaluate"] == "turbojev"
    assert routes["POST /v1/vercel/evaluate"] == "ai-sdk-evaluation-v4"
    assert routes["POST /v1/systemone"] == "typesafe-systemone-v1"
    assert routes["GET /v1/server/metrics"] == "metrics"

    native = execution["nativeSessionCapabilities"]
    assert native == metrics["capabilities"]
    assert native == native_caps["fixture"]
    assert native["fusedBatch"] is True
    assert native["maxFusedItems"] == 64
    assert native["cooperativeCancellation"] is False
    assert native["hardDeadlines"] is False
    assert native_caps["llamacpp"]["fusedBatch"] is False
    assert native_caps["llamacpp"]["maxFusedItems"] is None
    assert native_caps["llamacppOptIn"]["fusedBatch"] is True
    assert native_caps["llamacppOptIn"]["maxFusedItems"] == llama_multi_root["limits"]["maxRoots"]
    assert llama_multi_root["shimAbi"] == 5
    assert llama_multi_root["enabledByDefault"] is False
    assert llama_multi_root["realBackendEvidenceRequired"] is True

    for case in execution["planningCases"]:
        mode, turn_items = plan_case(case)
        assert mode == case["expected"]["mode"], case["id"]
        assert turn_items == case["expected"]["turnItems"], case["id"]

    assert lifecycle["probeBehavior"]["ready"] == {"healthz": 200, "readyz": 200}
    assert lifecycle["probeBehavior"]["draining"]["healthz"] == 200
    assert lifecycle["probeBehavior"]["draining"]["readyz"] == 503
    assert lifecycle["probeBehavior"]["draining"]["newInference"] == 503

    assert metrics["lifecycle"] == "ready"
    assert metrics["completedRequests"] <= metrics["submittedRequests"]
    assert metrics["fusedTurns"] >= 1
    assert metrics["callerDeadlines"] >= 1

    assert load_plan["baseUrl"].startswith("http://127.")
    assert load_report["completedRequests"] == load_report["successes"] + load_report["failures"]
    assert load_report["completedRequests"] <= load_report["totalRequests"]
    py_compile.compile(str(ROOT / "scripts" / "server_load_harness.py"), doraise=True)
    py_compile.compile(str(ROOT / "scripts" / "server_benchmark_bundle.py"), doraise=True)

    import hashlib
    for key, file_name in {
        "config": "config-fixture.json",
        "loadPlan": "load-plan.json",
        "loadReport": "load-report-sample.json",
    }.items():
        expected = hashlib.sha256((FIX / file_name).read_bytes()).hexdigest()
        assert benchmark_bundle["artifacts"][key]["sha256"] == expected

    server_source = (ROOT / "crates" / "turbojev-server" / "src" / "lib.rs").read_text(encoding="utf-8")
    host_source = (ROOT / "crates" / "turbojev-host-native" / "src" / "lib.rs").read_text(encoding="utf-8")
    protocol_source = (ROOT / "crates" / "turbojev-server-protocol" / "src" / "lib.rs").read_text(encoding="utf-8")
    llama_source = (ROOT / "crates" / "turbojev-runtime-llamacpp" / "src" / "lib.rs").read_text(encoding="utf-8")
    llama_ffi_source = (ROOT / "crates" / "turbojev-runtime-llamacpp" / "src" / "ffi.rs").read_text(encoding="utf-8")
    llama_shim_source = (ROOT / "crates" / "turbojev-runtime-llamacpp" / "native" / "turbojev_llama_shim.cpp").read_text(encoding="utf-8")
    for snippet in ["ShutdownToken", "SERVER_DRAINING", "X-TurboJev-Request-Id", "FORWARDED_HTTPS_REQUIRED", "drain_timeout_ms"]:
        assert snippet in server_source, snippet
    for snippet in ["NativeExecutionCapabilities", "execution_capabilities", "evaluate_fused", "FusedScoringBackend", "llamacpp_multi_root_fused"]:
        assert snippet in host_source, snippet
    assert '"/readyz"' in protocol_source
    for snippet in ["FusedRuntimeExecutor", "score_multi_root_next_token", "can_score_fused"]:
        assert snippet in llama_source, snippet
    for snippet in ["multi_root_enabled: false", "with_multi_root_enabled", "tj_llama_score_root_batch"]:
        assert snippet in llama_ffi_source, snippet
    for snippet in ["tj_llama_score_root_batch", "llama_n_ctx_seq", "root_output_indices", "llama_get_logits_ith"]:
        assert snippet in llama_shim_source, snippet

    print("OK: v0.25 llama.cpp multi-root contracts + fused server/runtime fixtures PASS")


if __name__ == "__main__":
    main()
