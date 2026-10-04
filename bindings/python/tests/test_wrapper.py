import json, sys, types

class FakeNative:
    @staticmethod
    def load_json(options): return FakeNative()
    def evaluate_json(self, request): return json.dumps({"answers": {"ok": {"type":"noul","noul":0.8}}})
    def evaluate_compatible_json(self, profile, request): return json.dumps({"profile": profile})
    def info_json(self): return json.dumps({"runtime":"fixture"})
    def close(self): pass

fake = types.ModuleType("turbojev._native"); fake.NativeEngine = FakeNative; sys.modules["turbojev._native"] = fake
from turbojev import TurboJev
from turbojev import LoadOptions

def test_worker_configuration_reaches_native_load():
    worker = {"program": "python", "args": ["scorer.py"]}
    assert LoadOptions(model="example/multimodal-model", worker=worker).to_wire()["worker"] == worker

def test_mmproj_configuration_reaches_native_load():
    assert LoadOptions(model="vision.gguf", mmproj="vision-mmproj.gguf").to_wire()["mmproj"] == "vision-mmproj.gguf"

def test_order_robustness_configuration_reaches_native_load():
    execution = {"orderRobustness": "cyclic"}
    assert LoadOptions(model="fixture://test", execution=execution).to_wire()["execution"] == execution

def test_default_execution_is_omitted_for_native_serde_defaults():
    assert "execution" not in LoadOptions(model="fixture://test").to_wire()

def test_cpu_is_the_default_and_gpu_is_explicit():
    assert LoadOptions(model="fixture://test").to_wire()["nGpuLayers"] == 0
    assert LoadOptions(model="fixture://test", device="gpu").to_wire()["device"] == "gpu"

def test_wrapper_roundtrip():
    jev = TurboJev.load("fixture://test")
    response = jev.evaluate({"state": {"message": "x"}, "questions": {"route": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}}})
    assert "answers" in response
    assert jev.info()["runtime"] == "fixture"


def test_thinking_defaults_off_and_is_forwarded_explicitly():
    assert LoadOptions(model="qwen.gguf").to_wire()["thinking"] is False
    wire = LoadOptions(model="qwen.gguf", thinking=True, max_thinking_tokens=256).to_wire()
    assert wire["thinking"] is True
    assert wire["maxThinkingTokens"] == 256
