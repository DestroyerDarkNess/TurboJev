import json, sys, types

class FakeNative:
    @staticmethod
    def load_json(options): return FakeNative()
    def evaluate_json(self, request): return json.dumps({"answers": {"ok": {"type":"noul","noul":0.8}}})
    def evaluate_compatible_json(self, profile, request): return json.dumps({"profile": profile})
    def classify_json(self, state, options): return json.dumps({"choice": json.loads(options)[-1], "probabilities": {}})
    def info_json(self): return json.dumps({"runtime":"fixture"})
    def close(self): pass

fake = types.ModuleType("turbojev._native"); fake.NativeEngine = FakeNative; sys.modules["turbojev._native"] = fake
from turbojev import TurboJev
from turbojev import LoadOptions

def test_worker_configuration_reaches_native_load():
    worker = {"program": "python", "args": ["scorer.py"]}
    assert LoadOptions(model="example/multimodal-model", worker=worker).to_wire()["worker"] == worker

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
    assert jev.classify("x", ["a", "b"])["choice"] == "b"
    assert jev.info()["runtime"] == "fixture"
