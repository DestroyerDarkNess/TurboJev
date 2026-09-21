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

def test_wrapper_roundtrip():
    jev = TurboJev.load("fixture://test")
    assert jev.classify("x", ["a", "b"])["choice"] == "b"
    assert jev.info()["runtime"] == "fixture"
