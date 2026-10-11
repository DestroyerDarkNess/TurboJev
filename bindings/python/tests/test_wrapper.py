import json, sys, types


class FakeNative:
    @staticmethod
    def load_json(options):
        return FakeNative()

    def evaluate_json(self, request):
        return json.dumps({"answers": {"ok": {"type": "noul", "noul": 0.8}}})

    def evaluate_compatible_json(self, profile, request):
        return json.dumps({"profile": profile})

    def info_json(self):
        return json.dumps({"runtime": "fixture"})

    def generate_json(self, request):
        self.request = json.loads(request)
        return json.dumps(
            {
                "result": {"roll": "six"},
                "thinking": {},
                "thinking_effort": {},
                "skipped": [],
                "usage": {"decisions": 1, "generated_tokens": 0},
                "prompts": ["<actual-runtime-prompt>"]
                if self.request.get("print_final_prompt")
                else [],
            }
        )

    def close(self):
        pass


fake = types.ModuleType("turbojev._native")
fake.NativeEngine = FakeNative
sys.modules["turbojev._native"] = fake
from turbojev import TurboJev
from turbojev import LoadOptions


def test_public_annotations_resolve_on_oldest_supported_python():
    from typing import get_type_hints

    from turbojev import benchmark_prefix_cache, compile_json_schema
    from turbojev.schema import ArrayField, Choice, Decision, Usage

    assert get_type_hints(LoadOptions)["model"] is str
    for public_type in (ArrayField, Choice, Decision, Usage):
        assert get_type_hints(public_type)
    for operation in (TurboJev.generate, compile_json_schema, benchmark_prefix_cache):
        assert get_type_hints(operation)


def test_worker_configuration_reaches_native_load():
    worker = {"program": "python", "args": ["scorer.py"]}
    assert (
        LoadOptions(model="example/multimodal-model", worker=worker).to_wire()["worker"] == worker
    )


def test_mmproj_configuration_reaches_native_load():
    assert (
        LoadOptions(model="vision.gguf", mmproj="vision-mmproj.gguf").to_wire()["mmproj"]
        == "vision-mmproj.gguf"
    )


def test_order_robustness_configuration_reaches_native_load():
    execution = {"orderRobustness": "cyclic"}
    assert (
        LoadOptions(model="fixture://test", execution=execution).to_wire()["execution"] == execution
    )


def test_default_execution_is_omitted_for_native_serde_defaults():
    assert "execution" not in LoadOptions(model="fixture://test").to_wire()


def test_cpu_is_the_default_and_gpu_is_explicit():
    assert LoadOptions(model="fixture://test").to_wire()["nGpuLayers"] == 0
    assert LoadOptions(model="fixture://test", device="gpu").to_wire()["device"] == "gpu"


def test_wrapper_roundtrip():
    jev = TurboJev.load("fixture://test")
    response = jev.evaluate(
        {
            "state": {"message": "x"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
        }
    )
    assert "answers" in response
    assert jev.info()["runtime"] == "fixture"


def test_thinking_defaults_off_and_is_forwarded_explicitly():
    assert LoadOptions(model="qwen.gguf").to_wire()["thinking"] is False
    wire = LoadOptions(model="qwen.gguf", thinking=True, max_thinking_tokens=256).to_wire()
    assert wire["thinking"] is True
    assert wire["maxThinkingTokens"] == 256


def test_dice_is_a_schema_builder_not_a_scorer():
    from turbojev import Dice

    schema = Dice(permutations=8).to_schema()
    assert schema["enum"] == ["one", "two", "three", "four", "five", "six"]
    assert schema["permutations"] == 8
    assert schema["return_probabilities"] is True


def test_sampling_and_numeric_options_reach_the_portable_core():
    jev = TurboJev.load("fixture://test")
    jev.generate(
        context="a die",
        questions={"n": {"type": "integer"}},
        mode="sample",
        temperature=0.7,
        numeric_max_digits=5,
        seed=42,
    )
    assert jev._native.request["numeric_max_digits"] == 5
    assert jev._native.request["mode"] == "sample"
    assert jev._native.request["temperature"] == 0.7
    assert jev._native.request["seed"] == 42


def test_runtime_prompts_are_printed_and_retained(capsys):
    jev = TurboJev.load("fixture://test")
    result = jev.generate_typed(
        context="x", questions={"n": {"type": "integer"}}, print_final_prompt=True
    )
    assert result.prompts == ("<actual-runtime-prompt>",)
    assert capsys.readouterr().out == "<actual-runtime-prompt>\n"


def test_error_usage_is_typed_and_missing_counts_stay_unknown():
    from turbojev import GenerationCancelled, TurboJevError

    error = GenerationCancelled(
        'GENERATION_CANCELLED: cancelled\nTURBOJEV_USAGE: {"decisions":1,"generated_tokens":2}'
    )
    assert error.usage.decisions == 1
    assert error.usage.generated_tokens == 2
    assert error.usage.input_tokens is None
    assert str(error) == "GENERATION_CANCELLED: cancelled"
    assert TurboJevError("failed").usage is None
