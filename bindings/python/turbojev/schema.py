"""Typed schema and response helpers; the Rust compiler owns validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

MAX_ENUM_CHOICES = 512
MAX_ARRAY_ITEMS = 50


class SchemaError(ValueError):
    """A schema failed validation before inference."""


@dataclass(frozen=True)
class Decision:
    """One tokenizer-independent decision returned by the Rust compiler."""

    name: str
    question: str
    choices: tuple[Any, ...]
    path: tuple[str, ...]
    output_schema: Mapping[str, Any]
    depends_on: tuple[str, ...] = ()
    levels: tuple[str, ...] = ()
    descriptions: tuple[tuple[Any, str], ...] = ()
    permutations: int | str = 1
    return_probabilities: bool = False
    thinking: bool | str = False
    thinking_budget: int | None = None
    dependency_groups: tuple[tuple[str, ...], ...] = ()
    when: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class ArrayField:
    """A bounded array workflow and its compiled item decisions."""

    name: str
    question: str
    items: tuple[Decision, ...]
    item_object: bool
    min_items: int = 0
    max_items: int | None = None
    continue_from: tuple[Any, ...] = ()
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Choice:
    """Convenient described choice field. Control labels remain runtime-owned.

    Example: Choice("Which team?", {"billing": "Payments", "outage": "Service down"}).
    Put ``choice.to_schema()`` in the request's questions map.
    """

    question: str
    choices: Mapping[str, str]
    return_probabilities: bool = False
    permutations: int | str | None = None

    def to_schema(self) -> dict[str, Any]:
        schema = {
            "type": "string",
            "instructions": self.question,
            "choices": [
                {"value": value, "description": description}
                for value, description in self.choices.items()
            ],
            "return_probabilities": self.return_probabilities,
        }
        if self.permutations is not None:
            schema["permutations"] = self.permutations
        return schema


@dataclass(frozen=True)
class Usage:
    """Reported inference counts. Missing token counts stay unknown."""

    decisions: int = 0
    input_tokens: int | None = None
    generated_tokens: int = 0


@dataclass(frozen=True)
class Generation:
    """Attribute access to a canonical generate response; dictionaries still work."""

    result: Mapping[str, Any]
    thinking: Mapping[str, str] = field(default_factory=dict)
    thinking_effort: Mapping[str, str] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_response(cls, response: Mapping[str, Any]) -> Generation:
        return cls(
            result=response["result"],
            thinking=response["thinking"],
            thinking_effort=response["thinking_effort"],
            skipped=tuple(response["skipped"]),
            usage=Usage(**response["usage"]),
        )


def _decision(description: Mapping[str, Any]) -> Decision:
    return Decision(
        name=description["name"],
        question=description["question"],
        choices=tuple(description["choices"] or ()),
        path=tuple(description["path"]),
        output_schema=description["output_schema"],
        depends_on=tuple(description["depends_on"]),
        levels=tuple(description["levels"]),
        descriptions=tuple(tuple(item) for item in description["descriptions"]),
        permutations=description["permutations"],
        return_probabilities=description["return_probabilities"],
        thinking=description["thinking"],
        thinking_budget=description["thinking_budget"],
        dependency_groups=tuple(tuple(group) for group in description["dependency_groups"]),
        when=tuple(tuple(gate) for gate in description["when"]),
    )


def compile_json_schema(schema: Mapping[str, Any]) -> list[Decision | ArrayField]:
    """Compile without loading a model; validation runs exclusively in Rust."""
    from . import NativeEngine, TurboJevError

    if NativeEngine is None:
        raise TurboJevError("TurboJev native extension is unavailable")
    try:
        compiled = json.loads(
            NativeEngine.compile_schema_json(
                json.dumps(schema, ensure_ascii=False, allow_nan=False)
            )
        )
    except Exception as error:
        raise SchemaError(str(error)) from error
    decisions = []
    for description in compiled:
        array = description["array"]
        if array is None:
            decisions.append(_decision(description))
        else:
            decisions.append(
                ArrayField(
                    name=description["name"],
                    question=description["question"],
                    items=tuple(_decision(item) for item in array["items"]),
                    item_object=array["item_object"],
                    min_items=array["minItems"],
                    max_items=array["maxItems"],
                    continue_from=tuple(array["continue_from"]),
                    depends_on=tuple(description["depends_on"]),
                )
            )
    return decisions


def run_schema(
    context: Any = "",
    schema: Mapping[str, Any] | None = None,
    *,
    model: str,
    questions: Mapping[str, Any] | None = None,
    load_options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    control: Any = None,
    **generation_options: Any,
) -> Generation:
    """Load, generate and close for a single call; reuse a client for repeated calls."""
    from . import TurboJev

    request = {"context": context, **generation_options}
    if schema is not None:
        request["schema"] = schema
    if questions is not None:
        request["questions"] = questions
    with TurboJev.load(model, **(load_options or {})) as client:
        return Generation.from_response(client.generate(request, timeout=timeout, control=control))


def candidate_softmax(logits: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Normalize candidate scores in Rust. Probabilities are not calibrated confidence."""
    from . import NativeEngine, TurboJevError

    if NativeEngine is None:
        raise TurboJevError("TurboJev native extension is unavailable")
    wire = json.dumps({"logits": dict(logits), "temperature": temperature}, allow_nan=False)
    try:
        return json.loads(NativeEngine.candidate_softmax_json(wire))
    except Exception as error:
        raise ValueError(str(error)) from error


def example_schema() -> dict[str, Any]:
    """A small customer-support schema using described choices and ordered levels."""
    return {
        "type": "object",
        "properties": {
            "urgent": {
                "type": "boolean",
                "instructions": "Does this need a reply within the hour?",
            },
            "team": Choice(
                "Which team should handle it?",
                {
                    "outage": "Service unavailable",
                    "billing": "Charges and refunds",
                    "feature": "Requests and how-to",
                },
            ).to_schema(),
            "tone": {
                "type": "number",
                "instructions": "How upset is the customer?",
                "levels": [{"label": "calm"}, {"label": "annoyed"}, {"label": "furious"}],
                "return_probabilities": True,
            },
        },
    }
