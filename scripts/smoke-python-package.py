"""Exercise an installed wheel's native extension (no mocked modules)."""
import json
from turbojev import TurboJev
from turbojev._native import NativeEngine

with TurboJev.load('fixture://package-smoke') as engine:
    assert isinstance(engine.info(), dict)

with TurboJev(NativeEngine.load_json(json.dumps({
    'model': 'fixture://package-smoke',
    # Keep the supplied positional logits in single-pass mode.
    'execution': {'orderRobustness': 'none'},
    'fixtureLogits': {'route': [0.0, 3.0]},
}))) as engine:
    response = engine.evaluate({'state': {'message': 'Package smoke test'}, 'questions': {
        'route': {'type': 'choice', 'instructions': 'Choose a route.', 'criteria': {'a': 'A', 'b': 'B'}}
    }})
    assert response['answers']['route']['choice'] == 'b', response
with TurboJev.load('fixture://balanced-package-smoke', execution={'orderRobustness': 'balanced'}, fixtureLogits={'route': [0.0, 0.0, 0.0]}) as engine:
    result = engine.evaluate({'state': 'facts', 'questions': {'route': {'type': 'choice', 'instructions': 'Choose', 'criteria': {'z': 'Z', 'a': 'A', 'b': 'B'}}}})
    assert result['answers']['route']['choice'] == 'a', result
    assert result['_turbojev']['order_robustness'] == 'balanced', result
    assert result['_turbojev']['order_evaluations'] == 6, result
print('Native Python package smoke test passed, including balanced ordering.')
