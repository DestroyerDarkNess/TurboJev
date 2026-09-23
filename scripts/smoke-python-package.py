"""Exercise an installed wheel's native extension (no mocked modules)."""
import json
from turbojev import TurboJev
from turbojev._native import NativeEngine

with TurboJev.load('fixture://package-smoke') as engine:
    assert isinstance(engine.info(), dict)

with TurboJev(NativeEngine.load_json(json.dumps({
    'model': 'fixture://package-smoke',
    'fixtureLogits': {'classification': [0.0, 3.0]},
}))) as engine:
    response = engine.classify('Package smoke test', ['a', 'b'])
    assert response['choice'] == 'b', response
print('Native Python package smoke test passed.')
