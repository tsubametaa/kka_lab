"""HTTP and FastAPI Request test doubles for offline unit tests."""

import json
from types import SimpleNamespace

from starlette.requests import Request


class FakeHttpResponse:
    """Context-managed urllib response stub returning JSON bytes."""

    def __init__(self, payload):
        self._raw = json.dumps(payload).encode('utf-8')

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def make_request(*, store=None, app_state=None, headers=None):
    """Build a minimal Starlette Request bound to mock app.state."""
    if app_state is None:
        app_state = SimpleNamespace(store=store, assistant=None)
    raw_headers = [
        (k.lower().encode('latin-1'), v.encode('latin-1'))
        for k, v in (headers or {}).items()
    ]
    scope = {
        'type': 'http',
        'method': 'GET',
        'path': '/',
        'headers': raw_headers,
        'query_string': b'',
        'app': SimpleNamespace(state=app_state),
    }
    return Request(scope)
