"""Tests for app.features.chat.router (routes and SSE formatting)."""

import unittest

from app.features.chat.router import _sse_event, router as chat_router


class ChatRouterTests(unittest.TestCase):
    def test_chat_routes_and_sse_event_format(self):
        chat_paths = [route.path for route in chat_router.routes if hasattr(route, 'path')]
        self.assertIn('/api/chat', chat_paths)
        self.assertIn('/api/chat/stream', chat_paths)
        self.assertEqual(_sse_event('delta', {'text': 'hi'}), 'event: delta\ndata: {"text": "hi"}\n\n')


if __name__ == '__main__':
    unittest.main()
