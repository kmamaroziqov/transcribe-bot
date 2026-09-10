from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from telegram_srt import api


class FakeEngine:
    device = 'cuda'
    compute_type = 'float16'
    folder = None
    no_speech = False

    def load(self):
        pass

    def transcribe(self, source, folder, language, layout="auto"):
        self.layout = layout
        self.folder = folder
        assert Path(source).read_bytes() == b'audio'
        if self.no_speech:
            return '', 'uz'
        return '1\n00:00:02,000 --> 00:00:03,000\nSalom!\n\n', language or 'uz'


class ApiTests(unittest.TestCase):
    def setUp(self):
        engine_patch = patch.object(api, 'RubaiEngine', FakeEngine)
        engine_patch.start()
        self.addCleanup(engine_patch.stop)
        languages = patch.dict(sys.modules, {'faster_whisper.tokenizer': SimpleNamespace(_LANGUAGE_CODES={'uz', 'en'})})
        languages.start()
        self.addCleanup(languages.stop)
        self.client = self.enterContext(TestClient(api.app))

    def test_health_and_srt(self):
        self.assertEqual(self.client.get('/health').json()['compute_type'], 'float16')
        response = self.client.post('/transcribe?layout=5', content=b'audio')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(api.app.state.engine.layout, '5')
        self.assertIn('00:00:02,000', response.text)
        self.assertEqual(response.headers['x-detected-language'], 'uz')
        self.assertFalse(api.app.state.engine.folder.exists())
        self.assertFalse(api.app.state.busy)

    def test_invalid_empty_large_and_busy(self):
        self.assertEqual(self.client.post('/transcribe?layout=invalid', content=b'audio').status_code, 422)
        self.assertEqual(self.client.post('/transcribe?language=invalid', content=b'audio').status_code, 422)
        self.assertEqual(self.client.post('/transcribe', content=b'').status_code, 400)
        with patch.object(api, 'MAX_BYTES', 2):
            self.assertEqual(self.client.post('/transcribe', content=b'audio').status_code, 413)
        self.assertFalse(api.app.state.busy)
        api.app.state.busy = True
        self.assertEqual(self.client.post('/transcribe', content=b'audio').status_code, 429)
        api.app.state.busy = False

    def test_no_speech_cleanup(self):
        api.app.state.engine.no_speech = True
        response = self.client.post('/transcribe', content=b'audio')
        self.assertEqual(response.status_code, 422)
        self.assertFalse(api.app.state.engine.folder.exists())
        self.assertFalse(api.app.state.busy)
