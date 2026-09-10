import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import download_model as downloader


class RangeDownloaderTests(unittest.TestCase):
    def test_resumes_partial_pieces_and_verifies_assembly(self):
        data = b'0123456789abcdefghijklmnop'
        sha = hashlib.sha256(data).hexdigest()
        requested = []

        class Response:
            status_code = 206
            def __init__(self, start, end):
                self.start, self.end = start, end
                self.headers = {'content-range': f'bytes {start}-{end}/{len(data)}'}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def iter_bytes(self, *args):
                yield data[self.start:self.end+1]

        class Client:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def stream(self, method, url, headers):
                start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
                requested.append((start, end))
                return Response(start, end)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / 'source'
            destination.mkdir()
            parts = root / 'models' / 'download-parts' / sha
            parts.mkdir(parents=True)
            (parts / '000000.part').write_bytes(data[:8])
            (parts / '000001.partial').write_bytes(data[8:11])
            with patch.multiple(downloader, ROOT=root, DEST=destination, PART_SIZE=8, WORKERS=2), \
                 patch('huggingface_hub.get_hf_file_metadata', return_value=SimpleNamespace(size=len(data), etag=sha)), \
                 patch('httpx.Client', Client):
                downloader.download_weights('test-token')
                self.assertEqual((destination/'model.safetensors').read_bytes(), data)
                self.assertNotIn((0, 7), requested)
                self.assertIn((11, 15), requested)
                self.assertEqual(len(list(parts.glob('*.part'))), 4)
                before = len(requested)
                downloader.download_weights('test-token')
                self.assertEqual(len(requested), before)
