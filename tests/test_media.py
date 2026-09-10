from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

from telegram_srt.media import MediaError, extract_audio


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def ffmpeg(self, args):
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True)

    def test_common_containers(self):
        for ext, codec in [("mp3", "libmp3lame"), ("ogg", "libopus"), ("flac", "flac"),
                           ("m4a", "aac"), ("mp4", "aac"), ("mov", "aac"),
                           ("mkv", "aac"), ("webm", "libopus"), ("avi", "mp3"), ("wav", "pcm_s16le")]:
            with self.subTest(format=ext):
                source = self.folder / f"sample.{ext}"
                self.ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", codec, str(source)])
                target = self.folder / "out.wav"
                extract_audio(source, target)
                with wave.open(str(target)) as wav:
                    self.assertEqual(wav.getframerate(), 16000)
                    self.assertEqual(wav.getnchannels(), 1)
                    self.assertAlmostEqual(wav.getnframes() / 16000, 1, delta=0.15)

    def test_delayed_audio_keeps_video_offset(self):
        source = self.folder / "delayed.mkv"
        self.ffmpeg(["-f", "lavfi", "-i", "color=black:s=64x64:d=4",
                     "-itsoffset", "2", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                     "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(source)])
        target = self.folder / "out.wav"
        extract_audio(source, target)
        with wave.open(str(target)) as wav:
            self.assertAlmostEqual(wav.getnframes() / 16000, 3, delta=0.1)
            self.assertEqual(set(wav.readframes(16000)), {0})

    def test_reject_corrupt_and_overlong(self):
        source = self.folder / "bad.mp4"
        source.write_bytes(b"not media")
        with self.assertRaises(MediaError):
            extract_audio(source, self.folder / "out.wav")
        self.ffmpeg(["-f", "lavfi", "-i", "sine=duration=3", str(self.folder / "long.wav")])
        with self.assertRaisesRegex(MediaError, "exceeds"):
            extract_audio(self.folder / "long.wav", self.folder / "out.wav", max_seconds=1)
