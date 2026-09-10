"""One local GPU model shared by the API's serialized transcription jobs."""
import os
from pathlib import Path

from .media import extract_audio
from .subtitles import Word, render_srt


class RubaiEngine:
    def __init__(self):
        self.model = None
        self.model_path = os.getenv("WHISPER_MODEL", "/models/rubai")
        self.device = os.getenv("WHISPER_DEVICE", "cuda")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        self.beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
        self.max_seconds = int(os.getenv("MAX_DURATION_SECONDS", "3600"))
        if self.beam_size < 1 or self.max_seconds < 1:
            raise ValueError("Beam size and maximum duration must be positive.")

    def load(self):
        import ctranslate2
        from faster_whisper import WhisperModel
        if not Path(self.model_path, "model.bin").is_file():
            raise RuntimeError("Rubai model missing. Run the download and conversion steps in README.md.")
        supported = ctranslate2.get_supported_compute_types(self.device)
        if self.compute_type not in supported:
            raise RuntimeError(f"{self.compute_type} is unsupported on {self.device}; supported: {sorted(supported)}")
        self.model = WhisperModel(
            self.model_path, device=self.device, compute_type=self.compute_type,
            num_workers=1, cpu_threads=4, local_files_only=True,
        )

    def transcribe(self, source, folder, language="uz", layout="auto"):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        audio = folder / "audio.wav"
        extract_audio(source, audio, self.max_seconds)
        segments, info = self.model.transcribe(
            str(audio), language=language, task="transcribe",
            beam_size=self.beam_size, word_timestamps=True,
            vad_filter=True, condition_on_previous_text=False,
        )
        words = (Word(w.start, w.end, w.word) for s in segments for w in (s.words or []))
        return render_srt(words, layout_mode=layout), info.language
