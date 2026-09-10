import os
import httpx

from .media import MediaError


class RemoteTranscriber:
    def __init__(self):
        self.url = os.getenv("STT_API_URL", "http://127.0.0.1:18000").rstrip("/")

    def transcribe(self, source, folder, language, layout="auto"):
        # Stream bytes from disk; no second in-memory copy of the upload.
        with source.open("rb") as audio, httpx.Client(timeout=7200) as client:
            response = client.post(
                f"{self.url}/transcribe", params={"language": language or "auto", "layout": layout},
                content=audio, headers={"Content-Type": "application/octet-stream"},
            )
        if response.status_code != 200:
            if response.status_code in {400, 413, 422, 429}:
                raise MediaError(response.json().get("detail", "Transcription failed."))
            raise MediaError("The local transcription service is unavailable. Please try again.")
        return response.text, response.headers.get("X-Detected-Language", language or "uz")
