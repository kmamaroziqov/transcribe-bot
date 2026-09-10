import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .engine import RubaiEngine
from .media import MediaError
from .subtitles import LAYOUTS

LOG = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    engine = RubaiEngine()
    await run_in_threadpool(engine.load)
    app.state.engine = engine
    app.state.busy = False
    yield


app = FastAPI(title="Local Rubai STT v2", lifespan=lifespan)
MAX_BYTES = int(os.getenv("STT_MAX_FILE_MB", "500")) * 1024 * 1024


@app.get("/health")
async def health():
    return {
        "status": "ready", "model": "islomov/rubaistt_v2_medium",
        "device": app.state.engine.device,
        "compute_type": app.state.engine.compute_type,
        "busy": app.state.busy,
    }


@app.post("/transcribe")
async def transcribe(request: Request, language: str = "uz", layout: str = "auto"):
    """Upload raw audio/video bytes; returns an SRT attachment."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES
    if language != "auto" and language not in _LANGUAGE_CODES:
        raise HTTPException(422, "Unsupported language code. Use uz or auto.")
    if layout != "auto" and layout not in LAYOUTS:
        raise HTTPException(422, "Unsupported subtitle layout. Use 1, 2, 3, 5, or 7.")
    if app.state.busy:
        raise HTTPException(429, "The GPU is busy. Try again when this job finishes.")
    # Reserve before awaiting upload bytes: exactly one request uses disk/GPU.
    app.state.busy = True
    try:
        with tempfile.TemporaryDirectory(prefix="rubai-stt-") as directory:
            folder = Path(directory)
            source = folder / "upload.media"
            size = 0
            with source.open("wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "File exceeds the local API upload limit.")
                    out.write(chunk)
            if not size:
                raise HTTPException(400, "Upload an audio or video file as the request body.")
            job = asyncio.create_task(asyncio.to_thread(
                app.state.engine.transcribe, source, folder,
                None if language == "auto" else language, layout,
            ))
            try:
                srt, detected = await asyncio.shield(job)
            except asyncio.CancelledError:
                await job
                raise
            if not srt:
                raise HTTPException(422, "No speech was detected.")
            return Response(
                srt.encode("utf-8"), media_type="application/x-subrip",
                headers={"Content-Disposition": 'attachment; filename="transcript.srt"',
                         "X-Detected-Language": detected},
            )
    except MediaError as exc:
        raise HTTPException(422, str(exc)) from None
    except HTTPException:
        raise
    except Exception as exc:
        LOG.error("Transcription failed (%s)", type(exc).__name__)
        raise HTTPException(500, "Transcription failed. Check the server logs and available GPU memory.") from None
    finally:
        app.state.busy = False
