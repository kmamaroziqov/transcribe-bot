# Local Rubai STT v2 → Telegram SRT

This project runs the official [islomov/rubaistt_v2_medium](https://huggingface.co/islomov/rubaistt_v2_medium) model locally in a Docker container on an NVIDIA GPU. It returns UTF-8 SRT subtitles for audio/video files and Telegram voice notes.

The defaults target an **RTX 5060 with 8 GB VRAM**: CUDA 12.8, FP16, beam size 1, one model instance and one active GPU request. Rubai is a 769M-parameter Whisper Medium fine-tune for **Uzbek**. Other language codes are accepted, but this is not a general multilingual accuracy guarantee. Runtime memory must be measured on your workload; these settings do not reserve all 8 GB.

## 1. Download the model with parallel requests

Create `.env` from `.env.example` if you do not already have one. Set `HF_API_TOKEN` in it. Existing `.env` files should be preserved.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "huggingface-hub[hf_xet]>=1,<2" "python-dotenv>=1,<2"
.\.venv\Scripts\python.exe scripts/download_model.py
```

The downloader privately maps `HF_API_TOKEN` to the Hugging Face CLI's `HF_TOKEN` environment variable. It uses 16 workers for configuration files and 64 HTTP range workers for the weights. Each 1 MiB piece is saved to disk under models/download-parts before assembly and SHA-256 verification. It downloads revision `af4601176140501f8ffbced420f6e2e23c9bb071` into `models/rubai-source` and verifies checksums. Re-run the same command to resume an interrupted download.

Your token is never passed as a command-line argument, included in the Docker build context, or sent to the STT container. Downloads contact Hugging Face; inference uses the local model.

## 2. Convert and start the GPU service

Start Docker Desktop with the WSL2 Linux-container backend and GPU support. Keep the Windows NVIDIA driver current. The runtime uses Python with the required CUDA 12.8/cuDNN libraries and a bundled FFmpeg binary. It does not need the full CUDA toolkit.

For slow or unreliable connections, pre-download the large dependencies in resumable, checksum-verified pieces (Docker automatically reuses these local wheels):

```powershell
.\.venv\Scripts\python.exe scripts/download_torch_wheel.py
.\.venv\Scripts\python.exe scripts/download_gpu_wheels.py
.\.venv\Scripts\python.exe scripts/download_dependencies.py
```

```powershell
docker compose --profile setup run --build --rm convert
docker compose up --build -d stt
docker compose logs -f stt
```

Conversion runs once on CPU and saves FP16 CTranslate2 weights in `models/rubai-ct2`. It uses the original model's tokenizer and alignment heads. The runtime mounts those files read-only and has Hugging Face offline mode enabled. Model files remain outside the image so a code rebuild does not download or convert the weights again. First setup needs several GB of downloads and additional disk space for Docker layers and both model copies.

The API is bound to **localhost only**, at [http://127.0.0.1:18000](http://127.0.0.1:18000). It loads the model before becoming healthy:

```powershell
curl.exe http://127.0.0.1:18000/health
curl.exe --fail-with-body -X POST "http://127.0.0.1:18000/transcribe?language=uz" -H "Content-Type: application/octet-stream" --data-binary "@C:\path\recording.mp4" --output transcript.srt
```

The request body is the raw file, not multipart form data. The response is SRT; errors are JSON with non-200 status codes. `language=auto` enables detection. A second simultaneous upload gets HTTP 429, preventing concurrent models from consuming your VRAM. The API does not have a public authentication layer; keep the localhost binding.

To stop: `docker compose down`. Local model files are retained.

## 3. Enable the Telegram bot

Set `TELEGRAM_BOT_TOKEN` in `.env` to your @BotFather token:

```powershell
docker compose --profile telegram up -d
```

Open the bot and send `/start`, then an audio/video file. Choose **Text only** for a plain transcript, or an inline subtitle button: 1, 2, or 3 words on one line; 5 words across two lines (3 + 2); or 7 words across three lines (3 + 2 + 2). Transcription starts after selection. Pauses, duration limits, and the final cue can produce fewer words. Buttons expire after 15 minutes; a new upload replaces your previous selection. Commands: `/language uz` (default), `/language auto`, and `/status`. The bot calls the STT service over Docker's private network and does not load a second model. It accepts up to three jobs, one per user; the queue and language preferences are in memory and reset on restart.

## Supported media and timing

Voice/video notes, MP3, WAV, M4A/AAC, OGG/Opus, FLAC, AIFF, AMR, WMA, MP4, MOV, MKV, WebM, AVI, MPEG, M4V, 3GP, TS/MTS and WMV are accepted when their codecs are decodable by FFmpeg. The first audio track is used.

Subtitles use word timestamps and preserve leading silence, gaps and audio delays relative to media start. Normal cues target two lines of 42 characters and up to six seconds. A single unusually long token may exceed these layout targets. Import the SRT alongside the **same untrimmed recording**, at the same timeline position. Source timecode metadata is not added to SRT.

Recognition and alignment are estimates, not guaranteed frame-accurate results. Review transcripts with noise, music, accents or overlapping voices. Speaker identification is not implemented. SRT is a caption format, not editor-specific transcript metadata.

## Limits and larger Telegram videos

- Local API: `STT_MAX_FILE_MB=500` by default.
- Telegram bot: `MAX_FILE_MB=20` with the hosted Bot API.
- Recording duration: `MAX_DURATION_SECONDS=3600`.
- Bot queue: `MAX_JOBS=3`.
- Optional access restriction: `ALLOWED_USER_IDS=123,456`.

Telegram's hosted Bot API only lets bots download files up to 20 MB. For larger Telegram files, run the official [local Bot API server](https://core.telegram.org/bots/api#using-a-local-bot-api-server), follow its hosted-to-local `logOut` migration procedure, set `TELEGRAM_API_URL`, and increase `MAX_FILE_MB`. Its downloaded-file directory must be accessible to the bot at the same absolute paths; add matching mounts to Compose. The optional `compose.local-telegram.yaml` provisions that server with a pinned aiogram image containing the official Telegram Bot API binary.

Temporary media is deleted after successful/failed jobs. Telegram retains sent files according to its own service behavior.

## Validation

```powershell
docker compose run --rm -v "${PWD}/tests:/app/tests:ro" -v "${PWD}/scripts:/app/scripts:ro" stt python3 -m unittest discover -s tests -v
```

Tests cover SRT formatting, original timing, common codecs, delayed audio, invalid uploads, queue behavior and cleanup. Test doubles are used for model/Telegram behavior. A real GPU transcription is a separate integration check.

## Troubleshooting

- **Missing model:** finish the downloader and conversion before starting `stt`.
- **GPU unavailable:** verify Docker Desktop uses WSL2 and `docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi` works.
- **Out of memory:** close other GPU-heavy applications. Keep beam size 1 and one API worker. Do not enable multiple replicas on an 8 GB GPU.
- **INT8 on RTX 50 series:** these defaults deliberately use FP16; CTranslate2 disables INT8 for SM120 Blackwell GPUs.
- **Windows credential-helper session error:** the agent session can use an isolated `.docker-build/config.json` containing `{"auths":{"https://index.docker.io/v1/":{}}}` and `$env:DOCKER_CONFIG = Join-Path (Get-Location) .docker-build` followed by `docker --host npipe:////./pipe/dockerDesktopLinuxEngine ...` for public image pulls. Your normal terminal may not need this workaround.
- **Network timeout:** retry; Hugging Face retains partial downloads and Docker retains completed layers.

References: [Rubai model](https://huggingface.co/islomov/rubaistt_v2_medium), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2 releases](https://github.com/OpenNMT/CTranslate2/releases).

Plain transcripts up to 4,000 UTF-16 units are sent as Telegram messages; longer transcripts are UTF-8 `.txt` attachments. Each SRT has a **Get plain text** button that reuses the transcript without another GPU request. Cached text is available to the uploader in the same chat for 30 minutes (up to 100 recent transcripts), and is cleared on bot restart.

## Enable Telegram uploads up to 4 GB

Add your own `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` from https://my.telegram.org/apps to `.env`. These are separate from the BotFather bot token. Then run:

```powershell
.\.venv\Scripts\python.exe scripts/enable_local_api.py
```

The script builds the updated application, waits for the local server, drains bot jobs, calls the required hosted `logOut`, and activates the local Compose configuration. It records `COMPOSE_FILE` and `COMPOSE_PATH_SEPARATOR` in `.env`, so ordinary subsequent Compose commands keep using the local server. The new file limit is 4096 MB for both the bot and STT API. The duration limit remains `MAX_DURATION_SECONDS` (60 minutes unless changed separately). Premium accounts can upload 4 GB files; Telegram account limits still apply.

The bot reads downloaded files from the local server's volume using the same unprivileged UID and a read-only mount. The server's own download cache persists in that volume. No API or statistics ports are published to the host. A full 4 GB upload has not been integration-tested until a real large file is submitted after activation. If activation fails after hosted logout, rerun the script to complete local startup; Telegram temporarily restricts returning to its hosted API.
