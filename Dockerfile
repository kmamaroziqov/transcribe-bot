FROM python:3.12-slim AS runtime-deps
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10 \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WHISPER_MODEL=/models/rubai WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /app
COPY telegram_srt/requirements.txt /app/requirements.txt
RUN --mount=type=bind,source=.wheels,target=/wheels \
    if [ -f /wheels/runtime.ready ]; then \
      pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt; \
    else pip install --no-cache-dir -r requirements.txt; fi
RUN python -c "import os, imageio_ffmpeg; os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), '/usr/local/bin/ffmpeg')"
FROM runtime-deps AS runtime
RUN --mount=type=bind,source=.wheels,target=/wheels \
    if [ -f /wheels/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl ] \
       && [ -f /wheels/nvidia_cudnn_cu12-9.8.0.87-py3-none-manylinux_2_27_x86_64.whl ] \
       && [ -f /wheels/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl ]; then \
      pip install --no-cache-dir /wheels/nvidia_*.whl; \
    else pip install --no-cache-dir nvidia-cublas-cu12==12.8.4.1 nvidia-cudnn-cu12==9.8.0.87 nvidia-cuda-runtime-cu12==12.8.90; fi
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN useradd --create-home bot
COPY telegram_srt /app/telegram_srt
USER bot
EXPOSE 8000
CMD ["uvicorn", "telegram_srt.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM python:3.12-slim AS converter
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10
RUN --mount=type=bind,source=.wheels,target=/wheels \
    if [ -f /wheels/converter.ready ]; then \
      pip install --no-cache-dir --no-index --find-links=/wheels torch==2.8.0+cpu; \
    elif [ -f /wheels/torch-2.8.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl ]; then \
      pip install --no-cache-dir /wheels/torch-2.8.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl; \
    else pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu; fi
RUN --mount=type=bind,source=.wheels,target=/wheels \
    if [ -f /wheels/converter.ready ]; then \
      pip install --no-cache-dir --no-index --find-links=/wheels ctranslate2==4.8.2 transformers==4.55.4 hf-xet==1.6.0; \
    else pip install --no-cache-dir ctranslate2==4.8.2 transformers==4.55.4 hf-xet==1.6.0; fi
COPY scripts/convert_model.py /app/convert_model.py
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
CMD ["python", "convert_model.py", "/source", "/output"]
