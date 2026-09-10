"""Authenticated, resumable parallel download with disk-backed byte ranges."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = "islomov/rubaistt_v2_medium"
REVISION = "af4601176140501f8ffbced420f6e2e23c9bb071"
DEST = ROOT / "models" / "rubai-source"
WORKERS = 64
PART_SIZE = 1024 * 1024


def download_weights(token):
    import httpx
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    url = hf_hub_url(REPO, "model.safetensors", revision=REVISION)
    metadata = get_hf_file_metadata(url, token=token, timeout=60)
    size = metadata.size
    expected_sha = metadata.etag
    if not size or not expected_sha or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RuntimeError("The Hub did not return a weight size and SHA-256 checksum.")
    return download_file(url, size, expected_sha, DEST / "model.safetensors", {"Authorization": f"Bearer {token}"})


def download_file(url, size, expected_sha, target, request_headers=None):
    """Serialize concurrent downloads of the same immutable file."""
    from filelock import FileLock
    lock_dir = ROOT / "models" / "download-parts"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_dir / (expected_sha + ".lock"))):
        return _download_file(url, size, expected_sha, target, request_headers)


def _download_file(url, size, expected_sha, target, request_headers=None):
    """Download an immutable file in persisted pieces; verify before publication."""
    import httpx
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() == expected_sha:
                print("Weights already verified.", flush=True)
                return
        raise RuntimeError("Existing weights failed verification; inspect them before replacing.")
    parts = ROOT / "models" / "download-parts" / expected_sha
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(i, start, min(start + PART_SIZE, size)) for i, start in enumerate(range(0, size, PART_SIZE))]
    saved = sum(end-start for i,start,end in ranges if (parts / f"{i:06}.part").exists() and (parts / f"{i:06}.part").stat().st_size == end-start)
    print(f"Weights: {size:,} bytes; already saved: {saved:,}; {WORKERS} workers; 1 MiB pieces.", flush=True)
    (parts / "download-info.txt").write_text(f"source={url.split(chr(63))[0]}\nsha256={expected_sha}\nsize={size}\n", encoding="utf-8")
    state = {"saved": saved, "failed": 0}
    state_lock = threading.Lock()
    stop = threading.Event()
    started = time.monotonic()
    initial_saved = saved

    def progress():
        while not stop.wait(15):
            with state_lock:
                done = state["saved"]
                failed = state["failed"]
            rate = (done - initial_saved) / max(1, time.monotonic() - started)
            print(f"Saved {done/1e6:.1f}/{size/1e6:.1f} MB ({100*done/size:.1f}%), {rate/1e6:.2f} MB/s average; failed pieces: {failed}", flush=True)

    def fetch(item):
        i, start, end = item
        part = parts / f"{i:06}.part"
        temporary = parts / f"{i:06}.partial"
        expected_length = end - start
        if part.exists() and part.stat().st_size == expected_length:
            return True
        # A separate client per worker/task avoids shared connection-pool bottlenecks.
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(90, connect=30),
                          headers={"Accept-Encoding": "identity"}) as client:
            for attempt in range(8):
                offset = temporary.stat().st_size if temporary.exists() else 0
                if offset > expected_length:
                    raise RuntimeError("A saved partial piece has an invalid size.")
                if offset == expected_length:
                    temporary.replace(part)
                    with state_lock:
                        state["saved"] += expected_length
                    return True
                try:
                    # httpx removes Authorization on cross-origin redirects. Only
                    # Hugging Face receives the token; its CDN receives a signed URL.
                    headers = {**(request_headers or {}), "Range": f"bytes={start+offset}-{end-1}"}
                    with client.stream("GET", url, headers=headers) as response:
                        expected_range = f"bytes {start+offset}-{end-1}/{size}"
                        if response.status_code != 206 or response.headers.get("content-range") != expected_range:
                            raise ValueError(f"Unexpected range response status {response.status_code}")
                        with temporary.open("ab") as file:
                            for chunk in response.iter_bytes(64 * 1024):
                                if file.tell() + len(chunk) > expected_length:
                                    raise ValueError("Server sent more data than the requested range.")
                                file.write(chunk)
                    if temporary.stat().st_size != expected_length:
                        raise ValueError("Incomplete range response.")
                    temporary.replace(part)
                    with state_lock:
                        state["saved"] += expected_length
                    return True
                except (httpx.HTTPError, ValueError):
                    time.sleep(min(2**attempt, 15))
        with state_lock:
            state["failed"] += 1
        return False

    reporter = threading.Thread(target=progress, daemon=True)
    reporter.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            success = all(list(pool.map(fetch, ranges)))
    finally:
        stop.set()
        reporter.join()
    if not success:
        raise RuntimeError("Some pieces failed. Re-run this script; saved pieces will be reused.")
    print("Assembling and verifying the SHA-256 checksum...", flush=True)
    assembling = target.with_name(target.name + ".assembling")
    digest = hashlib.sha256()
    with assembling.open("wb") as output:
        for i, start, end in ranges:
            with (parts / f"{i:06}.part").open("rb") as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError("Weight SHA-256 mismatch. The assembled file was not accepted.")
    assembling.replace(target)
    print(f"Weights verified: {target}", flush=True)


def run_cli(hf, args, env):
    # Redact signed query strings if the official CLI reports a transport error.
    with subprocess.Popen([str(hf), *args], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace") as process:
        for line in process.stdout:
            print(re.sub(r"(https?://[^\s?]+)\?[^\s)]+", r"\1?[redacted]", line), end="", flush=True)
        if process.wait():
            raise RuntimeError("Hugging Face CLI failed; completed local files are retained.")


def main():
    from dotenv import dotenv_values
    token = dotenv_values(ROOT / ".env").get("HF_API_TOKEN")
    if not token:
        raise SystemExit("HF_API_TOKEN is missing from .env.")
    env = os.environ.copy()
    env.update(HF_TOKEN=token, HF_HOME=str(ROOT / ".hf-cache"), HF_HUB_DISABLE_TELEMETRY="1")
    hf = Path(sys.executable).with_name("hf.exe" if os.name == "nt" else "hf")
    DEST.mkdir(parents=True, exist_ok=True)
    run_cli(hf, ["download", REPO, "--revision", REVISION, "--local-dir", str(DEST),
                 "--exclude", "model.safetensors", "--max-workers", "16"], env)
    download_weights(token)
    run_cli(hf, ["cache", "verify", REPO, "--revision", REVISION, "--local-dir", str(DEST),
                 "--fail-on-missing-files"], env)
    print("Rubai STT v2 download and verification complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never emit URLs/headers or a traceback with credentials from a dependency.
        print(f"Download stopped ({type(exc).__name__}). Saved pieces are retained for retry.", flush=True)
        raise SystemExit(1)
