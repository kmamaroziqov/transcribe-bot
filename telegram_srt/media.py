from pathlib import Path
import subprocess
import wave


class MediaError(ValueError):
    pass


def extract_audio(source: Path, target: Path, max_seconds: int = 3600) -> None:
    """Normalize the first audio track and preserve its source timeline."""
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-protocol_whitelist", "file,pipe", "-copyts", "-start_at_zero",
            "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000",
            "-t", str(max_seconds + 1), "-c:a", "pcm_s16le", str(target),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    except FileNotFoundError as exc:
        raise MediaError("FFmpeg is missing on the bot server. Please contact its operator.") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError("Audio extraction took too long. Please send a shorter clip.") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError("I could not decode an audio track. Send a playable audio or video file with sound.") from exc
    with wave.open(str(target), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    if duration > max_seconds:
        raise MediaError(f"This recording exceeds the {max_seconds // 60}-minute limit. Please split it into clips.")
    if duration <= 0:
        raise MediaError("The recording has no audio samples.")
