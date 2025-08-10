import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SplitResult:
    left_wav_path: str
    right_wav_path: str
    temp_dir: str


def ensure_ffmpeg_available() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required but not found on PATH. Install it (e.g., brew install ffmpeg)."
        ) from exc


def _build_filter(channel: Literal["FL", "FR"], normalizer: Literal["loudnorm", "dynaudnorm", "none"]) -> str:
    base = f"pan=mono|c0={channel}"
    if normalizer == "loudnorm":
        # EBU R128 single-pass normalization (targets voice-friendly loudness)
        return f"{base},loudnorm=I=-16:TP=-1.5:LRA=11:print_format=none"
    if normalizer == "dynaudnorm":
        # Dynamic normalizer to smooth out varying levels
        return f"{base},dynaudnorm=f=200:g=31:m=15:s=10"
    return base


def split_stereo_to_mono(
    input_path: str,
    sample_rate: int = 16000,
    normalizer: Literal["loudnorm", "dynaudnorm", "none"] = "loudnorm",
) -> SplitResult:
    """
    Split a stereo file into two mono PCM WAVs at a given sample rate using the pan filter,
    and optionally normalize loudness per channel.

    Returns paths to left and right channel wavs and the temp dir (caller should clean up).
    """
    ensure_ffmpeg_available()

    temp_dir = tempfile.mkdtemp(prefix="callsplit_")
    left_path = os.path.join(temp_dir, "left.wav")
    right_path = os.path.join(temp_dir, "right.wav")

    filt_left = _build_filter("FL", normalizer)
    filt_right = _build_filter("FR", normalizer)

    cmd_left = (
        f"ffmpeg -y -i {shlex.quote(input_path)} "
        f"-filter:a \"{filt_left}\" -ar {sample_rate} -c:a pcm_s16le {shlex.quote(left_path)}"
    )
    cmd_right = (
        f"ffmpeg -y -i {shlex.quote(input_path)} "
        f"-filter:a \"{filt_right}\" -ar {sample_rate} -c:a pcm_s16le {shlex.quote(right_path)}"
    )

    for cmd in (cmd_left, cmd_right):
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg error while splitting channels. Command: {cmd}\nSTDERR:\n{proc.stderr.decode('utf-8', errors='ignore')}"
            )

    return SplitResult(left_wav_path=left_path, right_wav_path=right_path, temp_dir=temp_dir)
