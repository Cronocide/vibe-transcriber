from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .whisper_utils import TranscriptSegment


def _fmt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    centis = int(round((secs - int(secs)) * 100))
    return f"[{minutes:02d}:{int(secs):02d}.{centis:02d}]"


def write_lrc(
    segments: Iterable[TranscriptSegment],
    output_path: str,
    title: Optional[str] = None,
    artists: Optional[str] = None,
) -> None:
    lines: List[str] = []
    if title:
        lines.append(f"[ti:{title}]")
    if artists:
        lines.append(f"[ar:{artists}]")

    for seg in segments:
        ts = _fmt_time(seg.start)
        speaker = seg.speaker or "Speaker"
        text = seg.text.strip()
        lines.append(f"{ts} {speaker}: {text}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
