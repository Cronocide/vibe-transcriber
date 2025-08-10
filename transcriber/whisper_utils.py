from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from statistics import mean
from typing import Iterable, List, Literal, Optional

from faster_whisper import WhisperModel


@dataclass
class WordToken:
    start: float
    end: float
    word: str
    probability: Optional[float] = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    speaker: Optional[str] = None  # filled by caller
    words: Optional[List[WordToken]] = None


_WORDLIKE = re.compile(r"[\w\-]+", re.UNICODE)
_BRACKETED = re.compile(r"\s*(\[[^\]]+\]|\([^\)]+\))\s*")


def load_model(
    model_size: str,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    compute_type: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> WhisperModel:
    if device == "auto":
        # Prefer CUDA if available, else CPU
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") else "cpu"
    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=model_dir,
    )


def _should_mark_indistinct(text: str, avg_logprob: Optional[float], avg_word_prob: Optional[float]) -> bool:
    if not text or not _WORDLIKE.search(text):
        return True
    if avg_word_prob is not None and avg_word_prob < 0.40:
        return True
    if avg_logprob is not None and avg_logprob < -1.2:
        return True
    return False


def _emphasize_bracketed(text: str) -> str:
    # Turn bracketed or parenthetical asides into *aside*
    def repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        inner = inner.strip()
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1].strip()
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1].strip()
        return f" *{inner}* "

    return _BRACKETED.sub(repl, text)


# Tighter VAD to avoid gluing across pauses
VAD_PARAMS = {
    "threshold": 0.6,
    "min_silence_duration_ms": 220,
    "speech_pad_ms": 80,
}

# Split a segment into multiple if gaps between words exceed this duration
SPLIT_GAP_SECONDS = 0.40


def transcribe_file(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str] = "en",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> List[TranscriptSegment]:
    segments: List[TranscriptSegment] = []

    seg_iter, _info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=VAD_PARAMS,
        word_timestamps=True,
    )

    for seg in seg_iter:  # type: ignore
        words = getattr(seg, "words", None)
        if words:
            current_words: List[WordToken] = []
            current_start: Optional[float] = None
            prev_end: Optional[float] = None

            def flush_from_words() -> None:
                if not current_words:
                    return
                start_ts = current_start if current_start is not None else float(getattr(seg, "start", 0.0))
                end_ts = float(current_words[-1].end if current_words[-1].end is not None else getattr(seg, "end", start_ts))
                raw_text = "".join(w.word for w in current_words).strip()
                text = _emphasize_bracketed(raw_text)
                avg_word_prob = None
                try:
                    probs = [float(w.probability) for w in current_words if w.probability is not None]
                    if probs:
                        avg_word_prob = float(mean(probs))
                except Exception:
                    avg_word_prob = None
                if _should_mark_indistinct(raw_text, getattr(seg, "avg_logprob", None), avg_word_prob):
                    text = f"*{text}*" if text else "*indistinct*"
                segments.append(
                    TranscriptSegment(
                        start=float(start_ts),
                        end=float(end_ts),
                        text=text,
                        avg_logprob=getattr(seg, "avg_logprob", None),
                        no_speech_prob=getattr(seg, "no_speech_prob", None),
                        words=list(current_words),
                    )
                )
                current_words.clear()

            for w in words:
                wstart = float(getattr(w, "start", getattr(seg, "start", 0.0)) or 0.0)
                wend = float(getattr(w, "end", wstart))
                wtext = getattr(w, "word", "")
                wprob = getattr(w, "probability", None)
                if prev_end is not None and (wstart - prev_end) >= SPLIT_GAP_SECONDS:
                    flush_from_words()
                    current_start = wstart
                if not current_words:
                    current_start = wstart
                current_words.append(WordToken(start=wstart, end=wend, word=wtext, probability=wprob))
                prev_end = wend

            flush_from_words()
        else:
            # Fallback if no word timestamps are present
            raw_text = (getattr(seg, "text", "") or "").strip()
            text = _emphasize_bracketed(raw_text)
            if _should_mark_indistinct(raw_text, getattr(seg, "avg_logprob", None), None):
                text = f"*{text}*" if text else "*indistinct*"
            segments.append(
                TranscriptSegment(
                    start=float(getattr(seg, "start", 0.0)),
                    end=float(getattr(seg, "end", 0.0)),
                    text=text,
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                    words=None,
                )
            )

    # Filter out any empty lines that survived
    return [s for s in segments if s.text.strip()]


def _split_segment_at_time(segment: TranscriptSegment, boundary_time: float) -> List[TranscriptSegment]:
    if boundary_time <= segment.start or boundary_time >= segment.end:
        return [segment]
    if not segment.words:
        # Without word alignment, just split timestamps and keep same text in the first part (avoid duplicating text)
        left = replace(segment, end=boundary_time)
        right = replace(segment, start=boundary_time, text="")
        return [left, right]

    left_words: List[WordToken] = []
    right_words: List[WordToken] = []
    for w in segment.words:
        if w.end <= boundary_time:
            left_words.append(w)
        elif w.start >= boundary_time:
            right_words.append(w)
        else:
            # Word straddles boundary: assign by midpoint
            mid = (w.start + w.end) / 2.0
            if mid <= boundary_time:
                left_words.append(WordToken(start=w.start, end=boundary_time, word=w.word, probability=w.probability))
            else:
                right_words.append(WordToken(start=boundary_time, end=w.end, word=w.word, probability=w.probability))

    parts: List[TranscriptSegment] = []
    if left_words:
        left_text = "".join(w.word for w in left_words).strip()
        parts.append(
            TranscriptSegment(
                start=segment.start,
                end=boundary_time,
                text=_emphasize_bracketed(left_text) if left_text else "*indistinct*",
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                speaker=segment.speaker,
                words=left_words,
            )
        )
    if right_words:
        right_text = "".join(w.word for w in right_words).strip()
        parts.append(
            TranscriptSegment(
                start=boundary_time,
                end=segment.end,
                text=_emphasize_bracketed(right_text) if right_text else "*indistinct*",
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                speaker=segment.speaker,
                words=right_words,
            )
        )
    if not parts:
        # If both sides are empty, keep left empty placeholder to preserve timing
        parts.append(replace(segment, end=boundary_time, text="*indistinct*"))
    return parts


def _split_segments_at_boundaries(segments: List[TranscriptSegment], boundaries: List[float]) -> List[TranscriptSegment]:
    if not segments or not boundaries:
        return segments
    boundaries = sorted(boundaries)
    result: List[TranscriptSegment] = []
    for seg in segments:
        stack = [seg]
        for b in boundaries:
            new_stack: List[TranscriptSegment] = []
            for part in stack:
                new_stack.extend(_split_segment_at_time(part, b))
            stack = new_stack
        result.extend(stack)
    # Drop empty-text parts if any
    return [s for s in result if s.text.strip()]


def merge_dialogue(
    left_segments: Iterable[TranscriptSegment],
    left_speaker: str,
    right_segments: Iterable[TranscriptSegment],
    right_speaker: str,
) -> List[TranscriptSegment]:
    left_list = [replace(s, speaker=left_speaker) for s in left_segments]
    right_list = [replace(s, speaker=right_speaker) for s in right_segments]

    # Cross-split at each other's start times to avoid gluing text across speaker turns
    left_boundaries = [s.start for s in right_list]
    right_boundaries = [s.start for s in left_list]

    left_list = _split_segments_at_boundaries(left_list, left_boundaries)
    right_list = _split_segments_at_boundaries(right_list, right_boundaries)

    merged: List[TranscriptSegment] = []
    merged.extend(left_list)
    merged.extend(right_list)

    # Sort by start time, then end time for stability
    merged.sort(key=lambda s: (s.start, s.end))
    return merged
