### Call Transcriber (Stereo Call -> LRC)

Transcribe stereo call recordings (M4A ALAC/AAC) where each party is on a separate channel, and produce a Simple Lyrics (.lrc) file with speaker labels and timestamps.

- **Splits stereo** into left/right mono with ffmpeg
- **Uses faster-whisper** with VAD to skip quiet space
- **Merges overlapping speech** into linear dialog
- **Labels speakers** using the filename (e.g., `Tel-From-Jane_Doe-...`) and `You`
- **Wraps noises/indistinct parts** with asterisks (e.g., `*laughter*`, `*indistinct*`)

### Requirements
- **Python**: 3.10+
- **ffmpeg** installed and on PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt-get install ffmpeg`

### Setup
```bash
cd /Users/cronocide/Desktop/call-transcriber-vibe-gpt5
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Usage
```bash
python -m transcriber.main --input "~/Downloads/Tel-From-John_Smith-2025-08-07-18-15-48.m4a" \
  --model medium.en \
  --other-on left \
  --output out/Tel-From-John_Smith-2025-08-07-18-15-48.lrc
```

Options:
- `--input PATH` (required): Stereo M4A (ALAC/AAC) with one speaker per channel
- `--output PATH` (default: alongside input, `.lrc` extension)
- `--model` [tiny|base|small|medium|large-v3|*-en] (default: `medium.en`)
- `--device` [auto|cpu|cuda] (default: `auto`)
- `--compute-type` [auto|int8|int8_float16|float16|float32] (default: auto)
- `--other-on` [left|right] (default: `left`): Which channel is the non-you party
- `--you-name` (default: `You`)
- `--other-name` (override auto-detected name from filename)

Filename convention (auto-detect):
- `Tel-From-<Name>-YYYY-MM-DD-...m4a` → other party name is `<Name>`
- `Tel-To-<Name>-YYYY-MM-DD-...m4a` → other party name is `<Name>`
(Underscores are converted to spaces.)

### Output
Produces `.lrc` with lines like:
```
[00:12.34] John Smith: Great to hear from you.
[00:13.20] You: Likewise!
[00:14.05] John Smith: *laughs* That was fast.
```

### Notes
- The first run of a given model downloads it (hundreds of MB to GB). Choose `small.en` for quick tests; `medium.en` or `large-v3` for quality.
- If you know your channel mapping differs, use `--other-on right`.
