from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Optional

import click

from .audio_utils import SplitResult, split_stereo_to_mono
from .lrc_writer import write_lrc
from .whisper_utils import load_model, merge_dialogue, transcribe_file


def _default_output_path(input_path: str) -> str:
    base, _ = os.path.splitext(os.path.basename(input_path))
    return os.path.join(os.path.dirname(input_path), base + ".lrc")


def _parse_other_party_from_filename(path: str) -> Optional[str]:
    name = None
    fname = os.path.basename(path)
    # Examples: Tel-From-Jonathan_Dayley-2025-08-07-18-15-48.m4a
    m = re.search(r"Tel-(?:From|To)-(.+?)-\d{4}-\d{2}-\d{2}-", fname)
    if m:
        name = m.group(1)
        name = name.replace("_", " ")
    return name


@click.command()
@click.option("--input", "input_path", type=str, required=True, help="Input stereo M4A path (ALAC/AAC)")
@click.option("--output", "output_path", type=str, default=None, help="Output .lrc path (default beside input)")
@click.option("--model", "model_size", type=str, default="medium.en", help="Whisper model size (e.g., small.en, medium.en, large-v3)")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
@click.option("--compute-type", "compute_type", type=str, default=None, help="faster-whisper compute type (auto)")
@click.option("--other-on", type=click.Choice(["left", "right"]), default="left", help="Which channel is the non-you party")
@click.option("--you-name", type=str, default="You", help="Override name for your channel label")
@click.option("--other-name", type=str, default=None, help="Override name parsed from filename for the other party")
@click.option("--normalize", type=click.Choice(["loudnorm", "dynaudnorm", "none"]), default="loudnorm", help="Per-channel normalization")
def main(
    input_path: str,
    output_path: Optional[str],
    model_size: str,
    device: str,
    compute_type: Optional[str],
    other_on: str,
    you_name: str,
    other_name: Optional[str],
    normalize: str,
) -> None:
    input_path = os.path.expanduser(input_path)
    if not os.path.exists(input_path):
        click.echo(f"Input not found: {input_path}", err=True)
        sys.exit(1)

    if output_path is None:
        output_path = _default_output_path(input_path)

    detected_other = _parse_other_party_from_filename(input_path)
    other_label = other_name or detected_other or "Other"

    # Split channels
    click.echo("Splitting stereo into left/right mono...")
    split: SplitResult = split_stereo_to_mono(input_path, normalizer=normalize)

    # Map speakers to channels
    if other_on == "left":
        left_label, right_label = other_label, you_name
        left_path, right_path = split.left_wav_path, split.right_wav_path
    else:
        left_label, right_label = you_name, other_label
        left_path, right_path = split.left_wav_path, split.right_wav_path

    # Load model
    click.echo(f"Loading whisper model: {model_size} ({device}) ...")
    model = load_model(model_size, device=device, compute_type=compute_type)

    # Transcribe each channel with VAD
    click.echo(f"Transcribing LEFT channel as '{left_label}' ...")
    left_segments = transcribe_file(model, left_path)

    click.echo(f"Transcribing RIGHT channel as '{right_label}' ...")
    right_segments = transcribe_file(model, right_path)

    # Merge into linear dialog
    click.echo("Merging dialogue and writing LRC...")
    merged = merge_dialogue(left_segments, left_label, right_segments, right_label)

    # Metadata
    title = os.path.splitext(os.path.basename(input_path))[0]
    artists = f"{left_label} & {right_label}"

    write_lrc(merged, output_path, title=title, artists=artists)

    click.echo(f"Done. Wrote: {output_path}")

    # Cleanup temp dir
    try:
        if os.path.isdir(split.temp_dir):
            shutil.rmtree(split.temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
