"""
eval_model.py — Standalone ASR spot-check for candidate CTranslate2 models
==========================================================================
A throwaway evaluation harness (NOT a pipeline stage) for auditioning a
fine-tuned Whisper model that has been converted to CTranslate2 format —
e.g. an Egyptian-Arabic large-v3 fine-tune on the feature/upgrade-stack
branch.

It transcribes a folder of short audio clips with the model you point it at
and prints each filename next to its transcription, so you can eyeball the
output against the current pipeline's transcripts.json for the same clips.
No JSON is read or written and no data contract is touched.

Usage:
  python eval_model.py data/eval_clips --model /path/to/ct2-model-dir
  python eval_model.py data/eval_clips --model ./models/egy-large-v3-ct2 --language ar --device cuda
  python eval_model.py data/eval_clips --model ./models/egy-ct2 --device cpu --compute-type int8

The --model path must be a LOCAL CTranslate2 model directory (the output of
ct2-transformers-converter, containing model.bin + config.json + the
tokenizer files). faster-whisper loads it directly from that path.
"""

import argparse
import sys
from pathlib import Path

# Arabic transcriptions crash a Windows cp1252 console on print(); force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Audio containers we sweep the folder for (case-insensitive).
AUDIO_EXTENSIONS = {".wav", ".mp3"}


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Return the audio clips in *audio_dir*, sorted for a stable run order."""
    files = [
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def load_ct2_model(model_path: str, device: str, compute_type: str):
    """
    Load a faster-whisper model from a LOCAL CTranslate2 directory.

    faster-whisper's WhisperModel accepts either a known size name (which it
    would download + convert) or a filesystem path to an already-converted
    CTranslate2 model dir — we use the latter so nothing is fetched remotely.
    """
    from faster_whisper import WhisperModel

    # Mirror asr_transcription.py: CTranslate2 has no fp16 CPU kernels.
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"
        print("  ⚠ float16 unsupported on CPU — using int8 compute type")

    print(f"  Loading CTranslate2 model from: {model_path}")
    print(f"  Device: {device}  |  compute_type: {compute_type}")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print("  ✔ Model loaded.\n")
    return model


def transcribe_file(model, audio_path: Path, language: str, beam_size: int) -> str:
    """Transcribe one clip and return the concatenated text of all segments."""
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
    )
    # segments is a lazy generator — materialise it and join the pieces.
    return " ".join(seg.text.strip() for seg in segments).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — standalone CTranslate2 ASR spot-check",
    )
    parser.add_argument(
        "audio_dir",
        help="Folder containing short .wav/.mp3 clips to transcribe",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a LOCAL CTranslate2 model directory (from ct2-transformers-converter)",
    )
    parser.add_argument(
        "--language",
        default="ar",
        help="Source language code passed to the model  (default: ar)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'cpu', 'cuda', or 'auto'  (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type: 'int8', 'float16', 'float32'  (default: float16)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding  (default: 5)",
    )
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────
    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"✖  Audio folder not found (or not a directory): {audio_dir}")
        sys.exit(1)

    if not Path(args.model).is_dir():
        print(f"✖  Model path is not a directory: {args.model}")
        print("   Expected a local CTranslate2 model dir (model.bin + config.json + tokenizer).")
        sys.exit(1)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        exts = "/".join(sorted(AUDIO_EXTENSIONS))
        print(f"✖  No {exts} files found in {audio_dir}")
        sys.exit(1)

    print(f"[1/2]  Found {len(audio_files)} audio clip(s) in {audio_dir}\n")

    # ── Load the candidate model ─────────────
    model = load_ct2_model(args.model, args.device, args.compute_type)

    # ── Transcribe + print each clip ─────────
    print(f"[2/2]  Transcribing (language={args.language})…\n")
    print("═" * 60)
    for idx, audio_path in enumerate(audio_files, start=1):
        try:
            text = transcribe_file(model, audio_path, args.language, args.beam_size)
        except Exception as exc:  # noqa: BLE001 — spot-check tool, keep sweeping
            print(f"[{idx}/{len(audio_files)}]  {audio_path.name}")
            print(f"    ✖ ERROR: {exc}\n")
            continue

        print(f"[{idx}/{len(audio_files)}]  {audio_path.name}")
        print(f"    → {text or '(empty transcription)'}\n")
    print("═" * 60)
    print(f"  ✅  Done — {len(audio_files)} clip(s) transcribed.")


if __name__ == "__main__":
    main()
