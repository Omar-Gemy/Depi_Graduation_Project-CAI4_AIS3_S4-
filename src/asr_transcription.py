"""
asr_transcription.py — Phase C: ASR Layer
==========================================
Offline speech-to-text transcription using faster-whisper
(CTranslate2 backend).  No third-party APIs are used.

Pipeline:
  1. Load the normalised audio and the VAD-generated segments.json
  2. Detect source language (or accept explicit --source-lang)
  3. For each segment, slice the audio by start_time / end_time
  4. Pass each audio slice to faster-whisper with language forcing
  5. Populate the ``text`` field in every segment
  6. Write the enriched data to artifacts/transcripts.json

Usage:
  python src/asr_transcription.py
  python src/asr_transcription.py --model small --source-lang ar
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "audio_out" / "_temp_normalised.wav"
DEFAULT_SEGMENTS = ARTIFACTS_DIR / "segments.json"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "transcripts.json"


# ──────────────────────────────────────────────
#  Step 1 — Load inputs
# ──────────────────────────────────────────────
def load_segments(segments_path: str) -> dict:
    """Read the VAD-generated segments.json data contract."""
    with open(segments_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Read the full audio file and return (samples, sample_rate).
    Uses soundfile for Windows compatibility.
    """
    data, sr = sf.read(audio_path, dtype="float32")
    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


# ──────────────────────────────────────────────
#  Step 2 — Slice audio by VAD timestamps
# ──────────────────────────────────────────────
def slice_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Extract a segment from the audio array using sample-accurate
    start/end times (in seconds).
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)
    return audio_data[start_sample:end_sample]


# ──────────────────────────────────────────────
#  Step 2b — Auto-detect source language
# ──────────────────────────────────────────────
LANGID_DURATION_SEC = 30  # seconds of audio used for language detection


def detect_language(
    audio_data: np.ndarray,
    sample_rate: int,
    model: WhisperModel,
) -> str:
    """
    Feed the first 30 seconds of the full audio to Whisper and
    return the detected language code (e.g. "ar", "en").

    This is a lightweight probe — we only need the ``info`` object,
    not the full transcription.
    """
    max_samples = int(LANGID_DURATION_SEC * sample_rate)
    probe_chunk = audio_data[:max_samples]

    # Run a minimal transcribe just to get the detected language
    _segments, info = model.transcribe(
        probe_chunk,
        vad_filter=False,
        without_timestamps=True,
    )
    # Consume the generator so CTranslate2 releases resources
    for _ in _segments:
        pass

    return info.language


# ──────────────────────────────────────────────
#  Step 3 — Transcribe each segment
# ──────────────────────────────────────────────
def transcribe_segments(
    audio_data: np.ndarray,
    sample_rate: int,
    segments_data: dict,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    source_lang: str = "auto",
) -> dict:
    """
    For each segment in *segments_data*, slice the audio and run
    faster-whisper transcription.  Whisper's internal VAD is disabled
    so we rely strictly on our Silero-based timestamps.

    If *source_lang* is ``"auto"``, the first 30 s of audio are used
    to detect the language.  The detected (or provided) language is
    then forced on every segment to prevent cross-lingual
    hallucinations.

    Returns a new dict with populated ``text`` fields, suitable for
    writing to ``transcripts.json``.
    """
    print(f"  Loading faster-whisper model '{model_size}' …")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    print("  ✔ Model loaded.\n")

    # ── Language identification ───────────────
    if source_lang == "auto":
        print("  Detecting source language from first 30s …")
        detected_lang = detect_language(audio_data, sample_rate, model)
        print(f"  ✔ Detected language: {detected_lang}\n")
    else:
        detected_lang = source_lang
        print(f"  ✔ Language forced to: {detected_lang}\n")

    # Store in the data contract for downstream stages
    segments_data["source_language"] = detected_lang

    segments = segments_data["segments"]
    total = len(segments)

    for idx, seg in enumerate(segments, start=1):
        start = seg["start_time"]
        end = seg["end_time"]
        duration = seg["duration"]

        print(
            f"  [{idx}/{total}]  Segment #{seg['segment_id']}  "
            f"({start:.3f}s → {end:.3f}s, {duration:.3f}s) … ",
            end="",
            flush=True,
        )

        # Slice the audio chunk for this VAD segment
        chunk = slice_audio(audio_data, sample_rate, start, end)

        if len(chunk) == 0:
            seg["text"] = ""
            print("⚠ empty slice, skipped.")
            continue

        # faster-whisper can accept a numpy array directly,
        # but it must be float32 mono at the model's expected rate (16 kHz).
        # If our audio is already 16 kHz mono (from ingestion), we can
        # pass it directly.  Otherwise, write a temp WAV.
        if sample_rate == 16000:
            audio_input = chunk
        else:
            # Write a temporary WAV file for non-16 kHz audio
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            try:
                sf.write(tmp.name, chunk, sample_rate, subtype="PCM_16")
                audio_input = tmp.name
            finally:
                tmp.close()

        # Transcribe — disable Whisper's internal VAD
        whisper_segments, info = model.transcribe(
            audio_input,
            language=detected_lang,    # force language — prevent cross-lingual hallucination
            vad_filter=False,          # rely on OUR VAD, not Whisper's
            without_timestamps=True,   # we already have timestamps
            beam_size=5,               # wider beam for short segments
            temperature=0.0,           # greedy — reduces hallucination on short audio
            initial_prompt="حديث بالعربية عن تحليل البيانات والأداء والإيرادات",
        )

        # Collect the text from Whisper's output segments
        text_parts = []
        for ws in whisper_segments:
            text_parts.append(ws.text.strip())

        full_text = " ".join(text_parts).strip()
        seg["text"] = full_text

        # Clean up temp file if we created one
        if sample_rate != 16000 and isinstance(audio_input, str):
            os.unlink(audio_input)

        # Truncate for display
        preview = full_text[:60] + "…" if len(full_text) > 60 else full_text
        print(f"✔ \"{preview}\"")

    return segments_data


# ──────────────────────────────────────────────
#  Step 4 — Save the enriched data contract
# ──────────────────────────────────────────────
def save_transcripts(data: dict, output_path: str) -> None:
    """Write the transcribed segments to a JSON file (UTF-8, pretty-printed)."""
    # Add ASR metadata
    data["asr_completed_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase C: ASR Transcription (faster-whisper)",
    )
    parser.add_argument(
        "--input-audio",
        default=str(DEFAULT_AUDIO),
        help="Path to the normalised WAV file  (default: data/audio_out/_temp_normalised.wav)",
    )
    parser.add_argument(
        "--input-segments",
        default=str(DEFAULT_SEGMENTS),
        help="Path to the VAD segments JSON  (default: artifacts/segments.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path  (default: artifacts/transcripts.json)",
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"],
        help="Whisper model size  (default: small)",
    )
    parser.add_argument(
        "--source-lang",
        default="auto",
        help="Source language code (e.g. 'ar', 'en') or 'auto' for detection  (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'cpu', 'cuda', or 'auto'  (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type: 'int8', 'float16', 'float32', or 'auto'  (default: auto)",
    )
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────
    if not os.path.isfile(args.input_audio):
        print(f"✖  Audio file not found: {args.input_audio}")
        sys.exit(1)

    if not os.path.isfile(args.input_segments):
        print(f"✖  Segments JSON not found: {args.input_segments}")
        sys.exit(1)

    # ── Step 1: Load inputs ──────────────────
    print(f"[1/3]  Loading audio: {args.input_audio}")
    audio_data, sample_rate = load_audio(args.input_audio)
    audio_duration = len(audio_data) / sample_rate
    print(f"       ✔ {audio_duration:.1f}s of audio at {sample_rate} Hz")

    print(f"       Loading segments: {args.input_segments}")
    segments_data = load_segments(args.input_segments)
    n_segs = segments_data["total_segments"]
    print(f"       ✔ {n_segs} segment(s) loaded.")

    # ── Step 2–3: Transcribe ─────────────────
    print(f"\n[2/3]  Transcribing with faster-whisper (model={args.model})…\n")
    segments_data = transcribe_segments(
        audio_data,
        sample_rate,
        segments_data,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        source_lang=args.source_lang,
    )

    # ── Step 4: Save output ──────────────────
    print(f"\n[3/3]  Saving transcripts → {args.output}")
    # Record which model was used
    segments_data["asr_model"] = f"faster-whisper/{args.model}"
    segments_data["source_language"] = segments_data.get("source_language", args.source_lang)
    save_transcripts(segments_data, args.output)

    # ── Summary ──────────────────────────────
    filled = sum(1 for s in segments_data["segments"] if s.get("text"))
    empty = n_segs - filled

    print()
    print(f"{'═' * 55}")
    print(f"  ✅  transcripts.json saved → {args.output}")
    print(f"  Segments transcribed : {filled}/{n_segs}")
    if empty:
        print(f"  ⚠  Empty transcriptions: {empty}")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
