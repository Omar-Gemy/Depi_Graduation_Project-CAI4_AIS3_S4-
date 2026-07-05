"""
asr_transcription.py — Phase C: ASR Layer
==========================================
Offline speech-to-text transcription using WhisperX (faster-whisper /
CTranslate2 backend + wav2vec2 forced alignment).  No third-party APIs.

Strategy — Full-File Transcription + Global-to-Local Word Mapping:
  1. Load the normalised audio and the diarization-enriched segments.json
  2. Resolve source language + registry config (initial_prompt, patterns)
  3. Transcribe the ENTIRE file in one pass with WhisperX so linguistic
     context is preserved (fixes the phonetic collapse caused by
     transcribing decontextualised per-VAD slices)
  4. Run wav2vec2 forced alignment to obtain precise word-level timestamps
  5. Intersect the aligned words with the existing diarization segments
     (each word is assigned to the segment its midpoint falls inside) —
     the same global-to-local pattern used by diarization.py
  6. Apply prompt-echo + hallucination guards, then write transcripts.json

Why full-file instead of per-slice:
  faster-whisper is as much a language model as an acoustic one. Feeding it
  2–4 s decontextualised chunks strips the surrounding words it needs to
  disambiguate colloquial / rare vocabulary and lets clipped VAD boundaries
  turn into garbled tokens. Transcribing the whole file keeps every 30 s
  window internally coherent; alignment then re-localises words to segments.

Usage:
  python src/asr_transcription.py
  python src/asr_transcription.py --model large-v3 --source-lang ar
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "audio_out" / "_temp_normalised.wav"
DEFAULT_SEGMENTS = ARTIFACTS_DIR / "segments.json"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "transcripts.json"
LANGUAGE_REGISTRY_PATH = PROJECT_ROOT / "config" / "language_registry.json"


# ──────────────────────────────────────────────
#  Language Registry — dynamic config loader
# ──────────────────────────────────────────────
REQUIRED_LANG_KEYS = {"name", "whisper_language", "initial_prompt", "hallucination_patterns"}

# Fallback config used when the detected/forced language is not in the registry.
FALLBACK_LANGUAGE_CONFIG = {
    "name": "Unknown",
    "whisper_language": None,  # set to the detected code at runtime
    "initial_prompt": "",
    "hallucination_patterns": [],
}


def load_language_registry(registry_path: Path = LANGUAGE_REGISTRY_PATH) -> dict:
    """
    Load and validate the language registry JSON file.

    Each language entry must contain: name, whisper_language,
    initial_prompt, and hallucination_patterns.

    Returns the parsed registry dict keyed by language code.
    Raises SystemExit on I/O or validation errors.
    """
    if not registry_path.is_file():
        print(f"✖  Language registry not found: {registry_path}")
        sys.exit(1)

    with open(registry_path, "r", encoding="utf-8") as fh:
        registry = json.load(fh)

    # Validate every language entry
    for lang_code, config in registry.items():
        missing = REQUIRED_LANG_KEYS - set(config.keys())
        if missing:
            print(
                f"✖  Language registry: '{lang_code}' is missing "
                f"required key(s): {', '.join(sorted(missing))}"
            )
            sys.exit(1)

    return registry


def resolve_source_language(
    source_lang_cli: str,
    segments_data: dict,
) -> tuple[str | None, str]:
    """
    Determine the source language WITHOUT touching the audio.

    Cascade:
      1. Explicit CLI --source-lang (not 'auto')
      2. segments.json 'source_language' (not 'auto')
      3. None → let WhisperX auto-detect during transcription

    Returns (language_code_or_None, source_description).
    """
    if source_lang_cli and source_lang_cli != "auto":
        return source_lang_cli, "CLI --source-lang"

    seg_lang = segments_data.get("source_language")
    if seg_lang and seg_lang != "auto":
        return seg_lang, "segments.json"

    return None, "auto-detect"


def lookup_lang_config(lang_code: str, registry: dict) -> tuple[dict, list[str]]:
    """
    Return (language_config, warnings) for *lang_code*, falling back to a
    neutral config (no prompt, no patterns) if the code is unknown.
    """
    warnings: list[str] = []
    if lang_code in registry:
        return registry[lang_code], warnings

    fallback = dict(FALLBACK_LANGUAGE_CONFIG)
    fallback["whisper_language"] = lang_code
    warn_msg = (
        f"Language '{lang_code}' not found in registry — using fallback "
        f"config (no initial_prompt, no hallucination patterns)"
    )
    warnings.append(warn_msg)
    return fallback, warnings


# ──────────────────────────────────────────────
#  Step 1 — Load inputs
# ──────────────────────────────────────────────
def load_segments(segments_path: str) -> dict:
    """Read the diarization-enriched segments.json data contract."""
    with open(segments_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────
#  Anti-hallucination guards
# ──────────────────────────────────────────────
# Signature fragments of the legacy "word-list" initial_prompt. Any transcript
# that reproduces these is a prompt-echo hallucination, not real speech.
PROMPT_ECHO_SIGNATURES = [
    "المتحدثين بيتكلموا",
    "وبيستخدموا كلمات",
]
PROMPT_ECHO_OVERLAP = 0.7  # token overlap with the prompt that flags an echo


def is_prompt_echo(text: str, initial_prompt: str) -> bool:
    """
    Return True if *text* is (mostly) a regurgitation of *initial_prompt*.

    Whisper can emit the initial_prompt verbatim on low-information audio.
    We catch it two ways:
      1. Known signature fragments of the old leaky prompt.
      2. High token overlap between the transcript and the active prompt.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    for sig in PROMPT_ECHO_SIGNATURES:
        if sig in stripped:
            return True

    if initial_prompt:
        prompt_tokens = set(initial_prompt.split())
        tokens = stripped.split()
        if len(tokens) >= 4 and prompt_tokens:
            overlap = sum(1 for t in tokens if t in prompt_tokens) / len(tokens)
            if overlap >= PROMPT_ECHO_OVERLAP:
                return True

    return False


def is_hallucination(text: str, hallucination_patterns: list[str]) -> bool:
    """
    Return True if *text* looks like a known Whisper hallucination:
      1. Known hallucinated phrases (substring match) from the registry.
      2. Excessive repetition (same trigram repeated ≥3 times).
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    for pattern in hallucination_patterns:
        if pattern in stripped:
            return True

    words = stripped.split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        for tri in set(trigrams):
            if trigrams.count(tri) >= 3:
                return True

    return False


# ──────────────────────────────────────────────
#  Global-to-Local: map aligned words → segments
# ──────────────────────────────────────────────
def assign_words_to_segments(
    segments: list[dict],
    words: list[dict],
) -> list[dict]:
    """
    Populate each segment's ``text`` field with the aligned words whose
    midpoint falls inside the segment's [start_time, end_time] window.

    Using the word midpoint (rather than raw overlap) gives each word to
    exactly one segment and is robust to small alignment jitter at
    boundaries.
    """
    for seg in segments:
        start = seg["start_time"]
        end = seg["end_time"]
        tokens = [
            w["word"]
            for w in words
            if start <= (w["start"] + w["end"]) / 2.0 <= end
        ]
        seg["text"] = " ".join(tokens).strip()
    return segments


# ──────────────────────────────────────────────
#  Steps 2–3 — Full-file transcription + alignment
# ──────────────────────────────────────────────
def transcribe_full_file(
    audio_path: str,
    segments_data: dict,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    source_lang: str = "auto",
    batch_size: int = 16,
    align_model_name: str | None = None,
) -> dict:
    """
    Transcribe the whole audio file with WhisperX, force-align to word level,
    then intersect words with the diarization segments in *segments_data*.

    Returns *segments_data* with populated ``text`` fields.
    """
    import torch
    import whisperx

    # ── Normalise device / compute_type ───────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and compute_type == "float16":
        # CTranslate2 has no fp16 CPU kernels — fall back to int8.
        compute_type = "int8"
        print("  ⚠ float16 unsupported on CPU — using int8 compute type")

    registry = load_language_registry()

    # ── Resolve language BEFORE loading (so initial_prompt can be applied) ──
    prelim_lang, lang_source = resolve_source_language(source_lang, segments_data)
    initial_prompt = None
    if prelim_lang is not None:
        prelim_config, _ = lookup_lang_config(prelim_lang, registry)
        initial_prompt = prelim_config["initial_prompt"] or None
        print(f"  ✔ Source language: {prelim_lang}  ({lang_source})")
    else:
        print("  Source language: auto — WhisperX will detect from audio")

    # ── Decoding options ──────────────────────────
    # Loosened fallback thresholds (were 1.6 / -0.5): the strict values made
    # hard colloquial segments fail the checks and escalate to higher
    # temperatures, injecting variance exactly where the audio was hardest.
    asr_options = {
        "beam_size": 8,
        "temperatures": [0.0, 0.2, 0.4],
        "compression_ratio_threshold": 2.4,   # was 1.6
        "log_prob_threshold": -1.0,           # was -0.5
        "no_speech_threshold": 0.6,
        "repetition_penalty": 1.2,
    }
    if initial_prompt:
        # Applied ONCE to the full-file pass (not per slice) — combined with
        # the short stylistic prompt + echo guard this avoids Obs-1 leakage.
        asr_options["initial_prompt"] = initial_prompt

    # ── Load model + transcribe the FULL file ─────
    print(f"  Loading WhisperX model '{model_size}' on {device} ({compute_type}) …")
    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type=compute_type,
        language=prelim_lang,
        asr_options=asr_options,
    )
    audio = whisperx.load_audio(audio_path)
    print("  ▶ Transcribing full audio …")
    result = model.transcribe(audio, batch_size=batch_size)
    detected_lang = result.get("language") or prelim_lang or "en"

    # Free the ASR model before alignment (VRAM budget).
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ── Finalise language config (covers the auto-detected case) ──
    lang_config, lang_warnings = lookup_lang_config(detected_lang, registry)
    hallucination_patterns = lang_config["hallucination_patterns"]
    active_prompt = lang_config["initial_prompt"] or (initial_prompt or "")
    print(f"  ✔ Language config: {lang_config['name']} ({detected_lang})")

    # ── Forced alignment → word-level timestamps ──
    # By default WhisperX picks its built-in wav2vec2 checkpoint for the
    # detected language. --align-model overrides this with an explicit HF
    # model name (e.g. a stronger Egyptian-Arabic wav2vec2) if the default
    # aligns colloquial speech poorly.
    if align_model_name:
        print(
            f"  ▶ Loading alignment model '{align_model_name}' "
            f"(override) for '{detected_lang}' …"
        )
    else:
        print(f"  ▶ Loading wav2vec2 alignment model for '{detected_lang}' …")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang,
        device=device,
        model_name=align_model_name,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    del align_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ── Flatten aligned words (absolute timestamps) ──
    words: list[dict] = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            # Alignment can leave a word without timing (e.g. digits/symbols).
            if w.get("start") is None or w.get("end") is None:
                continue
            token = (w.get("word") or "").strip()
            if token:
                words.append({"word": token, "start": w["start"], "end": w["end"]})
    print(f"  ✔ {len(words)} aligned word(s) across the file")

    # ── Global-to-local mapping onto diarization segments ──
    segments = segments_data["segments"]
    assign_words_to_segments(segments, words)

    # ── Prompt-echo + hallucination guards ────────
    n_echo = 0
    n_suspect = 0
    n_filled = 0
    for seg in segments:
        text = seg.get("text", "") or ""

        if is_prompt_echo(text, active_prompt):
            seg["_prompt_echo_filtered"] = text
            seg["text"] = ""
            n_echo += 1
            continue

        if is_hallucination(text, hallucination_patterns):
            seg["_hallucination_suspect"] = True
            seg["_hallucination_matched_pattern"] = text

        if seg.get("text", "").strip():
            n_filled += 1
        if seg.get("_hallucination_suspect"):
            n_suspect += 1

    print(f"  ✔ Segments with text     : {n_filled}/{len(segments)}")
    print(f"  ⚠ Prompt-echo filtered   : {n_echo}")
    print(f"  ⚠ Hallucination suspects : {n_suspect}")

    # ── Record contract metadata ──────────────────
    segments_data["source_language"] = detected_lang
    segments_data["language_config"] = {
        "name": lang_config["name"],
        "registry_language": detected_lang,
        "used_fallback": detected_lang not in registry,
    }
    if lang_warnings:
        segments_data["language_config"]["warnings"] = lang_warnings

    return segments_data


# ──────────────────────────────────────────────
#  Step 4 — Save the enriched data contract
# ──────────────────────────────────────────────
def save_transcripts(data: dict, output_path: str) -> None:
    """Write the transcribed segments to a JSON file (UTF-8, pretty-printed)."""
    data["asr_completed_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        def np_encoder(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError

        json.dump(data, fh, indent=2, ensure_ascii=False, default=np_encoder)


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase C: ASR Transcription (WhisperX)",
    )
    parser.add_argument(
        "--input-audio",
        default=str(DEFAULT_AUDIO),
        help="Path to the normalised WAV file  (default: data/audio_out/_temp_normalised.wav)",
    )
    parser.add_argument(
        "--input-segments",
        default=str(DEFAULT_SEGMENTS),
        help="Path to the diarization segments JSON  (default: artifacts/segments.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path  (default: artifacts/transcripts.json)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size  (default: large-v3)",
    )
    parser.add_argument(
        "--source-lang",
        default="auto",
        help="Source language code (e.g. 'ar', 'en') or 'auto' for detection  (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device: 'cpu', 'cuda', or 'auto'  (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type: 'int8', 'float16', 'float32'  (default: float16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="WhisperX batched-inference batch size  (default: 16)",
    )
    parser.add_argument(
        "--align-model",
        default=None,
        help=(
            "Override the wav2vec2 forced-alignment model with an explicit "
            "HuggingFace model name (e.g. a dedicated Egyptian-Arabic "
            "wav2vec2). Default: WhisperX's built-in model for the language."
        ),
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
    print(f"[1/3]  Loading segments: {args.input_segments}")
    segments_data = load_segments(args.input_segments)
    n_segs = segments_data["total_segments"]
    print(f"       ✔ {n_segs} segment(s) loaded.")

    # ── Steps 2–3: Transcribe + align + map ──
    print(f"\n[2/3]  Transcribing with WhisperX (model={args.model})…\n")
    segments_data = transcribe_full_file(
        args.input_audio,
        segments_data,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        source_lang=args.source_lang,
        batch_size=args.batch_size,
        align_model_name=args.align_model,
    )

    # ── Step 4: Save output ──────────────────
    print(f"\n[3/3]  Saving transcripts → {args.output}")
    segments_data["asr_model"] = f"whisperx/{args.model}"
    segments_data["align_model"] = args.align_model or "whisperx-default"
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
