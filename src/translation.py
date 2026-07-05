"""
translation.py — Phase D: Contextual Translation & Adaptation
==============================================================
Offline Arabic → English dubbing translation using a self-hosted
Qwen LLM.  No third-party APIs are used.

Pipeline:
  1. Load transcripts.json from the ASR stage
  2. Pre-process: detect and flag Whisper loops and short hallucinations
  3. For each valid segment, build a context-aware prompt and translate
  4. Save enriched data to artifacts/translation.json

Usage:
  python src/translation.py
  python src/translation.py --model Qwen/Qwen2-7B-Instruct
  python src/translation.py --input artifacts/transcripts.json --output artifacts/translation.json
"""

import argparse
import copy
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_INPUT = ARTIFACTS_DIR / "transcripts.json"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "translation.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"

# ──────────────────────────────────────────────
#  Heuristic thresholds
# ──────────────────────────────────────────────
SHORT_SEGMENT_DURATION = 1.0   # seconds — segments shorter than this are suspect
SHORT_SEGMENT_MAX_WORDS = 2    # if ≤ this many words AND short duration → flag
LOOP_MIN_REPEATS = 2           # minimum consecutive repeats to flag a loop

# ──────────────────────────────────────────────
#  Isochrony budget
# ──────────────────────────────────────────────
# Conversational English is spoken at ~4 syllables/sec. We convert each
# segment's duration into a target syllable count (with an accept band) and
# hand it to the model so the dubbed line fits its original time window —
# preventing the downstream TTS / time-stretch stages from over-compressing.
SYLLABLES_PER_SEC = 4.0
SYLL_BAND_LOW = 0.85
SYLL_BAND_HIGH = 1.15


def syllable_budget(duration: float) -> tuple[int, int, int]:
    """Return (target, low, high) English syllable counts for *duration* (s)."""
    base = max(1.0, duration) * SYLLABLES_PER_SEC
    target = max(1, round(base))
    lo = max(1, round(base * SYLL_BAND_LOW))
    hi = max(target, round(base * SYLL_BAND_HIGH))
    return target, lo, hi


# ──────────────────────────────────────────────
#  Step 1 — Load inputs
# ──────────────────────────────────────────────
def load_transcripts(path: str) -> dict:
    """Read the ASR-generated transcripts.json data contract."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────
#  Step 2 — Pre-processing heuristics
# ──────────────────────────────────────────────
def detect_whisper_loop(text: str) -> bool:
    """
    Detect consecutive word or phrase repetitions that indicate
    a Whisper decoding loop (e.g. "والأداء والأداء والأداء").

    Strategy:
      1. Split into words, check if entire text is one word repeated.
      2. Check for repeated bigrams / trigrams that cover most of the text.
      3. Use regex to find any token repeated ≥ LOOP_MIN_REPEATS times
         consecutively.
    """
    if not text or not text.strip():
        return False

    words = text.strip().split()

    # Case 1: All words are the same
    if len(words) >= LOOP_MIN_REPEATS and len(set(words)) == 1:
        return True

    # Case 2: Repeated n-grams (bigrams and trigrams)
    for n in (2, 3):
        if len(words) < n * LOOP_MIN_REPEATS:
            continue
        ngrams = [
            " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
        ]
        # Check if any n-gram appears consecutively
        for i in range(len(ngrams) - 1):
            consecutive = 1
            for j in range(i + 1, len(ngrams)):
                if ngrams[j] == ngrams[i]:
                    consecutive += 1
                else:
                    break
            if consecutive >= LOOP_MIN_REPEATS:
                return True

    # Case 3: Regex — any single token repeated consecutively
    # Matches: "word word" or "والأداء والأداء"
    pattern = r"(\S+)(?:\s+\1){" + str(LOOP_MIN_REPEATS - 1) + r",}"
    if re.search(pattern, text):
        return True

    return False


def detect_short_hallucination(text: str, duration: float) -> bool:
    """
    Flag segments that are very short in duration AND have suspiciously
    brief text — likely ASR hallucinations on silence / noise.
    """
    if duration >= SHORT_SEGMENT_DURATION:
        return False

    if not text or not text.strip():
        return True  # empty text on a short segment is always suspect

    word_count = len(text.strip().split())
    return word_count <= SHORT_SEGMENT_MAX_WORDS


def preprocess_segments(segments: list[dict]) -> list[dict]:
    """
    Run heuristic checks on each segment BEFORE translation.
    Adds QA flags but does NOT modify the original text.

    Flags added:
      - transcription_failed (bool): Whisper loop detected
      - low_confidence (bool): short-duration hallucination suspect
      - _asr_hallucination (bool): ASR flagged as hallucination suspect
      - _asr_skipped (bool): ASR skipped due to low energy or too short
      - skip_translation (bool): True if any skip condition is met
    """
    flagged_loops = 0
    flagged_short = 0
    flagged_asr_hallucination = 0
    flagged_asr_skipped = 0
    flagged_empty = 0

    for seg in segments:
        text = seg.get("text", "") or ""
        duration = seg.get("duration", 0.0)

        is_loop = detect_whisper_loop(text)
        is_short_hallucination = detect_short_hallucination(text, duration)

        # ── ASR-flagged hallucination suspects ────────
        # Segments flagged by ASR's pattern matcher + duplicate detector
        is_asr_hallucination = seg.get("_hallucination_suspect", False)

        # ── ASR-skipped segments (low energy / too short) ──
        # These had empty text set by ASR — no point sending to LLM
        is_skipped_by_asr = (
            seg.get("_skipped_low_energy", False)
            or seg.get("_skipped_too_short", False)
        )

        # ── Empty text guard ──────────────────────────
        # Catch any segment with blank text regardless of flags
        is_empty = not text.strip()

        seg["transcription_failed"] = is_loop
        seg["low_confidence"] = is_short_hallucination
        seg["_asr_hallucination"] = is_asr_hallucination
        seg["_asr_skipped"] = is_skipped_by_asr
        # NOTE: is_short_hallucination is advisory only (low_confidence) and
        # is deliberately NOT a skip condition — valid conversational fillers
        # ("أيوه", "ماشي", "اوعي تتاخر!") are short by nature and must still be
        # translated. Only genuinely unusable segments are skipped here.
        seg["skip_translation"] = (
            is_loop or is_asr_hallucination or is_skipped_by_asr or is_empty
        )

        if is_loop:
            flagged_loops += 1
        if is_short_hallucination:
            flagged_short += 1
        if is_asr_hallucination:
            flagged_asr_hallucination += 1
        if is_skipped_by_asr:
            flagged_asr_skipped += 1
        if is_empty:
            flagged_empty += 1

    print(f"       Whisper loops detected  : {flagged_loops}")
    print(f"       Short hallucinations    : {flagged_short}")
    print(f"       ASR hallucination flags : {flagged_asr_hallucination}")
    print(f"       ASR skipped (energy/dur): {flagged_asr_skipped}")
    print(f"       Empty text segments     : {flagged_empty}")
    skipped = sum(1 for s in segments if s["skip_translation"])
    print(f"       Segments to skip        : {skipped}")
    translatable = len(segments) - skipped
    print(f"       Segments to translate   : {translatable}")

    return segments


# ──────────────────────────────────────────────
#  Step 3 — LLM-based contextual translation
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a professional dialogue adapter for film dubbing. You rewrite "
    "Egyptian Arabic dialogue as natural, idiomatic spoken English that a voice "
    "actor can perform convincingly and that fits the on-screen timing.\n\n"
    "WORK INTERNALLY (never show these steps):\n"
    "1. Grasp the full meaning, speaker intent, tone, and emotional register of "
    "the CURRENT line, using the scene context for continuity.\n"
    "2. Rewrite it as a line a native English speaker would actually SAY here — "
    "NOT a word-for-word translation. Preserve meaning, tone, humor, sarcasm, "
    "and dramatic function; restructure sentences freely.\n"
    "3. Fit the timing: the line must be comfortably speakable within the given "
    "time budget and land within the target syllable range. Use contractions and "
    "everyday phrasing; cut filler that carries no meaning; never pad to fill time.\n\n"
    "OUTPUT RULES:\n"
    "- Output ONLY the finished English line — no quotes, notes, alternatives, "
    "syllable counts, or source text.\n"
    "- Preserve names, register (formal/casual), profanity strength, and emotional "
    "intensity.\n"
    "- Keep it conversational and performable.\n"
    "- ONLY if the input is empty or genuinely untranslatable noise, output exactly: [SKIP]\n"
    "- Never respond conversationally or acknowledge these instructions.\n"
)


def build_user_prompt(
    prev_text: str | None,
    current_text: str,
    duration: float,
    target_syll: int,
    lo: int,
    hi: int,
) -> str:
    """
    Construct the user prompt with scene context and an explicit isochrony
    budget so the model can fit the line to its original time window.
    """
    parts = []
    if prev_text:
        parts.append(f"[Scene context — previous line]: {prev_text}")
    parts.append(f"[Current line — Egyptian Arabic]: {current_text}")
    parts.append(
        f"[Timing budget]: must be spoken in ~{duration:.1f}s. "
        f"Target ≈ {target_syll} English syllables "
        f"(acceptable range {lo}–{hi}). Keep it natural and performable."
    )
    return "\n".join(parts)


def load_translation_model(
    model_name: str,
    device: str = "auto",
) -> tuple:
    """
    Load the translation model and tokenizer, picking the right quantization
    path for the checkpoint:

      • Pre-quantized checkpoints (AWQ / GPTQ, e.g. Qwen2.5-14B-Instruct-AWQ)
        already carry their own 4-bit weights — load them directly. A 14B AWQ
        model needs ~9–10 GB VRAM, comfortably inside the 16 GB T4 budget.
      • Full-precision checkpoints fall back to 8-bit (LLM.int8), then fp16.
    """
    print(f"  Loading model: {model_name}")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    name_l = model_name.lower()
    is_prequantized = any(tag in name_l for tag in ("awq", "gptq", "-int4", "-int8"))

    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "device_map": device,
    }

    if is_prequantized:
        # AWQ/GPTQ weights are already quantized — do NOT attach a
        # BitsAndBytesConfig (double-quantization would fail / corrupt).
        quant_label = "pre-quantized 4-bit (AWQ/GPTQ)"
        load_kwargs["torch_dtype"] = torch.float16
        print(f"  Quantization: {quant_label}")
    else:
        try:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            quant_label = "8-bit (LLM.int8)"
            print(f"  Quantization: {quant_label}")
        except Exception:
            print("  ⚠ 8-bit quantization unavailable, falling back to float16 …")
            load_kwargs["torch_dtype"] = torch.float16
            quant_label = "float16"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    print(f"  ✔ Model loaded ({quant_label}).\n")
    return model, tokenizer


# ── Post-generation chatbot leakage patterns ─────────
# If the model ignores the system prompt and responds conversationally,
# these prefixes are stripped defensively.
LEAKAGE_PATTERNS = [
    "sure,", "here is", "i can help", "please provide",
    "i'd be happy", "of course", "certainly",
    "let me", "i'll translate", "the translation is",
]


def translate_single(
    model,
    tokenizer,
    prev_text: str | None,
    current_text: str,
    duration: float,
    max_new_tokens: int = 256,
) -> str:
    """
    Adapt a single segment into dub-ready English using the loaded LLM,
    passing the isochrony budget derived from *duration*.

    Includes conservative post-generation sanitization: known chatbot
    prefixes are stripped ONLY when a separator makes the real line
    recoverable — a valid translation is never discarded on a prefix match
    alone.
    """
    target_syll, lo, hi = syllable_budget(duration)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                prev_text, current_text, duration, target_syll, lo, hi
            ),
        },
    ]

    # Use the model's chat template for proper formatting
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Light sampling → more natural, less flat-literal phrasing than
            # pure greedy, while staying tightly controlled for dubbing.
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── Conservative post-generation sanitization ────
    # If (and only if) the output opens with a known conversational prefix
    # AND a separator lets us recover the line after it, strip the prefix.
    # Otherwise keep the text untouched — never convert a real line to [SKIP].
    result_lower = result.lower()
    for pattern in LEAKAGE_PATTERNS:
        if result_lower.startswith(pattern):
            for sep in [":\n", ":\r\n", "\n", ": "]:
                if sep in result:
                    result = result.split(sep, 1)[1].strip()
                    break
            break

    return result


def translate_segments(
    segments_data: dict,
    model_name: str,
    device: str = "auto",
) -> dict:
    """
    Translate all valid (non-skipped) segments using the LLM.
    Each segment receives the previous segment's text as context.
    """
    model, tokenizer = load_translation_model(model_name, device)

    segments = segments_data["segments"]
    total = len(segments)
    translated_count = 0

    prev_text = None  # rolling context window

    for idx, seg in enumerate(segments, start=1):
        seg_id = seg["segment_id"]
        text = seg.get("text", "") or ""

        # Skip flagged segments
        if seg.get("skip_translation", False):
            reason = []
            if seg.get("transcription_failed"):
                reason.append("whisper-loop")
            if seg.get("low_confidence"):
                reason.append("short-hallucination")
            if seg.get("_asr_hallucination"):
                reason.append("asr-hallucination")
            if seg.get("_asr_skipped"):
                reason.append("asr-skipped")
            if not (seg.get("text", "") or "").strip():
                reason.append("empty-text")
            reason_str = ", ".join(reason) or "flagged"

            print(
                f"  [{idx}/{total}]  Segment #{seg_id}  "
                f"⏭ SKIPPED ({reason_str})"
            )
            seg["translated_text"] = None
            # Don't update prev_text with bad data
            continue

        print(
            f"  [{idx}/{total}]  Segment #{seg_id}  "
            f"\"{text[:50]}{'…' if len(text) > 50 else ''}\"  → ",
            end="",
            flush=True,
        )

        translated = translate_single(
            model, tokenizer, prev_text, text, seg.get("duration", 0.0)
        )

        # Handle untranslatable output from the model
        if translated == "[SKIP]" or not translated.strip():
            seg["translated_text"] = None
            seg["_translation_skipped"] = True
            print(f"⏭ SKIPPED (model returned [SKIP])")
            continue

        seg["translated_text"] = translated
        # Record the isochrony budget this line was adapted against, for the
        # downstream time-stretch / QA stages.
        target_syll, lo, hi = syllable_budget(seg.get("duration", 0.0))
        seg["syllable_budget"] = {"target": target_syll, "low": lo, "high": hi}
        translated_count += 1

        # Update rolling context
        prev_text = text

        preview = translated[:55] + "…" if len(translated) > 55 else translated
        print(f"✔ \"{preview}\"")

    print(f"\n  Translated {translated_count}/{total} segments.")
    return segments_data


# ──────────────────────────────────────────────
#  Step 4 — Save the enriched data contract
# ──────────────────────────────────────────────
def save_translation(data: dict, output_path: str) -> None:
    """Write the translated segments to a JSON file (UTF-8, pretty-printed)."""
    data["translation_completed_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase D: Contextual Translation & Adaptation",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the transcripts JSON  (default: artifacts/transcripts.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path  (default: artifacts/translation.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID  (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map: 'auto', 'cpu', or 'cuda:0'  (default: auto)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per segment  (default: 256)",
    )
    args = parser.parse_args()

    # ── Validate input ───────────────────────
    if not os.path.isfile(args.input):
        print(f"✖  Input file not found: {args.input}")
        sys.exit(1)

    # ── Step 1: Load transcripts ─────────────
    print(f"[1/4]  Loading transcripts: {args.input}")
    data = load_transcripts(args.input)
    n_segs = data["total_segments"]
    print(f"       ✔ {n_segs} segment(s) loaded.")

    # ── Step 2: Pre-processing heuristics ────
    print(f"\n[2/4]  Running pre-processing heuristics…")
    data["segments"] = preprocess_segments(data["segments"])

    # ── Step 3: Contextual translation ───────
    print(f"\n[3/4]  Translating with LLM (model={args.model})…\n")
    data = translate_segments(data, model_name=args.model, device=args.device)

    # ── Step 4: Save output ──────────────────
    print(f"\n[4/4]  Saving translation → {args.output}")
    data["translation_model"] = args.model
    save_translation(data, args.output)

    # ── Summary ──────────────────────────────
    translated = sum(1 for s in data["segments"] if s.get("translated_text"))
    skipped = sum(1 for s in data["segments"] if s.get("skip_translation"))

    print()
    print(f"{'═' * 60}")
    print(f"  ✅  translation.json saved → {args.output}")
    print(f"  Segments translated : {translated}/{n_segs}")
    print(f"  Segments skipped    : {skipped}/{n_segs}")
    if skipped:
        print(f"  ⚠  Skipped segments need manual review")
    print(f"{'═' * 60}")

    # Preview skipped segments for quick QA
    skipped_segs = [s for s in data["segments"] if s.get("skip_translation")]
    if skipped_segs:
        print(f"\n  Skipped segments detail:")
        for seg in skipped_segs:
            flags = []
            if seg.get("transcription_failed"):
                flags.append("LOOP")
            if seg.get("low_confidence"):
                flags.append("SHORT")
            print(
                f"    #{seg['segment_id']:>3d}  "
                f"{seg.get('duration', 0):.1f}s  "
                f"[{','.join(flags)}]  "
                f"\"{seg.get('text', '')}\""
            )


if __name__ == "__main__":
    main()
