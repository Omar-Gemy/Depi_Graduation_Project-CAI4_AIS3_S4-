"""
mix_render.py — Phase F, Step 2: Final Audio Mix & Video Render
================================================================
Assemble the time-stretched dubbed segments into a full-length
audio track, mix with the original video, and export the final
dubbed video.

Strategy (approved by Tech Lead):
  - Skipped segments → original Arabic audio passthrough (Q2: Option B)
  - No automated lip-sync QA (Q3: Option B — manual visual review only)
  - Final video codec → AAC 192k (Q4: Option A)
  - Segments placed at their original start_time positions
  - Crossfade transitions to avoid clicks at segment boundaries

Inputs:
  - artifacts/stretch_manifest.json   (Phase F Step 1 data contract)
  - artifacts/segments.json           (Phase B data contract — timing info)
  - artifacts/audio_out/stretched/    (time-fitted WAV segments)
  - data/audio_in/sample.mp4         (source video)

Outputs:
  - data/audio_out/final_dubbed.wav   (full-length dubbed audio)
  - data/audio_out/final_dubbed.mp4   (final video with dubbed audio track)
  - artifacts/mix_manifest.json       (Phase F Step 2 data contract)

Usage:
  python src/mix_render.py
  python src/mix_render.py --source data/audio_in/sample.mp4
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ──────────────────────────────────────────────
#  Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR      = PROJECT_ROOT / "artifacts"
AUDIO_OUT_DIR      = ARTIFACTS_DIR / "audio_out"
STRETCHED_DIR      = AUDIO_OUT_DIR / "stretched"
DATA_AUDIO_IN      = PROJECT_ROOT / "data" / "audio_in"
DATA_AUDIO_OUT     = PROJECT_ROOT / "data" / "audio_out"

DEFAULT_SOURCE     = DATA_AUDIO_IN / "sample.mp4"
SEGMENTS_FILE      = ARTIFACTS_DIR / "segments.json"
STRETCH_MANIFEST   = ARTIFACTS_DIR / "stretch_manifest.json"
MIX_MANIFEST       = ARTIFACTS_DIR / "mix_manifest.json"

# ──────────────────────────────────────────────
#  Audio constants
# ──────────────────────────────────────────────
SAMPLE_RATE        = 24000     # XTTS v2 native output rate → dubbed segs need no resample
CROSSFADE_MS       = 30        # Equal-power crossfade at real segment overlaps
DECLICK_MS         = 5         # Tiny head/tail fade to kill boundary pops
AAC_BITRATE        = "192k"    # Q4 decision: AAC 192k
LOUDNORM_I         = -16       # EBU R128 integrated loudness target (LUFS, web)
LOUDNORM_TP        = -1.5      # True-peak ceiling (dBTP)
LOUDNORM_LRA       = 11        # Loudness range target


# ──────────────────────────────────────────────
#  Step 1 — Extract original audio for passthrough
# ──────────────────────────────────────────────
def extract_segment_audio(
    source_video: Path,
    start_time: float,
    duration: float,
    output_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """
    Extract a segment of audio from the source video using FFmpeg.
    Used for Arabic passthrough of skipped segments (Q2 decision).
    Returns the output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_video),
        "-ss", str(start_time),
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg extract failed ({start_time:.2f}s, {duration:.2f}s):\n"
            f"{result.stderr[:500]}"
        )

    return output_path


def get_source_total_duration(source_video: Path) -> float:
    """Get the total duration of the source video/audio in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(source_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")

    return float(result.stdout.strip())


# ──────────────────────────────────────────────
#  Step 2 — Build the full-length audio timeline
# ──────────────────────────────────────────────
def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Anti-aliased resample. Uses scipy polyphase (resample_poly) when available;
    falls back to linear interpolation only if scipy is missing. At the 24 kHz
    pipeline rate, dubbed segments already match and skip this entirely.
    """
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if _HAVE_SCIPY:
        from math import gcd
        g = gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    # Fallback: linear interpolation (aliasing-prone) — kept only for safety.
    old_len = len(audio)
    new_len = int(old_len * dst_sr / src_sr)
    return np.interp(
        np.linspace(0, old_len - 1, new_len),
        np.arange(old_len),
        audio,
    ).astype(np.float32)


def apply_crossfade(
    audio: np.ndarray,
    position: int,
    segment: np.ndarray,
    fade_samples: int,
    declick_samples: int,
) -> None:
    """
    Prepare a segment for additive placement into the timeline.

    Two distinct operations (audit #29/#30):
      1. De-click: a very short (declick_samples) fade on the segment's head and
         tail so it can't pop when it starts/ends over silence.
      2. Equal-power crossfade: applied ONLY where the segment's head overlaps
         existing non-zero timeline content (a real adjacent-segment boundary).
         Both sides use sqrt gains so summed energy stays constant — no dip.
    """
    seg_len = len(segment)

    # 1. De-click head & tail (always) ────────────
    d = min(declick_samples, seg_len // 2)
    if d > 0:
        segment[:d] *= np.linspace(0.0, 1.0, d, dtype=np.float32)
        segment[-d:] *= np.linspace(1.0, 0.0, d, dtype=np.float32)

    # 2. Equal-power crossfade at real overlaps ───
    f = min(fade_samples, seg_len)
    if f > 0:
        head_end = min(position + f, len(audio))
        hlen = head_end - position
        if hlen > 0 and np.any(audio[position:head_end] != 0.0):
            t = np.linspace(0.0, 1.0, hlen, dtype=np.float32)
            segment[:hlen] *= np.sqrt(t)              # equal-power fade-in
            audio[position:head_end] *= np.sqrt(1.0 - t)  # equal-power fade-out


def build_audio_timeline(
    stretch_manifest: dict,
    segments_data: dict,
    source_video: Path,
    total_duration: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build the full-length dubbed audio timeline.

    Places each segment at its original start_time position.
    For skipped segments: extract and place original Arabic audio (Q2 decision).
    For dubbed segments: place the time-stretched WAV.

    Returns:
        (audio_timeline, placement_log)
    """
    total_samples = int(total_duration * SAMPLE_RATE) + SAMPLE_RATE  # +1s buffer
    timeline = np.zeros(total_samples, dtype=np.float32)
    fade_samples = int(CROSSFADE_MS / 1000.0 * SAMPLE_RATE)
    declick_samples = int(DECLICK_MS / 1000.0 * SAMPLE_RATE)

    # Build a lookup from segment_id → timing info
    timing_lookup = {}
    for seg in segments_data["segments"]:
        timing_lookup[seg["segment_id"]] = seg

    placement_log = []
    stretch_segments = stretch_manifest["segments"]
    passthrough_dir = AUDIO_OUT_DIR / "passthrough"

    for seg_entry in stretch_segments:
        seg_id = seg_entry["segment_id"]
        timing = timing_lookup.get(seg_id)

        if timing is None:
            print(f"  ⚠ No timing data for segment #{seg_id}, skipping")
            continue

        start_time = timing["start_time"]
        orig_duration = timing["duration"]
        start_sample = int(start_time * SAMPLE_RATE)

        # ── Dubbed segment (success) ─────────
        if seg_entry["status"] == "success":
            wav_path = PROJECT_ROOT / seg_entry["output_file"]

            if not wav_path.is_file():
                print(f"  ⚠ Seg #{seg_id}: WAV not found → {wav_path}")
                placement_log.append({
                    "segment_id": seg_id,
                    "type": "error",
                    "error": f"WAV not found: {wav_path}",
                })
                continue

            # Read the audio segment
            audio_data, sr = sf.read(str(wav_path), dtype="float32")

            # Resample to the pipeline rate if needed (dubbed = 24 kHz → no-op)
            if sr != SAMPLE_RATE:
                audio_data = _resample(audio_data, sr, SAMPLE_RATE)

            seg_audio = audio_data.copy()

            # Apply crossfade
            apply_crossfade(timeline, start_sample, seg_audio, fade_samples, declick_samples)

            # Place into timeline
            end_sample = start_sample + len(seg_audio)
            if end_sample > len(timeline):
                # Extend the timeline if this overflows
                extension = end_sample - len(timeline) + SAMPLE_RATE
                timeline = np.append(
                    timeline,
                    np.zeros(extension, dtype=np.float32)
                )

            timeline[start_sample:start_sample + len(seg_audio)] += seg_audio

            actual_dur = len(seg_audio) / SAMPLE_RATE
            print(f"  #{seg_id:<3}  🎙 DUBBED     "
                  f"@ {start_time:.2f}s  ({actual_dur:.2f}s)")

            placement_log.append({
                "segment_id": seg_id,
                "type": "dubbed",
                "start_time": start_time,
                "placed_duration_s": round(actual_dur, 3),
                "original_duration_s": orig_duration,
                "source_file": seg_entry["output_file"],
            })

        # ── Skipped segment → Arabic passthrough (Q2) ──
        elif seg_entry["status"] == "skipped":
            passthrough_file = passthrough_dir / f"passthrough_{seg_id:03d}.wav"

            try:
                extract_segment_audio(
                    source_video, start_time, orig_duration,
                    passthrough_file, SAMPLE_RATE,
                )

                audio_data, sr = sf.read(str(passthrough_file), dtype="float32")

                if sr != SAMPLE_RATE:
                    audio_data = _resample(audio_data, sr, SAMPLE_RATE)

                seg_audio = audio_data.copy()
                apply_crossfade(timeline, start_sample, seg_audio, fade_samples, declick_samples)

                end_sample = start_sample + len(seg_audio)
                if end_sample > len(timeline):
                    extension = end_sample - len(timeline) + SAMPLE_RATE
                    timeline = np.append(
                        timeline,
                        np.zeros(extension, dtype=np.float32)
                    )

                timeline[start_sample:start_sample + len(seg_audio)] += seg_audio

                actual_dur = len(seg_audio) / SAMPLE_RATE
                print(f"  #{seg_id:<3}  🌐 ARABIC     "
                      f"@ {start_time:.2f}s  ({actual_dur:.2f}s)  "
                      f"[passthrough]")

                placement_log.append({
                    "segment_id": seg_id,
                    "type": "arabic-passthrough",
                    "start_time": start_time,
                    "placed_duration_s": round(actual_dur, 3),
                    "original_duration_s": orig_duration,
                    "source_file": str(
                        passthrough_file.relative_to(PROJECT_ROOT)
                    ).replace("\\", "/"),
                })

            except Exception as e:
                print(f"  #{seg_id:<3}  ⚠ PASSTHROUGH FAILED: {e}")
                placement_log.append({
                    "segment_id": seg_id,
                    "type": "error",
                    "error": str(e),
                })

    return timeline, placement_log


# ──────────────────────────────────────────────
#  Step 3 — Normalise and save the final audio
# ──────────────────────────────────────────────
def clip_safety(audio: np.ndarray) -> np.ndarray:
    """
    Hard-limit samples to [-1, 1] to prevent WAV wrap-around before encoding.
    Perceived loudness is handled later by FFmpeg loudnorm (EBU R128), not by
    peak scaling here — a single loud sample no longer drags the whole track
    quiet (audit #32/#1).
    """
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def fit_to_duration(audio: np.ndarray, duration_s: float,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Pad (with silence) or trim the timeline to exactly match the video
    duration, so the mux never has to trim the video (audit #36).
    """
    target = int(round(duration_s * sample_rate))
    if len(audio) < target:
        return np.concatenate(
            [audio, np.zeros(target - len(audio), dtype=np.float32)]
        )
    return audio[:target]


def save_final_audio(
    audio: np.ndarray,
    output_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Save the final audio timeline as a WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")
    info = sf.info(str(output_path))
    print(f"  ✔ Final audio saved: {output_path}")
    print(f"    Duration: {info.duration:.2f}s  |  "
          f"Rate: {info.samplerate} Hz")
    return output_path


# ──────────────────────────────────────────────
#  Step 4 — Mux audio onto video (FFmpeg)
# ──────────────────────────────────────────────
def render_final_video(
    source_video: Path,
    dubbed_audio: Path,
    output_video: Path,
    audio_bitrate: str = AAC_BITRATE,
) -> Path:
    """
    Mux the dubbed audio track onto the source video.
    Replaces the original audio with the new dubbed track.
    Output codec: AAC at specified bitrate (Q4 decision).
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_video),        # video source
        "-i", str(dubbed_audio),         # new audio
        "-c:v", "copy",                  # copy video stream (no re-encode)
        "-af",                           # EBU R128 loudness normalization (web -16 LUFS)
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}",
        "-c:a", "aac",                   # encode audio as AAC
        "-b:a", audio_bitrate,           # bitrate (192k)
        "-map", "0:v:0",                 # take video from first input
        "-map", "1:a:0",                 # take audio from second input
        str(output_video),
    ]

    print(f"  FFmpeg: muxing video + dubbed audio…")
    print(f"    Video source : {source_video}")
    print(f"    Audio source : {dubbed_audio}")
    print(f"    Output       : {output_video}")
    print(f"    Audio codec  : AAC @ {audio_bitrate}")
    print(f"    Loudness     : EBU R128  I={LOUDNORM_I} TP={LOUDNORM_TP} LRA={LOUDNORM_LRA}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg mux failed:\n{result.stderr[:500]}"
        )

    print(f"  ✔ Final video rendered: {output_video}")
    return output_video


# ──────────────────────────────────────────────
#  Save mix manifest
# ──────────────────────────────────────────────
def _rel_or_abs(path: Path) -> str:
    """
    Return the path relative to PROJECT_ROOT when it lives inside the repo,
    otherwise fall back to the absolute path. The source video may sit outside
    the repo (e.g. on Google Drive), where relative_to() would raise.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def save_mix_manifest(
    placement_log: list[dict],
    output_audio: Path,
    output_video: Path,
    source_video: Path,
    output_path: Path,
) -> None:
    """Save the mix manifest as a JSON data contract."""
    dubbed_count = sum(1 for p in placement_log if p["type"] == "dubbed")
    passthrough_count = sum(
        1 for p in placement_log if p["type"] == "arabic-passthrough"
    )
    error_count = sum(1 for p in placement_log if p["type"] == "error")

    data = {
        "phase": "F",
        "step": "mix-render",
        "description": "Final audio mix and video render",
        "source_video": _rel_or_abs(source_video),
        "audio_codec": f"AAC {AAC_BITRATE}",
        "sample_rate": SAMPLE_RATE,
        "crossfade_ms": CROSSFADE_MS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_audio": str(
            output_audio.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "output_video": str(
            output_video.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "total_segments_placed": len(placement_log),
        "dubbed_segments": dubbed_count,
        "arabic_passthrough": passthrough_count,
        "errors": error_count,
        "skipped_segment_policy": "arabic-passthrough (Q2: Option B)",
        "lipsync_qa_policy": "manual-visual-review-only (Q3: Option B)",
        "placement_log": placement_log,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"  ✔ Mix manifest saved → {output_path}")


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase F, Step 2: Final Mix & Video Render",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Path to source video  (default: data/audio_in/sample.mp4)",
    )
    parser.add_argument(
        "--stretch-manifest",
        default=str(STRETCH_MANIFEST),
        help="Path to stretch_manifest.json  "
             "(default: artifacts/stretch_manifest.json)",
    )
    parser.add_argument(
        "--segments",
        default=str(SEGMENTS_FILE),
        help="Path to segments.json  "
             "(default: artifacts/segments.json)",
    )
    parser.add_argument(
        "--audio-bitrate",
        default=AAC_BITRATE,
        help=f"AAC audio bitrate  (default: {AAC_BITRATE})",
    )
    args = parser.parse_args()

    print()
    print(f"{'═' * 60}")
    print(f"  Dubly ME — Phase F, Step 2: Final Mix & Video Render")
    print(f"{'═' * 60}")

    # ── Validate inputs ──────────────────────
    source_video = Path(args.source)
    stretch_manifest_path = Path(args.stretch_manifest)
    segments_path = Path(args.segments)

    for label, path in [
        ("Source video", source_video),
        ("Stretch manifest", stretch_manifest_path),
        ("Segments file", segments_path),
    ]:
        if not path.is_file():
            print(f"\n  ✖ {label} not found: {path}")
            sys.exit(1)
        print(f"  ✔ {label}: {path}")

    # ── Load data ────────────────────────────
    with open(stretch_manifest_path, "r", encoding="utf-8") as fh:
        stretch_manifest = json.load(fh)

    with open(segments_path, "r", encoding="utf-8") as fh:
        segments_data = json.load(fh)

    # ── Get source duration ──────────────────
    print(f"\n[1/4]  Probing source video duration…\n")
    total_duration = get_source_total_duration(source_video)
    print(f"  Source duration: {total_duration:.2f}s")

    # ── Build audio timeline ─────────────────
    print(f"\n[2/4]  Building audio timeline…\n")
    t_start = time.perf_counter()

    timeline, placement_log = build_audio_timeline(
        stretch_manifest, segments_data, source_video, total_duration,
    )

    # Pad/trim timeline to EXACTLY the video duration (audit #36 — no -shortest)
    timeline = fit_to_duration(timeline, total_duration, SAMPLE_RATE)

    # Clip-safety only; perceptual loudness is applied by FFmpeg loudnorm at mux.
    timeline = clip_safety(timeline)

    t_build = time.perf_counter() - t_start
    print(f"\n  Timeline built in {t_build:.1f}s")

    # ── Save final audio ─────────────────────
    print(f"\n[3/4]  Saving final dubbed audio…\n")
    output_audio = DATA_AUDIO_OUT / "final_dubbed.wav"
    save_final_audio(timeline, output_audio, SAMPLE_RATE)

    # ── Render final video ───────────────────
    print(f"\n[4/4]  Rendering final video…\n")
    output_video = DATA_AUDIO_OUT / "final_dubbed.mp4"

    try:
        render_final_video(
            source_video, output_audio, output_video, args.audio_bitrate,
        )
    except Exception as e:
        print(f"\n  ⚠ Video render failed: {e}")
        print(f"  The dubbed audio WAV is still available at: {output_audio}")

    # ── Save mix manifest ────────────────────
    print(f"\n  Saving mix manifest…\n")
    save_mix_manifest(
        placement_log, output_audio, output_video,
        source_video, MIX_MANIFEST,
    )

    # ── Final banner ─────────────────────────
    dubbed = sum(1 for p in placement_log if p["type"] == "dubbed")
    passthrough = sum(
        1 for p in placement_log if p["type"] == "arabic-passthrough"
    )

    print()
    print(f"{'═' * 60}")
    print(f"  ✅  Phase F Step 2 complete — Mix & Render")
    print(f"{'─' * 60}")
    print(f"  Dubbed segments      : {dubbed}")
    print(f"  Arabic passthrough   : {passthrough}")
    print(f"  Audio output         : {output_audio}")
    print(f"  Video output         : {output_video}")
    print(f"  Audio codec          : AAC @ {args.audio_bitrate}")
    print(f"  Mix manifest         : {MIX_MANIFEST}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
