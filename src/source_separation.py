"""
source_separation.py — Phase F0: Music/SFX Bed vs Original Vocals
==================================================================
Split the source audio into two stems using self-hosted **Demucs**
(htdemucs, MIT-licensed, no cloud APIs):

  - background.wav  → music + sound-effects bed (Demucs "no_vocals")
  - vocals.wav      → original spoken dialogue (Demucs "vocals")

Downstream, `mix_render.py` lays `background.wav` as a CONTINUOUS bed under the
whole timeline (instead of dead silence between dubbed lines), and uses
`vocals.wav` for Arabic passthrough over that same bed — so the dubbed video
keeps its original music/ambience like a real dub.

This stage runs in its OWN isolated virtualenv (`/content/.venv_demucs`),
mirroring Phase E: Demucs is torch-heavy and must not perturb the pinned A–D
stack or the XTTS venv. `mix_render.py` (main kernel) only ever reads the WAV
artifacts this stage writes — never Demucs itself.

Inputs:
  - source video/audio (full-quality; extracted to 44.1 kHz for separation)

Outputs:
  - artifacts/audio_out/background.wav        (M&E bed, mono @ pipeline rate)
  - artifacts/audio_out/vocals.wav            (original dialogue, mono @ rate)
  - artifacts/separation_manifest.json        (Phase F0 data contract)

Usage:
  python src/source_separation.py --source data/audio_in/sample.mp4
  python src/source_separation.py --device cpu --segment 7
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
AUDIO_OUT_DIR  = ARTIFACTS_DIR / "audio_out"
DATA_AUDIO_IN  = PROJECT_ROOT / "data" / "audio_in"

DEFAULT_SOURCE   = DATA_AUDIO_IN / "sample.mp4"
DEFAULT_BG       = AUDIO_OUT_DIR / "background.wav"
DEFAULT_VOCALS   = AUDIO_OUT_DIR / "vocals.wav"
DEFAULT_MANIFEST = ARTIFACTS_DIR / "separation_manifest.json"

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
DEMUCS_MODEL   = "htdemucs"   # hybrid-transformer Demucs v4 (2-stem, fast, ~80MB)
EXTRACT_RATE   = 44100        # Demucs native working rate
PIPELINE_RATE  = 24000        # must match mix_render.SAMPLE_RATE (Phase 3A)


def _rel_or_abs(path: Path) -> str:
    """
    Path relative to PROJECT_ROOT when inside the repo, else the absolute path.
    The source video may sit outside the repo (e.g. on Google Drive), where
    relative_to() would raise.
    """
    try:
        return str(Path(path).relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Anti-aliased resample. scipy polyphase (resample_poly) when available; falls
    back to linear interpolation only if scipy is missing. Mirrors mix_render.
    """
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if _HAVE_SCIPY:
        from math import gcd
        g = gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    old_len = len(audio)
    new_len = int(old_len * dst_sr / src_sr)
    return np.interp(
        np.linspace(0, old_len - 1, new_len),
        np.arange(old_len),
        audio,
    ).astype(np.float32)


# ──────────────────────────────────────────────
#  Step 1 — Resolve device
# ──────────────────────────────────────────────
def resolve_device(device: str) -> str:
    """Resolve 'auto' → 'cuda' when available, else 'cpu'. Passes others through."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ──────────────────────────────────────────────
#  Step 2 — Extract full-quality audio for separation
# ──────────────────────────────────────────────
def extract_full_audio(source: Path, out_wav: Path) -> Path:
    """
    Extract the source's audio to 44.1 kHz stereo PCM via FFmpeg — the format
    Demucs works in. Kept separate from the video so Demucs never touches the
    container.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",                      # drop video
        "-acodec", "pcm_s16le",
        "-ar", str(EXTRACT_RATE),
        "-ac", "2",                 # stereo — let Demucs use full spatial info
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ✖ FFmpeg extraction failed:\n{result.stderr[:500]}")
        sys.exit(1)

    info = sf.info(str(out_wav))
    print(f"  ✔ Extracted audio: {info.duration:.1f}s @ {info.samplerate} Hz, "
          f"{info.channels}ch")
    return out_wav


# ──────────────────────────────────────────────
#  Step 3 — Run Demucs (two-stem separation)
# ──────────────────────────────────────────────
def run_demucs(
    input_wav: Path,
    out_dir: Path,
    model: str,
    device: str,
    segment: int | None,
) -> tuple[Path, Path]:
    """
    Run Demucs 2-stem separation (vocals / no_vocals). Invoked via
    `python -m demucs` using THIS interpreter — so it uses whatever venv is
    running this script (intended: /content/.venv_demucs).

    Returns (vocals_path, no_vocals_path) discovered from Demucs' output tree.
    """
    import importlib.util
    if importlib.util.find_spec("demucs") is None:
        print("  ✖ Demucs is not installed in this environment.")
        print("    Run the Phase F0 setup cell to build /content/.venv_demucs,")
        print("    or:  pip install demucs")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-d", device,
        "-o", str(out_dir),
        str(input_wav),
    ]
    # --segment chunks the audio to bound VRAM (needed on ≤4 GB GPUs).
    if segment is not None:
        cmd += ["--segment", str(segment)]

    print(f"  ▶ Demucs: model={model}  device={device}"
          f"{f'  segment={segment}' if segment else ''}")
    print(f"    (separating — this scales with clip length)…")

    # Stream progress; check=True surfaces failures to the caller.
    subprocess.run(cmd, check=True)

    # Demucs writes: <out_dir>/<model>/<track_stem>/{vocals,no_vocals}.wav
    model_dir = out_dir / model
    vocals = next(model_dir.glob("*/vocals.wav"), None)
    no_vocals = next(model_dir.glob("*/no_vocals.wav"), None)
    if vocals is None or no_vocals is None:
        print(f"  ✖ Demucs finished but stems not found under {model_dir}")
        sys.exit(1)

    print(f"  ✔ Stems produced:")
    print(f"      vocals    : {vocals}")
    print(f"      no_vocals : {no_vocals}")
    return vocals, no_vocals


# ──────────────────────────────────────────────
#  Step 4 — Post-process a stem → mono @ pipeline rate
# ──────────────────────────────────────────────
def finalize_stem(stem_path: Path, out_path: Path) -> float:
    """
    Downmix a Demucs stem to mono and resample to the pipeline rate (24 kHz) so
    it drops straight into mix_render's mono float32 timeline. Returns duration.
    """
    audio, sr = sf.read(str(stem_path), dtype="float32")
    if audio.ndim > 1:                      # stereo → mono
        audio = audio.mean(axis=1)
    if sr != PIPELINE_RATE:
        audio = _resample(audio, sr, PIPELINE_RATE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, PIPELINE_RATE, subtype="PCM_16")
    return round(len(audio) / PIPELINE_RATE, 3)


# ──────────────────────────────────────────────
#  Step 5 — Save the data contract
# ──────────────────────────────────────────────
def save_manifest(
    source: Path,
    bg_path: Path,
    vocals_path: Path,
    bg_dur: float,
    vocals_dur: float,
    model: str,
    device: str,
    segment: int | None,
    output_path: Path,
) -> None:
    """Write separation_manifest.json (Phase F0 data contract)."""
    data = {
        "phase": "F0",
        "description": "Source separation — music/SFX bed vs original vocals",
        "separation_model": model,
        "source_file": _rel_or_abs(source),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": PIPELINE_RATE,
        "channels": 1,
        "device": device,
        "segment": segment,
        "background_file": _rel_or_abs(bg_path),
        "vocals_file": _rel_or_abs(vocals_path),
        "background_duration_s": bg_dur,
        "vocals_duration_s": vocals_dur,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"  ✔ Manifest saved → {output_path}")


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase F0: Source Separation (Demucs)",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Source video/audio  (default: data/audio_in/sample.mp4)",
    )
    parser.add_argument(
        "--model",
        default=DEMUCS_MODEL,
        help=f"Demucs model name  (default: {DEMUCS_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'  (default: auto)",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Chunk length (s) to bound VRAM on small GPUs  (default: model max)",
    )
    parser.add_argument(
        "--bg-output",
        default=str(DEFAULT_BG),
        help="Output background (M&E) WAV  (default: artifacts/audio_out/background.wav)",
    )
    parser.add_argument(
        "--vocals-output",
        default=str(DEFAULT_VOCALS),
        help="Output vocals WAV  (default: artifacts/audio_out/vocals.wav)",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Output manifest path  (default: artifacts/separation_manifest.json)",
    )
    args = parser.parse_args()

    print()
    print(f"{'═' * 60}")
    print(f"  Dubly ME — Phase F0: Source Separation (Demucs)")
    print(f"{'═' * 60}")

    source = Path(args.source)
    if not source.is_file():
        print(f"\n  ✖ Source not found: {source}")
        sys.exit(1)

    device = resolve_device(args.device)
    bg_path = Path(args.bg_output)
    vocals_path = Path(args.vocals_output)

    t_start = time.perf_counter()

    # Work in a temp dir; only the finalized stems + manifest persist.
    with tempfile.TemporaryDirectory(prefix="dubly_sep_") as tmp:
        tmp_dir = Path(tmp)

        # ── Step 1/5: validate + device ─────────
        print(f"\n[1/5]  Preparing…\n")
        print(f"  ✔ Source : {source}")
        print(f"  ✔ Device : {device}")

        # ── Step 2/5: extract audio ─────────────
        print(f"\n[2/5]  Extracting full-quality audio…\n")
        extracted = extract_full_audio(source, tmp_dir / "full.wav")

        # ── Step 3/5: Demucs separation ─────────
        print(f"\n[3/5]  Separating stems with Demucs…\n")
        vocals_raw, no_vocals_raw = run_demucs(
            extracted, tmp_dir / "demucs_out", args.model, device, args.segment,
        )

        # ── Step 4/5: finalize stems ────────────
        print(f"\n[4/5]  Finalizing stems (mono @ {PIPELINE_RATE} Hz)…\n")
        bg_dur = finalize_stem(no_vocals_raw, bg_path)
        print(f"  ✔ Background (M&E) → {bg_path}  ({bg_dur:.1f}s)")
        vocals_dur = finalize_stem(vocals_raw, vocals_path)
        print(f"  ✔ Vocals           → {vocals_path}  ({vocals_dur:.1f}s)")

        # ── Step 5/5: manifest ──────────────────
        print(f"\n[5/5]  Saving separation manifest…\n")
        save_manifest(
            source, bg_path, vocals_path, bg_dur, vocals_dur,
            args.model, device, args.segment, Path(args.manifest),
        )

    t_total = time.perf_counter() - t_start
    print()
    print(f"{'═' * 60}")
    print(f"  ✅  Phase F0 complete — Source Separation")
    print(f"{'─' * 60}")
    print(f"  Background bed : {bg_path}")
    print(f"  Vocals stem    : {vocals_path}")
    print(f"  Model / device : {args.model} / {device}")
    print(f"  Elapsed        : {t_total:.1f}s")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
