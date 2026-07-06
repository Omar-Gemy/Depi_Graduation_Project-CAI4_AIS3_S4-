# -*- coding: utf-8 -*-
"""
colab_pipeline.py — Dubly ME: Sequential Pipeline (Phases A–D)
=====================================================================
Environment-agnostic: runs on Colab, RunPod, or any Linux/local box.
Each section is delimited with  # %%  for easy copy-paste into .ipynb.

Architecture:
  - Storage is a plain local directory (STORAGE_ROOT), resolved dynamically:
    DUBLY_STORAGE_ROOT env var → /workspace (RunPod) → <repo>/storage. No
    Google Drive mount is required on any platform.
  - HF_HOME/HF_HUB_CACHE/TORCH_HOME point into STORAGE_ROOT so heavy models
    are cached once per box.
  - The repo is auto-detected when the notebook runs from inside it (RunPod /
    local); on a bare Colab VM it is cloned on first run.
  - Each phase invokes the corresponding CLI script from  src/ .
"""

# %% [Cell 0] Environment Setup — Detect Env, Resolve Paths, Install Deps
"""
Auto-detect the runtime (Colab vs. server/local), resolve STORAGE_ROOT and
REPO_DIR dynamically, configure the HuggingFace cache, sync/clone the repo if
needed, and install dependencies via the staged setup_env.sh.
"""
import os
import subprocess
from pathlib import Path

# ── 0.1  Detect runtime environment ──────────────────────────────────
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print(f"✔ Environment: {'Colab' if IN_COLAB else 'server/local'}")

# ── 0.2  Resolve REPO_DIR (auto-detect; clone only on a bare Colab VM) ──
# Walk up from the current working dir looking for a repo marker (src/ +
# requirements.txt). On RunPod/local you run from inside the repo, so it is
# found here and never re-cloned. On a fresh Colab VM nothing is found → clone.
GIT_URL = "https://github.com/Omar-Gemy/Dubly_ME.git"
BRANCH = "feature/upgrade-stack"


def _find_repo_root(start: Path) -> Path | None:
    for base in (start, *start.parents):
        if (base / "src").is_dir() and (base / "requirements.txt").is_file():
            return base
    return None

_detected = _find_repo_root(Path.cwd())
if _detected is not None:
    REPO_DIR = str(_detected)
    print(f"✔ Repo detected at: {REPO_DIR}")
    if IN_COLAB:
        print("▶ Pulling latest changes…")
        subprocess.run(["git", "pull", "origin", BRANCH], cwd=REPO_DIR, check=False)
else:
    # Not inside a repo — only expected on a fresh Colab VM. Clone next to cwd.
    REPO_DIR = str(Path.cwd() / "Dubly_ME")
    if not os.path.isdir(REPO_DIR):
        print(f"▶ Cloning repo from {GIT_URL} (branch: {BRANCH})…")
        subprocess.run(["git", "clone", "-b", BRANCH, GIT_URL, REPO_DIR], check=True)
        print("✔ Repo cloned.")
    else:
        print(f"✔ Using existing repo dir: {REPO_DIR}")

# ── 0.3  Resolve STORAGE_ROOT (env var → /workspace → <repo>/storage) ──
# No Google Drive. Persistent, box-local storage for models + HF cache.
if os.environ.get("DUBLY_STORAGE_ROOT"):
    STORAGE_ROOT = os.environ["DUBLY_STORAGE_ROOT"]
    _storage_src = "DUBLY_STORAGE_ROOT env"
elif os.path.isdir("/workspace"):
    STORAGE_ROOT = "/workspace/Dubly_ME_Storage"
    _storage_src = "/workspace (RunPod)"
else:
    STORAGE_ROOT = str(Path(REPO_DIR) / "storage")
    _storage_src = "<repo>/storage (fallback)"
os.makedirs(STORAGE_ROOT, exist_ok=True)
print(f"✔ STORAGE_ROOT = {STORAGE_ROOT}  ({_storage_src})")

# ── 0.4  Persistent model cache (must be set BEFORE transformers imports) ──
os.environ["HF_HOME"]      = f"{STORAGE_ROOT}/models"
os.environ["HF_HUB_CACHE"] = f"{STORAGE_ROOT}/models/hub"
os.environ["TORCH_HOME"]   = f"{STORAGE_ROOT}/models/torch"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
print(f"✔ HF_HOME      = {os.environ['HF_HOME']}")
print(f"✔ HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")
print(f"✔ TORCH_HOME   = {os.environ['TORCH_HOME']}")

# ── 0.5  HF_TOKEN — Colab Secrets first, then plain env var ──────────
if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        print("✔ HF_TOKEN loaded from Colab Secrets.")
    except Exception:
        print("⚠ HF_TOKEN not set. Export it before diarization:")
        print("    export HF_TOKEN=hf_your_token   (or os.environ['HF_TOKEN']=…)")
else:
    print("✔ HF_TOKEN found in environment.")

# ── 0.6  Install dependencies via the staged installer ──────────────
# Replaces the rigid `pip install -r requirements-colab.txt` (dependency-hell
# on clean Linux). setup_env.sh installs in ordered, failure-tolerant stages.
setup_script = f"{REPO_DIR}/setup_env.sh"
print(f"▶ Installing dependencies via {setup_script} …")
subprocess.run(["bash", setup_script], check=True)

# ── 0.7  Verify FFmpeg (system dependency, not pip) ─────────────────
if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0:
    print("✔ FFmpeg is available.")
else:
    print("⚠ FFmpeg NOT found — install it:  apt-get install -y ffmpeg")

print("\n" + "═" * 60)
print("  ✅  Cell 0: Environment Setup Complete")
print("═" * 60)


# %% [Cell 0b] ASR Switchboard — Legacy vs. Experimental (reversible)
"""
Toggle which ASR engine Phase C uses, WITHOUT touching the legacy path.

  ASR_MODE = "legacy"        → existing WhisperX large-v3 + wav2vec2 forced
                                alignment (writes artifacts/transcripts.json —
                                required by Phase D).
  ASR_MODE = "experimental"  → local Egyptian-fine-tuned CTranslate2 model via
                                faster-whisper, TRANSCRIPTION ONLY (no forced
                                alignment, no transcripts.json). Diagnostic
                                output for manual comparison against legacy.

Flip ASR_MODE back to "legacy" at any time to restore the full pipeline.
"""
ASR_MODE = "legacy"   # "legacy" | "experimental"

# Local CTranslate2 model directory for experimental mode (the output of
# merge_lora.py + convert_to_ct2.py). Lives under STORAGE_ROOT so it persists
# per box; override EXPERIMENTAL_ASR_MODEL_PATH to any local path if you prefer.
EXPERIMENTAL_ASR_MODEL_PATH = f"{STORAGE_ROOT}/models/whisper-large-v3-egy-ct2"
EXPERIMENTAL_ASR_LANGUAGE = "ar"
EXPERIMENTAL_ASR_DEVICE = "cuda"
EXPERIMENTAL_ASR_COMPUTE_TYPE = "float16"

print(f"✔ ASR switchboard set: ASR_MODE = {ASR_MODE!r}")
if ASR_MODE == "experimental":
    print(f"  experimental model: {EXPERIMENTAL_ASR_MODEL_PATH}")
    print("  ⚠ experimental = transcription only; Phase D still needs a legacy run.")


# %% [Cell 0c] Model Conversion — Prepare Experimental ASR
"""
Build the experimental Egyptian CTranslate2 model IN-PLACE inside Colab.

Idempotent — only builds what is missing:
  1. CT2 model dir already exists      → skip (nothing to do).
  2. Merged HF checkpoint missing      → run merge_lora.py (LoRA merge).
  3. Merged exists but CT2 missing     → run convert_to_ct2.py (programmatic).

Only needed when you intend to run Phase C with ASR_MODE = "experimental".
Safe to run in legacy mode too — it just reports "already present / skipping".
"""
import os, subprocess

# REPO_DIR / STORAGE_ROOT / EXPERIMENTAL_ASR_MODEL_PATH come from Cell 0 / 0b.
# CT2 dir that experimental mode loads (defined in the Cell 0b switchboard),
# so we convert to exactly the path Phase C will read from.
CT2_MODEL_DIR = EXPERIMENTAL_ASR_MODEL_PATH
# merge_lora.py writes its merged HF checkpoint here (OUT is relative to its cwd).
MERGED_DIR = f"{REPO_DIR}/whisper-large-v3-egy-merged"

if os.path.isdir(CT2_MODEL_DIR):
    print(f"✔ CT2 model already present — skipping conversion:\n    {CT2_MODEL_DIR}")
else:
    print(f"▶ CT2 model not found at:\n    {CT2_MODEL_DIR}\n  Building it now.")

    # ── Step 1: LoRA merge (only if the merged folder is missing) ──
    # peft is installed by setup_env.sh (Cell 0); merge_lora.py needs it.
    if os.path.isdir(MERGED_DIR):
        print(f"✔ Merged HF model already present — skipping merge:\n    {MERGED_DIR}")
    else:
        print("▶ Running merge_lora.py (LoRA merge → merged HF checkpoint)…")
        subprocess.run(["python", f"{REPO_DIR}/merge_lora.py"], check=True, cwd=REPO_DIR)

    # ── Step 2: CTranslate2 conversion (programmatic — avoids the CLI's ──
    #    `dtype` TypeError / `Fast download` ValueError seen on the server). ──
    print(f"▶ Converting → CTranslate2:\n    {MERGED_DIR}\n    → {CT2_MODEL_DIR}")
    os.makedirs(os.path.dirname(CT2_MODEL_DIR), exist_ok=True)
    subprocess.run(
        [
            "python", f"{REPO_DIR}/convert_to_ct2.py",
            "--model", MERGED_DIR,
            "--output", CT2_MODEL_DIR,
            "--quantization", "float16",
            "--force",
        ],
        check=True,
    )
    print(f"✔ Conversion complete — CT2 model ready at:\n    {CT2_MODEL_DIR}")

print("\n" + "═" * 60)
print("  ✅  Cell 0c: Model Conversion check complete")
print("═" * 60)


# %% [Cell 1] Phase A — Audio Ingestion & Voice Activity Detection
"""
Extract audio from the source media file, normalise loudness (EBU R128),
and run Silero VAD to detect speech segments.

Script:   src/ingestion_vad.py
Input:    source media file (video or audio)
Output:   artifacts/segments.json  +  data/audio_out/_temp_normalised.wav
"""
import subprocess, os

# REPO_DIR / STORAGE_ROOT come from Cell 0. Source media defaults to
# <STORAGE_ROOT>/data/audio_in/source_media.mp4; override with DUBLY_INPUT_MEDIA.
INPUT_MEDIA = os.environ.get(
    "DUBLY_INPUT_MEDIA",
    f"{STORAGE_ROOT}/data/audio_in/source_media.mp4",
)

# Validate input exists
if not os.path.isfile(INPUT_MEDIA):
    raise FileNotFoundError(
        f"Source media not found: {INPUT_MEDIA}\n"
        f"Place your video/audio file there, or set DUBLY_INPUT_MEDIA to its path."
    )

print(f"▶ Phase A: Ingesting {INPUT_MEDIA}")
subprocess.run(
    [
        "python", f"{REPO_DIR}/src/ingestion_vad.py",
        INPUT_MEDIA,
        "--source-lang", "ar",
        "--target-lang", "en",
    ],
    check=True,
    cwd=REPO_DIR,
)

print("\n" + "═" * 60)
print("  ✅  Phase A: Audio Ingestion & VAD Complete")
print("═" * 60)


# %% [Cell 2] Phase B — Speaker Diarization
"""
Assign speaker identities to VAD segments using the pyannote.audio 4.x
ensemble (Segmentation → Embedding → Clustering).

Script:   src/diarization.py
Input:    data/audio_out/_temp_normalised.wav  +  artifacts/segments.json
Output:   artifacts/segments.json  (enriched with speaker_id)

Requires: HF_TOKEN set (for gated pyannote models).
"""
import subprocess

# REPO_DIR comes from Cell 0 (kernel global) — no hardcoded path.

print("▶ Phase B: Speaker Diarization")
subprocess.run(
    [
        "python", f"{REPO_DIR}/src/diarization.py",
        "--device", "cuda",
    ],
    check=True,
    cwd=REPO_DIR,
)

print("\n" + "═" * 60)
print("  ✅  Phase B: Speaker Diarization Complete")
print("═" * 60)


# %% [Cell 3] Phase C — ASR Transcription
"""
Transcribe the full audio in one pass with WhisperX (large-v3 backend),
run wav2vec2 forced alignment for precise word-level timestamps, then
intersect words with the diarization segments (global-to-local mapping).

Script:   src/asr_transcription.py
Input:    data/audio_out/_temp_normalised.wav  +  artifacts/segments.json
Output:   artifacts/transcripts.json
"""
import subprocess, os

# REPO_DIR comes from Cell 0 (kernel global) — no hardcoded path.


def run_experimental_asr(
    model_path,
    audio_path,
    language="ar",
    device="cuda",
    compute_type="float16",
    beam_size=5,
):
    """
    [EXPERIMENTAL] Transcribe the normalised WAV directly with a local
    CTranslate2 model (Egyptian fine-tune) via faster-whisper.

    Transcription ONLY — no wav2vec2 forced alignment and no diarization
    intersection, so this does NOT write artifacts/transcripts.json. It prints
    the transcript for manual comparison against the legacy WhisperX output;
    Phase D still requires a legacy ASR run to produce its input contract.
    """
    from faster_whisper import WhisperModel

    # CTranslate2 has no fp16 CPU kernels — mirror src/asr_transcription.py.
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"
        print("  ⚠ float16 unsupported on CPU — using int8 compute type")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Experimental CT2 model dir not found: {model_path}\n"
            f"Run merge_lora.py + ct2-transformers-converter first, or point "
            f"EXPERIMENTAL_ASR_MODEL_PATH at the converted model directory."
        )

    print(f"▶ [experimental] Loading CT2 model: {model_path}")
    print(f"  device={device}  compute_type={compute_type}")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)

    print("▶ [experimental] Transcribing (transcription only, NO alignment)…")
    segments, info = model.transcribe(audio_path, language=language, beam_size=beam_size)

    print(f"  language: {info.language}")
    print("═" * 60)
    for seg in segments:
        print(f"[{seg.start:7.2f}s → {seg.end:7.2f}s]  {seg.text.strip()}")
    print("═" * 60)
    print("  ⚠ Diagnostic only — no transcripts.json written.")
    print("    Set ASR_MODE = 'legacy' to produce the Phase D contract.")


print(f"▶ Phase C: ASR Transcription  [ASR_MODE = {ASR_MODE}]")

if ASR_MODE == "legacy":
    # ── LEGACY PATH (unchanged): WhisperX large-v3 + wav2vec2 forced alignment ──
    print("▶ Phase C: ASR Transcription")
    subprocess.run(
        [
            "python", f"{REPO_DIR}/src/asr_transcription.py",
            "--model", "large-v3",
            "--source-lang", "ar",
            "--device", "cuda",
            "--compute-type", "float16",
        ],
        check=True,
        cwd=REPO_DIR,
    )

elif ASR_MODE == "experimental":
    # ── EXPERIMENTAL PATH: local Egyptian CT2 model, transcription only ──
    run_experimental_asr(
        model_path=EXPERIMENTAL_ASR_MODEL_PATH,
        audio_path=f"{REPO_DIR}/data/audio_out/_temp_normalised.wav",
        language=EXPERIMENTAL_ASR_LANGUAGE,
        device=EXPERIMENTAL_ASR_DEVICE,
        compute_type=EXPERIMENTAL_ASR_COMPUTE_TYPE,
    )

else:
    raise ValueError(
        f"Unknown ASR_MODE={ASR_MODE!r} — set 'legacy' or 'experimental' "
        f"in the [Cell 0b] switchboard."
    )

print("\n" + "═" * 60)
print("  ✅  Phase C: ASR Transcription Complete")
print("═" * 60)


# %% [Cell 4] Phase D — Contextual Translation (Isochronous)
"""
Adapt the Arabic transcripts into natural spoken English for dubbing using
Qwen2.5-14B-Instruct-AWQ. The model is a pre-quantized 4-bit AWQ checkpoint
(~9–10 GB VRAM, fits the 16 GB T4) and is cached on Google Drive via HF_HOME
to prevent re-downloading on subsequent sessions.

Each line is translated against an explicit isochrony budget (target syllable
count derived from the segment duration) so the dub fits its time window.

Script:   src/translation.py
Input:    artifacts/transcripts.json
Output:   artifacts/translation.json
"""
import subprocess

# REPO_DIR comes from Cell 0 (kernel global) — no hardcoded path.

print("▶ Phase D: Contextual Translation (Isochronous)")
subprocess.run(
    [
        "python", f"{REPO_DIR}/src/translation.py",
        "--model", "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "--device", "auto",
    ],
    check=True,
    cwd=REPO_DIR,
)

print("\n" + "═" * 60)
print("  ✅  Phase D: Translation Complete")
print("═" * 60)


# %% [Cell 5] Pipeline Summary — Verify Artifacts
"""
Quick sanity check: verify that all expected output artifacts exist
and print a summary of what was produced.
"""
import json, os

# REPO_DIR comes from Cell 0 (kernel global) — no hardcoded path.
ARTIFACTS = f"{REPO_DIR}/artifacts"

expected_files = {
    "Phase A (VAD)":         f"{ARTIFACTS}/segments.json",
    "Phase C (ASR)":         f"{ARTIFACTS}/transcripts.json",
    "Phase D (Translation)": f"{ARTIFACTS}/translation.json",
}

print("Pipeline Artifact Verification")
print("─" * 50)
all_ok = True
for phase, path in expected_files.items():
    if os.path.isfile(path):
        size_kb = os.path.getsize(path) / 1024
        # Load to count segments
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_segs = data.get("total_segments", len(data.get("segments", [])))
        print(f"  ✔ {phase:<25s}  {n_segs:>3d} segments  ({size_kb:.1f} KB)")
    else:
        print(f"  ✖ {phase:<25s}  FILE MISSING: {path}")
        all_ok = False

print("─" * 50)
if all_ok:
    print("  ✅  All pipeline artifacts verified successfully!")
else:
    print("  ⚠  Some artifacts are missing — check the logs above.")
print("═" * 50)
