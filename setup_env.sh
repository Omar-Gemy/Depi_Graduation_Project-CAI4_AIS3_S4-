#!/usr/bin/env bash
# setup_env.sh — Dubly ME staged dependency installer
# =====================================================================
# Environment-agnostic replacement for `pip install -r requirements-colab.txt`.
# The rigid single-file install triggers dependency-resolution deadlocks
# (whisperx ↔ transformers ↔ autoawq ↔ pyannote) that fail hard on a clean
# Linux/RunPod box. This installs in ordered stages instead, so a failure in a
# late/optional stage (e.g. autoawq's compilation) does not wipe out the
# earlier, load-bearing stages (torch, whisperx, ctranslate2).
#
# Design:
#   • Idempotent — re-running skips satisfied stages where cheap to check.
#   • Non-fatal where it matters — Phase D (autoawq) is allowed to fail so that
#     ingestion/diarization/ASR still work; every failure is printed loudly.
#   • Pins overridable via env vars (TRANSFORMERS_VERSION, WHISPERX_VERSION …)
#     so you can re-pin on a specific CUDA image without editing this file.
#
# Usage:
#   bash setup_env.sh
#   TRANSFORMERS_VERSION=4.47.1 bash setup_env.sh
#
# NOTE: exact versions are validated against Colab T4; on a fresh RunPod CUDA
# image you may need to adjust a pin. Failures are reported per-stage, not
# swallowed, so you can see exactly which stage needs attention.

set -uo pipefail   # NOT -e: we handle failures per-stage so one bad optional
                   # stage doesn't abort the whole environment build.

# ── Overridable pins ────────────────────────────────────────────────
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.47.1}"
WHISPERX_VERSION="${WHISPERX_VERSION:-3.3.1}"
AUTOAWQ_VERSION="${AUTOAWQ_VERSION:-0.2.8}"
ACCELERATE_SPEC="${ACCELERATE_SPEC:-accelerate>=1.1.0,<1.3.0}"

PIP="python -m pip"
FAILED_STAGES=()

log()  { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
ok()   { printf '  ✔ %s\n' "$*"; }
warn() { printf '  ⚠ %s\n' "$*"; }
fail() { printf '  ✖ %s\n' "$*"; FAILED_STAGES+=("$1"); }

# Run a pip stage; record (but don't abort on) failure.
stage() {
  local name="$1"; shift
  log "Stage: ${name}"
  if $PIP install "$@"; then
    ok "${name} installed."
  else
    fail "${name}"
    warn "Stage '${name}' failed — continuing so earlier stages survive."
  fi
}

# ── 0. Tooling ──────────────────────────────────────────────────────
log "Upgrading pip / wheel / setuptools"
$PIP install -q --upgrade pip wheel setuptools || warn "pip self-upgrade failed (non-fatal)."

# ── 1. PyTorch (skip if a working CUDA/CPU torch is already present) ─
# Colab and most RunPod images ship torch already; reinstalling risks pulling a
# CUDA build that mismatches the driver. Only install if import fails.
log "Stage: PyTorch (detect existing)"
if python -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
  ok "Existing PyTorch detected — not reinstalling."
else
  warn "No PyTorch found — installing default CPU/CUDA wheel."
  warn "If this box has a specific CUDA, install the matching torch wheel FIRST, then re-run."
  stage "torch" torch torchaudio
fi

# ── 2. Core libraries (stable, rarely conflict) ─────────────────────
stage "core-libs" \
  "numpy<2.0" \
  soundfile \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0,<6.0" \
  "${ACCELERATE_SPEC}" \
  "bitsandbytes>=0.44.0" \
  onnxruntime

# ── 3. Conversion / experimental-ASR toolchain ──────────────────────
# ctranslate2 + faster-whisper are what WhisperX pins transitively; installing
# them here (before whisperx --no-deps) seeds a compatible pair, and peft is
# needed by merge_lora.py. If whisperx later disagrees on a pin, its own deps
# stage (4) will correct it.
stage "conversion-tools" ctranslate2 faster-whisper peft

# ── 4. WhisperX WITHOUT its dependency closure ──────────────────────
# The crux of the "dependency hell": letting whisperx resolve its full closure
# drags in a transformers/pyannote/torch combination that fights autoawq. We
# install whisperx with --no-deps, then hand-install the runtime deps it truly
# needs at versions the rest of the stack agrees on.
stage "whisperx (--no-deps)" --no-deps "whisperx==${WHISPERX_VERSION}"
stage "whisperx-runtime-deps" \
  "transformers==${TRANSFORMERS_VERSION}" \
  "pyannote.audio" \
  "pandas" \
  "nltk"

# ── 5. AutoAWQ LAST (Phase D; compilation-sensitive, allowed to fail) ─
# autoawq is the most fragile install (CUDA-compiled kernels). It is Phase D
# only — if it fails, ingestion → ASR still run, so we keep it last and
# non-fatal rather than letting it sink the whole environment.
log "Stage: AutoAWQ (Phase D — optional, may need a CUDA toolchain)"
if $PIP install "autoawq==${AUTOAWQ_VERSION}"; then
  ok "AutoAWQ installed."
else
  fail "autoawq"
  warn "AutoAWQ failed to install. Phases A–C (ingestion/diarization/ASR) are"
  warn "unaffected; only Phase D (Qwen AWQ translation) needs it. Fix the CUDA"
  warn "build toolchain on this box, then: pip install autoawq==${AUTOAWQ_VERSION}"
fi

# ── 6. Summary ──────────────────────────────────────────────────────
printf '\n%s\n' "════════════════════════════════════════════════════════════"
if [ "${#FAILED_STAGES[@]}" -eq 0 ]; then
  printf '  ✅  setup_env.sh: all stages installed successfully.\n'
else
  printf '  ⚠  setup_env.sh finished with failed stage(s): %s\n' "${FAILED_STAGES[*]}"
  printf '     Earlier stages are intact; address the above and re-run.\n'
fi
printf '%s\n' "════════════════════════════════════════════════════════════"

# FFmpeg is a SYSTEM dependency (not pip). Verify + hint, never hard-fail.
if command -v ffmpeg >/dev/null 2>&1; then
  ok "FFmpeg present: $(ffmpeg -version | head -n1)"
else
  warn "FFmpeg NOT found — install it:  apt-get update && apt-get install -y ffmpeg"
fi

# Exit non-zero only if a CORE stage failed (autoawq alone does not fail setup).
for s in "${FAILED_STAGES[@]:-}"; do
  if [ "${s}" != "autoawq" ] && [ -n "${s}" ]; then
    exit 1
  fi
done
exit 0
