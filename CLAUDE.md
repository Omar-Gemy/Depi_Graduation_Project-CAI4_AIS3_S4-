# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Dubly ME** is a self-hosted AI video dubbing pipeline (Arabic → English is the primary path, driven by `config/language_registry.json`). It ingests a video, detects who speaks when, transcribes, translates with timing-aware ("isochronous") rewriting, clones the voice, fits each line into its original time window, and renders a final dubbed video. The design goal is output that sounds like a real dubbing team, not an AI overlay — timing fit, emotional fidelity, and mix quality come before flashy features. See `docs/professional_ai_dubbing_project_plan_en.md` for the full vision.

## Hard Constraints (from `skills/dubbing_rules.md` — do not violate)

1. **Zero third-party APIs.** No commercial/cloud API calls (no DeepL, Google, OpenAI, etc.). Everything runs on self-hosted open models: faster-whisper, Silero VAD, pyannote.audio, Qwen, XTTS v2.
2. **Every stage emits a JSON data contract** into `artifacts/`. Never skip writing these — they are how stages hand off and how runs are debugged.
3. **Git workflow:** never push to `main`; branch per feature (current branch: `feature/architecture-refactor`).
4. **Keep it modular.** No end-to-end "magic" models — each stage must be independently runnable, testable, and replaceable.

## Architecture — a linear chain of CLI stages

Each `src/*.py` is a standalone CLI stage that reads the previous stage's JSON artifact and writes its own. Stages communicate **only** through files in `artifacts/` (plus audio under `data/audio_out/` and `artifacts/audio_out/`). There is no orchestrator module in `src/` — the notebook (`notebooks/colab_pipeline.py` / `.ipynb`) is the orchestrator, invoking each script via `subprocess` in order.

Run order, script, and the artifact each produces:

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 1 | `src/ingestion_vad.py` | source video/audio | `data/audio_out/_temp_normalised.wav`, `artifacts/segments.json` |
| 2 | `src/diarization.py` | `_temp_normalised.wav` + `segments.json` | rewrites `segments.json` (adds `speaker_id`) |
| 3 | `src/asr_transcription.py` | `_temp_normalised.wav` + `segments.json` | `artifacts/transcripts.json` |
| 4 | `src/translation.py` | `transcripts.json` | `artifacts/translation.json` |
| 5 | `src/tts_synthesis.py` | `translation.json` + `artifacts/voice_ref.wav` | `artifacts/audio_out/segment_XXX.wav`, `artifacts/tts_manifest.json` |
| 6 | `src/time_stretch.py` | `tts_manifest.json` + segment WAVs | `artifacts/audio_out/stretched/*.wav`, `artifacts/stretch_manifest.json` |
| 7 | `src/mix_render.py` | `stretch_manifest.json` + `segments.json` + source video | `data/audio_out/final_dubbed.{wav,mp4}`, `artifacts/mix_manifest.json` |
| 8 | `src/qa_report.py` | tts/stretch/mix manifests + `segments.json` | `artifacts/qa_report.{json,md}` |

The `segments.json` schema is the backbone contract: top-level run metadata (`source_file`, `source_language`, `target_language`, `vad_threshold`, `total_segments`) plus a `segments` array where each entry has `segment_id`, `start_time`, `end_time`, `duration`, `speaker_id`, and `text` (populated by ASR). Later stages enrich segments in place rather than reshaping them.

### Key stage behaviors worth knowing before editing

- **Ingestion** normalises to mono 16 kHz WAV via FFmpeg with two-pass **linear** EBU R128 (preserves SNR), then runs Silero VAD. It ships a Windows-safe `read_audio` (soundfile backend) because Silero's default uses `sox_effects`, which is unsupported on Windows.
- **Diarization** uses a *global-to-local intersection* strategy: run pyannote on the **full** audio for globally consistent speaker labels, then intersect with VAD segments, splitting multi-speaker segments at turn boundaries. Models load **sequentially** to stay within a 4 GB VRAM budget. Requires an HF token (`HF_TOKEN`) and accepted licenses (see requirements.txt).
- **ASR** transcribes the **full file in one pass** with WhisperX (`large-v3` backend) + wav2vec2 forced alignment, then intersects the aligned **word timestamps** with the diarization segments (same global-to-local pattern as diarization). This preserves linguistic context that per-slice transcription destroyed. Language config comes from `config/language_registry.json`: each entry supplies a short stylistic Whisper `initial_prompt` (kept content-free to avoid prompt-echo) and a `hallucination_patterns` list. A prompt-echo guard discards any transcript that regurgitates the prompt. Add new languages here, not in code.
- **Translation** adapts (not literally translates) with a self-hosted Qwen (`Qwen/Qwen2.5-14B-Instruct-AWQ` default, pre-quantized 4-bit). Each line is given an explicit **isochrony budget** — a target English syllable count derived from the segment `duration` (~4 syll/s ±15%) — so the dub fits its window. Short conversational fillers are translated, not skipped; only empty/loop/echo segments are hard-skipped.
- **Time-stretch** tiers: ≤1.15× → trivial/no stretch; 1.15–2.0× → WSOLA stretch; >2.0× → **capped at 2.0×**, overflow allowed (never sacrifice intelligibility). 50 ms fit tolerance.
- **Mix/render**: skipped segments fall back to original-audio passthrough; segments placed at original `start_time` with crossfades; final video is AAC 192k. Lip-sync QA is manual only.

## Running the pipeline

There are two execution environments; scripts are identical, only deps differ.

**Local (Windows, `requirements.txt`)** — run each stage in order from repo root:
```powershell
python src/ingestion_vad.py data/audio_in/sample.mp4   # --threshold 0.4
python src/diarization.py                               # --hf-token hf_xxx --max-speakers 3 --device cpu
python src/asr_transcription.py                        # --model large-v3 --source-lang ar --device cuda
python src/translation.py                              # --model Qwen/Qwen2.5-14B-Instruct-AWQ
python src/tts_synthesis.py                             # --device cpu
python src/time_stretch.py                              # --max-ratio 2.0
python src/mix_render.py                                # --source data/audio_in/sample.mp4
python src/qa_report.py
```
Every script accepts `--input`/`--output`/path overrides and defaults to the `artifacts/` locations above, so stages chain with no arguments once the input video is at `data/audio_in/sample.mp4`.

**Colab GPU (`requirements-colab.txt`)** — the intended full-run path. Open `notebooks/colab_pipeline.ipynb` (source of truth is `notebooks/colab_pipeline.py`, kept in sync). It mounts Drive, points `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME` at Drive so large models are cached once, clones this repo, installs deps, and runs Phases A–D. **`colab_pipeline.py` and the `.ipynb` must be edited together** — recent commits repeatedly re-sync them.

> Phase naming differs between the notebook and the source docstrings: the notebook labels ingestion "Phase A", while `ingestion_vad.py`/`diarization.py` docstrings call the speaker layer "Phase B". Follow the table above for actual run order.

## Setup

- **FFmpeg is a system dependency** (not pip). `winget install FFmpeg` (Win) / `brew install ffmpeg` (mac) / `apt install ffmpeg` (Linux); must be on PATH.
- **Hugging Face token** required for pyannote diarization: accept licenses for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then `$env:HF_TOKEN = "hf_..."` (PowerShell) or pass `--hf-token`.
- Local dev targets an RTX 3050 Ti / 4 GB VRAM budget; all stages are written to fit that (8-bit LLM, sequential model loading, CPU fallbacks via `--device cpu`).

## Conventions

- No test suite exists yet; verification is via the QA report (`artifacts/qa_report.md`) and manual review of `artifacts/audio_out/` and `data/audio_out/final_dubbed.mp4`.
- `artifacts/`, `.venv/`, and `data/audio_out/` are gitignored — the JSON contracts are runtime output, not committed source (the two committed `segments.json`/`transcripts.json` are sample fixtures).
- Stage scripts use `PROJECT_ROOT = Path(__file__).resolve().parent.parent`-relative paths; run them from anywhere but keep the `src/ … artifacts/ … data/` layout intact.
- User-facing console output uses ✔/✖/▶/✅ status glyphs and `[n/N]` step counters — match this style when adding stage output.

## Current Session State (2026-07-06) — Handoff: Phase E/F integration, pivot to Arabic-out

**Branch:** `feature/upgrade-stack` (all recent work is here, NOT `feature/architecture-refactor`).

### Current status — what works
- **Phases A, B, C, D are verified and working** end-to-end in
  `notebooks/colab_pipeline_(Colab).ipynb` on Colab (GPU):
  A = ingestion/VAD, B = diarization, C = ASR (WhisperX large-v3), D = translation
  (Qwen2.5-14B-Instruct-AWQ). Each phase is a `src/*.py` CLI stage the notebook
  invokes via `subprocess`, emitting its JSON contract into `artifacts/`.
- ASR confidence surfacing (`asr_confidence` / `low_confidence_asr`) is committed
  in `src/asr_transcription.py` (commit `5e1ffee`). NOTE: the committed
  `artifacts/*.json` are pre-fix runs, so those fields read null there — a fresh
  GPU run is needed to populate real numbers.

### THE BLOCKING DECISION — direction pivot (Arabic Video Out)
- The pipeline **currently outputs ENGLISH dubbing**: `translation.json` is
  `source_language: ar → target_language: en`, and `translated_text` is English
  (e.g. "You busy with the laundry now?"). Phase E (`src/tts_synthesis.py`)
  synthesizes **English** from that field.
- **The objective is Arabic Video Out** (Arabic source video → Arabic dubbed
  video). This is a direction change that must be resolved in Phase D BEFORE
  Phase E/F can produce Arabic audio. The current artifact cannot produce Arabic
  speech — its text is English.

### Required actions for next session (in order)
1. **Revisit Phase D (Translation) → make it output Arabic.** Change the target so
   the dubbed text is Arabic (AR→AR: an *isochronous Arabic rewrite/adaptation*
   that fits each segment's timing budget, NOT a literal copy of the source
   `text`). Touch points in `src/translation.py`: `SYSTEM_PROMPT`,
   `build_user_prompt`, and `config/language_registry.json` (target language).
   Re-run Phase D to regenerate `translation.json` with Arabic `translated_text`.
2. **Implement Phase E (TTS) in the notebook.** IMPORTANT: the stage script
   **already exists** — `src/tts_synthesis.py` (XTTS v2 via Coqui, self-hosted).
   XTTS v2 supports Arabic, so the engine does not change. Add a notebook cell
   that `subprocess`-calls it (same pattern as Phases A–D); do NOT write new
   inline TTS code. Needs `artifacts/voice_ref.wav` (speaker reference).
3. **Implement Phase F (Video Muxing) in the notebook.** Also **already exists** as
   scripts: `src/time_stretch.py` (WSOLA duration-fit → `stretch_manifest.json`),
   `src/mix_render.py` (mix + mux over source video → `final_dubbed.mp4`),
   `src/qa_report.py` (QA). Add notebook cells that `subprocess`-call these; no
   `moviepy` needed — `mix_render.py` muxes via FFmpeg directly.

### Hard constraints to respect during E/F (do not violate)
- **Zero third-party/cloud APIs** (rule #1). `edge-tts` is a Microsoft cloud
  endpoint and is **banned** — use the self-hosted **XTTS v2** already wired in
  `src/tts_synthesis.py`.
- **Reuse the existing `src/` stages**, don't reinvent them inline — the modular
  CLI-stage + JSON-contract design is a hard architectural rule.
- New TTS/video deps must be pinned/isolated to avoid conflicting with the
  pinned `whisperx==3.3.1` / `transformers==4.47.1` / `autoawq==0.2.8` set.

### Environment-agnostic refactor (this session, feature/upgrade-stack) — UNCOMMITTED
Created/edited but NOT yet committed (human commits explicitly, one concern per
commit). These make the pipeline runnable off Colab (RunPod/local):
- `setup_env.sh` (new) — staged, failure-tolerant dependency installer
  (PyTorch → core → whisperx `--no-deps` → autoawq last). Replaces rigid
  `requirements-colab.txt` install.
- `convert_to_ct2.py` (new) — programmatic HF→CTranslate2 conversion (fixes the
  CLI's `dtype` TypeError / `Fast download` ValueError).
- `merge_lora.py`, `eval_model.py` (new) — Egyptian-ASR experiment tooling
  (LoRA merge + standalone CT2 spot-check). Experimental ASR is opt-in only.
- `notebooks/colab_pipeline.py` + `notebooks/colab_pipeline.ipynb` (edited) —
  env auto-detection (`IN_COLAB`, `STORAGE_ROOT` = `DUBLY_STORAGE_ROOT` env →
  `/workspace` → `<repo>/storage`, `REPO_DIR` auto-detect), no Drive mount,
  ASR legacy/experimental switchboard. **NOTE:** the file the user runs on Colab
  is `colab_pipeline_(Colab).ipynb`, which is a SEPARATE copy still on the old
  `/content` Cell 0 — reconcile the two notebooks next session.

### Standing process rules (unchanged)
- One fix per commit. Show the full diff and wait for explicit "approved" before
  moving on.
- Never run git yourself (add/commit/push) — the human handles this after review.
- No `git add .` — explicit files only.
- Don't touch files outside the current task's stated scope.
