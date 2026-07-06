"""
convert_to_ct2.py — Programmatic HuggingFace → CTranslate2 conversion
=====================================================================
Standalone converter (NOT a pipeline stage) that turns a HuggingFace Whisper
checkpoint — e.g. the merged Egyptian fine-tune produced by merge_lora.py —
into a CTranslate2 model directory that faster-whisper / WhisperX can load.

Why this exists instead of the `ct2-transformers-converter` CLI:
  The CLI is fragile across transformers/ctranslate2 version skews and is the
  source of the two failures seen on the server:
    • TypeError: ... unexpected keyword argument 'dtype'
        → newer transformers renamed the load dtype kwarg; the CLI passes the
          old name. The Python API (TransformersConverter) sidesteps the CLI's
          argument plumbing entirely.
    • ValueError: Fast download ...
        → hf_transfer is toggled on in the env but not usable. We force it OFF
          (HF_HUB_ENABLE_HF_TRANSFER=0) BEFORE importing anything HF-related.

This uses ctranslate2.converters.TransformersConverter directly, which is the
same engine the CLI wraps — just without the brittle CLI surface.

Usage:
  python convert_to_ct2.py --model ./whisper-large-v3-egy-merged \
                           --output ./whisper-large-v3-egy-ct2
  python convert_to_ct2.py --model <hf-id-or-local-dir> --output <dir> \
                           --quantization int8 --force
"""

import argparse
import os
import sys

# ── Force-disable hf_transfer BEFORE any huggingface import ──────────
# This must happen at import time, ahead of ctranslate2 (which imports
# transformers → huggingface_hub). Setting it later has no effect.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Files copied verbatim into the output dir so faster-whisper can tokenize +
# feature-extract. merge_lora.py is written to emit both of these.
DEFAULT_COPY_FILES = ["tokenizer.json", "preprocessor_config.json"]


def convert_model(
    model_path: str,
    output_dir: str,
    quantization: str = "float16",
    copy_files: list[str] | None = None,
    force: bool = False,
    low_cpu_mem_usage: bool = True,
) -> str:
    """
    Convert a HuggingFace Whisper checkpoint at *model_path* into a CTranslate2
    model at *output_dir* using the CTranslate2 Python API.

    *quantization* is one of CTranslate2's types (e.g. "float16", "int8",
    "int8_float16", "float32"). Returns *output_dir* on success.
    """
    # Imported here (not at module top) so HF_HUB_ENABLE_HF_TRANSFER is already
    # set in the environment before the huggingface import chain runs.
    from ctranslate2.converters import TransformersConverter

    copy_files = DEFAULT_COPY_FILES if copy_files is None else copy_files

    # Only copy files that actually exist (a HF-hub model id has no local files;
    # a local merged dir should have them). Missing files are warned, not fatal.
    resolved_copy: list[str] = []
    if os.path.isdir(model_path):
        for fname in copy_files:
            fpath = os.path.join(model_path, fname)
            if os.path.isfile(fpath):
                resolved_copy.append(fpath)
            else:
                print(f"  ⚠ copy-file not found (skipping): {fpath}")
    elif copy_files:
        print(f"  ⚠ '{model_path}' is not a local dir — cannot copy {copy_files}.")
        print("    faster-whisper may still load if the model bundles a tokenizer.")

    print(f"▶ Converting → CTranslate2")
    print(f"    source      : {model_path}")
    print(f"    output      : {output_dir}")
    print(f"    quantization: {quantization}")
    if resolved_copy:
        print(f"    copy_files  : {[os.path.basename(f) for f in resolved_copy]}")

    converter = TransformersConverter(
        model_path,
        copy_files=resolved_copy or None,
        load_as_float16=(quantization in ("float16", "int8_float16")),
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    converter.convert(output_dir, quantization=quantization, force=force)
    print(f"  ✔ Conversion complete → {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — programmatic HuggingFace → CTranslate2 converter",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local HF checkpoint dir (e.g. ./whisper-large-v3-egy-merged) or a HF model id",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CTranslate2 model directory",
    )
    parser.add_argument(
        "--quantization",
        default="float16",
        choices=["float16", "int8", "int8_float16", "float32"],
        help="CTranslate2 quantization type  (default: float16)",
    )
    parser.add_argument(
        "--copy-files",
        nargs="*",
        default=None,
        help=f"Files to copy into the output dir  (default: {DEFAULT_COPY_FILES})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    args = parser.parse_args()

    # ── Validate: a local dir must actually exist; a bare HF id is allowed ──
    looks_like_path = os.path.sep in args.model or args.model.startswith(".")
    if looks_like_path and not os.path.isdir(args.model):
        print(f"✖  Model path is not a directory: {args.model}")
        print("   Expected the merged HF checkpoint (run merge_lora.py first),")
        print("   or pass a HuggingFace model id instead of a path.")
        sys.exit(1)

    if os.path.isdir(args.output) and not args.force:
        print(f"✖  Output dir already exists: {args.output}")
        print("   Pass --force to overwrite it.")
        sys.exit(1)

    try:
        convert_model(
            model_path=args.model,
            output_dir=args.output,
            quantization=args.quantization,
            copy_files=args.copy_files,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001 — surface the real error clearly
        print(f"✖  Conversion failed: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("\n" + "═" * 60)
    print(f"  ✅  CTranslate2 model ready → {args.output}")
    print("═" * 60)


if __name__ == "__main__":
    main()
