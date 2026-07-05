"""
merge_lora.py — Prepare Candidate A for local ASR evaluation
============================================================
Standalone conversion utility (NOT a pipeline stage). Merges the LoRA adapter
`AbdelrahmanHassan/whisper-large-v3-egyptian-arabic` into its base
`openai/whisper-large-v3`, then saves a full Hugging Face checkpoint plus the
tokenizer / feature-extractor files needed by the next step.

Run this in a SEPARATE conversion environment (not the pinned pipeline venv):
  pip install "transformers>=4.40" peft accelerate torch ctranslate2
  python merge_lora.py

Then convert the merged folder to CTranslate2 for eval_model.py:
  ct2-transformers-converter \
    --model whisper-large-v3-egy-merged \
    --output_dir whisper-large-v3-egy-ct2 \
    --copy_files tokenizer.json preprocessor_config.json \
    --quantization float16 --force

Nothing here reads or writes an artifacts/ data contract.
"""

from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
    WhisperFeatureExtractor,
)
from peft import PeftModel
import torch

BASE = "openai/whisper-large-v3"
ADAPTER = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"
OUT = "whisper-large-v3-egy-merged"


def main() -> None:
    # Load the base on CPU in fp16 (fits a 4GB machine; the LoRA merge math is
    # fine in fp16). The base auto-downloads from the HF hub on first run.
    print(f"▶ Loading base model: {BASE}")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Apply the adapter, fold its weights into the base, and drop the LoRA layers.
    print(f"▶ Applying + merging adapter: {ADAPTER}")
    peft_model = PeftModel.from_pretrained(base, ADAPTER)
    merged = peft_model.merge_and_unload()

    # Writes model.safetensors + config.json + generation_config.json.
    print(f"▶ Saving merged model → {OUT}")
    merged.save_pretrained(OUT)

    # Whisper's tokenizer + feature extractor come from the BASE model, not the
    # adapter. Save the FAST tokenizer explicitly so we get tokenizer.json (which
    # faster-whisper requires), and the feature extractor for preprocessor_config.json.
    print("▶ Saving tokenizer (tokenizer.json) + feature extractor (preprocessor_config.json)")
    WhisperTokenizerFast.from_pretrained(BASE).save_pretrained(OUT)
    WhisperFeatureExtractor.from_pretrained(BASE).save_pretrained(OUT)

    print(f"✔ Merged model + tokenizer written to: {OUT}")


if __name__ == "__main__":
    main()
