#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save full FP16 weights.

The merged model is what `convert_hf_to_gguf.py` then turns into a GGUF for
Ollama. Has to be run on CPU OR a GPU big enough to hold the *unquantized*
base — for Qwen2.5-3B that's ~6 GB FP16. The RTX 2070 (8 GB) can do this
but it'll be tight; passing --device cpu is the safe default.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="Directory written by finetune_summarizer.py (contains adapter/ + run_config.json).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the merged FP16 model.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("merge")

    run_cfg_path = args.adapter_dir / "run_config.json"
    if not run_cfg_path.exists():
        logger.error("run_config.json missing in %s — was this dir produced by finetune_summarizer.py?", args.adapter_dir)
        return 1
    run_cfg = json.loads(run_cfg_path.read_text())
    base_model = run_cfg["base_model"]

    adapter = args.adapter_dir / "adapter"
    if not adapter.exists():
        logger.error("adapter dir missing: %s", adapter)
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("Loading base %s onto %s", base_model, args.device)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    logger.info("Attaching adapter %s", adapter)
    merged = PeftModel.from_pretrained(base, str(adapter))
    merged = merged.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Merged model written → %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
