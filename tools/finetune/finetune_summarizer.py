#!/usr/bin/env python3
"""QLoRA fine-tune a summarizer on the rev16 distilled JSONL.

Defaults are sized for an RTX 2070 Super Mobile (8 GB VRAM):
  - 4-bit NF4 base + LoRA in fp16
  - Qwen2.5-3B-Instruct as the base (small enough to leave headroom)
  - micro-batch 1, grad-accum 8 (effective batch 8)
  - sequence length 2048 (enough for our 600-word windows + prompt + target)

Usage (after `uv sync --extra finetune` and `prepare_rev16.py`):
    uv run python tools/finetune/finetune_summarizer.py \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --train-file tools/finetune/data/train.jsonl \
        --val-file tools/finetune/data/val.jsonl \
        --output-dir tools/finetune/runs/qwen2.5-3b-rev16-r16

Resume from a previous run by passing the same --output-dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# torch is imported lazily inside main() so `--help` works without the
# finetune extra installed.


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_dataset(train_file: Path, val_file: Path | None, tokenizer, max_seq: int):
    """Load JSONL, render via Qwen2.5 chat template, return tokenized HF datasets.

    The chat template is what the model will see at inference time when called
    through Ollama, so training-time and inference-time prompts must match.
    """
    from datasets import load_dataset

    data_files = {"train": str(train_file)}
    if val_file is not None:
        data_files["validation"] = str(val_file)
    raw = load_dataset("json", data_files=data_files)

    def render(example):
        # Drop our `_meta` block so it can't bleed into the model's input.
        msgs = example["messages"]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    rendered = raw.map(render, remove_columns=raw["train"].column_names)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq,
            padding=False,
        )

    tokenized = rendered.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HF model ID. Use a 3B base for the RTX 2070; bump to 7B once you "
                             "move to the RTX 4090.")
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "tools/finetune/data/train.jsonl")
    parser.add_argument("--val-file", type=Path, default=REPO_ROOT / "tools/finetune/data/val.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit base quantization. Only do this on a 24GB+ GPU.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("finetune")

    if not args.train_file.exists():
        logger.error("train file missing: %s — run prepare_rev16.py first", args.train_file)
        return 1

    # Heavy imports deferred so --help is quick on a fresh checkout.
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForLanguageModeling, TrainingArguments, set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    set_seed(args.seed)

    if not torch.cuda.is_available():
        logger.error("CUDA not available — finetune requires a GPU. Did `nvidia-smi` work?")
        return 1
    logger.info("CUDA device: %s (%.1f GB)",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    quant_config = None
    if not args.no_4bit:
        # NF4 + double quant — Tim Dettmers' QLoRA recipe. Works on Turing
        # (RTX 2070) with bfloat16 compute even though the card prefers fp16:
        # bitsandbytes does the dtype handling internally.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        # Qwen2.5 has no default pad token — re-use eos so the collator
        # doesn't insert a token the model has never seen.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not quant_config else None,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    datasets = _build_dataset(args.train_file, args.val_file if args.val_file.exists() else None,
                              tokenizer, args.max_seq)
    logger.info("Train examples: %d", len(datasets["train"]))
    if "validation" in datasets:
        logger.info("Val examples:   %d", len(datasets["validation"]))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if "validation" in datasets else None,
        eval_strategy="steps" if "validation" in datasets else "no",
        save_strategy="steps",
        save_total_limit=2,
        bf16=True,
        report_to=["none"],
        seed=args.seed,
        load_best_model_at_end="validation" in datasets,
        metric_for_best_model="eval_loss" if "validation" in datasets else None,
        greater_is_better=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # TRL 0.12+ renamed `tokenizer` to `processing_class`; older versions still
    # accept the old name. Try the new name first, fall back for older TRL.
    sft_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        data_collator=collator,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **sft_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **sft_kwargs)

    last_ckpt = None
    if args.output_dir.exists() and any(p.name.startswith("checkpoint-") for p in args.output_dir.iterdir()):
        last_ckpt = True  # let HF auto-pick latest

    trainer.train(resume_from_checkpoint=last_ckpt)

    adapter_dir = args.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("LoRA adapter saved → %s", adapter_dir)

    # Stash the run config so eval / merge scripts can recover it without args.
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq": args.max_seq,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
