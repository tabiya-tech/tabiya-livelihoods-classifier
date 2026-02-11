"""
Fine-tune Gemma 3 1B (or Gemma 2 2B) for NER using LoRA/QLoRA on prompt+response JSONL
produced by prepare_gemma_ner_dataset.py. Saves the adapter and optionally merges
to a full model for evaluation with test/run_ner_eval_gemma.py --model <path>.

Usage (GCP GPU VM):
  python scripts/finetune_gemma_ner.py \
    --dataset data/gemma_ner_sft.jsonl \
    --base-model google/gemma-3-1b-it \
    --output-dir output/gemma_ner_lora \
    --merge-output output/gemma_ner_merged
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma for NER with LoRA")
    parser.add_argument("--dataset", type=str, required=True, help="JSONL with prompt and response")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-1b-it", help="HF causal LM (Gemma 3 1B it; no 2B in Gemma 3)")
    parser.add_argument("--output-dir", type=str, default="output/gemma_ner_lora", help="Where to save LoRA adapter")
    parser.add_argument("--merge-output", type=str, default=None, help="If set, merge LoRA and save full model here for eval")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-qlora", action="store_true", help="Use 4-bit quantization (saves VRAM)")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 (set if no bf16 support)")
    args = parser.parse_args()

    import json
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    token = os.getenv("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN for gated models (e.g. Gemma)")
        sys.exit(1)

    rows = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    for r in rows:
        r["text"] = r["prompt"] + r["response"]
    dataset = Dataset.from_list(rows)
    num_samples = len(dataset)
    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = max(1, (num_samples + eff_batch - 1) // eff_batch)
    total_steps = steps_per_epoch * args.num_epochs
    print(f"Dataset: {num_samples} examples, ~{steps_per_epoch} steps/epoch, ~{total_steps} total steps")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    model_kw = {"token": token, "trust_remote_code": True}
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if torch.cuda.is_available():
            model_kw["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kw)
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    response_template = "Output:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    use_cpu = not torch.cuda.is_available()
    if use_cpu:
        bf16, fp16 = False, False
        print("CUDA not available, training on CPU (will be slow)")
    else:
        bf16 = torch.cuda.is_bf16_supported() and not args.fp16
        fp16 = not bf16
        print(f"Training on GPU (bf16={bf16}, fp16={fp16})")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=bf16,
        fp16=fp16,
        use_cpu=use_cpu,
        logging_steps=5,
        logging_first_step=True,
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_output:
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                token=token,
            ),
            args.output_dir,
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge_output)
        tokenizer.save_pretrained(args.merge_output)
        print(f"Merged model saved to {args.merge_output}. Use: python test/run_ner_eval_gemma.py --model {args.merge_output}")


if __name__ == "__main__":
    main()
