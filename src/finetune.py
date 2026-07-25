"""
Lightweight LoRA fine-tuning, same recipe as
fine-tune-summary__1_.ipynb (peft + bitsandbytes + trl) and
FineTuning_NewKnowledge.ipynb (instruction-style Q&A formatting).

Purpose here is NOT to teach the model new facts (that's the RAG's
job) -- it's to nudge the model's answering *style* toward how a
university advisor should sound (concise, formal, matches the
student's language). Keep the dataset small (50-100 examples is
plenty); this is meant to run in well under an hour on a single GPU.

Run from the project root:
    python -m src.finetune
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from src import config

QA_DATA_PATH = "data/qa_finetune_sample.json"
OUTPUT_DIR = "outputs/university-assistant-lora"


def load_qa_dataset(path: str = QA_DATA_PATH) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    def format_example(example):
        text = f"[INST] {example['question']} [/INST] {example['answer']}"
        return {"text": text}

    dataset = Dataset.from_list(pairs).map(format_example)
    print(f"Loaded {len(dataset)} fine-tuning examples from {path}")
    return dataset


def load_base_model_for_training(model_name: str = config.LLM_MODEL_NAME):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )
    return tokenizer, model


def build_lora_model(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def run_finetuning():
    dataset = load_qa_dataset()
    tokenizer, base_model = load_base_model_for_training()
    peft_model = build_lora_model(base_model)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()

    peft_model.save_pretrained(f"{OUTPUT_DIR}/adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")
    print(f"LoRA adapter saved to {OUTPUT_DIR}/adapter")

    merged = peft_model.merge_and_unload()
    merged.save_pretrained(f"{OUTPUT_DIR}/merged")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged")
    print(f"Merged model saved to {OUTPUT_DIR}/merged")


if __name__ == "__main__":
    # python -m src.finetune
    # NOTE: needs a GPU. Run this on Kaggle (like FineTuning_NewKnowledge.ipynb) --
    # it will not run on a CPU-only laptop in reasonable time.
    run_finetuning()