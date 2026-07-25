import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src import config

tokenizer = None
model = None


def load_model(model_name=config.LLM_MODEL_NAME, use_4bit=config.USE_4BIT_QUANTIZATION):
    global tokenizer, model

    if model is not None:
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # إنشاء مجلد offload في الـ root لو مش موجود
    offload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "offload")
    os.makedirs(offload_folder, exist_ok=True)


    max_memory = {
        0: "6GiB",
        "cpu": "30GiB",
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )

    return tokenizer, model


def generate_text(prompt, max_new_tokens=config.LLM_MAX_NEW_TOKENS, temperature=config.LLM_TEMPERATURE):
    tok, mdl = load_model()

    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    inputs = tok(formatted_prompt, return_tensors="pt").to(mdl.device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )

    new_tokens = outputs[0][prompt_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    answer = generate_text("Explain model quantization in two sentences.", max_new_tokens=100)
    print(answer)