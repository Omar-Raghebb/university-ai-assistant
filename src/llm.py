import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src import config

tokenizer = None
model = None


def load_model(model_name=config.LLM_MODEL_NAME, use_4bit=config.USE_4BIT_QUANTIZATION):
    global tokenizer, model

    if model is not None:
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device_map = "auto"
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)

    return tokenizer, model


def generate_text(prompt, max_new_tokens=config.LLM_MAX_NEW_TOKENS, temperature=config.LLM_TEMPERATURE):
    tok, mdl = load_model()

    # Mistral uses different format than Qwen
    # Use simple instruction format for Mistral
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