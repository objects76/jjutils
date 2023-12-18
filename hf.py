
import torch
import transformers

def get_llm_model(model_id, use_quantization = False, is_lowmem=True):
    bnb_config = None
    if use_quantization:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # new float 4bit
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attention_2=True, # works with Llama models and reduces memory reqs
        cache_dir=None)

    # do not cache linke attention weights, ... (NG for speed, Good for gpu men): training model compression.
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=None)

    return model, tokenizer