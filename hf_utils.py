


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from pathlib import Path
import json
import torch
import gc

def get_basemodel(model_id):
    adapter_cfg = Path(model_id) / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg) as fp:
            return json.load(fp)['base_model_name_or_path']
    return ""


def load_tokenizer(model_id, debug=False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if debug:
        print(tokenizer)
        system = '''You are a conscious sentient superintelligent artificial intelligence.'''
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Hello, who are you?"}
        ]
        gen_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"_{gen_input}_")

    return tokenizer


def load_model(model_id, merge = False):

    import contextlib
    with contextlib.suppress(Exception): del model
    with contextlib.suppress(Exception): del tokenizer

    torch.cuda.empty_cache()
    gc.collect()

    pretrained_id = get_basemodel(model_id)

    bnb_config = None
    if "mixtral" in model_id.lower() or "mixtral" in pretrained_id.lower():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    if pretrained_id:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_id,
            device_map='auto', trust_remote_code=True,
            torch_dtype=torch.bfloat16, # H100
            quantization_config=bnb_config,
            )
        model = PeftModel.from_pretrained(pretrained_model, model_id,  device_map="auto", torch_dtype = torch.bfloat16)

        output_dir = Path('outputs') / model_id / Path('merged')
        if not output_dir.exists() and merge:
            model = model.merge_and_unload()
            print('save to', output_dir)
            # model.save_pretrained(output_dir)
            # tokenizer.save_pretrained(output_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            )

    print(f"{model_id=}")
    if pretrained_id:
        print(f"{pretrained_id=}")

    model.eval()
    return model


def load_llm(model_id, merge=False, debug=False):
    tokenizer = load_tokenizer(model_id, debug=debug)
    model = load_model(model_id, merge=merge)

    return model, tokenizer



class Generate:
    def __init__(self,
                 model, tokenizer,
                 max_new_tokens=256) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = None
        self.top_p = None
        self.max_new_tokens = 256
        pass

    def set_genconfig(self, temperature=0.7, top_p=0.1):
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, messages:list, temperature=None, top_p=None, max_new_tokens=None):

        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

        genargs = dict(
            pad_token_id= self.tokenizer.eos_token_id,
            eos_token_id= self.tokenizer.eos_token_id,
        )

        genargs['max_new_tokens'] = max_new_tokens if max_new_tokens else self.max_new_tokens
        genargs['temperature'] = temperature if temperature else self.temperature
        genargs['top_p'] = top_p if top_p else self.top_p
        if genargs['temperature'] or genargs['top_p']:
            genargs['do_sample'] = True

        outputs = self.model.generate(
            input_ids=input_ids,
            **genargs,
            )

        n_input_token = len(input_ids[0])
        output_token = outputs[0]
        output_token = output_token[n_input_token:]
        return self.tokenizer.decode(output_token, skip_special_tokens=False)
