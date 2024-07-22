from messages import create_message, create_response
from data_models import Message
from typing import Dict

import torch
from accelerate import Accelerator
from transformers import pipeline
from messages import create_message, create_response
from data_models import Message
from typing import Dict, List

def generate_text(args: Dict, model, tokenizer, accelerator: Accelerator) -> Dict[str, str]:
    prompt = tokenizer.apply_chat_template(
        args["messages"], tokenize=False, add_generation_prompt=True
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args["max_new_tokens"],
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=args["do_sample"],
            temperature=args["temperature"],
            top_p=args["top_p"],
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    message = Message(role="assistant", content=response)
    return create_response(message=message)