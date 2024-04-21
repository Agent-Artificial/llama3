import transformers
import torch
from accelerate.accelerator import Accelerator
from tqdm.auto import tqdm
from messages import create_message, create_response
from data_models import Message
from typing import Dict, List, Union


def generate_text(messages: Union[List[Message], Dict[str, str]]) -> Dict[str, str]:
    """
    Generates text based on a given prompt using a pre-trained model.

    Args:
        prompt (str, optional): The prompt for text generation. Defaults to "tell me a bit about yourself".

    Returns:
        None
    """

    accelerator = Accelerator()

    args = {
        "model_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "mode": "text-generation",
        "max_train_steps": 1000,
        "model_kwargs": {"torch_dtype": torch.bfloat16},
        "device": accelerator.device,
        "messages": messages,
        "max_new_tokens": 16000,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    tqdm(range(args['max_train_steps']), disable=not accelerator.is_local_main_process)

    pipeline = transformers.pipeline(
        args['mode'],
        model=args['model_id'],
        model_kwargs=args['model_kwargs'],
        device=args['device'],
    )

    prompt = pipeline.tokenizer.apply_chat_template(  # type: ignore
            args['messages'],
            tokenize=False, 
            add_generation_prompt=True
    )  

    terminators = [
        pipeline.tokenizer.eos_token_id,  # type: ignore
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # type: ignore
    ]

    pipeline = accelerator.prepare(pipeline)

    outputs = pipeline(
        prompt,
        max_new_tokens=args['max_new_tokens'],
        eos_token_id=terminators,
        do_sample=args['do_sample'],
        temperature=args['temperature'],
        top_p=args['top_p']

    )
    response = outputs[0]["generated_text"][len(prompt):]
    message = create_message(response, "assistant")
    return create_response(message=message)
