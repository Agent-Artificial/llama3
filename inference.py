from messages import create_message, create_response
from data_models import Message
from typing import Dict


def generate_text(args, pipeline, accelerator) -> Dict[str, str]:
    """
    Generates text based on a given prompt using a pre-trained model.

    Args:
        prompt (str, optional): The prompt for text generation. Defaults to "tell me a bit about yourself".

    Returns:
        None
    """

    prompt = pipeline.tokenizer.apply_chat_template(  # type: ignore
        args["messages"], tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,  # type: ignore
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # type: ignore
    ]
    pipeline = accelerator.prepare(pipeline)
    outputs = pipeline(
        prompt,
        max_new_tokens=args["max_new_tokens"],
        eos_token_id=terminators,
        pad_token_id=pipeline.tokenizer.pad_token_id,
        do_sample=args["do_sample"],
        temperature=args["temperature"],
        top_p=args["top_p"],
    )
    response = outputs[0]["generated_text"][len(prompt) :]
    message = Message(role="assistant", content=response)
    return create_response(message=message)


if __name__ == "__main__":
    generate_text([create_message("hello")])
