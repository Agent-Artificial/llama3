from datetime import date
from data_models import Message, ResponseObject, RequestObject
from typing_extensions import Dict, Union, List



def create_response(
    message: Message,
    id="chatcmpl-123",
    object="chat.completion",
    created=11111,
    model="Llama3-70b",
    system_fingerprint="fp_44709d6fcb",
    prompt_tokens=9,
    completion_tokens=12,
    total_tokens=21
    ):
    """
    Creates a response object based on the provided message and additional parameters.
    
    Args:
        message (Dict[str, str]): The message content.
        id (str): The ID of the response object.
        object (str): The type of object.
        created (date): The creation date.
        model (str): The model associated with the response.
        system_fingerprint (str): The system fingerprint.
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        total_tokens (int): Total number of tokens.
    
    Returns:
        None
    """

    choices: List[Dict[str, Union[int, Dict[str, str], None, str]]] = [{
        "index": 0,
        "message": message.dict(exclude_unset=True),
        "logprobs": None,
        "finish_reason": "stop"
    }]
    usage: Dict[str, Union[str, int]] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    return ResponseObject(
        id = id,
        object = object,
        created = created,
        model = model,
        system_fingerprint = system_fingerprint,
        choices = choices,
        usage = usage
        ).dict(exclude_unset=True)

def create_message(input_text: str, role: str = "user") -> Message:
    """
    Creates a message with the given input text and role.

    Args:
        input_text (str): The input text content.
        role (str, optional): The role of the message. Defaults to "user".

    Returns:
        None
    """
    return Message(
        role = role,
        content = input_text
    )

