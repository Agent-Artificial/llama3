from pydantic import BaseModel
from typing_extensions import Literal, Optional, Union, Iterable, List, Dict, override, Mapping, Self
from typing import TypeVar
from datetime import date


class Message(BaseModel):
    """
    A class representing a message with a role and content.

    Args:
        role (str): The role of the message.
        content (str): The content of the message.

    Returns:
        None
    """

    role: str
    content: str


class ResponseObject(BaseModel):
    """
    A class representing a response object with various attributes.

    Args:
        id (str): The ID of the response object.
        object (str): The type of object.
        created (int): The creation timestamp.
        model (str): The model associated with the response.
        system_fingerprint (str): The system fingerprint.
        choices (List[Dict[str, Union[str, int, float]]]): List of choices with their attributes.
        usage (Dict[str, Union[str, int]]): Dictionary of usage information.

    Returns:
        None
    """
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[Dict[str, Union[int, Dict[str, str], None, str]]]
    usage: Dict[str, Union[str, int]]


"""
Code used is from OpenAI's API repo
https://github.com/openai/openai-python/blob/main/LICENSE

Copyright (c) 2021 OpenAI
"""

class Omit(BaseModel):
    """In certain situations you need to be able to represent a case where a default value has
    to be explicitly removed and `None` is not an appropriate substitute, for example:

    ```py
    # as the default `Content-Type` header is `application/json` that will be sent
    client.post("/upload/files", files={"file": b"my raw file content"})

    # you can't explicitly override the header as it has to be dynamically generated
    # to look something like: 'multipart/form-data; boundary=0d8382fcf5f8c3be01ca2e11002d2983'
    client.post(..., headers={"Content-Type": "multipart/form-data"})

    # instead you can remove the default `application/json` header by passing Omit
    client.post(..., headers={"Content-Type": Omit()})
    ```
    """
    Self: bool


class Query(BaseModel):
    """
    A class representing a query with key-value pairs.

    Explanation:
    This class inherits from BaseModel and Mapping, and represents a query with key-value pairs.
    """
    Self: Mapping[str, object]


class Body(BaseModel):
    """
    A class representing a body with an unspecified object.

    Explanation:
    This class inherits from BaseModel and represents a body with an unspecified object.
    """
    Self: object


class Headers(BaseModel):
    """
    A class representing headers with key-value pairs where the values can be strings or omitted.

    Explanation:
    This class inherits from BaseModel and Mapping, and represents headers with key-value pairs where the values can be strings or omitted.
    """
    Self: Mapping[str, Union[str, Omit]]
    

class _T(BaseModel):
    """
    A class representing a type variable.

    Explanation:
    This class inherits from BaseModel and represents a type variable.
    """
    Self: TypeVar
    class Config:
        arbitrary_types_allowed = True
        

class NotGiven(BaseModel):
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response:
        ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr = Union[_T, NotGiven]
NOT_GIVEN = NotGiven()

class Function(BaseModel):
    """
    A class representing a function with a name, description, and parameters.

    Explanation:
    This class inherits from BaseModel and represents a function with a name, description, and parameters stored as key-value pairs.
    """
    name: str
    description: str
    parameters: Dict[str, str]

class FunctionCall(BaseModel):
    """
    A class representing a function call with options for "none", "auto", or a ResponseObject.

    Explanation:
    This class inherits from BaseModel and represents a function call with options for "none", "auto", or a ResponseObject.
    """
    Self: Union[Literal["none", "auto"], ResponseObject]

class ResponseFormat(BaseModel):
    """
    A class representing a response format with options for "text" or "json_object".

    Explanation:
    This class inherits from BaseModel and represents a response format with options for "text" or "json_object".
    """
    type: Literal["text", "json_object"]

class RequestObject(BaseModel):
    """
    A class representing a request object with options for "text" or "json_object".

    Args:
        model (str): The model to use for the request.
        messages (List[Message]): The messages to use for the request.
        frequency_penalty (Optional[float]): The frequency penalty to use for the request.
        function_call (FunctionCall): The function call to use for the request.
        functions (Iterable[Function]): The functions to use for the request.
        logit_bias (Optional[Dict[str, int]]): The logit bias to use for the request.
        logprobs (Optional[bool]): The logprobs to use for the request.
        max_tokens (Optional[int]): The max tokens to use for the request.
        n (Optional[int]): The n to use for the request.
        presence_penalty (Optional[float]): The presence penalty to use for the request.
        response_format (ResponseFormat): The response format to use for the request.
        seed (Optional[int]): The seed to use for the request.
        stop (Union[Optional[str], List[str]]): The stop to use for the request.
        temperature (Optional[float]): The temperature to use for the request.
        tool_choice (Dict[str, str]): The tool choice to use for the request.
        tools (Iterable[Dict[str, str]]): The tools to use for the request.
        top_logprobs (Optional[int]): The top logprobs to use for the request.
    """
    model: str
    messages: List[Message]
    frequency_penalty: Optional[float]=None
    function_call: Optional[FunctionCall]=None
    functions: Optional[Iterable[Function]]=None
    logit_bias: Optional[Dict[str, int]]=None
    logprobs: Optional[bool]=None
    max_tokens: Optional[int]=None
    n: Optional[int]=None
    presence_penalty: Optional[float]=None
    response_format: Optional[ResponseFormat]=None
    seed: Optional[int]=None
    stop: Optional[Union[str, List[str]]]=None
    temperature: Optional[float]=None
    tool_choice: Optional[Dict[str, str]]=None
    tools: Optional[Iterable[Dict[str, str]]]=None
    top_logprobs: Optional[int]=None
    top_p: Optional[float]=None
    user: Optional[str]=None

class RequestObjectStreaming(RequestObject):
    """
    A class representing a streaming request object with options for "text" or "json_object".
    Args:
        stream: Literal[True] 
    """
    stream: Literal[True]
class RequestObjectNonStreaming(RequestObject):
    """
    A class representing a streaming request object with options for "text" or "json_object".
    Args:
        stream: Literal[False] 
    """
    stream: Optional[Literal[False]]