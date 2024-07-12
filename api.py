import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference import generate_text
from data_models import RequestObject, Message
import torch
import transformers
import tqdm
from accelerate import Accelerator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = 8000
HOST = "0.0.0.0"
RELOAD = True


def load_model():
    accel = Accelerator()

    tqdm.tqdm(range(1000), disable=not accel.is_local_main_process)

    pipe = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=accel.device,
    )

    model = accel.prepare(pipe.model)
    model.eval()
    return accel, pipe


accelerator, pipeline = load_model()


@app.post("/text")
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
def text_generation(request: RequestObject):
    messages = request.messages
    args = {
        "messages": messages,
        "max_new_tokens": 16000,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    try:
        response = generate_text(args, pipeline, accelerator)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=9099, reload=RELOAD)
#    response = text_generation(RequestObject(messages=[Message(role="user", content="hello")], model = "Llama3-70b"))
#
#    print(response.body)
