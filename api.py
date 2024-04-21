import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference import generate_text
from data_models import RequestObject


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


@app.post("/text")
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def text_generation(request: RequestObject):
    messages = request.messages
    try:
        response = generate_text(messages)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT, reload=RELOAD)