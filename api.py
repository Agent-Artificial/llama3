import os
import torch
from accelerate import Accelerator
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import generate_text

app = FastAPI()

# Initialize Accelerator
accelerator = Accelerator()

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Prepare model for distributed training
model = accelerator.prepare(model)

class GenerationRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 100
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    
@app.post("/text")
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        args = request.model_dump()
        response = generate_text(args, model, tokenizer, accelerator)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7099)