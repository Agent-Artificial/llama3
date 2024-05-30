import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset 
from random import randint
from 


model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")


# Load our test dataset
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))
messages = eval_dataset[rand_idx]["messages"][:2]

tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

# Test on sample 
input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(model.device)
attention = model.forward(input_ids)
outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id= tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id

)
response = outputs[0][input_ids.shape[-1]:]

print(f"**Query:**\n{eval_dataset[rand_idx]['messages'][1]['content']}\n")
print(f"**Original Answer:**\n{eval_dataset[rand_idx]['messages'][2]['content']}\n")
print(f"**Generated Answer:**\n{tokenizer.decode(response,skip_special_tokens=True)}")