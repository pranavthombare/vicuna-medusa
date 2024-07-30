from fastapi import FastAPI, Request
import uvicorn
from medusa.model.medusa_model import MedusaModel
import torch
from pydantic import BaseModel

global model, tokenizer
model = MedusaModel.from_pretrained(
    "FasterDecoding/medusa-vicuna-7b-v1.3",
    torch_dtype=torch.float16,
).to("cuda")
tokenizer = model.get_tokenizer()
app = FastAPI()

# Create a class for the input data
class InputData(BaseModel):
    prompt: str
    parameters: dict

# Create a class for the output data
class OutputData(BaseModel):
    response: str

@app.post('/generate', response_model=OutputData)
async def generate(request: Request, input_data: InputData):
    input_ids = tokenizer.encode(input_data.prompt, return_tensors="pt").to(
        model.base_model.device
    )
    output_stream = model.medusa_generate(
        input_ids,
        temperature=input_data.parameters['temperature'],
        max_steps=input_data.parameters['max_steps'],
    )
    response = ""
    for output in output_stream:
        response = output['text']
    print(response)
    return OutputData(response=response)

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)