from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.prediction import load_class, predicter

app = FastAPI()

class input_text(BaseModel):
    text: str


class outputResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict", response_model=outputResponse)
def predict(request: input_text, model: predicter = Depends(load_class)):
    sentiment, probabbility = predicter.predict(request.text)
    return outputResponse(
        sentiment=sentiment, probabilities=probabbility,
    )


