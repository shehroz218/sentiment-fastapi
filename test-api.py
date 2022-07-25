from typing import Dict, Union

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from model.prediction import load_class, predicter

app = FastAPI()

class input_text(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str

@app.post("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: input_text, model: predicter = Depends(load_class)):
    sentiment, probabbility = predicter.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, probabilities=probabbility,
    )
