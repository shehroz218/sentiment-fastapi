from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.prediction import load_class, predicter

app = FastAPI()

class input_text(BaseModel):
    """
    loads string from api call
    """
    text: str


class outputResponse(BaseModel):
    """
    returns probabilites and sentiment for api output
    """
    probabilities: Dict[str, float]
    sentiment: str

@app.get("/")
def read_root():
    """
    root api call
    """
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}

@app.post("/predict", response_model=outputResponse)
def predict(request: input_text, model: predicter = Depends(load_class)):
    """
    calls the predict function to create predictions on text

    Attributes:
    ----------
    input_text: str
        text/tweet to be classified
    predicter: hugging face pipeline
        uses hugging face pipeline from load_model function to create predictions
    
    """
    sentiment, probabbility = predicter.predict(request.text)
    return outputResponse(
        sentiment=sentiment, probabilities=probabbility,
    )
