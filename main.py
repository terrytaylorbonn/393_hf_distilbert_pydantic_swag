# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import Literal

# Load Hugging Face sentiment analysis model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define request and response models
class ChatMessage(BaseModel):
    role: Literal["user", "system"]
    content: str

class SentimentResponse(BaseModel):
    sentiment: Literal["POSITIVE", "NEGATIVE"]
    score: float

# Create FastAPI app
app = FastAPI(
    title="LLM Personal Tools API",
    description="An example of using Hugging Face models with FastAPI and Swagger",
    version="1.0"
)

@app.post("/chat/sentiment", response_model=SentimentResponse, tags=["Chatbot"])
def get_sentiment(message: ChatMessage):
    result = classifier(message.content)[0]
    return SentimentResponse(sentiment=result["label"], score=result["score"])

