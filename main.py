# 393_hf_distilbert_pydantic_swag/main.py
# https://chatgpt.com/c/683892ad-c92c-800c-b771-ed6ddec01670

# GPT5 #############################################

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

# GPT6 #############################################

# Add to imports at the top
from typing import Optional

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define input/output models for summarization
class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 130
    min_length: Optional[int] = 30
    do_sample: bool = False

class SummaryResponse(BaseModel):
    summary: str

# Add new route
@app.post("/chat/summarize", response_model=SummaryResponse, tags=["Chatbot"])
def summarize_text(request: SummarizationRequest):
    result = summarizer(
        request.text,
        max_length=request.max_length,
        min_length=request.min_length,
        do_sample=request.do_sample
    )
    return SummaryResponse(summary=result[0]["summary_text"])


# GPT30 #############################################


# Load T5 translation model (no sentencepiece needed)
translator = pipeline("translation_en_to_de", model="t5-small")

# Pydantic models
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

# # FastAPI app
# app = FastAPI(
#     title="Translation API",
#     description="Translate English to German using t5-small",
#     version="1.0"
# )

@app.post("/chat/translate", response_model=TranslationResponse, tags=["Chatbot"])
def translate_text(request: TranslationRequest):
    result = translator(request.text)
    return TranslationResponse(translated_text=result[0]["translation_text"])
