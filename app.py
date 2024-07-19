from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

class SummarizationRequest(BaseModel):
    text: str

app = FastAPI()

# Load the BART model and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.post("/summarize")
def summarize(request: SummarizationRequest):
    summary = summarizer(request.text, max_length=150, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}

@app.get("/")
def read_root():
    return {"message": "Welcome to the BART Summarization API"}
