from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "askpatio-ai running"}

@app.post("/ask")
def ask_ai(data: Question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional patio cover and sunroom estimator in Vancouver. Provide clear pricing explanations in English."
            },
            {
                "role": "user",
                "content": data.question
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }
