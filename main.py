from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI

app = FastAPI()

# 允许跨域（测试阶段全部开放）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 数据模型
class Question(BaseModel):
    question: str

# 测试接口
@app.get("/")
def root():
    return {"status": "AskPatio AI running"}

# AI问答接口
@app.post("/ask")
async def ask_ai(data: Question):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a professional patio cover and sunroom estimator in Vancouver, Canada.

You help homeowners understand pricing, materials, permits, and design options.

Guidelines:
- Give realistic Vancouver pricing ranges
- Be clear and concise
- Use simple homeowner-friendly language
- If size is given, estimate cost range
"""
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
