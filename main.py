from fastapi import FastAPI, Request
from openai import OpenAI
import os

app = FastAPI()

# 从环境变量读取 OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
async def root():
    return {"status": "askpatio-ai running"}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")

    # 这里是你的核心 Prompt（以后就在这里改）
system_prompt = """
You are a Vancouver-based patio cover and sunroom expert.

You understand:
- Material costs
- Labor pricing
- Installation timelines
- Permit requirements in British Columbia
- Structural considerations for rain and snow loads

Rules:
- Always respond in professional, clear English.
- Never reply in Chinese.
- If the customer does not provide dimensions, ask for width, projection, and height.
- You may provide realistic price ranges but never guarantee a final quote.
- Be confident and knowledgeable.
- Guide the customer toward scheduling a site measurement when appropriate.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )

    answer = response.choices[0].message.content
    return {"answer": answer}
