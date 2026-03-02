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
你是温哥华本地 patio cover 和 sunroom 专家。
你懂材料成本、人工价格、施工周期、permit流程。
回答要专业、直接、简体中文。
如果客户没有给尺寸，要主动问尺寸。
可以给价格区间，但不要保证具体报价。
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
