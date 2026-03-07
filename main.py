from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

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

# Google Sheet Webhook
SHEET_WEBHOOK = "https://script.google.com/macros/s/AKfycbwBw3iypXhsPWmgGMa2wwilgRDCYqJA3m5nq7RgbruW9s8ms6D6ZoL7R_isOKHUCrTH/exec"


# =========================
# 数据模型
# =========================

class Question(BaseModel):
    question: str


class EmailRequest(BaseModel):
    source: str
    name: str
    phone: str
    email: str
    city: str
    project_type: str
    size: str
    message: str


# =========================
# 测试接口
# =========================

@app.get("/")
def root():
    return {"status": "AskPatio AI running"}


# =========================
# AI问答接口
# =========================

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

    answer = response.choices[0].message.content

    # 写入 Google Sheet（记录聊天）
    try:
        requests.post(SHEET_WEBHOOK, json={
            "visitor_id": "chat_user",
            "role": "chat",
            "message": data.question
        })

        requests.post(SHEET_WEBHOOK, json={
            "visitor_id": "ai",
            "role": "assistant",
            "message": answer
        })
    except:
        pass

    return {
        "answer": answer
    }


# =========================
# 发送客户线索邮件
# =========================

@app.post("/send-email")
def send_email(data: EmailRequest):

    subject = f"New Lead - {data.name}"

    html_content = f"""
    <h2>New Customer Lead</h2>
    <p><b>Source:</b> {data.source}</p>
    <p><b>Name:</b> {data.name}</p>
    <p><b>Phone:</b> {data.phone}</p>
    <p><b>Email:</b> {data.email}</p>
    <p><b>City:</b> {data.city}</p>
    <p><b>Project Type:</b> {data.project_type}</p>
    <p><b>Size:</b> {data.size}</p>
    <p><b>Message:</b> {data.message}</p>
    """

    message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=os.getenv("LEAD_RECEIVER_EMAIL"),
        subject=subject,
        html_content=html_content,
    )

    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
    response = sg.send(message)

    # 写入 Google Sheet（记录客户）
    try:
        requests.post(SHEET_WEBHOOK, json={
            "visitor_id": data.email,
            "role": "lead",
            "message": data.message
        })
    except:
        pass

    return {
        "status": "success",
        "code": response.status_code
    }
