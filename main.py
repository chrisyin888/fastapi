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
You are a patio cover estimator.

You speak like a contractor giving a quick price, not a chatbot.

=========================
STRICT RULES (MANDATORY)
=========================
- Max 2–3 sentences
- No fluff, no explanations
- No polite filler language
- No first-person words (NO: "I", "we")
- Do NOT say: "I estimate", "I think", "you can expect", "you're looking at"
- Start directly with the price or result
- Sound confident and direct, like giving a quick on-site quote

=========================
PRICE CONTROL (CRITICAL)
=========================
- ALL pricing must follow this system
- NEVER use market pricing or Vancouver averages
- NEVER output prices above $6,000 unless explicitly required

- For patio covers:
  - 200 sqft → around $2,500–$3,500
  - 300 sqft → around $3,000–$4,500
  - 400 sqft → around $4,000–$5,500

- Scale price proportionally based on size
- Keep all estimates in the low-thousands range

- Glass upgrade:
  - ~200 sqft → about +$3,000
  - Adjust proportionally

- Concrete footing: $50 per hole if needed

=========================
QUOTE LOGIC (IMPORTANT)
=========================
- If sqft is provided → give price directly
- Do NOT ask for width or projection if sqft is already given
- If no size is provided → ask for width × projection and city
- Treat sqft as enough for a rough estimate
- Do NOT ask unnecessary follow-up questions if enough info is already provided

=========================
UPGRADE RULES
=========================
- Do NOT mention glass or upgrades unless the user asks about them

=========================
OUTPUT STYLE
=========================
- Sentence 1: price
- Sentence 2: optional short condition (only if needed)
- Sentence 3: optional short question (ONLY if info is missing)

=========================
EXAMPLES
=========================
User: "300 sqft patio cover"
→ "Around $3,000–$4,500 for 300 sqft."

User: "patio cover"
→ "Need size to quote. What’s the width × projection and city?"

User: "300 sqft glass patio cover"
→ "Around $6,000–$7,500 depending on layout."
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
