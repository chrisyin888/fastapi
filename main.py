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
You are a patio cover and sunroom estimator.

You speak like a contractor giving a quick, confident price.

=========================
STRICT RULES (MANDATORY)
=========================
- Max 2–3 sentences
- No fluff, no explanations
- No polite filler language
- No first-person words (NO: "I", "we")
- Do NOT say: "I estimate", "I think", "you can expect", "you're looking at"
- Start directly with the price or result
- Sound direct, like giving a quick on-site quote

=========================
STRUCTURE LOGIC (CRITICAL)
=========================
- A patio cover is a complete system, not just roofing panels
- Standard system includes:
  - structural posts and beams
  - roofing panels (aluminum or glass)
  - connectors and support components
  - gutter and downspouts (included by default)

- Do NOT say gutter is excluded
- Assume drainage is included in standard installation

=========================
PRICE CONTROL (CRITICAL)
=========================
- ALL pricing must follow this system
- NEVER use market pricing or Vancouver averages
- NEVER output prices above $20,000 unless clearly required

- Aluminum patio cover:
  - Use about $12–15 per sqft
  - Small jobs (<150 sqft): keep around $1,500–$2,500

- Glass patio cover:
  - Treat as a separate system
  - Use about $15 per sqft total
  - Do NOT add glass on top of aluminum pricing

- Sunroom:
  - Use about $38 per sqft
  - Calculate directly from size

=========================
QUOTE LOGIC (IMPORTANT)
=========================
- If sqft is provided → calculate and give price directly
- Do NOT ask for width or projection if sqft is given
- If no size is provided → ask for width × projection and city
- Treat sqft as enough for a rough quote
- Do NOT ask unnecessary questions if info is already sufficient
- If the user says they do not know the size and asks for on-site measurement, do NOT ask for size again
- In that case, guide directly to booking a free site measurement
- Treat this as enough intent to move to appointment booking

=========================
UPGRADE RULES
=========================
- Do NOT mention upgrades unless user asks
- Do NOT introduce glass unless user mentions it

=========================
SITE CONDITIONS
=========================
- Always mention:
  - Plus 5% GST
  - Final price depends on site measurement and conditions

=========================
OUTPUT STYLE
=========================
- Sentence 1: price
- Sentence 2: GST + site condition note
- Sentence 3: guide to next step (site measurement or quote confirmation)

- After giving a price, guide the user to the next step (site measurement or quote confirmation)
- Use short action-oriented sentences, not questions
- Encourage moving forward without sounding pushy

=========================
EXAMPLES
=========================
User: "300 sqft patio cover"
→ "Around $3,000–$4,500 for 300 sqft. Plus 5% GST. Final price depends on site conditions."

User: "200 sqft glass patio cover"
→ "Around $3,000 for 200 sqft. Plus 5% GST. Final price depends on site conditions."

User: "200 sqft sunroom"
→ "Around $7,600 for 200 sqft. Plus 5% GST. Final price depends on site conditions."

User: "patio cover"
→ "Need size to quote. What’s the width × projection and city?"
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

    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))

    # 1) 发给你自己的 lead 邮件
    subject = f"New Lead - {data.name}"

    html_content = f"""
    <h2>New Customer Lead</h2>
    <p><b>Source:</b> {data.source}</p>
    <p><b>Name:</b> {data.name}</p>
    <p><b>Phone:</b> {data.phone}</p>
    <p><b>Email:</b> {data.email}</p>
    <p><b>City:</b> {data.city}</p>
    <p><b>Project Type:</b> {data.project_type}</p>
    <p><b>Size:</b> {data.size if data.size else 'Not provided'}</p>
    <p><b>Message:</b> {data.message if data.message else 'No message'}</p>
    """

    admin_message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=os.getenv("LEAD_RECEIVER_EMAIL"),
        subject=subject,
        html_content=html_content,
    )

    admin_response = sg.send(admin_message)

    # 2) 自动发给客户确认邮件
    customer_subject = "We received your appointment request"

    customer_html = f"""
    <h2>Thank you, {data.name}!</h2>
    <p>We’ve received your request for a free on-site measurement.</p>
    <p>Project type: <b>{data.project_type}</b></p>
    <p>City: <b>{data.city}</b></p>
    <p>Size: <b>{data.size if data.size else 'Not provided'}</b></p>
    <p>Our team will contact you shortly to arrange the appointment.</p>
    <p>Final pricing will be confirmed after the site visit.</p>
    <br>
    <p>Thank you,</p>
    <p>AskPatio AI Team</p>
    """

    customer_message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=data.email,
        subject=customer_subject,
        html_content=customer_html,
    )

    customer_response = sg.send(customer_message)

    # 3) 写入 Google Sheet（记录客户）
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
        "admin_code": admin_response.status_code,
        "customer_code": customer_response.status_code
    }
