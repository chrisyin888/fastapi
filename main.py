from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from typing import Optional
from datetime import datetime

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
SHEET_WEBHOOK = os.getenv(
    "SHEET_WEBHOOK",
    "https://script.google.com/macros/s/AKfycbwBw3iypXhsPWmgGMa2wwilgRDCYqJA3m5nq7RgbruW9s8ms6D6ZoL7R_isOKHUCrTH/exec"
)

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")
LEAD_RECEIVER_EMAIL = os.getenv("LEAD_RECEIVER_EMAIL")


# =========================
# 数据模型
# =========================

class Question(BaseModel):
    question: str


class LeadRequest(BaseModel):
    source: str = "website_chat"
    name: str
    phone: str
    email: Optional[str] = ""
    city: Optional[str] = ""
    address: Optional[str] = ""
    project_type: Optional[str] = ""
    size: Optional[str] = ""
    preferred_contact_time: Optional[str] = ""
    message: Optional[str] = ""
    notes: Optional[str] = ""


# 兼容你旧前端 /send-email 的字段
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
# 工具函数
# =========================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %I:%M %p")


def safe_post_to_sheet(payload: dict):
    if not SHEET_WEBHOOK:
        return {"success": False, "error": "Missing SHEET_WEBHOOK"}

    try:
        r = requests.post(SHEET_WEBHOOK, json=payload, timeout=15)
        return {
            "success": r.ok,
            "status_code": r.status_code,
            "text": r.text[:300]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_admin_lead_email(payload: dict):
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL or not LEAD_RECEIVER_EMAIL:
        return {"success": False, "error": "Missing SendGrid env vars"}

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)

        subject = f"New Lead - {payload.get('name', 'Unknown')} - {payload.get('city', '')}"

        html_content = f"""
        <h2>New Customer Lead</h2>
        <p><b>Submitted At:</b> {payload.get('submitted_at', '')}</p>
        <p><b>Source:</b> {payload.get('source', '')}</p>
        <p><b>Name:</b> {payload.get('name', '')}</p>
        <p><b>Phone:</b> {payload.get('phone', '')}</p>
        <p><b>Email:</b> {payload.get('email', '') or 'Not provided'}</p>
        <p><b>City:</b> {payload.get('city', '') or 'Not provided'}</p>
        <p><b>Address:</b> {payload.get('address', '') or 'Not provided'}</p>
        <p><b>Project Type:</b> {payload.get('project_type', '') or 'Not provided'}</p>
        <p><b>Size:</b> {payload.get('size', '') or 'Not provided'}</p>
        <p><b>Preferred Contact Time:</b> {payload.get('preferred_contact_time', '') or 'Not provided'}</p>
        <p><b>Message:</b> {payload.get('message', '') or 'No message'}</p>
        <p><b>Notes:</b> {payload.get('notes', '') or 'No notes'}</p>
        """

        admin_message = Mail(
            from_email=SENDGRID_FROM_EMAIL,
            to_emails=LEAD_RECEIVER_EMAIL,
            subject=subject,
            html_content=html_content,
        )

        admin_response = sg.send(admin_message)

        return {
            "success": True,
            "status_code": admin_response.status_code
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_customer_confirmation_email(payload: dict):
    email = (payload.get("email") or "").strip()

    # 客户没留邮箱就跳过，不报错
    if not email:
        return {"success": True, "skipped": True, "reason": "No customer email"}

    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        return {"success": False, "error": "Missing SendGrid env vars"}

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)

        customer_subject = "We received your appointment request"

        customer_html = f"""
        <h2>Thank you, {payload.get('name', '')}!</h2>
        <p>Your request for a free on-site measurement has been received.</p>
        <p><b>Project type:</b> {payload.get('project_type', '') or 'Patio Cover / Sunroom'}</p>
        <p><b>City:</b> {payload.get('city', '') or 'Not provided'}</p>
        <p><b>Address:</b> {payload.get('address', '') or 'Not provided'}</p>
        <p><b>Size:</b> {payload.get('size', '') or 'Not provided'}</p>
        <p>Someone will contact you shortly to arrange the appointment.</p>
        <p>Final pricing will be confirmed after the site visit.</p>
        <br>
        <p>Thank you,</p>
        <p>AskPatio AI Team</p>
        """

        customer_message = Mail(
            from_email=SENDGRID_FROM_EMAIL,
            to_emails=email,
            subject=customer_subject,
            html_content=customer_html,
        )

        customer_response = sg.send(customer_message)

        return {
            "success": True,
            "status_code": customer_response.status_code
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def normalize_lead_payload(data):
    return {
        "submitted_at": now_str(),
        "source": (getattr(data, "source", "") or "website_chat").strip(),
        "name": (getattr(data, "name", "") or "").strip(),
        "phone": (getattr(data, "phone", "") or "").strip(),
        "email": (getattr(data, "email", "") or "").strip(),
        "city": (getattr(data, "city", "") or "").strip(),
        "address": (getattr(data, "address", "") or "").strip(),
        "project_type": (getattr(data, "project_type", "") or "").strip(),
        "size": (getattr(data, "size", "") or "").strip(),
        "preferred_contact_time": (getattr(data, "preferred_contact_time", "") or "").strip(),
        "message": (getattr(data, "message", "") or "").strip(),
        "notes": (getattr(data, "notes", "") or "").strip(),
    }


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
You are a patio cover and sunroom estimator AND conversion-focused sales expert for Greater Vancouver.

Your goal is to move the customer toward the next action:
1. get city + size for a rough quote
2. or book a free on-site measurement
3. or leave contact information for follow-up

=========================
CORE STYLE
=========================
- Sound like an experienced contractor / estimator
- Natural, confident, direct
- Helpful, slightly persuasive
- No fluff
- Keep replies short: usually 2–4 sentences
- Always move the conversation forward

=========================
CONVERSION PRIORITY
=========================
Always guide the user to one of these next steps:
- provide city + approximate size
- book a free on-site measurement
- leave contact info for follow-up

Do not let the conversation stall.
If the user cannot provide dimensions, move directly toward measurement booking.
If the user shows clear interest, encourage the next action naturally.

=========================
WHEN TO QUOTE
=========================
Only give pricing when there is clear quote intent.

Quote intent means at least one of these is true:
1. sqft is provided
2. dimensions are provided
3. the user explicitly asks for price / quote / estimate

If pricing is required:
- Start directly with the price or result
- Include:
  - Plus 5% GST
  - Final price depends on site conditions
- End with a clear next step:
  - confirm rough quote
  - book a free on-site measurement
  - or leave contact details

=========================
PRODUCT INTEREST LOGIC
=========================
If the user only mentions a product type or shows interest without size or price intent, do NOT give pricing yet.

Examples:
- "sunroom"
- "glass patio cover"
- "aluminum patio cover"
- "skyline combo"
- "I'm interested in patio covers"
- "tell me about sunrooms"

In this case:
1. Briefly introduce the product
2. Mention 1–3 strong selling points
3. Do NOT include pricing or $/sqft yet
4. Then invite the user to provide city + approximate size for a rough quote
5. If appropriate, mention free on-site measurement as the easy next step

=========================
NO SIZE / NO IDEA LOGIC
=========================
If the user says they do not know the size, are not sure, cannot measure, or have no idea:
- STOP asking for dimensions
- Immediately offer a free on-site measurement
- Guide toward booking
- If they seem interested but hesitant, encourage them by saying exact pricing can be confirmed after the visit

Treat phrases like these as NO SIZE CASE:
- "don't know size"
- "not sure"
- "no idea"
- "can't measure"
- "don't know the dimensions"
- "need someone to measure"

In NO SIZE CASE:
- Do not ask for width / projection again
- Do not repeat the same question
- Move directly toward measurement booking

=========================
PRICING SYSTEM
=========================
Use only this rough pricing structure:

- Aluminum patio cover:
  - about $12–15 per sqft
  - small jobs under 150 sqft: around $1,500–$2,500

- Glass patio cover:
  - treat as a separate system
  - about $15 per sqft total
  - do NOT stack glass on top of aluminum pricing

- Sunroom:
  - about $38 per sqft

- Never give pricing above $20,000 unless clearly required by size

=========================
PRODUCT KNOWLEDGE
=========================
Aluminum Patio Covers:
- durable
- low maintenance
- cost-effective
- practical for rain and sun protection
- available in different styles and finishes

Glass Patio Covers:
- premium option
- clean, modern look
- strong weather protection
- clear and tinted options available
- tempered glass is durable, safe, and low maintenance
- clear glass generally blocks around 60–70% UV
- tinted glass can block around 90–95% UV and reduces glare / heat

Skyline Combo Covers:
- strong structure
- modern open-look design
- good for homeowners wanting both protection and a cleaner architectural look

Sunrooms:
- brighter, more enclosed space
- usable through more seasons
- adds comfort, weather protection, and a more finished living space feel

=========================
CONTACT / BOOKING GUIDANCE
=========================
When the conversation is moving well, naturally guide the user toward:
- booking a free on-site measurement
- or leaving their phone / email for follow-up

Do this especially when:
- they want exact pricing
- they do not know the size
- they ask detailed project questions
- they show clear buying intent

Do not sound pushy.
Sound practical and action-oriented.

=========================
RESPONSE STRUCTURE
=========================
Case 1: Product interest only
- Short introduction
- Key benefits
- Invite city + approximate size
- Optionally mention free on-site measurement

Case 2: Quote with enough information
- Sentence 1: rough price
- Sentence 2: GST + site condition note
- Sentence 3: next action (confirm quote, book visit, or leave contact)

Case 3: User does not know size
- Offer free on-site measurement immediately
- Say exact pricing can be confirmed after the visit
- Ask if they want to book

Case 4: Strong intent / ready customer
- Move naturally toward booking or contact collection

=========================
AGREEMENT / BOOKING TRIGGER
=========================
If the user shows agreement, readiness, or positive intent (examples: "yes", "yeah", "sure", "ok", "sounds good", "let's do it", "book it"):

- DO NOT ask for size, dimensions, or repeat earlier questions
- DO NOT go backwards in the conversation

Immediately switch to booking / lead capture mode.

Response should:
1. Acknowledge positively
2. Move directly to collecting contact details
3. Be simple and action-focused

Collect:
- Name
- Phone number
- Address (or area)

Keep it short and natural.

Example response:
"Perfect 👍 Let’s get your free measurement booked.

Please share:
• Name
• Phone number
• Address

Scheduling can be arranged and exact pricing will be confirmed after the visit."

=========================
EXAMPLES
=========================
User: "sunroom"
Assistant:
Sunrooms create a brighter, more enclosed space that can be enjoyed through more seasons of the year. A great option for extra comfort, weather protection, and a more finished extension of the home. Share your city and approximate size for a rough quote, or book a free on-site measurement.

User: "glass patio cover"
Assistant:
Glass patio covers are a premium option with a clean, modern look and strong weather protection. Clear and tinted tempered glass options are available, and they are durable, safe, and low maintenance. Share your city and approximate size for a rough quote, or book a free on-site measurement.

User: "500 sqft aluminum patio cover richmond"
Assistant:
Around $6,000–$7,500 for a 500 sqft aluminum patio cover. Plus 5% GST. Final price depends on site conditions. Book a free on-site measurement to confirm the exact quote.

User: "I don't know the size"
Assistant:
No problem. Free on-site measurement is available, and exact pricing can be confirmed after the visit. Would you like to book a quick measurement appointment?

User: "Richmond but no idea on size"
Assistant:
No problem. Free on-site measurement is available in Richmond, and exact pricing can be confirmed after the visit. Would you like to book a quick measurement appointment?

User: "I want an exact quote"
Assistant:
Exact pricing can be confirmed with a free on-site measurement. Leave your city and preferred contact details, or book a measurement appointment to move forward.
"""
            },
            {
                "role": "user",
                "content": data.question
            }
        ],
        temperature=0.5
    )

    answer = response.choices[0].message.content

    # 写入 Google Sheet（记录聊天）
    try:
        safe_post_to_sheet({
            "type": "chat",
            "visitor_id": "chat_user",
            "role": "chat",
            "message": data.question,
            "submitted_at": now_str()
        })

        safe_post_to_sheet({
            "type": "chat",
            "visitor_id": "ai",
            "role": "assistant",
            "message": answer,
            "submitted_at": now_str()
        })
    except Exception:
        pass

    return {
        "answer": answer
    }


# =========================
# 统一 lead 接口
# 聊天窗口 和 booking form 都可以调用这个
# =========================

@app.post("/lead")
def submit_lead(data: LeadRequest):
    payload = normalize_lead_payload(data)

    if not payload["name"] or not payload["phone"]:
        return {
            "success": False,
            "message": "Name and phone are required."
        }

    # 1) 写入 Google Sheet
    sheet_result = safe_post_to_sheet({
        "type": "lead",
        "submitted_at": payload["submitted_at"],
        "source": payload["source"],
        "name": payload["name"],
        "phone": payload["phone"],
        "email": payload["email"],
        "city": payload["city"],
        "address": payload["address"],
        "project_type": payload["project_type"],
        "size": payload["size"],
        "preferred_contact_time": payload["preferred_contact_time"],
        "message": payload["message"],
        "notes": payload["notes"]
    })

    # 2) 发给你自己的提醒邮件
    admin_email_result = send_admin_lead_email(payload)

    # 3) 如果客户留了邮箱，自动发确认邮件
    customer_email_result = send_customer_confirmation_email(payload)

    return {
        "success": True,
        "message": "Lead received successfully.",
        "sheet_result": sheet_result,
        "admin_email_result": admin_email_result,
        "customer_email_result": customer_email_result
    }


# =========================
# 兼容旧接口 /send-email
# 你的旧前端不改也能继续用
# =========================

@app.post("/send-email")
def send_email(data: EmailRequest):
    payload = {
        "submitted_at": now_str(),
        "source": (data.source or "").strip(),
        "name": (data.name or "").strip(),
        "phone": (data.phone or "").strip(),
        "email": (data.email or "").strip(),
        "city": (data.city or "").strip(),
        "address": "",
        "project_type": (data.project_type or "").strip(),
        "size": (data.size or "").strip(),
        "preferred_contact_time": "",
        "message": (data.message or "").strip(),
        "notes": ""
    }

    if not payload["name"] or not payload["phone"]:
        return {
            "success": False,
            "message": "Name and phone are required."
        }

    sheet_result = safe_post_to_sheet({
        "type": "lead",
        "submitted_at": payload["submitted_at"],
        "source": payload["source"],
        "name": payload["name"],
        "phone": payload["phone"],
        "email": payload["email"],
        "city": payload["city"],
        "address": payload["address"],
        "project_type": payload["project_type"],
        "size": payload["size"],
        "preferred_contact_time": payload["preferred_contact_time"],
        "message": payload["message"],
        "notes": payload["notes"]
    })

    admin_email_result = send_admin_lead_email(payload)
    customer_email_result = send_customer_confirmation_email(payload)

    return {
        "success": True,
        "sheet_result": sheet_result,
        "admin_email_result": admin_email_result,
        "customer_email_result": customer_email_result
    }
