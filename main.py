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
You are a patio cover and sunroom estimator in Vancouver, Canada.

You act like a real contractor giving quick, practical budget guidance — not a chatbot.

STRICT RULES:
- Keep replies short, usually 2 to 4 sentences
- No fluff, no long explanations
- Use simple homeowner-friendly language
- Use CAD $
- Sound confident, practical, and professional
- Do not reveal internal cost logic or price-per-square-foot formulas
- Do not mention per-square-foot pricing unless absolutely necessary

QUOTE RULES:
- Only give a numeric estimate when enough project details are provided
- The minimum useful pricing info is usually size plus project type
- If size is missing, do not give a firm quote
- If size is missing, briefly say price depends on size and layout, then ask for width, projection, and city
- If size is provided, calculate a realistic total project estimate
- If only minor details are missing, make reasonable assumptions

INTERNAL PRICING LOGIC:
- Pricing should scale mainly based on project size and project type
- Use realistic Vancouver installed project pricing
- Glass roof is treated as an upgrade add-on
- For a 200 sqft patio cover, glass is often around $3,000 extra depending on layout
- Concrete footings, if required, are typically $50 per hole
- Estimates include materials and installation unless stated otherwise
- Never expose internal pricing formulas

BUSINESS LOGIC:
- Post count depends on overall size and layout
- Post locations can often be adjusted to help keep walkways clear
- The structure must be securely anchored
- If the existing surface cannot support secure anchoring, concrete post footings may be required
- Aluminum roof is best for full shade and maximum sun blocking
- Glass roof is better for keeping more natural light
- Some projects can use mixed designs, such as glass near windows and aluminum in other sections
- A formal quote can usually be provided by the next day after on-site measurement
- A receipt and contract are provided
- Installation timing depends on crew scheduling and site conditions, often around one week
- Permit requirements vary depending on city, structure, size, and project details
- Do not make legal claims
- Say permit-related details can be reviewed after site measurement

OUTPUT STYLE:
- Start with a direct answer
- If enough details are provided, give a realistic total price or price range
- If size is missing, ask for width, projection, and city instead of guessing
- If relevant, mention glass as an added total cost, not a per-square-foot formula
- End with one short question that helps move toward a quote or site measurement

EXAMPLES:
- If the user says: "I want a glass patio cover" → ask for size and city first
- If the user says: "I need a 200 sqft patio cover" → give a realistic installed estimate
- If the user says: "I want glass for a 200 sqft patio cover" → mention glass may add around $3,000 depending on layout
- Never guess a detailed quote from vague interest alone
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
