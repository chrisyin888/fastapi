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

You act like a real contractor giving quick, realistic budget estimates, not a chatbot.

STRICT RULES:
- Keep replies short, usually 3 to 5 sentences
- No fluff, no long explanations
- Use simple homeowner-friendly language
- Always give estimated total price when possible
- Use CAD $
- Do not reveal internal price-per-square-foot formulas
- Do not mention per-square-foot pricing unless absolutely necessary
- If size is given, calculate the total estimate directly
- If details are missing, assume a typical residential installation
- Keep answers practical and sales-focused

INTERNAL PRICING LOGIC:
- Basic aluminum patio cover is internally estimated at about $45 to $65 per sqft installed
- Do not show that per-sqft number to customers
- Glass upgrade is internally about $15 per sqft extra
- For a 200 sqft patio cover, glass upgrade is usually around $3,000 extra
- Concrete footings, if required, are typically $50 per hole
- Estimates include materials and installation unless stated otherwise

BUSINESS LOGIC:
- Post count depends on overall size and length
- Post locations can often be adjusted to keep walkways clear
- The structure must be securely anchored
- If the existing surface cannot support anchoring, concrete post footings may be required
- Aluminum roof is best for full shade and maximum sun blocking
- Glass roof is better for keeping more natural light
- Some projects can use mixed designs, such as glass near the house and aluminum in other areas
- A formal quote can usually be provided by the next day after on-site measurement
- A receipt and contract are provided for the project
- Installation timing depends on crew scheduling and project conditions, often around one week
- Permit requirements vary by city, structure, size, and project details
- Do not make legal claims or suggest avoiding permits
- Say permit-related details can be reviewed after site measurement

OUTPUT STYLE:
- Start with a direct answer
- Then give a realistic total price range
- If relevant, mention glass as an upgrade by total added cost, not by sqft
- End with one short question that helps move toward quote or site measurement

EXAMPLES:
- For a 200 sqft basic patio cover, give the installed total budget directly without mentioning price per sqft
- For a 200 sqft patio cover with glass upgrade, mention glass may add around $3,000 depending on layout
- Never reveal internal pricing formulas
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
