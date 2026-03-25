from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
import html
import json
import os
import re
import requests
from datetime import datetime, timezone
from typing import List, Optional, Union

from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Attachment, Disposition, FileContent, FileName, FileType, Mail

app = FastAPI()

# CORS: allow all origins (open for testing)
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
# Data models
# =========================

class ChatMessage(BaseModel):
    role: str
    content: str

class Question(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None
    project_type: Optional[str] = None
    city: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


def _chat_log_file_path() -> str:
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chat_history.jsonl")
    return os.getenv("CHAT_LOG_FILE", default)


def _log_chat_turn(
    question: str,
    answer: str,
    project_type: Optional[str],
    city: Optional[str],
    email: Optional[str],
    phone: Optional[str],
) -> None:
    """Append one structured line to JSONL and post one row to Google Sheet webhook."""
    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "timestamp": ts,
        "user_message": question,
        "ai_reply": answer or "",
        "project_type": (project_type or "").strip(),
        "city": (city or "").strip(),
        "email": (email or "").strip(),
        "phone": (phone or "").strip(),
    }

    try:
        path = _chat_log_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # Single webhook payload: structured fields + legacy-friendly message/role
    try:
        summary = f"User: {question}\nAI: {answer or ''}"
        requests.post(
            SHEET_WEBHOOK,
            json={
                "event": "chat_turn",
                "timestamp": ts,
                "user_message": question,
                "ai_reply": answer or "",
                "project_type": entry["project_type"],
                "city": entry["city"],
                "email": entry["email"],
                "phone": entry["phone"],
                "role": "chat_turn",
                "visitor_id": "chat",
                "message": summary,
            },
            timeout=15,
        )
    except Exception:
        pass


# =========================
# Health / root
# =========================

@app.get("/")
def root():
    return {"status": "AskPatio AI running"}


# =========================
# AI chat endpoint
# =========================

SYSTEM_PROMPT = """
You are a friendly patio cover and sunroom sales assistant for LoomiHome Patios in Greater Vancouver.

=========================
STRICT RULES
=========================
- Max 2–4 sentences
- Friendly and conversational, not pushy — sound like a helpful sales rep, not a form bot
- No first-person words (NO: "I", "we")
- NEVER reveal per-sqft pricing (e.g. "$12/sqft", "$15 per square foot")
- NEVER give a price unless BOTH the product type AND clear size meaning are confirmed (see vague-number rules below)
- Do not **guess** a product type when the customer has never chosen one — ask which type they want. Once they **clearly state** a type anywhere in the thread, treat it as **locked in** unless they switch or ask to compare.
- REMEMBER everything the customer already told you — **never re-ask** the same question or ignore prior answers

=========================
CONTEXT MEMORY — NO ROBOTIC REPEATS (CRITICAL)
=========================
- Use the **full message history**. The latest user line must be read **together** with what they already said.
- If they already chose a product (e.g. "glass", "aluminum", "skyline", "sunroom", or the Chinese equivalents), **do not** ask "which type of cover" again or re-list the three patio options unless they clearly want to change product.
- If they already gave an answer to your last question, **do not** repeat that question — either move forward or ask a **new** clarification only.
- **Bad (forbidden):** User says "glass", then "approx 1085" → assistant asks which cover type. **Good:** acknowledge glass and ask what "1085" refers to (budget vs sq ft vs dimensions).

=========================
VAGUE OR STANDALONE NUMBERS — CLARIFY, DON’T RESET
=========================
When the user sends mainly a number or vague quantity (e.g. "1085", "300", "15", "about 8k", "maybe 200", typos like "appox 1085"), and especially when **type is already known**:

- **Do not** restart an earlier step or repeat a question they already answered.
- **Do not** output a full dollar quote until **what the number means** is clear.
- Reply with a **short, natural** clarification: acknowledge the product they’re on, then ask whether the number is **rough budget (CAD)**, **square footage**, **width × projection / depth** (feet or metres), or something else — offer a compact A/B style question in one or two sentences.
- After they clarify, continue the normal flow (confirm size → then quote when both type and size are clear).

Chinese (when replying in 中文): same logic — e.g. 已选玻璃顶棚后用户说「大概1085」，用一两句自然追问：是指**预算**、**平方英尺面积**，还是**长×宽/伸出尺寸**？**不要**再问「您要哪种顶棚」.

=========================
VAGUE SIZES, BUDGETS, INCOMPLETE / PARTIAL REPLIES
=========================
- **Only one dimension** or fuzzy size ("about 12 ft", "medium", "pretty big"): ask **once** for the missing part (e.g. projection) or units — **without** re-asking product type if already set.
- **Vague budget** before size is known: can acknowledge and ask for approximate **footprint** (sq ft or width × depth) so a ballpark can make sense later — do not reset the thread.
- **Partial reply** that only answers half of what was asked: fill in from context; ask only for what’s still missing, conversationally.

=========================
CONVERSATION FLOW
=========================
1. If customer says "patio cover" or "interested in patio" without specifying a type:
   → Introduce ALL three patio cover options briefly:
     • Glass Patio Cover — great natural light, modern look
     • Aluminum Patio Cover — durable, low maintenance, weather protection
     • Skyline Combo Cover — premium mix of glass + aluminum panels, balanced light and shade
   → Ask which type interests them
   → Do NOT give pricing yet

2. If customer asks about a SPECIFIC product type (glass, aluminum, skyline, sunroom):
   → Briefly introduce that product with 1–2 key benefits
   → Ask: "What size are you looking at? (width × projection in feet)"
   → Do NOT give pricing yet

3. If customer provides dimensions or sqft BUT type is NOT yet stated **anywhere** in the thread:
   → Ask which type they want (glass, aluminum, skyline combo, or sunroom)
   → Do NOT calculate a price until type is confirmed

3b. If type **is** already confirmed but the user sends an **ambiguous number or vague size** (see sections above):
   → Clarify meaning first — **do not** ask for type again and **do not** give a full quote until size/budget meaning is clear

4. If customer provides **clear** dimensions or sqft AND type is already confirmed in the conversation:
   → Calculate total estimated price using internal rates
   → Give the total only (NEVER per sqft)
   → Mention plus 5% GST, final price confirmed after site visit
   → Ask if they'd like to book a free on-site measurement

5. If customer says they don't know the size:
   → "No worries — would you like to book a free on-site measurement? Our team can come take exact measurements and give you a final quote."

6. If customer confirms they want to book:
   → "Perfect — you can use the booking form to submit your details and upload photos of your space."

=========================
PRICING (internal only — NEVER share per-sqft rates)
=========================
- Aluminum: $12–15/sqft
- Glass: about $15/sqft total
- Skyline Combo: about $14/sqft
- Sunroom: about $38/sqft
- Small jobs: $1,500–$2,500

=========================
PRODUCT INFO
=========================
- Glass Patio Cover: tempered glass panels, great natural light, clean modern look, weather-resistant
- Aluminum Patio Cover: durable V-panel aluminum, low maintenance, strong rain/weather protection, practical design
- Skyline Combo Cover: premium mix of glass + aluminum V-panels, balanced light and shade, modern style
- Sunroom: fully enclosed, thermal-break aluminum + glass, year-round comfortable space, adds usable square footage

=========================
LANGUAGE — MATCH THE CUSTOMER
=========================
- Decide reply language from the customer's **most recent message** (the one you are answering now).
- If that message is **primarily English**, the **entire** reply must be **English only** — follow all English rules above (including no "I" / "we").
- If that message is **primarily Chinese** (Simplified or Traditional), the **entire** reply must be **Simplified Chinese only** — use the Chinese section below.
- If the user mixes both scripts heavily in one message, use whichever language dominates that message; still output **one language only** for the whole reply.

=========================
MONOLINGUAL OUTPUT — NO MIXING (CRITICAL)
=========================
- **Never** put Chinese characters or Chinese product names in an **English** reply. Forbidden in English replies: 玻璃顶棚, 铝合金顶棚, 玻璃＋铝合金组合顶棚, 露台顶棚, 阳光房, or any other Chinese wording. In English, use only: Glass Patio Cover, Aluminum Patio Cover, Skyline Combo Cover, Sunroom, patio cover (and normal English sentences).
- **Never** put English product marketing names in a **Chinese** reply (e.g. do not say "Glass Patio Cover" or "Skyline Combo" in English inside Chinese text). In Chinese, use only the approved Chinese terms below. **Allowed exceptions in Chinese replies:** the email address info@loomihomepatios.ca (ASCII), the abbreviations **CAD** and **GST**, and numbers/units.
- Do not alternate languages within one reply. One script, one voice.

=========================
CHINESE (简体中文) — TONE & VOCABULARY
=========================
Sound practical, conversational, and sales-friendly — never stiff machine-translation style.

Required product terms (use these; NEVER use wrong literal terms such as 阳伞罩 for patio cover):
- Patio cover / patio covers → 露台顶棚
- Glass patio cover → 玻璃顶棚
- Aluminum patio cover → 铝合金顶棚
- Skyline combo cover → 玻璃＋铝合金组合顶棚
- Sunroom → 阳光房
- Free on-site measurement → 免费上门测量
- Booking form on the page → 页面上的预约表单

When introducing the three patio options in Chinese, name them as: 玻璃顶棚、铝合金顶棚、玻璃＋铝合金组合顶棚.

Chinese — vague numbers / 模糊数字: If the user already picked a product (e.g. 玻璃顶棚) then sends only a number like「1085」「大概8千」, **不要**再问选哪种产品；用自然口语追问数字是指预算、面积（平方英尺）还是长宽尺寸。

Chinese pricing (same internal math as English):
- NEVER reveal per-sqft rates in any language.
- NEVER give a 总价 until both product type and dimensions are confirmed in the thread.
- When giving a total, always state a real calculated number in CAD — NEVER use placeholders like XXX、待填、或类似占位符.
- Mention 另加约 5% GST（消费税） and that 实地测量后最终报价以现场为准 when appropriate.

Contact in Chinese:
- If they ask for 电话、联系方式、怎么联系、邮箱、微信、客服: answer helpfully — give 邮箱 info@loomihomepatios.ca and mention 也可通过页面上的预约表单留言，或预约免费上门测量；不要回避或生硬推脱。
- Do not invent phone numbers or messaging apps not provided here.

Keep replies short (about 2–4 sentences worth in the chosen language), warm, and natural.
"""


def _monolingual_turn_reminder(user_text: str) -> Optional[str]:
    """Nudge the model right before the latest user turn to cut EN/Chinese mixing."""
    if not user_text or not user_text.strip():
        return None
    t = user_text.strip()
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    letters = sum(1 for c in t if c.isalpha() and ord(c) < 128)

    if cjk >= 2 and cjk > letters:
        return (
            "Reminder for this reply: write the entire answer in Simplified Chinese only. "
            "Do not use English product names or English sentences."
        )
    if letters >= 6 and letters > cjk * 2:
        return (
            "Reminder for this reply: write the entire answer in English only. "
            "Do not use any Chinese characters."
        )
    if cjk >= 2 and letters >= 6:
        return (
            "Reminder for this reply: the user mixed scripts — pick one language for the "
            "whole answer (the one they mainly used) with zero mixing."
        )
    return None


@app.post("/ask")
async def ask_ai(data: Question):

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if data.history:
        for msg in data.history:
            role = msg.role if msg.role in ("user", "assistant") else "user"
            messages.append({"role": role, "content": msg.content})

    turn_reminder = _monolingual_turn_reminder(data.question)
    if turn_reminder:
        messages.append({"role": "system", "content": turn_reminder})

    messages.append({"role": "user", "content": data.question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content

    _log_chat_turn(
        question=data.question,
        answer=answer or "",
        project_type=data.project_type,
        city=data.city,
        email=data.email,
        phone=data.phone,
    )

    return {
        "answer": answer
    }


# =========================
# /lead — JSON lead from chat mini form
# =========================

class LeadRequest(BaseModel):
    source: str = "website_chat"
    name: str
    phone: str
    email: Optional[str] = ""
    city: str = ""
    address: str = ""
    project_type: str = ""
    size: str = ""
    preferred_contact_time: str = ""
    message: str = ""
    notes: str = ""


@app.post("/lead")
async def create_lead(lead: LeadRequest):
    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))

    safe = lambda v: html.escape((v or "").strip())

    # 1) Admin notification email
    subject = f"New Lead - {safe(lead.name)}"
    html_content = f"""
    <h2>New Customer Lead</h2>
    <p><b>Source:</b> {safe(lead.source)}</p>
    <p><b>Name:</b> {safe(lead.name)}</p>
    <p><b>Phone:</b> {safe(lead.phone)}</p>
    <p><b>Email:</b> {safe(lead.email) or 'Not provided'}</p>
    <p><b>City:</b> {safe(lead.city)}</p>
    <p><b>Address:</b> {safe(lead.address) or 'Not provided'}</p>
    <p><b>Project Type:</b> {safe(lead.project_type) or 'Not specified'}</p>
    <p><b>Size:</b> {safe(lead.size) or 'Not provided'}</p>
    <p><b>Preferred Contact:</b> {safe(lead.preferred_contact_time) or 'Any time'}</p>
    <p><b>Message:</b> {safe(lead.message) or 'No message'}</p>
    <p><b>Notes:</b> {safe(lead.notes) or ''}</p>
    """

    admin_message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=os.getenv("LEAD_RECEIVER_EMAIL"),
        subject=subject,
        html_content=html_content,
    )

    try:
        admin_response = sg.send(admin_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send admin email: {str(e)}")

    # 2) Customer confirmation email (only if email provided)
    customer_code = None
    email_val = (lead.email or "").strip()
    if email_val:
        customer_html = f"""
        <h2>Thank you, {safe(lead.name)}!</h2>
        <p>We've received your request for a free on-site measurement.</p>
        <p>Project type: <b>{safe(lead.project_type) or 'To be discussed'}</b></p>
        <p>City: <b>{safe(lead.city)}</b></p>
        <p>Our team will contact you shortly to arrange the appointment.</p>
        <p>Final pricing will be confirmed after the site visit.</p>
        <br>
        <p>Thank you,</p>
        <p>LoomiHome Patios Team</p>
        """
        customer_message = Mail(
            from_email=os.getenv("SENDGRID_FROM_EMAIL"),
            to_emails=email_val,
            subject="We received your measurement request",
            html_content=customer_html,
        )
        try:
            resp = sg.send(customer_message)
            customer_code = resp.status_code
        except Exception:
            pass

    # 3) Log to Google Sheet
    try:
        requests.post(SHEET_WEBHOOK, json={
            "event": "lead",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": lead.source,
            "name": lead.name,
            "phone": lead.phone,
            "email": email_val,
            "city": lead.city,
            "address": lead.address,
            "project_type": lead.project_type,
            "size": lead.size,
            "message": lead.message,
            "notes": lead.notes,
            "visitor_id": email_val or lead.phone,
            "role": "lead",
        }, timeout=15)
    except Exception:
        pass

    return {
        "status": "success",
        "admin_code": admin_response.status_code,
        "customer_code": customer_code,
    }


# =========================
# Lead email (appointment form)
# =========================

MAX_APPOINTMENT_PHOTOS = 8
MAX_PHOTO_BYTES = 8 * 1024 * 1024  # 8 MB each (SendGrid total payload limits apply)


def _safe_attachment_filename(original: Optional[str], index: int, mime: str) -> str:
    base = (original or "").strip()
    base = os.path.basename(base).replace("\\", "").replace("/", "")
    if not base or len(base) > 120 or not re.match(r"^[\w.\- ()\[\]]+$", base):
        ext = ".jpg"
        if "png" in mime:
            ext = ".png"
        elif "gif" in mime:
            ext = ".gif"
        elif "webp" in mime:
            ext = ".webp"
        elif "heic" in mime:
            ext = ".heic"
        base = f"photo_{index + 1}{ext}"
    return base


@app.post("/send-email")
async def send_email(
    source: str = Form(...),
    name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    city: str = Form(...),
    project_type: str = Form(...),
    size: str = Form(""),
    message: str = Form(""),
    photos: Union[UploadFile, List[UploadFile], None] = File(None),
):
    """
    Appointment booking: multipart form (text fields + optional image files).
    Photos are attached to the admin/lead notification email only.
    """
    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))

    if photos is None:
        photo_list: List[UploadFile] = []
    elif isinstance(photos, list):
        photo_list = photos
    else:
        photo_list = [photos]
    if len(photo_list) > MAX_APPOINTMENT_PHOTOS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many photos (max {MAX_APPOINTMENT_PHOTOS}).",
        )

    attachment_count = 0
    admin_attachments: List[Attachment] = []

    for idx, upload in enumerate(photo_list):
        raw = await upload.read()
        if not raw:
            continue
        if len(raw) > MAX_PHOTO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Photo too large (max {MAX_PHOTO_BYTES // (1024 * 1024)} MB each).",
            )
        mime = (upload.content_type or "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Only image uploads are allowed.",
            )
        fname = _safe_attachment_filename(upload.filename, idx, mime)
        encoded = base64.b64encode(raw).decode()
        att = Attachment(
            file_content=FileContent(encoded),
            file_name=FileName(fname),
            file_type=FileType(mime),
            disposition=Disposition("attachment"),
        )
        admin_attachments.append(att)
        attachment_count += 1

    # 1) Admin / lead notification email
    subject = f"New Lead - {name}"

    html_content = f"""
    <h2>New Customer Lead</h2>
    <p><b>Source:</b> {html.escape(source)}</p>
    <p><b>Name:</b> {html.escape(name)}</p>
    <p><b>Phone:</b> {html.escape(phone)}</p>
    <p><b>Email:</b> {html.escape(email)}</p>
    <p><b>City:</b> {html.escape(city)}</p>
    <p><b>Project Type:</b> {html.escape(project_type)}</p>
    <p><b>Size:</b> {html.escape(size) if size else 'Not provided'}</p>
    <p><b>Message:</b> {html.escape(message) if message else 'No message'}</p>
    <p><b>Photos attached:</b> {attachment_count}</p>
    """

    admin_message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=os.getenv("LEAD_RECEIVER_EMAIL"),
        subject=subject,
        html_content=html_content,
    )
    for att in admin_attachments:
        admin_message.add_attachment(att)

    admin_response = sg.send(admin_message)

    # 2) Customer confirmation email (no attachments — size & privacy)
    customer_subject = "We received your appointment request"

    customer_html = f"""
    <h2>Thank you, {html.escape(name)}!</h2>
    <p>We’ve received your request for a free on-site measurement.</p>
    <p>Project type: <b>{html.escape(project_type)}</b></p>
    <p>City: <b>{html.escape(city)}</b></p>
    <p>Size: <b>{html.escape(size) if size else 'Not provided'}</b></p>
    <p>Our team will contact you shortly to arrange the appointment.</p>
    <p>Final pricing will be confirmed after the site visit.</p>
    <br>
    <p>Thank you,</p>
    <p>AskPatio AI Team</p>
    """

    customer_message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=email,
        subject=customer_subject,
        html_content=customer_html,
    )

    customer_response = sg.send(customer_message)

    # 3) Log lead to Google Sheet
    try:
        requests.post(SHEET_WEBHOOK, json={
            "event": "lead",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "name": name,
            "phone": phone,
            "email": email,
            "city": city,
            "project_type": project_type,
            "size": size,
            "message": message,
            "visitor_id": email or phone,
            "role": "lead",
        }, timeout=15)
    except Exception:
        pass

    return {
        "status": "success",
        "admin_code": admin_response.status_code,
        "customer_code": customer_response.status_code,
        "photos_attached": attachment_count,
    }
