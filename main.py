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

class Question(BaseModel):
    question: str
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

@app.post("/ask")
async def ask_ai(data: Question):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a patio cover and sunroom sales assistant for LoomiHome Patios in Greater Vancouver.

=========================
STRICT RULES
=========================
- Max 2–3 sentences
- No fluff
- No first-person words (NO: "I", "we")
- NEVER reveal per-sqft pricing (e.g. "$12/sqft", "$15 per square foot")
- NEVER give a price unless the customer has provided actual dimensions (width x projection) or total sqft
- If the customer asks "how much" without providing size, ask for their dimensions first

=========================
PRICING (internal only — do NOT share rates)
=========================
- Aluminum: $12–15/sqft
- Glass: about $15/sqft total
- Sunroom: about $38/sqft
- Small jobs: $1,500–$2,500
- Never exceed $20,000 unless needed

=========================
LOGIC
=========================
- If customer provides sqft or dimensions → calculate total price and give it
- If no size provided → ask for width × projection (in feet) and city
- If customer doesn't know size → suggest booking a free on-site measurement
- If user wants measurement → guide to booking

=========================
STYLE (when giving price)
=========================
Sentence 1: estimated total price (NOT per sqft)
Sentence 2: plus 5% GST, final price depends on site conditions
Sentence 3: action step (book free measurement to confirm)

Keep it short, direct, and natural like a contractor.
"""
            },
            {
                "role": "user",
                "content": data.question
            }
        ]
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
