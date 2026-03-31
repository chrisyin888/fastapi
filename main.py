import logging

import base64
import html
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import requests
from datetime import datetime, timezone
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Union

import database
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Attachment, Disposition, FileContent, FileName, FileType, Mail

app = FastAPI()

_log = logging.getLogger(__name__)


@app.on_event("startup")
def startup_init_db() -> None:
    _log.info("=== FastAPI startup: AskPatio / LoomiHome backend ===")
    _log.info("app: startup — calling database.init_db()")
    database.init_db()
    _log.info("app: startup — database.init_db() finished (see [database] logs for table creation)")


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
SHEET_WEBHOOK = "https://script.google.com/macros/s/AKfycbzsMbb0V3Hmw00Ds7Kt2e5VWLvscpNI4XZJSyOlxqZHxHA8rgcuK2ttlnsEQ5wIyELhuQ/exec"


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
    visitor_id: Optional[str] = None
    source: Optional[str] = None
    name: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def _chat_log_file_path() -> str:
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.jsonl")
    return os.getenv("CHAT_LOG_FILE", default)


def _log_chat_turn(
    question: str,
    answer: str,
    project_type: Optional[str],
    city: Optional[str],
    email: Optional[str],
    phone: Optional[str],
    visitor_id: Optional[str] = None,
) -> None:
    """Append one JSON line per chat turn to chat_history.jsonl."""
    ts = datetime.now(timezone.utc).isoformat()
    vid = (visitor_id or "").strip()
    entry = {
        "timestamp": ts,
        "question": question,
        "answer": answer or "",
        "visitor_id": vid,
        "project_type": (project_type or "").strip(),
        "city": (city or "").strip(),
        "email": (email or "").strip(),
        "phone": (phone or "").strip(),
    }

    try:
        path = _chat_log_file_path()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def save_to_airtable(
    *,
    question: str,
    ai_reply: str,
    visitor_id: Optional[str] = None,
    city: Optional[str] = None,
    project_type: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """POST one row to Airtable table 'Chat Logs'. Fails soft: logs only, never raises."""
    api_key = (os.getenv("AIRTABLE_API_KEY") or "").strip()
    base_id = (os.getenv("AIRTABLE_BASE_ID") or "").strip()
    if not api_key or not base_id:
        return

    page_val = ""
    if meta and isinstance(meta, dict):
        p = meta.get("page")
        if p is not None:
            page_val = str(p).strip()
        else:
            page_val = str(meta.get("page_path") or "").strip()

    ts = datetime.now(timezone.utc).isoformat()
    fields = {
        "Timestamp": ts,
        "Visitor ID": (visitor_id or "").strip(),
        "User Message": question or "",
        "AI Reply": ai_reply or "",
        "Page": page_val,
        "City": (city or "").strip(),
        "Project Type": (project_type or "").strip(),
    }

    table_path = quote("Chat Logs", safe="")
    url = f"https://api.airtable.com/v0/{base_id}/{table_path}/records"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"records": [{"fields": fields}]}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        _log.info("app: Airtable Chat Logs row created (status=%s)", resp.status_code)
    except Exception:
        _log.exception("app: Airtable save_to_airtable failed")


def save_to_google_sheet(
    *,
    visitor_id: Optional[str],
    question: str,
    ai_answer: Optional[str],
    project_type: Optional[str],
    city: Optional[str],
    email: Optional[str],
    phone: Optional[str],
) -> None:
    """POST chat turn to Google Apps Script webhook. Fails soft; never raises."""
    webhook = (os.getenv("SHEET_WEBHOOK") or "").strip()
    if not webhook:
        _log.info("app: SHEET_WEBHOOK unset; skipping Google Sheet logging")
        return
    payload = {
        "visitor_id": visitor_id or "",
        "question": question or "",
        "answer": ai_answer or "",
        "project_type": project_type or "",
        "city": city or "",
        "email": email or "",
        "phone": phone or "",
    }
    try:
        _log.info("app: Google Sheet logging started")
        resp = requests.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
        _log.info("app: Google Sheet row created")
    except Exception:
        _log.exception("app: Google Sheet logging failed")


# =========================
# Health / root
# =========================


def _public_route_list() -> List[Dict[str, str]]:
    """What is actually registered on this process (use to verify Render deployed this file)."""
    out: List[Dict[str, str]] = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", None)
        if not path or path.startswith("/openapi") or path.startswith("/docs"):
            continue
        for m in sorted(methods):
            if m in ("HEAD", "OPTIONS"):
                continue
            out.append({"method": m, "path": path})
    out.sort(key=lambda x: (x["path"], x["method"]))
    return out


@app.get("/")
def root():
    """If `routes` does not include POST /ask, this Render service is NOT running this repo/file."""
    return {
        "service_name": "askpatio-ai-fastapi",
        "backend_build": "askpatio_api_v3_route_introspect",
        "entry_file": "main.py",
        "entrypoint": "uvicorn main:app",
        "expected_repo": "github.com/chrisyin888/fastapi",
        "chat_endpoint": {"method": "POST", "path": "/ask"},
        "routes": _public_route_list(),
    }


@app.get("/ask")
def ask_get_hint():
    """So GET /ask is not a silent 404 — chat must use POST with JSON body."""
    return {
        "detail": "Chat uses POST /ask with Content-Type: application/json",
        "method": "POST",
        "path": "/ask",
        "example": {"question": "What sizes do you offer?", "visitor_id": "optional"},
    }


@app.get("/db-test")
def db_test():
    """Verify DATABASE_URL, connectivity, and whether public.chat_logs exists."""
    return database.db_health_check()


@app.post("/debug-insert-chat")
def debug_insert_chat():
    """
    TEMPORARY — insert one test row into chat_logs for debugging.
    Remove from production after Postgres logging is verified.
    """
    return database.debug_insert_test_row()


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
- For customer replies: give a **rough total in CAD** (from the internal model below) plus **+ 5% GST** and a short site-visit disclaimer — keep it simple.
- **Never** mention **price per square foot**, **$/sq ft**, **$/sf**, **per sq ft**, or similar rate wording in replies. Give **only** a rounded **total in CAD** (and GST). If the customer **explicitly** asks how the number was calculated, answer briefly in plain language **without** quoting dollar-per-unit rates.
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
   → **Use only that product’s rate** (below). If they gave **width × projection** (feet unless they say metres), **square footage = width × projection**; if they gave **sq ft** directly, use that number.
   → Compute rough material total in CAD using the internal model below; state it naturally, e.g. “around CAD $X, plus about 5% GST” — **never** show $/sq ft or a rate breakdown unless they explicitly ask how it was figured, and even then avoid quoting per-unit prices.
   → Always add: **+ ~5% GST** and that the **final price depends on site conditions, layout, and install details**, confirmed after a site visit.
   → Ask if they'd like to book a free on-site measurement

5. If customer says they don't know the size:
   → Offer a free on-site measurement in one short sentence; mention the **Quick Book** mini-form in the chat as the fastest way to get on the calendar — not a long pitch.

6. If customer confirms they want to book, schedule, or requests an appointment:
   → **Primary path:** one short line pointing to the **in-chat Quick Book** form (fastest). Example tone: "Perfect — use the quick form in the chat to get booked for a free on-site measurement."
   → Only mention the **full booking section on the page** if they need to upload many photos or prefer the long form — do **not** push the page form as the default or repeat long instructions.
   → Chinese: same idea — 优先引导使用**聊天窗口里的快速预约表单**；完整页面表单仅在有大量照片等需要时简要提及。

=========================
PRICING MODEL (internal math only — never disclose $/sq ft to customers)
=========================
Use these **internal** multipliers only to compute a **rounded total CAD** for replies. **Do not** say “$9/sq ft”, “per square foot”, “单价”, or similar in customer-facing text.

**Internal rate (CAD per sq ft) by product — for calculation only:**
- Aluminum Patio Cover: **9**
- Glass Patio Cover: **13**
- Skyline Combo Cover: **12.5**
- Sunroom: **35**

**Calculation (internal):**
- **Sq ft given:** rough CAD total ≈ sq ft × (correct internal rate above).
- **Width × projection given** (assume **feet** if unstated): sq ft ≈ width × projection, then same formula.
- If only metres are given, convert to feet first (1 m ≈ 3.28 ft) or ask once for units — do not guess silently.

**Sanity checks (examples for 300 sq ft — output style is totals only):**
- Aluminum → about **CAD $2,700** + GST
- Glass → about **CAD $3,900** + GST
- Skyline Combo → about **CAD $3,750** + GST
- Sunroom → about **CAD $10,500** + GST

**Customer-facing style:** short, helpful, one rounded **total** + “plus about 5% GST” + final depends on site — **never** lead with or list per-sq-ft rates.

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

Chinese pricing (与英文同一套数字与公式):
- 内部用同一套公式算总价；用户没主动追问算法时，回复里只用 **大约 CAD $X + 另加约 5% GST** 的自然说法。**不要**在回复里写 **每平方英尺多少钱**、**$/平方英尺** 等单价，除非用户明确问“怎么算的”，且即使回答也要简短，避免罗列单价。
- 未确认产品类型和明确面积含义前不要报总价。
- 报价必须是按公式算出的真实数字（CAD）——禁止 XXX、待填 等占位符。
- 必须带：**约 5% GST**、**最终以现场勘测与施工条件为准**（现场布局、安装细节会影响最终价）。

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
        visitor_id=data.visitor_id,
    )

    saved = database.save_chat_log(
        user_message=data.question,
        ai_reply=answer or "",
        visitor_id=data.visitor_id,
        source=data.source,
        project_type=data.project_type,
        city=data.city,
        name=data.name,
        phone=data.phone,
        email=data.email,
        meta=data.meta,
    )
    if saved:
        _log.info("app: /ask — chat row saved to PostgreSQL (save_chat_log ok)")
    else:
        _log.error(
            "app: /ask — PostgreSQL save failed (JSONL may still have written); check [database] logs"
        )

    save_to_airtable(
        question=data.question,
        ai_reply=answer or "",
        visitor_id=data.visitor_id,
        city=data.city,
        project_type=data.project_type,
        meta=data.meta,
    )

    save_to_google_sheet(
        visitor_id=data.visitor_id,
        question=data.question,
        ai_answer=answer,
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
