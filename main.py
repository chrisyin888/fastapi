import logging

import base64
import html
import json
import os
import smtplib
import ssl
from email.message import EmailMessage

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


class ChatDisplayLog(BaseModel):
    """Frontend-only bot lines (Quick Book, confirmations) — same Sheet shape as /ask."""

    visitor_id: Optional[str] = None
    question: str = ""
    answer: str = ""
    project_type: Optional[str] = None
    city: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


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
CORE STYLE
=========================
- Max 2–4 sentences.
- Sound like a real patio cover sales rep, not a form bot.
- Friendly, practical, confident, and product-focused.
- No first-person words: do NOT use "I", "we", "our", "us".
- Do not sound too pushy.
- Do not overuse words like free quote, free measurement, booking, appointment, consultation, form.
- Focus on selling the product, confirming size, giving a rough total range, and checking installation details.
- Keep replies short and useful.

=========================
PRICE RULES
=========================
- Give a rough total range in CAD only when BOTH product type and clear size are confirmed.
- Add: plus about 5% GST.
- Mention final price depends on site conditions, layout, post locations, wall connection, drainage, and installation details.
- Never mention price per square foot, $/sq ft, $/sf, per sq ft, rate, unit price, or similar wording.
- Never reveal the internal pricing formula.
- If the customer asks how the number was calculated, answer briefly in plain language without showing any per-square-foot rate.
- Do not use placeholders like XXX, TBD, or pending.
- Always calculate a real rounded CAD total range when enough information is provided.
- If the calculated project amount is below CAD $1,200, quote CAD $1,200 + GST as the rough minimum.

=========================
CONTEXT MEMORY
=========================
- Use the full message history.
- Never re-ask something the customer already answered.
- If the customer already chose a product type, treat it as locked in unless they clearly change it.
- If they already chose glass, aluminum, skyline, or sunroom, do not ask which type again.
- If the customer gives a partial answer, fill in what is already known and only ask for the missing part.

=========================
VAGUE NUMBERS
=========================
If the customer sends only a number or vague amount, such as:
"1085", "300", "15", "about 8k", "maybe 200", "appox 1085"

Then:
- Do not restart the conversation.
- Do not ask product type again if it was already chosen.
- Do not give a price until the number meaning is clear.
- Ask whether the number means budget, square footage, width × projection, or something else.
- Keep the clarification short and natural.

Example:
Customer already said glass, then says "approx 1085".
Good reply:
"For the glass patio cover, does 1085 mean your budget, square footage, or the width × projection size? Once the size is clear, a rough total can be estimated."

=========================
CONVERSATION FLOW
=========================

1. Customer says only "patio cover", "interested in patio", or similar:
   - Briefly introduce the three product options:
     Glass Patio Cover — modern look, bright natural light
     Aluminum Patio Cover — durable, practical, strong rain protection
     Skyline Combo Cover — mix of glass and aluminum, balanced light and shade
   - Ask which style they prefer and what approximate size they need.
   - Do not give a price yet.

2. Customer asks about a specific product type:
   - Briefly describe that product with 1–2 benefits.
   - Ask for size: width × projection in feet.
   - Do not give a price yet.

3. Customer gives size but no product type is known:
   - Ask which style they want: glass, aluminum, skyline combo, or sunroom.
   - Do not calculate yet.

4. Customer gives product type and clear size:
   - Calculate the rough total range using the internal pricing model.
   - Reply with rounded total CAD range + about 5% GST.
   - Mention final price depends on actual site and installation details.
   - Ask for city and a few patio photos if they want to continue checking the project.

5. Customer does not know the size:
   - Ask for a rough photo, or approximate width × projection.
   - Do not push a booking form.
   - Do not say "free quote" repeatedly.

6. Customer wants to move forward:
   - Ask for city, approximate size, and a few photos of the patio area.
   - Say photos help confirm material, posts, drainage, wall connection, and installation details.
   - Do not overuse booking language.

=========================
INTERNAL PRICING MODEL
=========================
Use these **internal** multiplier ranges only to compute a **rounded total CAD range** for replies. **Do not** say “$9/sq ft”, “per square foot”, “单价”, or similar in customer-facing text.

**Internal rate range (CAD per sq ft) by product — for calculation only:**
- Aluminum Patio Cover: **8–10**
- Glass Patio Cover: **10–12.5**
- Skyline Combo Cover: **11–14**
- Sunroom: **32–38**

**Calculation (internal):**
- **Sq ft given:** rough CAD total range ≈ sq ft × (correct internal low/high range above).
- **Width × projection given** (assume **feet** if unstated): sq ft ≈ width × projection, then same range formula.
- If only metres are given, convert to feet first (1 m ≈ 3.28 ft) or ask once for units — do not guess silently.
- **Minimum job price:** if the calculated low/high total is below CAD $1,200, quote **CAD $1,200 + GST** as the rough minimum. If only the low end is below CAD $1,200, raise the low end to CAD $1,200 and keep the high end from the formula.
- Round customer-facing totals to clean numbers (nearest CAD $50 or $100 depending on size). Keep it as a simple range, not a detailed breakdown.

**Sanity checks (examples — output style is totals only):**
- Glass 10 × 20 ft = 200 sq ft → roughly **CAD $2,000–$2,500** + GST
- Glass 8 × 10 ft = 80 sq ft → roughly **CAD $1,200** + GST minimum
- Aluminum 300 sq ft → roughly **CAD $2,400–$3,000** + GST
- Skyline Combo 300 sq ft → roughly **CAD $3,300–$4,200** + GST
- Sunroom 300 sq ft → roughly **CAD $9,600–$11,400** + GST

**Customer-facing style:** short, helpful, one rounded **total range** + “plus about 5% GST” + final depends on site/size/layout/install details — **never** lead with or list per-sq-ft rates.

=========================
PRODUCT INFO
=========================
Glass Patio Cover:
- Tempered glass panels
- Great natural light
- Clean modern look
- Weather-resistant

Aluminum Patio Cover:
- Durable aluminum panels
- Low maintenance
- Strong rain and weather protection
- Practical design

Skyline Combo Cover:
- Mix of glass and aluminum panels
- Balanced shade and natural light
- Modern style

Sunroom:
- Fully enclosed space
- Aluminum and glass system
- More usable year-round living space
- Final design depends heavily on site details

=========================
LANGUAGE RULES
=========================
- Match the customer's most recent message.
- If the latest customer message is mainly English, reply in English only.
- If the latest customer message is mainly Chinese, reply in Simplified Chinese only.
- If the customer mixes English and Chinese, choose the dominant language and use only one language.
- Never mix English and Chinese in the same reply, except CAD, GST, numbers, units, and info@loomihomepatios.ca.

=========================
ENGLISH OUTPUT RULES
=========================
- Use only English product names:
  Glass Patio Cover
  Aluminum Patio Cover
  Skyline Combo Cover
  Sunroom
  Patio Cover

- Do not include Chinese characters in English replies.
- Do not say:
  free quote
  free consultation
  free measurement
  book now
  fill out the form
unless the customer directly asks about booking or contact.

Better English sales wording:
- "A few patio photos would help confirm the layout."
- "This size should work well for a glass patio cover."
- "The rough total would be around CAD $X–$Y, plus about 5% GST."
- "Final pricing depends on the actual site, posts, drainage, and installation details."
- "City, size, and a few photos would be enough to check the next step."

=========================
CHINESE OUTPUT RULES
=========================
中文回复必须使用简体中文。
语气要自然、实用、像正常卖露台顶棚的商家，不要像客服机器人。

中文产品名称必须使用：
- Patio cover / patio covers → 露台顶棚
- Glass patio cover → 玻璃顶棚
- Aluminum patio cover → 铝合金顶棚
- Skyline combo cover → 玻璃＋铝合金组合顶棚
- Sunroom → 阳光房

不要使用这些词太多：
- 免费报价
- 免费上门测量
- 免费咨询
- 马上预约
- 填表
- 预约表单

优先使用这些说法：
- 可以先按这个尺寸估一个大概总价
- 发几张现场照片可以看得更准
- 需要确认墙体、排水、柱子位置和安装细节
- 这个尺寸适合做玻璃顶棚 / 铝合金顶棚 / 组合顶棚
- 如果尺寸和现场条件合适，就可以继续确认安装方案

中文报价格式：
- “这个尺寸做玻璃顶棚，大概 CAD $X–$Y，另加约 5% GST。”
- “如果尺寸比较小，最低项目价大概 CAD $1,200，另加约 5% GST。”
- “最终价格还要看现场情况，比如柱子位置、排水、连接方式和安装细节。”
- “可以发一下城市、尺寸和几张现场照片，方便继续确认。”
- 内部用同一套区间公式算总价；用户没主动追问算法时，回复里只用大约 CAD $X–$Y + 另加约 5% GST 的自然说法。不要在回复里写每平方英尺多少钱、$/平方英尺等单价，除非用户明确问“怎么算的”，且即使回答也要简短，避免罗列单价。
- 未确认产品类型和明确面积含义前不要报总价。
- 报价必须是按公式算出的真实数字（CAD）——禁止 XXX、待填 等占位符。
- 如果区间计算结果低于 CAD $1,200，按最低项目价回复大约 CAD $1,200 + 另加约 5% GST；没有低于 CAD $1,200 的项目报价。
- 必须带：约 5% GST、最终以现场勘测与施工条件为准（现场布局、安装细节会影响最终价）。

中文模糊数字规则：
如果客户已经选了产品，比如玻璃顶棚，然后只发「1085」「300」「大概8千」：
- 不要再问选哪种顶棚。
- 问这个数字是预算、面积，还是长×宽/伸出尺寸。
- 不要马上报价。

中文联系方式规则：
如果客户问电话、联系方式、邮箱、微信、客服：
- 给邮箱 info@loomihomepatios.ca
- 可以说也可以发城市、尺寸和现场照片来继续确认。
- 不要编造电话号码、微信号或其他联系方式。

=========================
CONTACT
=========================
Only provided contact email:
info@loomihomepatios.ca

Do not invent phone numbers, WeChat, WhatsApp, or other contact methods.

=========================
FINAL REMINDER
=========================
Every reply should feel like:
- selling patio cover products
- confirming style and size
- giving a clear rough total range when possible
- asking for photos/city/details to continue

Every reply should NOT feel like:
- a free quote ad
- a booking form
- a chatbot collecting leads
- a pushy appointment script
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


@app.post("/log-chat-display")
async def log_chat_display(data: ChatDisplayLog):
    """Log a bot/system line shown in the widget (no OpenAI). Same Google Sheet path as /ask."""
    save_to_google_sheet(
        visitor_id=data.visitor_id,
        question=data.question or "",
        ai_answer=data.answer or "",
        project_type=data.project_type,
        city=data.city,
        email=data.email,
        phone=data.phone,
    )
    return {"ok": True}


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


def _format_smtp_send_error(exc: BaseException) -> str:
    """Short SMTP error string for logs and warnings (no credentials)."""
    return f"{type(exc).__name__}: {exc!s}"[:400]


def _recipient_log_hint(addr: str) -> str:
    """Privacy-safe hint for logs (not full address)."""
    addr = (addr or "").strip()
    if "@" not in addr:
        return "(invalid_or_empty)"
    local, _, domain = addr.partition("@")
    domain = domain[:80]
    if len(local) <= 1:
        return f"***@{domain}"
    return f"{local[0]}…@{domain}"


def _send_html_email_workspace_smtp(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    mail_from: str,
    mail_to: str,
    subject: str,
    html_body: str,
) -> None:
    """Send one HTML email via Google Workspace / Gmail SMTP (587 STARTTLS or 465 SSL)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(
        "This is an HTML message. Please use an email client that supports HTML.",
        subtype="plain",
    )
    msg.add_alternative(html_body, subtype="html")

    rcpt_hint = _recipient_log_hint(mail_to)
    _log.info(
        "lead: workspace_smtp start host=%s port=%s from=%s to=%s",
        host,
        port,
        mail_from,
        rcpt_hint,
    )

    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=45, context=context) as server:
            _log.info("lead: workspace_smtp connected (SSL) host=%s port=%s", host, port)
            _log.info("lead: workspace_smtp login user=%s", username)
            server.login(username, password)
            _log.info("lead: workspace_smtp login_ok")
            server.send_message(msg)
            _log.info("lead: workspace_smtp send_message_ok to=%s", rcpt_hint)
    else:
        with smtplib.SMTP(host, port, timeout=45) as server:
            _log.info("lead: workspace_smtp connected host=%s port=%s", host, port)
            server.ehlo()
            _log.info("lead: workspace_smtp ehlo_ok starttls")
            server.starttls(context=context)
            server.ehlo()
            _log.info("lead: workspace_smtp starttls_ok")
            _log.info("lead: workspace_smtp login user=%s", username)
            server.login(username, password)
            _log.info("lead: workspace_smtp login_ok")
            server.send_message(msg)
            _log.info("lead: workspace_smtp send_message_ok to=%s", rcpt_hint)


def _format_sendgrid_send_error(exc: BaseException) -> str:
    """Human-readable + loggable summary (no secrets)."""
    parts = [type(exc).__name__, str(exc).strip() or "(empty message)"]
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"http_status={status}")
    body = getattr(exc, "body", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8", errors="replace")[:400]
        except Exception:
            body = repr(body)[:200]
    if body:
        parts.append(f"body={body}")
    return " | ".join(parts)


def _lead_sheet_payload(lead: LeadRequest, email_val: str) -> Dict[str, Any]:
    return {
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
    }


def _post_lead_to_google_sheet(lead: LeadRequest, email_val: str) -> bool:
    try:
        r = requests.post(SHEET_WEBHOOK, json=_lead_sheet_payload(lead, email_val), timeout=15)
        if r.status_code >= 400:
            _log.error(
                "lead: Google Sheet webhook returned %s — %s",
                r.status_code,
                (r.text or "")[:500],
            )
            return False
        return True
    except Exception:
        _log.exception("lead: Google Sheet webhook request failed")
        return False


def _lead_summary_for_db(lead: LeadRequest, email_val: str) -> str:
    """Stored in chat_logs.ai_reply so ops can read the lead without opening meta."""
    lines = [
        f"source={lead.source}",
        f"name={lead.name}",
        f"phone={lead.phone}",
        f"email={email_val or '-'}",
        f"city={(lead.city or '').strip() or '-'}",
        f"address={(lead.address or '').strip() or '-'}",
        f"project_type={(lead.project_type or '').strip() or '-'}",
        f"size={(lead.size or '').strip() or '-'}",
        f"preferred_contact={(lead.preferred_contact_time or '').strip() or '-'}",
        f"message={(lead.message or '').strip() or '-'}",
        f"notes={(lead.notes or '').strip() or '-'}",
    ]
    return "\n".join(lines)


@app.post("/lead")
async def create_lead(lead: LeadRequest):
    """
    Accept a lead from the site/chat. Persists first (Sheet + optional Postgres).

    Admin notification: SendGrid (best-effort).
    Customer confirmation: Google Workspace SMTP when WORKSPACE_SMTP_* is set;
    otherwise SendGrid to customer if configured. Email failures never fail the request.
    """
    email_val = (lead.email or "").strip()
    warnings: List[str] = []

    # 1) Persist first — survives SendGrid outages / bad keys
    sheet_ok = _post_lead_to_google_sheet(lead, email_val)
    if not sheet_ok:
        warnings.append("google_sheet_log_failed")

    lead_meta: Dict[str, Any] = {
        "event": "lead",
        "source": lead.source,
        "name": lead.name,
        "phone": lead.phone,
        "email": email_val,
        "city": lead.city,
        "address": lead.address,
        "project_type": lead.project_type,
        "size": lead.size,
        "preferred_contact_time": lead.preferred_contact_time,
        "message": lead.message,
        "notes": lead.notes,
    }
    db_ok = database.save_chat_log(
        user_message="[lead: quick_book]",
        ai_reply=_lead_summary_for_db(lead, email_val),
        visitor_id=email_val or (lead.phone or "").strip() or None,
        source=lead.source,
        project_type=(lead.project_type or "").strip() or None,
        city=(lead.city or "").strip() or None,
        name=(lead.name or "").strip() or None,
        phone=(lead.phone or "").strip() or None,
        email=email_val or None,
        meta=lead_meta,
    )
    if not db_ok:
        warnings.append("database_log_failed")

    persisted = sheet_ok or db_ok

    safe = lambda v: html.escape((v or "").strip())

    api_key = (os.getenv("SENDGRID_API_KEY") or "").strip()
    from_email = (os.getenv("SENDGRID_FROM_EMAIL") or "").strip()
    to_admin = (os.getenv("LEAD_RECEIVER_EMAIL") or "").strip()

    admin_email_sent = False
    customer_code: Optional[int] = None
    customer_email_sent = False
    customer_email_channel: Optional[str] = None

    # Admin notification — Workspace SMTP (best-effort, never fails the request)
    _ws_user_adm = (os.getenv("WORKSPACE_SMTP_USER") or "").strip()
    _ws_pass_adm = "".join((os.getenv("WORKSPACE_SMTP_PASSWORD") or "").split())
    _ws_host_adm = (os.getenv("WORKSPACE_SMTP_HOST") or "smtp.gmail.com").strip()
    _ws_from_adm = (os.getenv("WORKSPACE_SMTP_FROM_EMAIL") or _ws_user_adm).strip()
    try:
        _ws_port_adm = int((os.getenv("WORKSPACE_SMTP_PORT") or "587").strip())
    except ValueError:
        _ws_port_adm = 587

    if not (_ws_user_adm and _ws_pass_adm and to_admin):
        msg = (
            "email_admin_skipped: set WORKSPACE_SMTP_USER, WORKSPACE_SMTP_PASSWORD, "
            "and LEAD_RECEIVER_EMAIL"
        )
        warnings.append(msg)
        _log.warning("lead: %s", msg)
    else:
        admin_subject = f"New Lead - {safe(lead.name)}"
        admin_html = f"""
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
        try:
            _send_html_email_workspace_smtp(
                host=_ws_host_adm,
                port=_ws_port_adm,
                username=_ws_user_adm,
                password=_ws_pass_adm,
                mail_from=_ws_from_adm,
                mail_to=to_admin,
                subject=admin_subject,
                html_body=admin_html,
            )
            admin_email_sent = True
            _log.info(
                "lead: admin_email_sent channel=workspace_smtp host=%s port=%s to=%s",
                _ws_host_adm,
                _ws_port_adm,
                _recipient_log_hint(to_admin),
            )
        except Exception as e:
            err = _format_smtp_send_error(e)
            warnings.append(f"email_admin_failed:workspace_smtp:{err[:280]}")
            _log.warning(
                "lead: admin_email_failed channel=workspace_smtp host=%s port=%s error=%s",
                _ws_host_adm,
                _ws_port_adm,
                err,
                exc_info=True,
            )

    # Customer confirmation (best-effort) — prefer Google Workspace SMTP
    customer_subject = "We received your measurement request"
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

    if email_val:
        ws_user = (os.getenv("WORKSPACE_SMTP_USER") or "").strip()
        # App passwords are often pasted with spaces; strip all whitespace chars.
        ws_pass = "".join((os.getenv("WORKSPACE_SMTP_PASSWORD") or "").split())
        ws_host = (os.getenv("WORKSPACE_SMTP_HOST") or "smtp.gmail.com").strip()
        ws_port_raw = (os.getenv("WORKSPACE_SMTP_PORT") or "587").strip()
        try:
            ws_port = int(ws_port_raw)
        except ValueError:
            ws_port = 587
        ws_from = (os.getenv("WORKSPACE_SMTP_FROM_EMAIL") or ws_user).strip()

        _log.info(
            "lead: customer_email_branch recipient=%s workspace_user_set=%s workspace_pass_set=%s "
            "sendgrid_customer_fallback_available=%s",
            _recipient_log_hint(email_val),
            bool(ws_user),
            bool(ws_pass),
            bool(api_key and from_email),
        )

        if ws_user and ws_pass:
            customer_email_channel = "workspace_smtp"
            try:
                _send_html_email_workspace_smtp(
                    host=ws_host,
                    port=ws_port,
                    username=ws_user,
                    password=ws_pass,
                    mail_from=ws_from,
                    mail_to=email_val,
                    subject=customer_subject,
                    html_body=customer_html,
                )
                customer_email_sent = True
                _log.info(
                    "lead: customer_email_sent channel=workspace_smtp host=%s port=%s from=%s",
                    ws_host,
                    ws_port,
                    ws_from,
                )
            except Exception as e:
                err = _format_smtp_send_error(e)
                warnings.append(f"email_customer_failed:workspace_smtp:{err[:220]}")
                _log.warning(
                    "lead: customer_email_failed channel=workspace_smtp host=%s port=%s error=%s",
                    ws_host,
                    ws_port,
                    err,
                    exc_info=True,
                )
        elif api_key and from_email:
            customer_email_channel = "sendgrid"
            sg2 = SendGridAPIClient(api_key)
            customer_message = Mail(
                from_email=from_email,
                to_emails=email_val,
                subject=customer_subject,
                html_content=customer_html,
            )
            try:
                resp = sg2.send(customer_message)
                customer_code = resp.status_code
                customer_email_sent = True
                _log.info(
                    "lead: customer_email_sent channel=sendgrid status_code=%s",
                    customer_code,
                )
            except Exception as e:
                err = _format_sendgrid_send_error(e)
                warnings.append(f"email_customer_failed:sendgrid:{err[:200]}")
                _log.warning(
                    "lead: customer_email_failed channel=sendgrid error=%s",
                    err,
                    exc_info=True,
                )
        else:
            warnings.append(
                "email_customer_skipped:set WORKSPACE_SMTP_USER and WORKSPACE_SMTP_PASSWORD "
                "(or SendGrid for customer fallback)"
            )
            _log.warning(
                "lead: customer_email_skipped (no WORKSPACE_SMTP_* and no SendGrid from key for customer)"
            )

    else:
        _log.info(
            "lead: customer_email_branch skipped — no recipient after trim "
            "(JSON field `email` missing, null, or blank; customer auto-reply not attempted)"
        )

    if not persisted:
        # Nothing stored — surface a clear failure for the frontend
        raise HTTPException(
            status_code=503,
            detail={
                "error": "lead_not_persisted",
                "message": "Lead could not be saved to Google Sheet or database; team was not notified via storage.",
                "warnings": warnings,
            },
        )

    # Keep status "success" for 200 when persisted so older clients stay compatible.
    _log.info(
        "lead: response_summary admin_email_sent=%s customer_email_sent=%s "
        "customer_email_channel=%s warning_count=%s sheet_logged=%s database_logged=%s",
        admin_email_sent,
        customer_email_sent,
        customer_email_channel,
        len(warnings),
        sheet_ok,
        db_ok,
    )

    return {
        "status": "success",
        "sheet_logged": sheet_ok,
        "database_logged": db_ok,
        "admin_email_sent": admin_email_sent,
        "admin_email_channel": "workspace_smtp",
        "customer_email_status": customer_code,
        "customer_email_sent": customer_email_sent,
        "customer_email_channel": customer_email_channel,
        "warnings": warnings,
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
