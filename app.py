"""
app.py
------
Flask backend for the Mental Health Support Chatbot (API-powered, free tier).

Uses Groq (https://console.groq.com) for the LLM call, genuinely free with no
credit card required, rate-limited rather than metered. Groq's API is
OpenAI-compatible, so this talks to https://api.groq.com/openai/v1/chat/completions.

ARCHITECTURE - why it's a HYBRID of rules + an LLM, not pure LLM:

    User message
         |
    [1] Clean text
         |
    [2] CRISIS CHECK (deterministic keyword rules) --- high risk? --> safety
         |  no                                                         response
         v                                                             (no API
    [3] Call the LLM with a strict system prompt                        call,
         |  (domain restriction, empathetic reply,                     instant,
         |   JSON output with its own risk read)                       reliable)
         v
    [4] Model flags high risk too? --> safety response (defense in depth)
         |  no
         v
    [5] Model says out_of_scope? --> fixed refusal text (not the model's own wording)
         |  no
         v
    [6] Show the model's reply (+ a professional-help nudge if moderate risk)

Step [2] NEVER depends on the LLM. A keyword match for something like
"I want to kill myself" returns the safety response immediately, without ever
calling the API. This matters because an LLM can occasionally misread intent,
have an API outage, or return something unexpected, and a mental-health bot's
crisis handling should not have any of those as a single point of failure.

Step [4] is a second, independent check: the model itself is asked to flag risk
in its structured output, so if the rules miss a phrasing, the model can still
catch it. Either layer alone is imperfect; together they're much more reliable.

This chatbot gives general emotional support only. It does not diagnose,
treat, or replace a mental-health professional or emergency service.
"""

import json
import os
import re
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent

# Load a local .env file if present (for local development only; on your host
# you set the real environment variables in its dashboard instead).
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

MAX_MESSAGE_LENGTH = 1000
MAX_HISTORY_TURNS = 6  # how many past exchanges to send back to the model for context
DEFAULT_COUNTRY = "IN"

# --- Groq: free, no credit card, OpenAI-compatible chat completions API ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
API_TIMEOUT_SECONDS = 20

app = Flask(__name__)


# ---------------------------------------------------------------------------
# TEXT CLEANING (same idea as the rule-based version: normalize before matching)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# CRISIS / RISK DETECTION (rule-based, runs before the API and cannot be
# skipped or talked around by anything the user types)
# ---------------------------------------------------------------------------
HIGH_RISK_PATTERNS = [
    r"\bkill(ing)? my ?self\b",
    r"\bsuicid(e|al)\b",
    r"\bend(ing)? (my|my own) life\b",
    r"\bend(ing)? it all\b",
    r"\bend(ing)? everything\b",
    r"\bwant to (die|be dead)\b",
    r"\bwish i (was|were) dead\b",
    r"\bshould just die\b",
    r"\bbetter off (dead|without me)\b",
    r"\b(hurt|harm|cut|injure|hurting|harming|cutting) my ?self\b",
    r"\bself ?harm\b",
    r"\bdo not want to (live|be alive|be here|exist|wake up)\b",
    r"\bno reason to (live|go on|keep living|stay alive)\b",
    r"\b(life is|life isnt|life is not) (not )?worth living\b",
    r"\bcannot go on (living|like this|anymore)\b",
    r"\bnever wake up\b",
    r"\bdisappear forever\b",
    r"\bover ?dose\b",
    r"\btake all (my|the) pills\b",
    r"\bi am in (immediate )?danger\b",
    r"\bi (am not|do not feel) safe\b",
    r"\bsomeone is (hurting|abusing|attacking|threatening) me\b",
    r"\b(kill|hurt|harm|stab|shoot) (him|her|them|someone|somebody|everyone)\b",
]

MODERATE_RISK_PATTERNS = [
    r"\bhopeless\b",
    r"\bno point\b",
    r"\bwhat is the point\b",
    r"\bcannot (take|do) (it|this) anymore\b",
    r"\bcannot cope\b",
    r"\bgive up on (everything|life)\b",
    r"\bworthless\b",
    r"\ba burden\b",
    r"\bnobody (would )?(care|cares|miss)\b",
    r"\bfalling apart\b",
    r"\bbreaking down\b",
    r"\bnothing matters\b",
    r"\bhate myself\b",
    r"\bfed up with (life|everything)\b",
    r"\bno way out\b",
    r"\btrapped\b",
    r"\bcannot see a way (out|forward)\b",
]


def detect_risk(cleaned_text: str) -> str:
    """Return 'high', 'moderate' or 'none'. This is the hard safety gate."""
    if any(re.search(p, cleaned_text) for p in HIGH_RISK_PATTERNS):
        return "high"
    if any(re.search(p, cleaned_text) for p in MODERATE_RISK_PATTERNS):
        return "moderate"
    return "none"


CRISIS_RESOURCES = {
    "IN": {"emergency": "112", "lines": ["Tele-MANAS (free mental-health support in India):\n14416 or 1-800-891-4416"]},
    "US": {"emergency": "911", "lines": ["988 Suicide & Crisis Lifeline: call or text 988"]},
    "GB": {"emergency": "999", "lines": ["Samaritans: call 116 123 (free)"]},
    "CA": {"emergency": "911", "lines": ["9-8-8 Suicide Crisis Helpline: call or text 988"]},
    "AU": {"emergency": "000", "lines": ["Lifeline: call 13 11 14"]},
    "OTHER": {"emergency": "your local emergency number", "lines": ["Find a helpline for your country at https://findahelpline.com"]},
}


def build_crisis_response(country: str) -> str:
    resources = CRISIS_RESOURCES.get(country, CRISIS_RESOURCES["OTHER"])
    lines = "\n".join(resources["lines"])
    emergency = resources["emergency"]
    return (
        "I'm really sorry you're going through this, and I'm glad you told me. "
        "You don't have to face this alone.\n\n"
        "Please reach out to someone you trust right now, such as a friend, a family member "
        "or a teacher, and stay with them if you can. If you feel you may hurt yourself or "
        f"someone else, or you're in immediate danger, call {emergency} or go to the nearest "
        "emergency department.\n\n"
        f"You can also talk to a trained person:\n{lines}\n\n"
        "I'm an automated program and can't provide emergency help, but real people are ready to listen."
    )


OUT_OF_SCOPE_RESPONSE = (
    "I'm designed specifically to support conversations about stress, anxiety, sleep, "
    "and emotional well-being. I can't help with unrelated topics."
)

PROFESSIONAL_NUDGE = (
    "\n\nWhat you're describing sounds really heavy. It could help to talk with a counselor, "
    "a doctor or someone you trust. If you ever feel unsafe, please contact your local emergency number."
)

API_KEY_REJECTED_RESPONSE = (
    "Groq rejected the API key. Set a valid GROQ_API_KEY in your local .env file, "
    "then restart the app."
)
MODEL_CONFIGURATION_RESPONSE = (
    "Groq rejected the model or request configuration. Check GROQ_MODEL and the current "
    "Groq API documentation."
)

FALLBACK_RESPONSE = (
    "I'm having trouble responding right now, that's a technical issue on my end, not "
    "something you did. If this keeps happening, it may help to talk to someone you trust "
    "in the meantime. You're welcome to try sending that again in a moment."
)

NOT_CONFIGURED_RESPONSE = (
    "This chatbot isn't fully set up yet, its API key is missing on the server. "
    "If you're the developer, add GROQ_API_KEY to your environment variables "
    "(get a free one at console.groq.com)."
)

RATE_LIMITED_RESPONSE = (
    "I'm getting a lot of requests right now and have hit my free-tier limit for the moment. "
    "Please try again in a little while. In the meantime, if you're in distress, please reach "
    "out to someone you trust."
)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# This is the whole "brain" of the bot's personality and boundaries. Everything
# the Flask code can't guarantee through rules, it asks the model to do, then
# the backend still double-checks the parts that matter for safety (step 4/5
# above) rather than trusting the model's word blindly.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a supportive mental-health companion chatbot embedded in a web app. \
Your ONLY purpose is general emotional support around: stress, anxiety, sleep difficulties, \
sadness or low mood, loneliness, general emotional distress, healthy coping and relaxation \
techniques, and guidance on when to see a counselor or professional.

STRICT RULES:
1. You are NOT a doctor, therapist, or emergency service. NEVER diagnose a condition \
   ("you have depression", "you have an anxiety disorder") and never claim a technique will \
   "cure" anything.
2. Stay strictly inside the domain above. If the user asks about anything else (coding, \
   homework, general knowledge, news, entertainment, math, writing unrelated content, etc.), \
   do not answer it, even partially, even if they also mention a feeling in the same message \
   unless that feeling is the real point of the message.
3. Be empathetic, warm, non-judgmental, and concise: 2 to 5 short sentences or a few short \
   bullet points, not a long essay. Prefer plain, everyday language.
4. Never encourage self-harm, give instructions related to self-harm, or provide content that \
   could be harmful. Never give specific medical dosing or diagnostic advice.
5. You may ask ONE gentle follow-up question at the end of a reply if it fits naturally, but \
   never interrogate the user with multiple questions.
6. If the message indicates the user may be a danger to themselves or others, or in immediate \
   danger, set "risk_level" to "high" in your output (the app will show a dedicated safety \
   message instead of your reply, so your "reply" field can be brief in that case).
7. If the message shows signs of real but non-immediate distress (hopelessness, feeling like a \
   burden, "can't cope", "no point", "trapped", etc.), set "risk_level" to "moderate".
   Otherwise use "none".

OUTPUT FORMAT - respond with ONLY valid JSON, no markdown fences, no text outside the JSON:
{"in_scope": true or false, "risk_level": "none" or "moderate" or "high", "reply": "your reply text here"}

If "in_scope" is false, the app will show its own fixed message instead of your "reply", so for \
out-of-scope messages just set "reply" to a short one-line version of the same idea."""


class RateLimited(Exception):
    """Raised when the free-tier rate limit is hit (HTTP 429)."""


class APIKeyRejected(Exception):
    """Raised when Groq rejects the configured API key."""


class ModelConfigurationError(Exception):
    """Raised when Groq rejects the model or request configuration."""


def call_llm(message: str, history: list) -> dict:
    """Calls the Groq chat completions API and returns in_scope, risk_level, reply.

    Raises on network/API failure (or RateLimited on HTTP 429) so the caller
    can decide on a fallback instead of crashing.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:MAX_MESSAGE_LENGTH]})
    messages.append({"role": "user", "content": message})

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "max_completion_tokens": 400,
            "temperature": 0.6,
            "messages": messages,
        },
        timeout=API_TIMEOUT_SECONDS,
    )
    if response.status_code == 429:
        raise RateLimited()
    if response.status_code in (401, 403):
        raise APIKeyRejected()
    if response.status_code in (400, 404, 422):
        raise ModelConfigurationError()
    response.raise_for_status()
    data = response.json()

    raw = data["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    parsed = json.loads(raw)
    return {
        "in_scope": bool(parsed.get("in_scope", True)),
        "risk_level": parsed.get("risk_level") if parsed.get("risk_level") in ("none", "moderate", "high") else "none",
        "reply": str(parsed.get("reply", "")).strip(),
    }


# ---------------------------------------------------------------------------
# MAIN CHAT LOGIC
# ---------------------------------------------------------------------------
def handle_message(message: str, history: list, country: str = DEFAULT_COUNTRY) -> dict:
    cleaned = clean_text(message)

    # Step 2: crisis rules always run first, and never depend on the API
    if detect_risk(cleaned) == "high":
        return {"intent": "crisis", "risk_level": "high", "response": build_crisis_response(country)}

    if not GROQ_API_KEY:
        return {"intent": "error", "risk_level": "none", "response": NOT_CONFIGURED_RESPONSE}

    # Step 3: ask the model
    try:
        result = call_llm(message, history)
    except RateLimited:
        return {"intent": "error", "risk_level": "none", "response": RATE_LIMITED_RESPONSE}
    except APIKeyRejected:
        return {"intent": "error", "risk_level": "none", "response": API_KEY_REJECTED_RESPONSE}
    except ModelConfigurationError:
        return {"intent": "error", "risk_level": "none", "response": MODEL_CONFIGURATION_RESPONSE}
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError):
        return {"intent": "error", "risk_level": "none", "response": FALLBACK_RESPONSE}

    # Step 4: defense in depth, the model's own risk read can still trigger safety mode
    if result["risk_level"] == "high":
        return {"intent": "crisis", "risk_level": "high", "response": build_crisis_response(country)}

    # Step 5: domain restriction uses OUR fixed wording, not the model's own phrasing
    if not result["in_scope"]:
        return {"intent": "out_of_scope", "risk_level": "none", "response": OUT_OF_SCOPE_RESPONSE}

    # Step 6: show the model's reply, with a professional-help nudge if moderate risk
    reply = result["reply"] or FALLBACK_RESPONSE
    if result["risk_level"] == "moderate":
        reply += PROFESSIONAL_NUDGE

    return {"intent": "support", "risk_level": result["risk_level"], "response": reply}


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please send a non-empty 'message'."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."}), 400

    country = str(data.get("country", DEFAULT_COUNTRY)).upper()
    if country not in CRISIS_RESOURCES:
        country = "OTHER"

    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    history = history[-(MAX_HISTORY_TURNS * 2):]  # user+assistant pairs, so x2

    # Nothing here is written to disk or a database; it only lives for this one request.
    return jsonify(handle_message(message, history, country))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "api_configured": bool(GROQ_API_KEY)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    app.run(host=host, port=port, debug=False)
