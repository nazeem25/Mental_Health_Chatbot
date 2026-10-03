# Mental Health Support Chatbot (API-powered, free tier)

A Flask web chatbot that gives **general emotional support** for stress, anxiety, sleep
difficulties, sadness, loneliness and everyday emotional distress — powered by a real LLM
through **Groq's free API** (no credit card required), instead of a fixed set of canned
replies.

> **Important:** this chatbot is not a doctor, therapist or emergency service. It never
> diagnoses anything and is not a substitute for professional mental-health care. If you are
> in immediate danger, contact your local emergency number.

## Why Groq

Anthropic, OpenAI and most frontier-model providers don't offer a reliable free API tier, you
need a paid account to call Claude or GPT programmatically. **Groq** (console.groq.com) does:
it's genuinely free, requires no credit card to start, and is rate-limited rather than billed
per token. It's an open-source model host (Llama, in this app's default config), not the same
company as Google Gemini, don't mix them up when searching.

If you'd rather use Google's **Gemini** free tier instead (also no-card, via
aistudio.google.com), the architecture below still applies, you'd just swap the `call_llm`
function in `app.py` to call Gemini's API format instead of Groq's.

## How this differs from a canned-reply chatbot

Earlier versions of this project used a TF-IDF + Logistic Regression classifier trained on a
few hundred example sentences to pick a reply from a fixed dictionary. This version calls a
real LLM, so replies are generated fresh for whatever you actually type.

**Crisis detection is still 100% rule-based and runs *before* any API call.** This is
deliberate: a free-tier API can hit rate limits, time out, or have an outage, and a
mental-health bot's safety response should never depend on that. The flow is:

```
message -> clean text -> RULE-BASED crisis check (no API call, instant)
                             |
                        not high risk
                             |
                             v
                   call Groq's API (domain restriction + empathetic reply,
                   returns structured JSON including its OWN risk read)
                             |
              model also flags high risk? -> same safety response (defense in depth)
              model says off-topic? -> our fixed refusal text, not the model's wording
                             |
                             v
                      show the model's reply
```

So the app never trusts the model alone for the two things that matter most for safety
(crisis detection and staying on-topic), it uses the model for the open-ended part (writing
an empathetic, relevant response) and keeps deterministic rules in charge of the guardrails.

## Get a free API key (no credit card)

1. Go to <https://console.groq.com>.
2. Sign in with Google or GitHub, no card needed.
3. Open the **API Keys** tab, click **Create**, and copy the key (starts with `gsk_`).
4. That's it, you're on the free developer tier immediately.

Free-tier limits are generous for a student project (roughly 30 requests/minute, 1,000/day
at the time of writing, shared across models), but can change, check the console if you hit
a limit. The app already handles a rate-limit response gracefully (see
`RATE_LIMITED_RESPONSE` in `app.py`) instead of crashing.

## Run it locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env from the example only if you don't already have one.
cp .env.example .env              # Windows: copy .env.example .env
# Edit .env and set GROQ_API_KEY to your new key.
python app.py
```

Open **http://127.0.0.1:5000**. Without a key set, the app still runs and the crisis layer
still works, but normal messages will show a "not fully set up yet" message instead of a
generated reply, this is intentional so the app never silently fails.

Keep `.env` private and never commit it. If an API key is exposed, revoke it in the Groq
console and replace it with a newly generated key.

## Deploying (Render)

1. Push this project to a GitHub repo.
2. On [render.com](https://render.com), **New > Web Service > Build and deploy from a Git
   repository**, pick this repo.
3. Build Command: `pip install -r requirements.txt`
   Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Before deploying, add an environment variable: `GROQ_API_KEY` = your real key. Never commit
   it to GitHub, this is the only safe place to put it.
5. Select the Free instance type and deploy. Open the generated `*.onrender.com` URL once live.

### Deploying (Railway)

Same idea: **New Project > Deploy from GitHub repo**, Railway auto-detects the `Procfile`,
then add `GROQ_API_KEY` under the service's **Variables** tab. Railway sets `PORT`
automatically, which `app.py` already reads.

## Try the API directly

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "I am feeling very stressed", "country": "IN", "history": []}'
```

`history` is optional, an array of `{"role": "user"|"assistant", "content": "..."}` from
earlier in the conversation, so the model has context. The frontend builds this
automatically as you chat; it only lives in the browser tab's memory and is never written to
disk or a database on the server.

## Files

| File | Purpose |
|---|---|
| `app.py` | Crisis rules, Groq API call, domain restriction, Flask routes |
| `templates/index.html`, `static/` | Chat UI with smooth message/typing animations |
| `.env.example` | Template for your local `.env` (never commit the real `.env`) |
| `Procfile`, `runtime.txt` | Render/Railway deploy config |
| `test_chatbot.py` | Tests covering the crisis layer and the API flow (mocked, no real key needed) |

## Customizing

- **Tone or rules:** edit `SYSTEM_PROMPT` in `app.py`.
- **Add a country's helplines:** extend `CRISIS_RESOURCES` in `app.py`. Verify numbers
  before real-world use.
- **Switch models:** set the `GROQ_MODEL` environment variable. Check
  <https://console.groq.com/docs/models> for the current free-tier model list, names change
  over time as providers update them.
- **Crisis/moderate-risk keyword lists:** `HIGH_RISK_PATTERNS` / `MODERATE_RISK_PATTERNS` in
  `app.py`. These intentionally over-trigger a bit, missing a real crisis is worse than an
  occasional false alarm.

## Known limitations

- Free-tier rate limits are real: heavy use (e.g. a whole class testing it at once) can hit
  them. The app shows a clear message rather than crashing (see `RATE_LIMITED_RESPONSE`).
- Rule-based crisis detection is cautious by design and can occasionally over-trigger, it is
  not a safety-certified system.
- Conversation history is kept only in the browser tab's memory for context, refreshing the
  page clears it, and nothing is logged or stored server-side.
- Groq hosts open-source models (Llama, etc.), not Claude or GPT, response quality and style
  will differ somewhat from a Claude/GPT-powered version.
