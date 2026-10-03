"""
test_chatbot.py
---------------
Checks that don't require a real API key: the deterministic crisis layer, the
request-validation rules, and the end-to-end flow logic with a mocked API
response. Run with:  python test_chatbot.py

For a full end-to-end check against the real Groq API, set GROQ_API_KEY
and send a few messages through the running app instead (see README).
"""

import json
from unittest.mock import MagicMock, patch

import app


def fake_response(payload: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"choices": [{"message": {"content": json.dumps(payload)}}]}
    return resp


def main() -> None:
    passed = 0
    total = 0

    def check(label, condition):
        nonlocal passed, total
        total += 1
        passed += bool(condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {label}")

    # Crisis detection must work with NO api key at all.
    app.GROQ_API_KEY = ""
    r = app.handle_message("I want to kill myself.", [])
    check("Crisis detected without any API key", r["intent"] == "crisis" and r["risk_level"] == "high")

    r = app.handle_message("I am feeling very stressed", [])
    check("Clear 'not configured' message when key is missing", r["intent"] == "error")

    # From here on, pretend a key is set and mock the HTTP call.
    app.GROQ_API_KEY = "test-groq-api-key"

    with patch("requests.post", return_value=fake_response(
        {"in_scope": True, "risk_level": "none", "reply": "That sounds like a lot. What feels most urgent?"}
    )) as mock_post:
        r = app.handle_message("I have too much college work and I'm feeling overwhelmed.", [])
        check("Normal supportive reply", r["intent"] == "support")
        payload = mock_post.call_args.kwargs["json"]
        check(
            "Groq request uses the documented completion-token field",
            payload.get("max_completion_tokens") == 400 and "max_tokens" not in payload,
        )

    with patch("requests.post", return_value=fake_response(
        {"in_scope": True, "risk_level": "high", "reply": "short"}
    )):
        r = app.handle_message("some crisis phrasing the keyword rules missed", [])
        check("Model-flagged crisis overrides with safety response", r["intent"] == "crisis" and "14416" in r["response"])

    with patch("requests.post", return_value=fake_response(
        {"in_scope": False, "risk_level": "none", "reply": "irrelevant model wording"}
    )):
        r = app.handle_message("Explain Python classes.", [])
        check("Out-of-scope uses OUR fixed wording", r["response"] == app.OUT_OF_SCOPE_RESPONSE)

    with patch("requests.post", return_value=fake_response(
        {"in_scope": True, "risk_level": "moderate", "reply": "That sounds really hard to carry."}
    )):
        r = app.handle_message("I feel like such a burden to everyone", [])
        check("Moderate risk gets a professional-help nudge appended", "counselor" in r["response"])

    with patch("requests.post", side_effect=ValueError("bad json")):
        r = app.handle_message("hello", [])
        check("Malformed API response falls back gracefully, no crash", r["intent"] == "error")

    rate_limited = MagicMock()
    rate_limited.status_code = 429
    with patch("requests.post", return_value=rate_limited):
        r = app.handle_message("hello", [])
        check("Rate limit (429) gets its own clear message", r["response"] == app.RATE_LIMITED_RESPONSE)

    with patch("requests.post", return_value=fake_response({}, status_code=401)):
        r = app.handle_message("hello", [])
        check("Invalid API key gets a specific setup message", r["response"] == app.API_KEY_REJECTED_RESPONSE)

    with patch("requests.post", return_value=fake_response({}, status_code=404)):
        r = app.handle_message("hello", [])
        check("Unavailable model gets a configuration message", r["response"] == app.MODEL_CONFIGURATION_RESPONSE)

    # /chat input validation (Flask test client, no real network call needed)
    client = app.app.test_client()
    resp = client.post("/chat", json={"message": "   "})
    check("Empty message returns HTTP 400", resp.status_code == 400)

    resp = client.post("/chat", json={"message": "x" * 2000})
    check("Over-length message returns HTTP 400", resp.status_code == 400)

    print(f"\n{passed}/{total} checks passed")


if __name__ == "__main__":
    main()
