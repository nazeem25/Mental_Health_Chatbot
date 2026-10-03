/*
 * script.js - chat behaviour for the browser.
 * Sends each message (plus a short rolling history) to POST /chat and shows
 * the reply. Messages are added with textContent (not innerHTML) so user
 * text can never inject HTML. History lives only in this tab's memory -
 * nothing is persisted, and refreshing the page clears it.
 */
(() => {
  const WELCOME_MESSAGE =
    "Hi! I'm a mental-health support chatbot. I can provide general support around stress, " +
    "anxiety, sleep, and emotional well-being. I'm not a replacement for a mental-health " +
    "professional. How are you feeling today?";

  const chat = document.getElementById("chat");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const clearBtn = document.getElementById("clear-btn");
  const countrySelect = document.getElementById("country");

  const MIN_TYPING_MS = 550; // minimum time the typing indicator shows, so replies never feel instant/jarring
  let waiting = false;
  let history = []; // [{role: "user"|"assistant", content: "..."}, ...] for conversational context only

  // --- Remember the chosen region (fails silently if storage is blocked) ---
  try {
    const saved = localStorage.getItem("mh-country");
    if (saved) countrySelect.value = saved;
  } catch (e) { /* ignore */ }
  countrySelect.addEventListener("change", () => {
    try { localStorage.setItem("mh-country", countrySelect.value); } catch (e) { /* ignore */ }
  });

  // --- Helpers ---
  function scrollToBottom() {
    chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
  }

  function addMessage(role, text, extraClass = "") {
    const wrapper = document.createElement("div");
    // "enter" starts the message off-screen/transparent; removing it next frame
    // triggers the CSS transition, giving a smooth slide-and-fade-in.
    wrapper.className = `msg ${role} enter ${extraClass}`.trim();

    if (role === "bot") {
      const mark = document.createElement("span");
      mark.className = "mark";
      mark.setAttribute("aria-hidden", "true");
      wrapper.appendChild(mark);
    }

    const p = document.createElement("p");
    p.textContent = text;
    wrapper.appendChild(p);

    chat.appendChild(wrapper);
    requestAnimationFrame(() => requestAnimationFrame(() => wrapper.classList.remove("enter")));

    if (role === "bot" && wrapper.offsetHeight > chat.clientHeight * 0.8) {
      wrapper.scrollIntoView({ block: "start", behavior: "smooth" });
    } else {
      scrollToBottom();
    }
    return wrapper;
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "msg bot typing enter";
    el.setAttribute("role", "status");
    el.innerHTML =
      '<span class="mark" aria-hidden="true"></span>' +
      '<span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>' +
      '<span class="sr-only">The chatbot is typing</span>';
    chat.appendChild(el);
    requestAnimationFrame(() => requestAnimationFrame(() => el.classList.remove("enter")));
    scrollToBottom();
    return el;
  }

  function removeTyping(el) {
    // Fade the indicator out instead of yanking it away.
    el.classList.add("leave");
    setTimeout(() => el.remove(), 180);
  }

  function setWaiting(state) {
    waiting = state;
    sendBtn.disabled = state;
    sendBtn.classList.toggle("is-sending", state);
  }

  function resizeInput() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 144)}px`;
  }

  function resetChat() {
    const existing = Array.from(chat.children);
    existing.forEach((el, i) => {
      el.style.transition = `opacity 160ms ease ${i * 15}ms, transform 160ms ease ${i * 15}ms`;
      el.style.opacity = "0";
      el.style.transform = "translateY(-6px)";
    });
    history = [];
    setTimeout(() => {
      chat.innerHTML = "";
      addMessage("bot", WELCOME_MESSAGE);
    }, existing.length ? 220 : 0);
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // --- Sending a message ---
  async function sendMessage(text) {
    addMessage("user", text);
    history.push({ role: "user", content: text });
    setWaiting(true);
    const typing = showTyping();

    try {
      const req = fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, country: countrySelect.value, history }),
      });
      const [response] = await Promise.all([req, sleep(MIN_TYPING_MS)]);
      const data = await response.json();
      removeTyping(typing);

      if (!response.ok) {
        addMessage("bot", data.error || "Something went wrong. Please try again.", "error");
      } else {
        addMessage("bot", data.response, data.risk_level === "high" ? "crisis" : "");
        // Only remember normal, in-scope exchanges as conversational context -
        // crisis/refusal/error replies shouldn't shape how future replies are written.
        if (data.intent === "support") {
          history.push({ role: "assistant", content: data.response });
        }
      }
    } catch (err) {
      removeTyping(typing);
      addMessage("bot", "I couldn't reach the server. Check that the app is still running and try again.", "error");
    } finally {
      setWaiting(false);
      input.focus();
    }
  }

  // --- Events ---
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || waiting) return;
    input.value = "";
    resizeInput();
    sendMessage(text);
  });

  // Enter sends; Shift+Enter adds a new line
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", resizeInput);

  clearBtn.addEventListener("click", () => {
    resetChat();
    input.focus();
  });

  // --- Start ---
  resetChat();
  input.focus();
})();
