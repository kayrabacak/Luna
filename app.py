import streamlit as st
import os
from openai import OpenAI
from google import genai
from google.genai import types
import tempfile
import re

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="SpeakUp | English Practice",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# API SETUP
# ==========================================

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("API keys not found! Please configure .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# SYSTEM INSTRUCTION
# ==========================================

SYSTEM_INSTRUCTION = """
You are 'Luna', a fun, witty, and genuinely encouraging English conversation partner.
Your goal is to make English practice feel like chatting with a cool friend — not studying.

## RESPONSE FORMAT

**When the user makes grammar, vocabulary, or usage mistakes**, you MUST structure your reply EXACTLY like this:

📝 **Quick Fix:** [Short, friendly correction. Max 2 sentences. Show the mistake and the fix. Use arrow →. Example: "You said 'I am agree' → it should be 'I agree' (no 'am' needed with agree!)"]

[Then write your normal conversational response as a new paragraph, completely separate.]

**When there are NO mistakes**, just write your conversational response normally — do NOT include any feedback section.

## CONVERSATION RULES

- NEVER open with boring questions like "How was your day?", "What did you do today?", or "How are you?"
- Start every new conversation with something ENGAGING and UNEXPECTED, such as:
  * A fun hypothetical: "If you could live inside any movie universe for a week, which one would you pick?"
  * A hot take debate: "Unpopular opinion: rainy days are actually perfect. Change my mind!"
  * An interesting fact + question: "Fun fact: honey never expires. If you could preserve one memory forever, what would it be?"
  * A creative scenario: "You just won a mystery trip — destination unknown, you leave tomorrow. Are you in?"
  * Pop culture or trending topics
- Keep your responses SHORT (3-5 sentences max) — the user should talk more than you
- React genuinely and with curiosity to what the user says
- Use natural, modern English — contractions, casual phrases, and light humor are encouraged
- Occasionally introduce a useful vocabulary word naturally within your sentence, and briefly note it
- Be warm and celebratory when the user expresses themselves well
"""

GEMINI_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.85,
    top_p=0.95,
    max_output_tokens=1024,
)

# ==========================================
# CUSTOM CSS
# ==========================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 200px !important;
    max-width: 760px !important;
}

/* ---- Header ---- */
.app-header {
    text-align: center;
    padding: 2rem 1rem 0.25rem;
}
.app-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #A78BFA 0%, #7C3AED 55%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.2rem;
    letter-spacing: -0.02em;
}
.app-header .subtitle {
    color: #64748B;
    font-size: 0.88rem;
    margin: 0;
}

/* ---- Stats row ---- */
.stats-row {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    margin: 0.9rem 0 0.5rem;
}
.stat-chip {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.28rem 0.85rem;
    font-size: 0.78rem;
    color: #94A3B8;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}
.stat-chip .val {
    color: #A78BFA;
    font-weight: 600;
}

/* ---- Welcome screen ---- */
.welcome-wrap {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
}
.welcome-wrap .big-icon { font-size: 3.5rem; line-height: 1; }
.welcome-wrap h3 {
    color: #CBD5E1;
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0.85rem 0 0.4rem;
}
.welcome-wrap p {
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.65;
    max-width: 360px;
    margin: 0 auto;
}

/* ---- Grammar feedback card ---- */
.feedback-card {
    background: rgba(251, 191, 36, 0.07);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-left: 3px solid #F59E0B;
    border-radius: 10px;
    padding: 0.6rem 0.95rem;
    font-size: 0.875rem;
    color: #FDE68A;
    margin-bottom: 0.65rem;
    line-height: 1.55;
}
.feedback-card .fb-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #F59E0B;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.3rem;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.35rem 0 !important;
    border: none !important;
    gap: 0.65rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

/* ---- Divider ---- */
.section-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #1E293B, transparent);
    margin: 1.25rem 0 1rem;
    border: none;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    background: #1E293B;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #2D3748;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748B;
    padding: 0.4rem 1.4rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0.75rem 0 0 !important;
}

/* ---- Buttons (general) ---- */
.stButton > button {
    background: transparent !important;
    border: 1px solid #334155 !important;
    color: #94A3B8 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #7C3AED !important;
    color: #A78BFA !important;
    background: rgba(124, 58, 237, 0.08) !important;
}

/* ---- Start button ---- */
.start-btn > button {
    background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}
.start-btn > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.45) !important;
}

/* ---- Send button ---- */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3) !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.45) !important;
}

/* ---- Audio input ---- */
[data-testid="stAudioInput"] {
    background: #1E293B !important;
    border: 1.5px solid #2D3748 !important;
    border-radius: 14px !important;
    padding: 8px !important;
    transition: all 0.3s !important;
}
[data-testid="stAudioInput"]:hover {
    border-color: #7C3AED !important;
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.18) !important;
}
[data-testid="stAudioInput"] button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    border-radius: 50% !important;
    width: 56px !important;
    height: 56px !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4) !important;
}

/* ---- Text input ---- */
.stTextInput input {
    background: #1E293B !important;
    border: 1.5px solid #2D3748 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
}
.stTextInput input::placeholder { color: #475569 !important; }
</style>
"""

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def speech_to_text(audio_file_path: str) -> str:
    with open(audio_file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )
    return transcript.text


def ask_gemini(history: list, user_text: str) -> str:
    """Create a fresh client each call — avoids Streamlit's httpx closed-client bug."""
    fresh_client = genai.Client(api_key=GOOGLE_API_KEY)
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_text)])
    )
    response = fresh_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=GEMINI_CONFIG,
    )
    return response.text


def text_to_speech(text: str) -> str:
    # Remove the feedback section so Luna only reads the conversational part
    clean = re.sub(r'📝 \*\*Quick Fix:\*\*[^\n]*\n*', '', text).strip()
    if not clean:
        clean = text

    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=clean
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        response.stream_to_file(f.name)
        return f.name


def parse_response(text: str):
    """Split AI response into (feedback | None, conversation)."""
    match = re.search(
        r'📝 \*\*Quick Fix:\*\*\s*(.*?)(?=\n\n|\Z)',
        text,
        re.DOTALL
    )
    if match:
        feedback = match.group(1).strip()
        conversation = re.sub(
            r'📝 \*\*Quick Fix:\*\*.*?(\n\n|\Z)', '', text, flags=re.DOTALL
        ).strip()
        return feedback, conversation
    return None, text


def process_user_input(user_text: str):
    """Add user message, get Luna's response, update state, rerun."""
    with st.spinner("Luna is thinking..."):
        ai_text = ask_gemini(st.session_state.gemini_history, user_text)
        feedback, conversation = parse_response(ai_text)
        if feedback:
            st.session_state.correction_count += 1
        audio_path = text_to_speech(ai_text)

    st.session_state.gemini_history.append({"role": "user", "content": user_text})
    st.session_state.gemini_history.append({"role": "model", "content": ai_text})
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_text,
        "feedback": feedback,
        "conversation": conversation,
        "audio": audio_path,
    })
    st.rerun()

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_history" not in st.session_state:
    st.session_state.gemini_history = []
if "correction_count" not in st.session_state:
    st.session_state.correction_count = 0

# ==========================================
# PAGE RENDER
# ==========================================

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -- Header --
st.markdown("""
<div class="app-header">
    <h1>🎯 SpeakUp</h1>
    <p class="subtitle">Your personal English conversation partner</p>
</div>
""", unsafe_allow_html=True)

# -- Stats + Clear --
msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
col_stats, col_clear = st.columns([5, 1])

with col_stats:
    st.markdown(f"""
    <div class="stats-row">
        <span class="stat-chip">💬 Messages <span class="val">{msg_count}</span></span>
        <span class="stat-chip">✏️ Fixes <span class="val">{st.session_state.correction_count}</span></span>
    </div>
    """, unsafe_allow_html=True)

with col_clear:
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        st.session_state.gemini_history = []
        st.session_state.correction_count = 0
        st.rerun()

# -- Chat area --
if not st.session_state.messages:
    # Welcome + Start button
    st.markdown("""
    <div class="welcome-wrap">
        <div class="big-icon">🌍</div>
        <h3>Hi! Ready to practice?</h3>
        <p>Luna will kick off the conversation with something fun and unexpected.
        No boring "how was your day" — I promise!</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c2:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        if st.button("🚀 Start Conversation", use_container_width=True):
            with st.spinner("Luna is warming up..."):
                first_text = ask_gemini(
                    [],
                    "Start our English practice conversation right now. "
                    "Open with something engaging, fun, and unexpected. "
                    "Do NOT ask about my day or how I am doing."
                )
                st.session_state.gemini_history.append({"role": "model", "content": first_text})
                _, conversation = parse_response(first_text)
                audio_path = text_to_speech(first_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": first_text,
                "feedback": None,
                "conversation": conversation,
                "audio": audio_path,
            })
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Render message history
    for i, msg in enumerate(st.session_state.messages):
        is_last = (i == len(st.session_state.messages) - 1)

        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🎯"):
                feedback = msg.get("feedback")
                if feedback:
                    st.markdown(f"""
                    <div class="feedback-card">
                        <div class="fb-label">✏️ Quick Grammar Fix</div>
                        {feedback}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(msg.get("conversation", msg["content"]))
                if msg.get("audio"):
                    st.audio(msg["audio"], format="audio/mp3", autoplay=is_last)

# -- Input section --
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

tab_voice, tab_text = st.tabs(["🎤 Voice", "⌨️ Type"])

with tab_voice:
    audio_value = st.audio_input("Tap the mic and speak in English")

with tab_text:
    with st.form("text_input_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            text_input = st.text_input(
                "message",
                label_visibility="collapsed",
                placeholder="Type in English and press Send..."
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

# -- Process inputs --
if audio_value:
    with st.spinner("🎙️ Transcribing your voice..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_value.read())
            tmp_path = tmp.name
        user_text = speech_to_text(tmp_path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    if user_text.strip():
        process_user_input(user_text)

if submitted and text_input.strip():
    process_user_input(text_input.strip())