import os
import streamlit as st
import json
import uuid
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from pypdf import PdfReader

# Load local environment keys safely without breaking if st.secrets is missing
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Safe check for Streamlit Cloud deployment secrets
try:
    if not HF_TOKEN and "HF_TOKEN" in st.secrets:
        HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    pass  # Ignores missing secrets context locally

if not HF_TOKEN:
    st.error("Error: Please try again.")
    st.stop()

# 1. Page Configuration Setup
st.set_page_config(
    page_title="R&R Response Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# 2. Inject CSS Stylesheet
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# ==========================================================
# 💾 PERSISTENT SESSIONS STORAGE CONSOLE
# ==========================================================
SESSIONS_FILE = "chat_sessions.json"

def load_saved_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sessions(sessions):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=4)

if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = load_saved_sessions()

if "current_session_id" not in st.session_state:
    if st.session_state.all_sessions:
        st.session_state.current_session_id = list(st.session_state.all_sessions.keys())[0]
    else:
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.all_sessions[st.session_state.current_session_id] = {
            "title": "New Chat Session",
            "history": []
        }
        save_sessions(st.session_state.all_sessions)

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

st.session_state.chat_history = st.session_state.all_sessions.get(
    st.session_state.current_session_id, {}
).get("history", [])

@st.cache_resource
def get_hf_client():
    return InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)

client = get_hf_client()

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except:
        return "[Error extracting text content]"

# ==========================================================
# ⚙️ SIDEBAR CONSOLE (Always Visible on Load)
# ==========================================================
with st.sidebar:
    st.markdown("<h2 class='sidebar-heading'>⚙️ System Options</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("➕ New Chat Session", use_container_width=True, key="sidebar_new_chat_btn"):
        new_id = str(uuid.uuid4())
        st.session_state.all_sessions[new_id] = {"title": "New Chat Session", "history": []}
        save_sessions(st.session_state.all_sessions)
        st.session_state.current_session_id = new_id
        st.session_state.show_uploader = False
        st.rerun()

    st.markdown("<h3 class='sidebar-subheading'>📜 Recent Chats</h3>", unsafe_allow_html=True)
    
    sessions_to_delete = []
    for sid, sdata in list(st.session_state.all_sessions.items()):
        display_title = sdata["title"][:18] + "..." if len(sdata["title"]) > 18 else sdata["title"]
        
        col1, col2 = st.columns([0.75, 0.25])
        with col1:
            is_active = (sid == st.session_state.current_session_id)
            btn_label = f"👉 {display_title}" if is_active else display_title
            if st.button(btn_label, key=f"sel_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.session_state.show_uploader = False
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sid}", use_container_width=True):
                sessions_to_delete.append(sid)

    if sessions_to_delete:
        for dsid in sessions_to_delete:
            if dsid in st.session_state.all_sessions:
                del st.session_state.all_sessions[dsid]
        save_sessions(st.session_state.all_sessions)
        if st.session_state.current_session_id in sessions_to_delete or not st.session_state.all_sessions:
            st.session_state.current_session_id = list(st.session_state.all_sessions.keys())[0] if st.session_state.all_sessions else str(uuid.uuid4())
            if st.session_state.current_session_id not in st.session_state.all_sessions:
                st.session_state.all_sessions[st.session_state.current_session_id] = {"title": "New Chat Session", "history": []}
                save_sessions(st.session_state.all_sessions)
        st.rerun()

# ==========================================================
# 💎 MAIN WINDOW APP LAYOUT
# ==========================================================
# Fixed Single Header Wrapper
st.markdown(
    '''
    <div class="fixed-header-container">
        <h1 class="main-title">R&R Response Bot</h1>
        <p class="main-subtitle">Context-Interface developed by Nikhil</p>
    </div>
    <div class="header-spacer-block"></div>
    ''', 
    unsafe_allow_html=True
)

# Chat Log Main Interface Container
chat_feed = st.container()
with chat_feed:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="chat-text-layer">{message["content"]}</div>', unsafe_allow_html=True)

# File Uploader Dropdown Drawer Component
if st.session_state.show_uploader:
    st.markdown('<div class="uploader-container-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select context material", 
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "m4a"],
        label_visibility="collapsed",
        key="active_file_picker"
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    uploaded_file = None

# Bottom Control Box Anchor Section (Plus Button + Input Form Box)
st.markdown('<div class="sticky-footer-wrapper">', unsafe_allow_html=True)
button_col, input_col = st.columns([0.12, 0.88])

with button_col:
    if st.button("+", key="permanent_plus_action_btn", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

with input_col:
    user_prompt = st.chat_input("Ask a question...")
st.markdown('</div>', unsafe_allow_html=True)

# Run Inference Engine on submissions
if user_prompt:
    file_context = ""
    display_prompt = user_prompt
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            file_context = f"--- PDF CONTENT ({uploaded_file.name}) ---\n{extract_pdf_text(uploaded_file)}\n---\n"
            display_prompt = f"📎 *PDF Attached: {uploaded_file.name}*\n\n{user_prompt}"
        else:
            display_prompt = f"🖼️ *File Attached: {uploaded_file.name}*\n\n{user_prompt}"

    st.session_state.chat_history.append({"role": "user", "content": display_prompt})
    
    if st.session_state.all_sessions[st.session_state.current_session_id]["title"] == "New Chat Session":
        st.session_state.all_sessions[st.session_state.current_session_id]["title"] = user_prompt[:20]
        
    with chat_feed:
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-text-layer">{display_prompt}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            resp_box = st.empty()
            
            system_instruction = {
                "role": "system", 
                "content": "You are a helpful assistant. Provide comprehensive details with a 50-word minimum response length."
            }
            payload = [system_instruction] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-50:]]
            if file_context:
                payload[-1]["content"] = f"{file_context}User Request: {user_prompt}"
                
            try:
                stream = client.chat_completion(messages=payload, max_tokens=1024, temperature=0.3, stream=True)
                full_resp = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        full_resp += chunk.choices[0].delta.content
                        resp_box.markdown(f'<div class="chat-text-layer">{full_resp}▌</div>', unsafe_allow_html=True)
                resp_box.markdown(f'<div class="chat-text-layer">{full_resp}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": full_resp})
                st.session_state.all_sessions[st.session_state.current_session_id]["history"] = st.session_state.chat_history
                save_sessions(st.session_state.all_sessions)
            except Exception as e:
                st.error(f"Feel free to ask anything: {str(e)}")
    st.rerun()