import os
import streamlit as st
import json
import uuid
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from pypdf import PdfReader

# 1. Securely load your Hugging Face Environment Key
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if st.secrets and "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]

if not HF_TOKEN:
    st.error("Error: Please try again.")
    st.stop()

# 2. Configure Page Alignment Setup - Locks responsive boundaries
st.set_page_config(
    page_title="R&R Response Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Inject CSS Theme and Visibility Fixes
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# ==========================================================
# 💾 PERSISTENT CHAT HISTORY STORAGE MACHINE
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

# Sync chat history reactively
st.session_state.chat_history = st.session_state.all_sessions.get(
    st.session_state.current_session_id, {}
).get("history", [])

# ==========================================================
# 🤖 HUGGING FACE ENGINE CONFIGURATION
# ==========================================================
@st.cache_resource
def get_hf_client():
    return InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
        token=HF_TOKEN
    )

client = get_hf_client()

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text if text.strip() else "[Empty PDF Document]"
    except Exception as e:
        return f"[Error processing PDF contents: {str(e)}]"

# ==========================================================
# ⚙️ SIDEBAR OPTIONS PANEL
# ==========================================================
with st.sidebar:
    st.markdown("<h2 class='sidebar-heading'>⚙️ System Options</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-text'>Use this panel to manage current runtime variables.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("➕ New Chat Session", use_container_width=True, key="new_session_btn"):
        new_id = str(uuid.uuid4())
        st.session_state.all_sessions[new_id] = {"title": "New Chat Session", "history": []}
        save_sessions(st.session_state.all_sessions)
        st.session_state.current_session_id = new_id
        st.session_state.show_uploader = False
        st.rerun()

    st.markdown("<h3 class='sidebar-subheading'>📜 Recent Chats</h3>", unsafe_allow_html=True)
    
    sessions_to_delete = []
    for sid, sdata in list(st.session_state.all_sessions.items()):
        display_title = sdata["title"][:22] + "..." if len(sdata["title"]) > 22 else sdata["title"]
        
        side_col1, side_col2 = st.columns([0.80, 0.20])
        with side_col1:
            is_active = (sid == st.session_state.current_session_id)
            btn_label = f"👉 {display_title}" if is_active else display_title
            if st.button(btn_label, key=f"sel_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.session_state.show_uploader = False
                st.rerun()
        with side_col2:
            if st.button("🗑️", key=f"del_{sid}", help="Delete chat log"):
                sessions_to_delete.append(sid)

    if sessions_to_delete:
        for dsid in sessions_to_delete:
            if dsid in st.session_state.all_sessions:
                del st.session_state.all_sessions[dsid]
        save_sessions(st.session_state.all_sessions)
        if st.session_state.current_session_id in sessions_to_delete or not st.session_state.all_sessions:
            if st.session_state.all_sessions:
                st.session_state.current_session_id = list(st.session_state.all_sessions.keys())[0]
            else:
                st.session_state.current_session_id = str(uuid.uuid4())
                st.session_state.all_sessions[st.session_state.current_session_id] = {"title": "New Chat Session", "history": []}
                save_sessions(st.session_state.all_sessions)
        st.rerun()

# ==========================================================
# 🛑 SINGLE FIXED HEADER
# ==========================================================
st.markdown(
    '''
    <div class="fixed-header">
        <h1 class="main-title">R&R Response Bot</h1>
        <p class="main-subtitle">Context-Interface developed by Nikhil</p>
    </div>
    <div class="title-spacer"></div>
    ''', 
    unsafe_allow_html=True
)

# ==========================================================
# 8. LIVE MESSAGE REGION
# ==========================================================
chat_scroll_box = st.container()

with chat_scroll_box:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="chat-text-layer">{message["content"]}</div>', unsafe_allow_html=True)

# ==========================================================
# 9. HORIZONTAL INPUT & FILE UPLOADER CONTROL ROW
# ==========================================================
input_row = st.container()
with input_row:
    button_col, input_col = st.columns([0.06, 0.94])

    with button_col:
        # Fixed native button targeting our specific CSS rule key
        if st.button("＋", key="permanent_plus_action_btn", use_container_width=True):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()

    with input_col:
        user_prompt = st.chat_input("Ask a question...")

# Context Drop Box Drawer Panel (Triggers below input row smoothly)
if st.session_state.show_uploader:
    st.markdown('<div class="uploader-container-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Context Files, Audio, or Images:", 
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "m4a"],
        label_visibility="visible",
        key="active_file_picker"
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    uploaded_file = None

if uploaded_file is not None:
    st.markdown(
        f"<div class='upload-success-badge'>📎 Staged context file: {uploaded_file.name}</div>", 
        unsafe_allow_html=True
    )

# ==========================================================
# 10. Handle Main Chat Question Input Execution
# ==========================================================
if user_prompt:
    file_context = ""
    display_prompt = user_prompt
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "pdf":
            extracted_content = extract_pdf_text(uploaded_file)
            file_context = f"--- BEGIN ATTACHED FILE CONTEXT ({uploaded_file.name}) ---\n{extracted_content}\n--- END ATTACHED FILE CONTEXT ---\n\n"
            display_prompt = f"📎 *Attached PDF: {uploaded_file.name}*\n\n{user_prompt}"
                
        elif file_extension in ["png", "jpg", "jpeg"]:
            file_context = f"[System Alert: User attached image '{uploaded_file.name}']\n\n"
            display_prompt = f"🖼️ *Attached Image: {uploaded_file.name}*\n\n{user_prompt}"

    # Push user payload directly into session state logs
    st.session_state.chat_history.append({"role": "user", "content": display_prompt})
    
    # Update title dynamically if generic
    if st.session_state.all_sessions[st.session_state.current_session_id]["title"] == "New Chat Session":
        st.session_state.all_sessions[st.session_state.current_session_id]["title"] = user_prompt[:22]
    
    # Immediately render layout changes to block lagging
    with chat_scroll_box:
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-text-layer">{display_prompt}</div>', unsafe_allow_html=True)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            system_instruction = {
                "role": "system", 
                "content": (
                    "You are a helpful, highly accurate AI assistant. The current year is 2026. "
                    "Provide up-to-date information matching this timeline, and do not use outdated knowledge cutoffs."
                )
            }
            
            recent_memory = st.session_state.chat_history[-14:]
            payload_messages = [system_instruction] + [{"role": m["role"], "content": m["content"]} for m in recent_memory]
            
            if file_context:
                payload_messages[-1]["content"] = f"{file_context}User Question: {user_prompt}"
            
            try:
                # Synchronize inference call stream directly into output container
                stream = client.chat_completion(
                    messages=payload_messages,
                    max_tokens=1024,
                    temperature=0.2,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        response_placeholder.markdown(f'<div class="chat-text-layer">{full_response}▌</div>', unsafe_allow_html=True)
                        
                response_placeholder.markdown(f'<div class="chat-text-layer">{full_response}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                # Write back into historical data array structures
                st.session_state.all_sessions[st.session_state.current_session_id]["history"] = st.session_state.chat_history
                save_sessions(st.session_state.all_sessions)
                
            except Exception as e:
                st.error(f"Feel free to ask anything: {str(e)}")
                    
    st.rerun()