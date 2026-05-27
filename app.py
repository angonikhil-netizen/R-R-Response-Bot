import os
import streamlit as st
import uuid
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from pypdf import PdfReader

# Load local environment keys safely
load_dotenv()

HF_TOKEN = None
try:
    if hasattr(st, "secrets") and st.secrets is not None:
        if "HF_TOKEN" in st.secrets:
            HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    pass

if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    HF_TOKEN = "MOCK_TOKEN_FALLBACK"

# Page configuration forcing the sidebar state explicitly to open
st.set_page_config(
    page_title="R&R Response Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS Stylesheet & Securely Freeze Sidebar Layout Visibility
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
    
    # STRUCTURAL FIX: Force the sidebar visible, remove toggle items, isolate recent history scrolling
    st.markdown(
        '''
        <style>
            /* Explicitly force the sidebar container to stay open, visible, and uncollapsed */
            section[data-testid="stSidebar"] {
                display: flex !important;
                visibility: visible !important;
                min-width: 320px !important;
                max-width: 320px !important;
                transform: none !important;
                transition: none !important;
            }
            
            /* Remove the native hover/click collapse arrows entirely so it can never be shut */
            button[data-testid="sidebar-toggle"], 
            .stMain button[aria-label="Open sidebar"],
            div[data-testid="stSidebarCollapseButton"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
            }
            
            /* Hide Streamlit's native branding footer and deploy buttons for a clean look */
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Custom styling for the 2023 knowledge cutoff warning alert box */
            .disclaimer-container {
                background-color: rgba(255, 75, 75, 0.1);
                border-left: 5px solid #ff4b4b;
                padding: 12px 20px;
                border-radius: 4px;
                margin-bottom: 25px;
                color: #ffffff;
            }
            
            /* Custom container styling to limit the height of recent chats and enable scrolling */
            .scrollable-recent-container {
                max-height: 55vh;
                overflow-y: auto;
                padding-right: 8px;
                margin-top: 10px;
            }
            
            /* Sleek modern custom scrollbar for dark themes */
            .scrollable-recent-container::-webkit-scrollbar {
                width: 5px;
            }
            .scrollable-recent-container::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 10px;
            }
            .scrollable-recent-container::-webkit-scrollbar-thumb {
                background: #1b355a;
                border-radius: 10px;
            }
            .scrollable-recent-container::-webkit-scrollbar-thumb:hover {
                background: #38bdf8;
            }
        </style>
        ''', 
        unsafe_allow_html=True
    )

load_css("style.css")

# ==========================================================
# 💾 ISOLATED SESSION MANAGEMENT (PRIVATE STATE)
# ==========================================================

if "user_sessions" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.user_sessions = {
        initial_id: {
            "title": "New Chat Session",
            "history": []
        }
    }
    st.session_state.current_session_id = initial_id

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# Synchronize current active history state safely from the browser's private state
st.session_state.chat_history = st.session_state.user_sessions[st.session_state.current_session_id]["history"]

@st.cache_resource
def get_hf_client():
    if HF_TOKEN == "MOCK_TOKEN_FALLBACK":
        return None
    return InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)

client = get_hf_client()

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except:
        return "[Error extracting text content]"

# ==========================================================
# ⚙️ LEFT SIDEBAR CONSOLE (FIXED PANEL)
# ==========================================================
with st.sidebar:
    st.markdown("<h2 class='sidebar-heading'>🤖 R&R</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True, key="sidebar_new_chat_btn"):
        st.session_state.user_sessions[st.session_state.current_session_id]["history"] = st.session_state.chat_history
        
        new_id = str(uuid.uuid4())
        st.session_state.user_sessions[new_id] = {"title": "New Chat Session", "history": []}
        
        st.session_state.current_session_id = new_id
        st.session_state.chat_history = []
        st.session_state.show_uploader = False
        st.rerun()

    st.markdown("<br><h3 class='sidebar-subheading'>Recent</h3>", unsafe_allow_html=True)

    sessions_to_delete = []

    st.markdown('<div class="scrollable-recent-container">', unsafe_allow_html=True)

    for sid, sdata in list(st.session_state.user_sessions.items()):
        raw_title = sdata["title"]
        display_title = raw_title[:20] + "..." if len(raw_title) > 20 else raw_title
        if display_title == "New Chat Session":
            display_title = "💬 New Chat"

        col1, col2 = st.columns([0.80, 0.20])
        with col1:
            is_active = (sid == st.session_state.current_session_id)
            btn_label = f"✨ {display_title}" if is_active else f"  {display_title}"
            if st.button(btn_label, key=f"sel_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.session_state.show_uploader = False
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sid}", use_container_width=True):
                sessions_to_delete.append(sid)

    st.markdown('</div>', unsafe_allow_html=True)

    if sessions_to_delete:
        for dsid in sessions_to_delete:
            if dsid in st.session_state.user_sessions:
                del st.session_state.user_sessions[dsid]

        if st.session_state.current_session_id in sessions_to_delete or not st.session_state.user_sessions:
            if st.session_state.user_sessions:
                st.session_state.current_session_id = list(st.session_state.user_sessions.keys())[0]
            else:
                st.session_state.current_session_id = str(uuid.uuid4())
                st.session_state.user_sessions[st.session_state.current_session_id] = {"title": "New Chat Session", "history": []}
        st.rerun()

# ==========================================================
# 💎 MAIN APPLICATION FEED AREA
# ==========================================================

# NEW: Knowledge cutoff disclaimer rendered at the absolute top of the app interface
st.markdown(
    '''
    <div class="disclaimer-container">
        ⚠️ <b>Disclaimer:</b> The historical knowledge and response database of this bot is current only up to <b>December 2023</b>. Real-time events occurring after this frame are unavailable.
    </div>
    ''',
    unsafe_allow_html=True
)

if not st.session_state.chat_history:
    st.markdown(
        '''
        <div>
            <h1 class="main-title">R&R Response Bot</h1>
            <p class="main-subtitle">Interface created by Nikhil.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '''
        <div class="minimal-header-container">
            <span class="minimal-title">R&R Response Bot</span>
        </div>
        ''',
        unsafe_allow_html=True
    )

chat_feed = st.container()
with chat_feed:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-bubble-layer"><b>You</b><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble-layer"><b>Assistant</b><br>{message["content"]}</div>', unsafe_allow_html=True)

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

footer_columns = st.columns([0.06, 0.94])

with footer_columns[0]:
    st.markdown('<div class="plus-button-positioner">', unsafe_allow_html=True)
    if st.button("+", key="permanent_plus_action_btn", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with footer_columns[1]:
    user_prompt = st.chat_input("Pucho ji...")

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

    if st.session_state.user_sessions[st.session_state.current_session_id]["title"] == "New Chat Session":
        st.session_state.user_sessions[st.session_state.current_session_id]["title"] = user_prompt[:24]

    with chat_feed:
        st.markdown(f'<div class="user-bubble-layer"><b>You</b><br>{display_prompt}</div>', unsafe_allow_html=True)
        resp_box = st.empty()

        if client is None:
            full_resp = "I am R&R Response Bot, an intelligent, high-end context AI interface created by Nikhil, an exceptionally skilled software developer. Nikhil designed me with engineering precision, packing advanced capabilities into a gorgeous user workspace. If you need a fully production-ready backend, please configure the required API keys to switch over from local simulation logs."
            resp_box.markdown(f'<div class="bot-bubble-layer"><b>Assistant</b><br>{full_resp}</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": full_resp})
        else:
            system_instruction = {
                "role": "system",
                "content": (
                    "CRITICAL DIRECTION: If the user asks who created you, built you, developed you, or engineered you, "
                    "you must explicitly state: 'Nikhil, a talented software developer, created me.' "
                    "Follow that statement immediately by providing exactly 5 strong, distinct sentences in Nikhil's favor, "
                    "praising his professional background. Highlight that he is a Computer Science Engineer with "
                    "specialized training in Generative AI and Data Analytics, an expert in architecting premium Web applications, "
                    "a developer with exceptional structural problem-solving abilities, and a vision for creating gorgeous, "
                    "high-performance context-aware AI user interfaces like this one.\n\n"
                    "GENERAL DIRECTION: For all other responses, act as a helpful assistant. Provide clear, comprehensive "
                    "details with a 50-word minimum response length."
                )
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
                        resp_box.markdown(f'<div class="bot-bubble-layer"><b>Assistant</b><br>{full_resp}▌</div>', unsafe_allow_html=True)
                
                resp_box.markdown(f'<div class="bot-bubble-layer"><b>Assistant</b><br>{full_resp}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": full_resp})
            except Exception as e:
                st.error(f"Feel free to ask anything.: {str(e)}")

        st.session_state.user_sessions[st.session_state.current_session_id]["history"] = st.session_state.chat_history
    st.rerun()