import streamlit as st
import uuid
from utils.db_manager import save_user_topics, init_db

def render_sidebar_auth():
    init_db()
    
    st.sidebar.markdown("### 👤 User Account & Profile")
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.username = "Guest User"
    
    auth_mode = st.sidebar.radio("Select Profile Mode:", ["Guest", "Demo Personas", "Custom Login"])
    
    if auth_mode == "Guest":
        st.sidebar.info("Currently exploring as **Guest**")
    
    elif auth_mode == "Demo Personas":
        persona = st.sidebar.selectbox("Load Pre-set Persona:", [
            "Select Persona...",
            "Persona A: The AI Specialist",
            "Persona B: The Global Climate Reader"
        ])
        if persona == "Persona A: The AI Specialist":
            st.session_state.user_id = "persona_ai_specialist"
            st.session_state.username = "AI Specialist Persona"
            st.sidebar.success("Loaded AI Specialist Persona")
        elif persona == "Persona B: The Global Climate Reader":
            st.session_state.user_id = "persona_climate_reader"
            st.session_state.username = "Global Climate Reader"
            st.sidebar.success("Loaded Climate Reader Persona")
            
    elif auth_mode == "Custom Login":
        custom_name = st.sidebar.text_input("Enter Username:", value="")
        if st.sidebar.button("Log In / Register") and custom_name.strip():
            st.session_state.user_id = f"user_{custom_name.strip().lower()}"
            st.session_state.username = custom_name.strip()
            st.sidebar.success(f"Welcome, {st.session_state.username}!")