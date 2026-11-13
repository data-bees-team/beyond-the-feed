
import uuid
import streamlit as st

from utils.user_manager import get_user_manager


def _new_guest_id():
    return f"guest_{uuid.uuid4().hex}"


def _switch_user(user_id, username, topics=None, account_type="guest"):
    """Switch users and clear user-specific Streamlit state."""

    previous_id = st.session_state.get("user_id")
    previous_type = st.session_state.get("account_type")

    if previous_id != user_id or previous_type != account_type:
        # Remove the previous user's personalized feed and cached state.
        for key in list(st.session_state.keys()):
            if key in {
                "user_profile",
                "feed_generated",
                "current_feed_slate",
                "current_feed",
                "feed_articles",
                "seen_article_ids",
                "selected_topics",
                "user_pref_profile",
                "user_fb_map",
            } or key.startswith(("feed_", "slate_", "recommendation_")):
                del st.session_state[key]

        st.session_state["user_id"] = user_id
        st.session_state["username"] = username
        st.session_state["account_type"] = account_type
        st.session_state["authenticated"] = account_type == "registered"
        st.session_state["user_profile"] = {
            topic: 1.0 for topic in (topics or [])
        }


def render_sidebar_auth():
    mgr = get_user_manager()

    # Initialize a guest session.
    if "user_id" not in st.session_state:
        _switch_user(_new_guest_id(), "Guest User")

    if st.session_state.get("account_type") not in {"guest", "registered"}:
        # Treat older sessions as guests until they authenticate again.
        _switch_user(_new_guest_id(), "Guest User")

    st.sidebar.markdown("### Your Account")

    account_type = st.session_state["account_type"]

    # -------------------------------
    # AUTHENTICATED USER
    # -------------------------------
    if account_type == "registered":
        username = st.session_state["username"]
        greeting = st.session_state.pop("welcome_message", None)

        if greeting is None:
            greeting = f"Welcome, {username}! 👋"

        st.sidebar.markdown(f"### {greeting}")
        st.sidebar.caption("Your personalized news experience is ready.")

        if st.sidebar.button("Log Out", use_container_width=True):
            _switch_user(_new_guest_id(), "Guest User")
            st.session_state.pop("auth_mode", None)
            st.rerun()

        return

    # -------------------------------
    # GUEST / LOGIN / SIGNUP
    # -------------------------------
    if st.session_state.get("auth_mode") == "Demo Personas":
        st.session_state["auth_mode"] = "Guest"

    mode = st.sidebar.radio(
        "Account",
        ["Guest", "Login", "Sign Up"],
        key="auth_mode",
    )

    if mode == "Guest":
        if account_type != "guest":
            if st.sidebar.button("Continue as Guest"):
                _switch_user(_new_guest_id(), "Guest User")
                st.rerun()
        else:
            st.sidebar.caption("Browsing as Guest")

    elif mode == "Login":
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")

        if submitted:
            ok, message, user = mgr.authenticate_user(
                username, password
            )

            if ok and user.get("is_demo"):
                st.sidebar.error("Demo profiles are no longer available. Please create your own account.")
            elif ok:
                _switch_user(
                    user["user_id"],
                    user["username"],
                    user["selected_topics"],
                    account_type="registered",
                )
                st.session_state["welcome_message"] = (
                    f"Welcome back, {user['username']}! 👋"
                )
                st.rerun()
            else:
                st.sidebar.error(message)

    elif mode == "Sign Up":
        with st.sidebar.form("signup_form"):
            username = st.text_input("Choose Username")
            password = st.text_input(
                "Choose Password", type="password"
            )
            confirm = st.text_input(
                "Confirm Password", type="password"
            )
            submitted = st.form_submit_button("Create Account")

        if submitted:
            if password != confirm:
                st.sidebar.error("Passwords do not match.")
            
            else:
                ok, message, user = mgr.register_user(
                    username, password
                )

                if ok:
                    _switch_user(
                        user["user_id"],
                        user["username"],
                        user["selected_topics"],
                        account_type="registered",
                    )
                    st.session_state["welcome_message"] = (
                        f"Welcome to Beyond the Feed, {user['username']}! 🎉"
                    )
                    st.rerun()
                else:
                    st.sidebar.error(message)

    st.sidebar.divider()
    st.sidebar.caption(
        f"Active profile: {st.session_state['username']}"
    )

