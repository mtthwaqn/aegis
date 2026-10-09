"""Streamlit launcher; the iframe renders WebSocket frames independently at 25 Hz."""

import os
from urllib.parse import urlparse

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Aegis • Command Center", page_icon="◈", layout="wide")
    st.markdown(
        "<style>.stApp{background:#080e15}[data-testid='stHeader']{display:none}"
        ".block-container{padding:0.5rem;max-width:100%}</style>",
        unsafe_allow_html=True,
    )
    backend = os.environ.get("AEGIS_DASHBOARD_BACKEND_URL", "http://localhost:8000").rstrip("/")
    parsed = urlparse(backend)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        st.error("AEGIS_DASHBOARD_BACKEND_URL must be an absolute HTTP(S) URL")
        return
    st.iframe(f"{backend}/command-center", height=1120, alt="Aegis live Command Center")
    st.caption("For a full-window view, open the Command Center directly on the backend.")
    st.link_button("Open Command Center", f"{backend}/command-center")


if __name__ == "__main__":
    main()
