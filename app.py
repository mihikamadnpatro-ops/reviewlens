"""
ReviewLens: Streamlit web app for the review insights assistant.

Run:
    python -m streamlit run app.py
Needs logo.svg next to this file and .streamlit/config.toml for the theme.
"""
import base64
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from graph import ask

APP_NAME = "ReviewLens"
TAGLINE = "Ask your reviews anything. Get the themes, not the wall of text."

st.set_page_config(page_title=APP_NAME, page_icon="🔍", layout="wide")

# ---------- styling ----------
st.markdown(
    """
    <style>
    /* light purple Ask button with dark navy text */
    button[data-testid="stBaseButton-primary"], .stButton button[kind="primary"] {
        background-color: #C4B5FD; color: #0B1230; border: none;
        font-weight: 700; border-radius: 12px; padding: 0.5rem 1.6rem;
    }
    button[data-testid="stBaseButton-primary"]:hover, .stButton button[kind="primary"]:hover {
        background-color: #D8CEFF; color: #0B1230;
    }
    /* softer example buttons in the sidebar */
    button[data-testid="stBaseButton-secondary"] {
        border-radius: 12px; border: 1px solid #3A4380; background-color: #151D45;
    }
    .app-title { font-size: 2.6rem; font-weight: 800; margin: 0; line-height: 1.1;
        background: linear-gradient(90deg, #C4B5FD, #F9A8D4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .app-tagline { color: #A9B0DD; font-size: 1.05rem; margin-top: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_path = Path(__file__).parent / "logo.svg"
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:1.2rem;">
        <img src="data:image/svg+xml;base64,{logo_b64}" width="72" height="72"/>
        <div>
            <p class="app-title">{APP_NAME}</p>
            <p class="app-tagline">{TAGLINE}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- sidebar ----------
with st.sidebar:
    st.header("About the data")

    @st.cache_data
    def load_data():
        return pd.read_csv("reviews_sample_5000.csv")

    data = load_data()
    st.write(f"**{len(data):,}** reviews from **{data['product_id'].nunique():,}** products")
    st.write("Rating distribution")
    st.bar_chart(data["rating"].value_counts().sort_index(), color="#C4B5FD")
    st.caption("Sample balanced at 1,000 reviews per star rating.")

    st.header("Try these")
    examples = [
        "What do customers complain about regarding coffee?",
        "What do people like about the tea?",
        "What are the problems with packaging?",
        "Do customers think the snacks are good value?",
        "What do people say about the taste of the chocolate?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

# ---------- main area ----------
question = st.text_input(
    "Your question", key="question", placeholder="e.g. What do people dislike about the dog food?"
)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_ask(q: str):
    return ask(q)  # repeated questions are served from memory, no API call


if st.button("Ask", type="primary") and question.strip():
    start = time.time()
    with st.spinner("Reading through the reviews..."):
        try:
            result = cached_ask(question.strip())
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    elapsed = time.time() - start
    c1, c2, c3 = st.columns(3)
    c1.metric("Question type", result["route"].capitalize())
    c2.metric("Reviews retrieved", len(result["docs"]))
    c3.metric("Query rewrites", result["retries"])

    st.caption(f"Answered in {elapsed:.1f} seconds")
    st.subheader("Answer")
    st.markdown(result["answer"])

    with st.expander(f"Source reviews ({len(result['docs'])})"):
        for d in result["docs"]:
            st.markdown(
                f"**[{d['id']}]** {'⭐' * d['rating']} &nbsp; *{d['summary']}*  \n"
                f"{d['text']}  \n"
                f"<small>similarity: {d['score']:.2f}</small>",
                unsafe_allow_html=True,
            )
            st.divider()
