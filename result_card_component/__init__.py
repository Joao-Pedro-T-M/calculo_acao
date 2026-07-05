from pathlib import Path
import html
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


def load_styles():
    css = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_asset_card(symbol, name, exchange, typedisp, currency="—"):
    template = (FRONTEND_DIR / "card.html").read_text(encoding="utf-8")
    return template.format(
        symbol=html.escape(symbol),
        name=html.escape(name),
        exchange=html.escape(exchange or "—"),
        typedisp=html.escape(typedisp or "—"),
        currency=html.escape(currency or "—"),
    )