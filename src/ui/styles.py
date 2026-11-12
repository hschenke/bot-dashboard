"""Statisches Theme-CSS für eine moderne Desktop- & Smartphone-Darstellung."""

from __future__ import annotations

import streamlit as st

_STATIC_CSS = """
<style>
.block-container {
    padding-top: 4.5rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 12px;
    padding: 12px 16px;
}
</style>
"""


def inject_global_styles() -> None:
    """Injiziert ausschließlich statisches Layout-CSS (niemals Benutzereingaben)."""
    st.markdown(_STATIC_CSS, unsafe_allow_html=True)
