"""Statisches Theme-CSS für eine moderne Desktop- & Smartphone-Darstellung."""

from __future__ import annotations

import streamlit as st

_STATIC_CSS = """
<style>
.block-container {
    padding-top: 3.5rem;
    padding-bottom: 2rem;
    max-width: 1240px;
}
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 12px;
    padding: 8px 14px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.45rem !important; }
.st-key-ki_kpi_row div[data-testid="stMetric"] { padding: 4px 10px !important; border-radius: 8px !important; min-height: 0 !important; }
.st-key-ki_kpi_row div[data-testid="stMetric"] [data-testid="stMetricLabel"] p { font-size: 0.74rem !important; line-height: 1.15 !important; }
.st-key-ki_kpi_row div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.05rem !important; line-height: 1.2 !important; }

/* Einheitliche Button-, Link- & Selectbox-Höhe für saubere horizontale Linien */
div[data-testid="stButton"] > button,
div[data-testid="stLinkButton"] > a {
    min-height: 42px !important;
    height: 42px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.6rem !important;
    border-radius: 8px !important;
    box-sizing: border-box !important;
    width: 100% !important;
    text-decoration: none !important;
    white-space: nowrap !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 42px !important;
}

/* Kompakte Feedback-Banner ohne vertikales Aufblähen */
div[data-testid="stAlert"] {
    margin-bottom: 0.4rem !important;
}
div[data-testid="stAlert"] > div {
    padding: 0.55rem 0.9rem !important;
    border-radius: 10px !important;
    min-height: 42px !important;
}
div[data-testid="stAlert"] p {
    margin: 0 !important;
    font-size: 0.92rem !important;
    line-height: 1.35 !important;
}
div[data-testid="stSpinner"] {
    margin: 0.25rem 0 !important;
}

/* Kompakte App-Cards im Dashboard */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.9rem 1.1rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] ul {
    margin: 0.35rem 0 !important;
    padding-left: 1.15rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] li {
    margin: 0.2rem 0 !important;
    line-height: 1.45 !important;
    font-size: 0.92rem !important;
}

/* Sidebar Navigation: Exakte vertikale Ausrichtung aller Icons & Labels */
section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button {
    min-height: 48px !important;
    height: 48px !important;
    padding: 10px 16px !important;
    margin-bottom: 4px !important;
    border-radius: 10px !important;
    justify-content: flex-start !important;
    align-items: center !important;
    text-align: left !important;
    font-size: 1.02rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease-in-out !important;
}
section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button > div,
section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button > div > div {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    gap: 12px !important;
}
section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button :is([data-testid="stIconEmoji"], [data-testid="stIconMaterial"]) {
    width: 24px !important;
    min-width: 24px !important;
    height: 24px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.15rem !important;
    line-height: 1 !important;
    flex-shrink: 0 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button p {
    font-size: 1.02rem !important;
    font-weight: 500 !important;
    line-height: 1.35 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}
section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.09) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}
section[data-testid="stSidebar"] [class*="st-key-nav_page_"] div[data-testid="stButton"] > button[kind="primary"] {
    background: rgba(59, 130, 246, 0.25) !important;
    border: 1px solid #3b82f6 !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.25) !important;
    color: #ffffff !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }
    section[data-testid="stSidebar"] :is([class*="st-key-nav_page_"], [class*="st-key-nav_logout_btn"]) div[data-testid="stButton"] > button {
        min-height: 52px !important;
        height: 52px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
    }
}

/* Kompakte, horizontal scrollbare Markdown-Tabellen (z. B. Health-Übersicht) */
div[data-testid="stTable"],
div[data-testid="stTable"] > div {
    overflow-x: auto !important;
    max-width: 100% !important;
}
div[data-testid="stElementContainer"]:has(.bot-dash-health-marker)
    + div[data-testid="stElementContainer"]
    table[data-testid="stTableStyledTable"] :is(th, td, [data-testid="stMarkdownContainer"] p) {
    white-space: nowrap !important;
}

/* Unsichtbare Marker & Auth-Storage-Bridge ohne Layout-Abstand */
div[data-testid="stElementContainer"]:has(:is(.bot-dash-health-marker, .bot-dash-auth-bridge)) {
    display: none !important; margin: 0 !important; padding: 0 !important; height: 0 !important;
}
div[data-testid="stElementContainer"]:has(:is(.ki-task-backdrop, #ki-brain-instant-backdrop)) {
    height: 0 !important; margin: 0 !important; padding: 0 !important;
}

/* KI-View Task-Overlay: blockierter Hintergrund + lesbare Fortschritts-Box im Vordergrund */
.ki-task-backdrop,
#ki-brain-instant-backdrop.active {
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    background-color: rgba(15, 23, 42, 0.74);
    backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px);
    pointer-events: all; cursor: wait;
}
.ki-task-backdrop { z-index: 999990; }
#ki-brain-instant-backdrop { display: none; z-index: 999985; align-items: center; justify-content: center; }
#ki-brain-instant-backdrop.active { display: flex; }
body:has(.st-key-ki_task_overlay) #ki-brain-instant-backdrop .ki-instant-card { display: none; }
.ki-instant-card {
    width: min(500px, 90vw); background: rgba(15, 23, 42, 0.96); color: #e2e8f0;
    border: 1px solid rgba(56, 189, 248, 0.45); border-radius: 14px; padding: 1.1rem 1.35rem;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.7);
}
.st-key-ki_task_overlay {
    position: fixed !important; top: 50% !important; left: 50% !important;
    transform: translate(-50%, -50%) !important; width: min(560px, 92vw) !important; z-index: 1000001 !important;
    background: rgba(15, 23, 42, 0.96) !important; border: 1px solid rgba(56, 189, 248, 0.45) !important;
    border-radius: 14px !important; padding: 1.15rem 1.35rem !important;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.7) !important; pointer-events: all !important;
}
</style>
"""


def inject_global_styles() -> None:
    """Injiziert ausschließlich statisches Layout-CSS (niemals Benutzereingaben)."""
    st.markdown(_STATIC_CSS, unsafe_allow_html=True)
