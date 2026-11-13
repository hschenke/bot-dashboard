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
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
}

/* Einheitliche Button- & Link-Button-Größe & Geometrie */
div[data-testid="stButton"] > button,
div[data-testid="stLinkButton"] > a {
    min-height: 42px !important;
    height: 42px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.85rem !important;
    border-radius: 8px !important;
    box-sizing: border-box !important;
    width: 100% !important;
    text-decoration: none !important;
    white-space: nowrap !important;
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

/* Sidebar Navigation: Vollflächig, gleichmäßig lang & touch-freundlich */
[data-testid="stSidebar"] div[data-testid="stRadio"],
[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"],
[data-testid="stSidebar"] div[role="radiogroup"] {
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 8px !important;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div,
[data-testid="stSidebar"] div[role="radiogroup"] > div {
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label,
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-width: 100% !important;
    box-sizing: border-box !important;
    min-height: 48px !important;
    padding: 10px 16px !important;
    margin-bottom: 4px !important;
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}

/* Radio-Button-Kreis vollständig ausblenden */
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div:first-child {
    display: none !important;
}

/* Text in Navi-Punkten: lesbar & sauber formatiert */
[data-testid="stSidebar"] div[role="radiogroup"] label p,
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    line-height: 1.4 !important;
    width: 100% !important;
}

/* Hover-Status */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover,
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background: rgba(255, 255, 255, 0.09) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Aktiver / Ausgewählter Navi-Punkt */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked),
[data-testid="stSidebar"] div[data-selected="true"] [data-testid="stRadioOption"] {
    background: rgba(59, 130, 246, 0.25) !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.25) !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        min-height: 52px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
    }
}
</style>
"""


def inject_global_styles() -> None:
    """Injiziert ausschließlich statisches Layout-CSS (niemals Benutzereingaben)."""
    st.markdown(_STATIC_CSS, unsafe_allow_html=True)
