"""Bidirektionale Browser-Speicher- & WebCrypto-Brücke via Components v2."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import streamlit as st

from src.config.constants import AUTH_COOKIE_MAX_AGE_SECONDS, AUTH_COOKIE_NAME
from src.ui.auth_bridge_js import AUTH_BRIDGE_CSS, AUTH_BRIDGE_HTML, AUTH_BRIDGE_JS

logger = logging.getLogger(__name__)

_PERSISTENT_ADMIN_TOKEN_RE = re.compile(
    r"^adm2:\d+:[0-9a-f]{24}:[0-9a-f]{16}:[0-9a-f]{32}:[0-9a-f]{64}$"
)
_BRIDGE_COMPONENT_NAME = "bot_dashboard_auth_storage_bridge"
_BRIDGE_MOUNT_KEY = "auth_storage_bridge"
_BRIDGE_COMPONENT: Any = None


def is_valid_admin_token_format(token: str) -> bool:
    """Prüft rein syntaktisch, ob ein String dem gerätegebundenen `adm2`-Persistenzformat entspricht."""
    return bool(token and _PERSISTENT_ADMIN_TOKEN_RE.fullmatch(token.strip()))


def parse_bridge_message(raw: str) -> dict[str, Any] | None:
    """Parst eine JSON-Antwort der Components-v2-Auth-Brücke sicher."""
    cleaned = (raw or "").strip()
    if not cleaned or len(cleaned) > 4096:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _noop_callback() -> None:
    return None


def _get_bridge_component() -> Any:
    global _BRIDGE_COMPONENT
    if _BRIDGE_COMPONENT is None and hasattr(st, "components") and hasattr(st.components, "v2"):
        _BRIDGE_COMPONENT = st.components.v2.component(
            _BRIDGE_COMPONENT_NAME,
            html=AUTH_BRIDGE_HTML,
            css=AUTH_BRIDGE_CSS,
            js=AUTH_BRIDGE_JS,
            isolate_styles=False,
        )
    return _BRIDGE_COMPONENT


def mount_auth_storage_bridge(
    action: str,
    token: str = "",
    nonce: str = "",
    *,
    challenge: str = "",
    candidate_token: str = "",
) -> str:
    """Führt Save/Clear/Bind/Restore im Browser aus und liefert die Client-Antwort."""
    clean_token = token.strip()
    if action == "save" and not is_valid_admin_token_format(clean_token):
        return ""
    clean_candidate = candidate_token.strip()
    if clean_candidate and not is_valid_admin_token_format(clean_candidate):
        clean_candidate = ""
    component = _get_bridge_component()
    if component is None:
        return ""
    payload = {
        "action": action,
        "cookieName": AUTH_COOKIE_NAME,
        "maxAge": AUTH_COOKIE_MAX_AGE_SECONDS if action == "save" else 0,
        "token": clean_token if action == "save" else "",
        "nonce": nonce,
        "challenge": challenge,
        "candidateToken": clean_candidate if action == "restore" else "",
    }
    try:
        result = component(
            key=_BRIDGE_MOUNT_KEY,
            data=payload,
            height="content",
            on_restored_change=_noop_callback,
        )
        restored = (
            result.get("restored")
            if isinstance(result, dict)
            else getattr(result, "restored", None)
        )
        return str(restored or "").strip()
    except Exception as exc:
        logger.debug("Auth-Storage-Bridge konnte nicht gemountet werden: %s", exc)
        return ""
