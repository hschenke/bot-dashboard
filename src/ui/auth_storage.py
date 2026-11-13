"""Bidirektionale Browser-Speicher-Brücke (localStorage & Cookie) via Components v2."""

from __future__ import annotations

import logging
import re
from typing import Any

import streamlit as st

from src.config.constants import AUTH_COOKIE_MAX_AGE_SECONDS, AUTH_COOKIE_NAME

logger = logging.getLogger(__name__)

_ADMIN_TOKEN_RE = re.compile(r"^adm:\d+:[0-9a-f]{64}$")
_BRIDGE_COMPONENT_NAME = "bot_dashboard_auth_storage_bridge"
_BRIDGE_MOUNT_KEY = "auth_storage_bridge"

_BRIDGE_HTML = '<div class="bot-dash-auth-bridge" aria-hidden="true"></div>'
_BRIDGE_CSS = ".bot-dash-auth-bridge { display: none !important; height: 0 !important; }"

_BRIDGE_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const TOKEN_RE = /^adm:\\d+:[0-9a-f]{64}$/;

  try {
    const container = parentElement?.closest?.('[data-testid="stElementContainer"]');
    if (container) {
      container.style.display = "none";
      container.style.margin = "0";
      container.style.padding = "0";
      container.style.height = "0";
    }
  } catch (_) {}

  const isValidToken = (value) => typeof value === "string" && TOKEN_RE.test(value.trim());

  const getAccessibleWindows = () => {
    const wins = [];
    for (const candidate of [window, window.parent, window.top]) {
      try {
        if (candidate && candidate.document && !wins.includes(candidate)) {
          wins.push(candidate);
        }
      } catch (_) {}
    }
    return wins;
  };

  const readCookie = (doc, name) => {
    try {
      const prefix = `${name}=`;
      for (const part of String(doc.cookie || "").split(";")) {
        const trimmed = part.trim();
        if (trimmed.startsWith(prefix)) {
          const raw = trimmed.slice(prefix.length).trim();
          try { return decodeURIComponent(raw).trim(); } catch (_) { return raw; }
        }
      }
    } catch (_) {}
    return "";
  };

  const writeCookie = (doc, win, name, value, maxAgeSeconds) => {
    try {
      const secureAttr = win?.location?.protocol === "https:" ? "; Secure" : "";
      if (maxAgeSeconds > 0 && value) {
        const expires = new Date(Date.now() + maxAgeSeconds * 1000).toUTCString();
        doc.cookie = `${name}=${value}; max-age=${maxAgeSeconds}; expires=${expires}; path=/; SameSite=Lax${secureAttr}`;
      } else {
        doc.cookie = `${name}=; max-age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax${secureAttr}`;
        doc.cookie = `${name}=; max-age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
      }
    } catch (_) {}
  };

  const saveToken = (name, token, maxAgeSeconds) => {
    for (const win of getAccessibleWindows()) {
      try { win.localStorage?.setItem(name, token); } catch (_) {}
      writeCookie(win.document, win, name, token, maxAgeSeconds);
    }
  };

  const clearToken = (name) => {
    for (const win of getAccessibleWindows()) {
      try { win.localStorage?.removeItem(name); } catch (_) {}
      writeCookie(win.document, win, name, "", 0);
    }
  };

  const readToken = (name) => {
    const wins = getAccessibleWindows();
    for (const win of wins) {
      try {
        const stored = String(win.localStorage?.getItem(name) || "").trim();
        if (isValidToken(stored)) return stored;
      } catch (_) {}
    }
    for (const win of wins) {
      const cookieVal = readCookie(win.document, name);
      if (isValidToken(cookieVal)) return cookieVal;
    }
    return "";
  };

  const action = String(data?.action || "noop");
  const cookieName = String(data?.cookieName || "bot_dashboard_auth_token");
  const maxAgeSeconds = Number(data?.maxAge || 0);
  const token = String(data?.token || "").trim();
  const nonce = String(data?.nonce || "");

  if (action === "save" && isValidToken(token)) {
    saveToken(cookieName, token, maxAgeSeconds);
  } else if (action === "clear") {
    clearToken(cookieName);
    window.__botDashRestoreNonce = "";
  } else if (action === "restore") {
    if (nonce && window.__botDashRestoreNonce === nonce) return () => {};
    const saved = readToken(cookieName);
    if (saved) {
      window.__botDashRestoreNonce = nonce || "restored";
      setTriggerValue("restored", saved);
    }
  }
  return () => {};
}
"""

_BRIDGE_COMPONENT: Any = None


def is_valid_admin_token_format(token: str) -> bool:
    """Prüft rein syntaktisch, ob ein String dem signierten `adm:<ts>:<hex>`-Format entspricht."""
    return bool(token and _ADMIN_TOKEN_RE.fullmatch(token.strip()))


def _noop_callback() -> None:
    return None


def _get_bridge_component() -> Any:
    global _BRIDGE_COMPONENT
    if _BRIDGE_COMPONENT is None and hasattr(st, "components") and hasattr(st.components, "v2"):
        _BRIDGE_COMPONENT = st.components.v2.component(
            _BRIDGE_COMPONENT_NAME,
            html=_BRIDGE_HTML,
            css=_BRIDGE_CSS,
            js=_BRIDGE_JS,
            isolate_styles=False,
        )
    return _BRIDGE_COMPONENT


def mount_auth_storage_bridge(action: str, token: str = "", nonce: str = "") -> str:
    """Führt Save/Clear/Restore im Browser aus und liefert ggf. ein wiederhergestelltes Token."""
    clean_token = token.strip()
    if action == "save" and not is_valid_admin_token_format(clean_token):
        return ""
    component = _get_bridge_component()
    if component is None:
        return ""
    payload = {
        "action": action,
        "cookieName": AUTH_COOKIE_NAME,
        "maxAge": AUTH_COOKIE_MAX_AGE_SECONDS if action == "save" else 0,
        "token": clean_token if action == "save" else "",
        "nonce": nonce,
    }
    try:
        result = component(
            key=_BRIDGE_MOUNT_KEY,
            data=payload,
            height="content",
            on_restored_change=_noop_callback,
        )
        restored = result.get("restored") if isinstance(result, dict) else getattr(result, "restored", None)
        return str(restored or "").strip()
    except Exception as exc:
        logger.debug("Auth-Storage-Bridge konnte nicht gemountet werden: %s", exc)
        return ""
