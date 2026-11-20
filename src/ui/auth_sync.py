"""Browser-Storage- & PoP-Synchronisation für die Streamlit-Authentifizierung."""

from __future__ import annotations

import hmac
import secrets
from collections.abc import Callable

import streamlit as st

from src.config.constants import ADMIN_TOKEN_MAX_AGE_SECONDS
from src.config.settings import AppSettings
from src.security.auth import generate_admin_token, parse_admin_token, verify_admin_token
from src.security.device_pop import compute_device_key_tag, verify_device_pop_signature
from src.storage.auth_sessions import get_auth_session_store
from src.ui.auth_storage import is_valid_admin_token_format, parse_bridge_message

BridgeFn = Callable[..., str]


def schedule_storage_sync(action: str) -> None:
    """Plant eine asynchrone Browser-Speicher-Aktion für den nächsten Renderlauf."""
    st.session_state["_pending_storage_action"] = action
    st.session_state["_storage_sync_nonce"] = secrets.token_hex(4)


def _handle_bind_response(
    settings: AppSettings,
    session_token: str,
    raw_msg: str,
    client_fp: str,
) -> None:
    msg = parse_bridge_message(raw_msg)
    if msg is None or msg.get("kind") != "bind":
        return

    expected_challenge = str(st.session_state.pop("_auth_bind_challenge", "") or "")
    st.session_state.pop("_auth_bind_nonce", None)
    st.session_state.pop("_pending_storage_action", None)
    if msg.get("unsupported") or not expected_challenge:
        return

    msg_challenge = str(msg.get("challenge", "") or "")
    pubkey = str(msg.get("pubkey", "") or "")
    signature = str(msg.get("signature", "") or "")
    if not (
        msg_challenge
        and hmac.compare_digest(msg_challenge, expected_challenge)
        and verify_device_pop_signature(pubkey, signature, f"bind:{expected_challenge}")
    ):
        schedule_storage_sync("clear")
        return

    device_tag = compute_device_key_tag(pubkey)
    parsed = parse_admin_token(session_token)
    if not device_tag or parsed is None:
        return

    bound_token = generate_admin_token(
        settings.app_password,
        client_fingerprint=client_fp,
        device_key_tag=device_tag,
        session_id=parsed.session_id,
    )
    bound_parts = parse_admin_token(bound_token)
    if bound_parts is None:
        return

    get_auth_session_store(settings).register_session(
        session_id=bound_parts.session_id,
        fp_tag=bound_parts.fp_tag,
        device_key_tag=device_tag,
        issued_at=bound_parts.issued_at,
        expires_at=bound_parts.issued_at + ADMIN_TOKEN_MAX_AGE_SECONDS,
    )
    st.session_state["auth_token"] = bound_token
    schedule_storage_sync("save")
    st.rerun()


def sync_authenticated_storage(
    settings: AppSettings,
    token: str,
    client_fp: str,
    bridge_fn: BridgeFn,
) -> None:
    """Bindet Geräte-Keys bei Remember-Me oder führt geplante Save/Clear-Syncs aus."""
    if st.session_state.get("_pending_storage_action") == "bind" and st.session_state.get(
        "_auth_bind_challenge"
    ):
        bind_nonce = str(st.session_state.setdefault("_auth_bind_nonce", secrets.token_hex(4)))
        raw_msg = bridge_fn(
            action="bind",
            nonce=bind_nonce,
            challenge=str(st.session_state["_auth_bind_challenge"]),
        )
        _handle_bind_response(settings, token, raw_msg, client_fp)
        return

    pending = st.session_state.pop("_pending_storage_action", None)
    sync_nonce = str(st.session_state.pop("_storage_sync_nonce", "") or "")
    if pending == "save" and is_valid_admin_token_format(token):
        bridge_fn(action="save", token=token, nonce=sync_nonce)
    elif pending == "clear":
        bridge_fn(action="clear", nonce=sync_nonce)
    else:
        bridge_fn(action="noop")


def _verify_restore_payload(
    settings: AppSettings,
    msg: dict[str, object],
    expected_challenge: str,
    client_fp: str,
) -> str:
    if msg.get("kind") != "restore" or msg.get("missing_key") or not expected_challenge:
        return ""
    msg_challenge = str(msg.get("challenge", "") or "")
    token = str(msg.get("token", "") or "").strip()
    pubkey = str(msg.get("pubkey", "") or "")
    signature = str(msg.get("signature", "") or "")
    if not (
        msg_challenge
        and hmac.compare_digest(msg_challenge, expected_challenge)
        and verify_admin_token(
            token, settings.app_password, client_fingerprint=client_fp, allow_session_only=False
        )
    ):
        return ""

    parsed = parse_admin_token(token)
    if parsed is None or compute_device_key_tag(pubkey) != parsed.device_key_tag:
        return ""
    if not verify_device_pop_signature(
        pubkey, signature, f"restore:{expected_challenge}:{parsed.session_id}"
    ):
        return ""

    store = get_auth_session_store(settings)
    if not store.is_session_active(
        parsed.session_id, parsed.issued_at, require_registered=store.is_remote_persistent
    ):
        return ""
    store.touch_session(parsed.session_id)
    return token


def sync_unauthenticated_storage(
    settings: AppSettings,
    client_fp: str,
    cookie_candidate: str,
    bridge_fn: BridgeFn,
) -> None:
    """Prüft gespeicherte Tokens per WebCrypto-Challenge-Response und stellt sie sicher wieder her."""
    pending = st.session_state.pop("_pending_storage_action", None)
    sync_nonce = str(st.session_state.pop("_storage_sync_nonce", "") or "")
    if pending == "clear":
        bridge_fn(action="clear", nonce=sync_nonce or secrets.token_hex(4))
        return

    restore_nonce = str(st.session_state.setdefault("_auth_restore_nonce", secrets.token_hex(4)))
    challenge = str(st.session_state.setdefault("_auth_restore_challenge", secrets.token_hex(16)))
    raw = bridge_fn(
        action="restore",
        nonce=restore_nonce,
        challenge=challenge,
        candidate_token=cookie_candidate,
    )
    if not raw:
        return

    st.session_state.pop("_auth_restore_nonce", None)
    expected_challenge = str(st.session_state.pop("_auth_restore_challenge", "") or "")
    msg = parse_bridge_message(raw)
    verified = _verify_restore_payload(settings, msg, expected_challenge, client_fp) if msg else ""
    if verified:
        st.session_state["auth_token"] = verified
        st.session_state["remember_me"] = True
        st.session_state.pop("_ignore_context_cookie", None)
        schedule_storage_sync("save")
    else:
        schedule_storage_sync("clear")
        st.session_state["_auth_notice"] = (
            "Gespeicherte Sitzung ist ungültig, abgemeldet oder nicht für dieses Gerät autorisiert."
        )
    st.rerun()
