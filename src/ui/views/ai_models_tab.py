"""AI-Brain Tab 2 (AI): Dynamische Verwaltung aller Free- & Paid-Tier AI-Modelle."""

from __future__ import annotations

import streamlit as st
from ai_brain import (
    AIModelSpec,
    AIWebsearchBrain,
    BrainError,
    BrainTier,
    ModelProvider,
    ProbeStatus,
    estimate_blended_cost_per_1m,
    format_model_budget_label,
)

from src.ui.views.ai_sequences_tab import format_probe_badge
from src.ui.views.ai_task_overlay import invalidate_ai_ui_cache


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ai_brain_feedback"] = {"level": level, "message": message}


def render_models_tab(
    brain: AIWebsearchBrain,
    is_admin: bool,
    *,
    models: list[AIModelSpec] | None = None,
) -> None:
    """Zeigt alle registrierten AI-Modelle (Free-/Paid-Key-vorbereitet) und erlaubt dynamisches CRUD."""
    resolved = models if models is not None else brain.list_models()
    st.caption(
        "Zentrale Modell-Liste (Standardmäßig kostenlos über den Free-Tier-Key; `💳 Paid-Only` nutzt den Paid-Tier-Key erst bei `allow_paid=True`)."
    )
    rows = [
        {
            "Name": m.name,
            "Provider": m.provider.value,
            "Tier": m.tier_hint.value.capitalize(),
            "Status": format_probe_badge(m.last_status, enabled=m.enabled),
            "Latenz (ms)": m.last_latency_ms,
            "Score": m.quality_score,
            "Kosten (~$/1M)": estimate_blended_cost_per_1m(m),
            "Budget": format_model_budget_label(m),
            "Diagnose": m.last_error or "–",
        }
        for m in resolved
    ]
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Tier": st.column_config.TextColumn("Tier", width="small"),
            "Latenz (ms)": st.column_config.NumberColumn("Latenz", format="%d ms", width="small", help="Antwortzeit des letzten Live-Tests in Millisekunden."),
            "Score": st.column_config.NumberColumn("Score", format="%.1f", width="small", help="Qualitätsscore 0–100 (höher = besser)."),
            "Kosten (~$/1M)": st.column_config.NumberColumn("Kosten (~$/1M)", format="$%.2f", width="small", help="Geschätzte Mischkosten je 1 Mio. Tokens (USD, 75 % Input + 25 % Output)."),
            "Budget": st.column_config.TextColumn("Budget", width="small", help="🆓 Free Tier / 🆓 $0 Forever standardmäßig kostenlos aktiv; 💳 Paid-Only nur bei Opt-In (`allow_paid=True`)."),
        },
    )

    if any(m.last_status == ProbeStatus.AUTH_MISSING for m in resolved):
        st.caption("ℹ️ Fehlende API-Keys (Free-Tier & Paid-Tier) werden zentral im Tab **🔐 Vault** verwaltet.")

    if not is_admin:
        return

    st.subheader("⚙️ Modell aktivieren, bearbeiten oder entfernen")
    model_ids = [m.id for m in resolved]
    if model_ids:
        sel_col, toggle_col, del_col = st.columns([2, 1, 1], vertical_alignment="bottom")
        with sel_col:
            selected_id = st.selectbox("Modell wählen", options=model_ids, key="ai_model_quick_sel")
        selected_obj = next(m for m in resolved if m.id == selected_id)
        with toggle_col:
            do_toggle = st.button("⛔ Deaktivieren" if selected_obj.enabled else "✅ Aktivieren", key="ai_model_toggle_btn", width="stretch")
        with del_col:
            do_delete = st.button("🗑️ Entfernen", key="ai_model_delete_btn", width="stretch")

        if do_toggle:
            brain.set_model_enabled(selected_id, not selected_obj.enabled)
            invalidate_ai_ui_cache()
            _set_feedback("success", f"Modell '{selected_id}' aktualisiert.")
            st.rerun()
        if do_delete:
            brain.remove_model(selected_id)
            invalidate_ai_ui_cache()
            _set_feedback("success", f"Modell '{selected_id}' aus dem Katalog entfernt.")
            st.rerun()

    with st.expander("➕ Neues AI-Modell manuell eintragen oder bearbeiten", expanded=False):
        edit_options = ["➕ Neues Modell hinzufügen", *model_ids]
        target_mode = st.selectbox("Vorlage / Bearbeitung", options=edit_options, key="ai_model_edit_target")
        existing = next((m for m in resolved if m.id == target_mode), None)

        with st.form(f"ai_model_crud_form_{target_mode}"):
            c1, c2 = st.columns(2)
            with c1:
                m_id = st.text_input("Modell-ID (eindeutig)", value=existing.id if existing else "", placeholder="z. B. groq-llama-3.3-70b")
                m_name = st.text_input("Anzeigename", value=existing.name if existing else "", placeholder="z. B. Groq Llama 3.3 70B")
                m_provider = st.selectbox("Provider", options=list(ModelProvider), index=list(ModelProvider).index(existing.provider) if existing else 0, format_func=lambda p: p.value)
                m_model_id = st.text_input("API Model-Slug", value=existing.model_id if existing else "", placeholder="llama-3.3-70b-versatile")
            with c2:
                m_tier = st.selectbox("Ziel-Stufe (Tier Hint)", options=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP], index=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP].index(existing.tier_hint) if existing else 0, format_func=lambda t: t.value.capitalize())
                m_key_env = st.text_input("Free-Tier API-Key Env", value=existing.api_key_env if existing else "GEMINI_FREE_API_KEY")
                m_paid_key_env = st.text_input("Paid-Tier API-Key Env (optional)", value=existing.paid_api_key_env if existing else "GEMINI_PAID_API_KEY")
                m_base_url = st.text_input("Base-URL (für OpenAI-Compat)", value=existing.base_url if existing else "", placeholder="https://api.groq.com/openai/v1")
            c_rpm, c_notes = st.columns([1, 2])
            with c_rpm:
                m_rpm = st.number_input("Max RPM", min_value=1, max_value=240, value=existing.max_rpm if existing else 15)
            with c_notes:
                m_notes = st.text_input("Notiz", value=existing.notes if existing else "")
            c3, c4, c5 = st.columns(3)
            with c3:
                m_struct = st.checkbox("Structured JSON Output", value=existing.supports_structured_output if existing else True)
            with c4:
                m_is_free = st.checkbox("Kostenlos im Free-Tier (ohne allow_paid)", value=existing.is_free if existing else True)
            with c5:
                m_enabled = st.checkbox("Aktiviert", value=existing.enabled if existing else True)
            submitted = st.form_submit_button("💾 Modell speichern", width="stretch")

        if submitted:
            try:
                spec = AIModelSpec(
                    id=m_id.strip().lower(),
                    name=m_name.strip() or m_id.strip(),
                    provider=m_provider,
                    model_id=m_model_id.strip() or m_id.strip(),
                    tier_hint=m_tier,
                    is_free=bool(m_is_free),
                    free_quota_limit=existing.free_quota_limit if existing else 0,
                    free_quota_period="unlimited" if m_is_free else "none",
                    free_quota_unit=existing.free_quota_unit if existing else "requests",
                    requires_api_key=bool(m_key_env.strip() or m_paid_key_env.strip()),
                    api_key_env=m_key_env.strip(),
                    paid_api_key_env=m_paid_key_env.strip(),
                    base_url=m_base_url.strip(),
                    supports_structured_output=bool(m_struct),
                    supports_google_search=(m_provider == ModelProvider.GOOGLE_GEMINI),
                    supports_url_context=(m_provider == ModelProvider.GOOGLE_GEMINI),
                    is_deep_agent=(m_provider == ModelProvider.GEMINI_DEEP_AGENT),
                    max_rpm=int(m_rpm),
                    enabled=bool(m_enabled),
                    source="custom" if existing is None else existing.source,
                    origin_apps=existing.origin_apps if existing else ["manual"],
                    notes=m_notes.strip(),
                    input_cost_per_1m=existing.input_cost_per_1m if existing else None,
                    output_cost_per_1m=existing.output_cost_per_1m if existing else None,
                )
                saved = brain.add_or_update_model(spec)
                invalidate_ai_ui_cache()
                _set_feedback("success", f"Modell '{saved.id}' gespeichert.")
                st.rerun()
            except (BrainError, ValueError) as exc:
                _set_feedback("error", f"Ungültiges Modell: {exc}")
                st.rerun()
