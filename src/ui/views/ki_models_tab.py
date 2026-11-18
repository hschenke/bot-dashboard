"""KI-View Tab 2: Dynamische Verwaltung aller kostenfreien KI-Modelle."""

from __future__ import annotations

import streamlit as st
from ki_brain import (
    AIModelSpec,
    BrainError,
    BrainTier,
    KIWebsearchBrain,
    ModelProvider,
    SecretStatusEntry,
    estimate_blended_cost_per_1m,
    format_model_budget_label,
    is_preview_or_experimental_model,
)

from src.ui.views.ki_secret_dialog import render_missing_keys_action_bar
from src.ui.views.ki_sequences_tab import format_probe_badge
from src.ui.views.ki_task_overlay import (
    filter_retryable_excluded_models,
    format_discover_feedback,
    invalidate_ki_ui_cache,
    run_with_task_overlay,
)


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _render_model_maintenance_bar(
    brain: KIWebsearchBrain,
    resolved: list[AIModelSpec],
    *,
    excluded_model_ids: tuple[str, ...],
    excluded_model_reasons: dict[str, str] | None,
    exclude_previews: bool,
) -> None:
    preview_models = [m for m in resolved if is_preview_or_experimental_model(m.model_id, m.name)]
    retry_targets = filter_retryable_excluded_models(
        excluded_model_ids, excluded_model_reasons, exclude_previews=exclude_previews
    )
    c_clean, c_retry = st.columns(2)
    with c_clean:
        do_clean = st.button(
            f"🧹 Clean Previews ({len(preview_models)})", key="ki_purge_previews_btn", disabled=not preview_models,
            use_container_width=True,
            help=(f"Entfernt {len(preview_models)} Preview-/Experimental-Modell(e) dauerhaft (Deep Research Agent bleibt erhalten)." if preview_models else "Keine Preview-/Experimental-Modelle vorhanden (Deep Research Agent bleibt erhalten)."),
        )
    with c_retry:
        do_retry = st.button(
            f"🔄 Retry Excluded ({len(retry_targets)})", key="ki_retry_excluded_btn", disabled=not retry_targets,
            use_container_width=True,
            help=(f"Prüft {len(retry_targets)} ausgeschlossene Modell(e) inkl. früherer Quota-429-Einträge erneut und aktiviert funktionierende Modelle." if retry_targets else "Aktuell keine ausgeschlossenen Modelle zum erneuten Prüfen vorhanden."),
        )
    # Overlay, Cache-Invalidierung & Feedback strikt außerhalb schmaler Spalten
    if do_clean:
        removed = brain.remove_preview_models()
        invalidate_ki_ui_cache()
        _set_feedback("success" if removed else "info", f"Clean Previews: {len(removed)} Modell(e) entfernt ({', '.join(removed) or '–'}).")
        st.rerun()
    if do_retry:
        try:
            summary = run_with_task_overlay(
                "🔄 Retry Excluded", "Prüfe ausgeschlossene & Quota-429-Modelle erneut...",
                lambda cb: brain.auto_sync(verify_discovered=True, exclude_previews=exclude_previews, retry_excluded=True, on_progress=cb),
            )
            _set_feedback(*format_discover_feedback(summary, prefix="Retry Excluded fertig"))
        except BrainError as exc:
            _set_feedback("error", f"Retry Excluded fehlgeschlagen: {exc}")
        invalidate_ki_ui_cache()
        st.rerun()


def render_models_tab(
    brain: KIWebsearchBrain,
    is_admin: bool,
    *,
    models: list[AIModelSpec] | None = None,
    secret_statuses: list[SecretStatusEntry] | None = None,
    active_tab_label: str | None = None,
    excluded_model_ids: tuple[str, ...] = (),
    excluded_model_reasons: dict[str, str] | None = None,
    exclude_previews: bool = True,
) -> None:
    """Zeigt alle registrierten KI-Modelle und erlaubt dynamisches CRUD."""
    resolved = models if models is not None else brain.list_models()
    st.caption("Zentrale Modell-Liste (Standard-Sequencing strikt 0% Kosten; `💳 Paid-Only` nur mit Opt-In `allow_paid=True`).")
    rows = [
        {
            "Name": m.name, "Provider": m.provider.value, "Tier": m.tier_hint.value.capitalize(),
            "Status": format_probe_badge(m.last_status, enabled=m.enabled), "Latenz (ms)": m.last_latency_ms,
            "Score": m.quality_score, "Kosten (~$/1M)": estimate_blended_cost_per_1m(m),
            "Budget": format_model_budget_label(m), "Diagnose": m.last_error or "–",
        }
        for m in resolved
    ]
    st.dataframe(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "Tier": st.column_config.TextColumn("Tier", width="small"),
            "Latenz (ms)": st.column_config.NumberColumn("Latenz", format="%d ms", width="small", help="Antwortzeit des letzten Live-Tests in Millisekunden."),
            "Score": st.column_config.NumberColumn("Score", format="%.1f", width="small", help="Qualitätsscore 0–100 (höher = besser): 45 % Latenz + 55 % Tier-Stärke (Fast 76 / Standard 84 / Deep 92) + 6 JSON-Bonus + Kapazitätsausgleich. Fixwerte: Untested=55, Quota 429=12, Error/Blocked=5, Missing Key=0."),
            "Kosten (~$/1M)": st.column_config.NumberColumn("Kosten (~$/1M)", format="$%.2f", width="small", help="Geschätzte Mischkosten je 1 Mio. Tokens (USD, 75 % Input + 25 % Output; $0.00 = Free/Gemma). Im Free Tier entstehen $0.00; relevant für Rate/Opt-In."),
            "Budget": st.column_config.TextColumn("Budget", width="small", help="🆓 Free Tier / 🆓 $0 Forever im 0%-Standard aktiv; 💳 Paid-Only nur mit Opt-In (`allow_paid=True`)."),
        },
    )

    model_envs = {m.api_key_env.strip().upper() for m in resolved if m.api_key_env.strip()}
    render_missing_keys_action_bar(brain, is_admin=is_admin, env_filter=model_envs, section_key="models", statuses=secret_statuses, active_tab_label=active_tab_label)
    if not is_admin:
        return

    _render_model_maintenance_bar(brain, resolved, excluded_model_ids=excluded_model_ids, excluded_model_reasons=excluded_model_reasons, exclude_previews=exclude_previews)

    st.subheader("⚙️ Modell aktivieren, bearbeiten oder entfernen")
    model_ids = [m.id for m in resolved]
    if model_ids:
        sel_col, toggle_col, del_col = st.columns([2, 1, 1], vertical_alignment="bottom")
        with sel_col:
            selected_id = st.selectbox("Modell wählen", options=model_ids, key="ki_model_quick_sel")
        selected_obj = next(m for m in resolved if m.id == selected_id)
        with toggle_col:
            do_toggle = st.button("⛔ Deaktivieren" if selected_obj.enabled else "✅ Aktivieren", key="ki_model_toggle_btn", use_container_width=True)
        with del_col:
            do_delete = st.button("🗑️ Entfernen", key="ki_model_delete_btn", use_container_width=True)

        if do_toggle:
            brain.set_model_enabled(selected_id, not selected_obj.enabled)
            invalidate_ki_ui_cache()
            _set_feedback("success", f"Modell '{selected_id}' aktualisiert.")
            st.rerun()
        if do_delete:
            brain.remove_model(selected_id)
            invalidate_ki_ui_cache()
            _set_feedback("success", f"Modell '{selected_id}' aus dem Katalog entfernt.")
            st.rerun()

    with st.expander("➕ Neues Modell / KI manuell eintragen oder bearbeiten", expanded=False):
        edit_options = ["➕ Neues Modell hinzufügen", *model_ids]
        target_mode = st.selectbox("Vorlage / Bearbeitung", options=edit_options, key="ki_model_edit_target")
        existing = next((m for m in resolved if m.id == target_mode), None)

        with st.form(f"ki_model_crud_form_{target_mode}"):
            c1, c2 = st.columns(2)
            with c1:
                m_id = st.text_input("Modell-ID (eindeutig)", value=existing.id if existing else "", placeholder="z. B. groq-llama-3.3-70b")
                m_name = st.text_input("Anzeigename", value=existing.name if existing else "", placeholder="z. B. Groq Llama 3.3 70B")
                m_provider = st.selectbox("Provider", options=list(ModelProvider), index=list(ModelProvider).index(existing.provider) if existing else 0, format_func=lambda p: p.value)
                m_model_id = st.text_input("API Model-Slug", value=existing.model_id if existing else "", placeholder="llama-3.3-70b-versatile")
            with c2:
                m_tier = st.selectbox("Ziel-Stufe (Tier Hint)", options=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP], index=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP].index(existing.tier_hint) if existing else 0, format_func=lambda t: t.value.capitalize())
                m_key_env = st.text_input("API-Key Env-Variable", value=existing.api_key_env if existing else "GEMINI_API_KEY")
                m_base_url = st.text_input("Base-URL (für OpenAI-Compat)", value=existing.base_url if existing else "", placeholder="https://api.groq.com/openai/v1")
                m_rpm = st.number_input("Max RPM", min_value=1, max_value=240, value=existing.max_rpm if existing else 15)
            m_notes = st.text_input("Notiz", value=existing.notes if existing else "")
            c3, c4, c5 = st.columns(3)
            with c3:
                m_struct = st.checkbox("Structured JSON Output", value=existing.supports_structured_output if existing else True)
            with c4:
                m_is_free = st.checkbox("Free-Tier / 0%-Kosten (ohne allow_paid)", value=existing.is_free if existing else True)
            with c5:
                m_enabled = st.checkbox("Aktiviert", value=existing.enabled if existing else True)
            submitted = st.form_submit_button("💾 Modell speichern", use_container_width=True)

        if submitted:
            try:
                spec = AIModelSpec(
                    id=m_id.strip().lower(), name=m_name.strip() or m_id.strip(), provider=m_provider,
                    model_id=m_model_id.strip() or m_id.strip(), tier_hint=m_tier,
                    is_free=bool(m_is_free),
                    free_quota_limit=existing.free_quota_limit if existing else 0,
                    free_quota_period="unlimited" if m_is_free else "none",
                    free_quota_unit=existing.free_quota_unit if existing else "requests",
                    requires_api_key=bool(m_key_env.strip()), api_key_env=m_key_env.strip(), base_url=m_base_url.strip(),
                    supports_structured_output=bool(m_struct),
                    supports_google_search=(m_provider == ModelProvider.GOOGLE_GEMINI),
                    supports_url_context=(m_provider == ModelProvider.GOOGLE_GEMINI),
                    is_deep_agent=(m_provider == ModelProvider.GEMINI_DEEP_AGENT), max_rpm=int(m_rpm),
                    enabled=bool(m_enabled), source="custom" if existing is None else existing.source,
                    origin_apps=existing.origin_apps if existing else ["manual"], notes=m_notes.strip(),
                    input_cost_per_1m=existing.input_cost_per_1m if existing else None,
                    output_cost_per_1m=existing.output_cost_per_1m if existing else None,
                )
                saved = brain.add_or_update_model(spec)
                invalidate_ki_ui_cache()
                _set_feedback("success", f"Modell '{saved.id}' gespeichert.")
                st.rerun()
            except (BrainError, ValueError) as exc:
                _set_feedback("error", f"Ungültiges Modell: {exc}")
                st.rerun()
