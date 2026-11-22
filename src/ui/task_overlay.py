"""Gemeinsames Task-Overlay: grauer Blur-Backdrop im Hintergrund & Live-Status im Vordergrund."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TypeVar

import streamlit as st

_T = TypeVar("_T")

INSTANT_OVERLAY_HTML = """
<div id="ai-brain-instant-backdrop">
  <div class="ai-instant-card">
    <strong>⚙️ Running task…</strong>
    <div id="ai-instant-sub" style="font-size:0.9rem;color:#94a3b8;margin-top:0.35rem;">Please wait — locking background.</div>
  </div>
</div>
<script>
(function () {
  if (window.__aiBrainOverlayBound) return;
  window.__aiBrainOverlayBound = true;
  function releaseBackdrop() {
    var cur = document.getElementById("ai-brain-instant-backdrop");
    if (cur) cur.classList.remove("active");
    if (window.__aiBrainWatch) {
      clearInterval(window.__aiBrainWatch);
      window.__aiBrainWatch = null;
    }
  }
  document.addEventListener("click", function (ev) {
    var btn = ev.target && ev.target.closest
      ? ev.target.closest('[class*="st-key-ai_top_btn_"] button, [class*="st-key-overview_top_btn_"] button')
      : null;
    if (!btn || btn.disabled) return;
    var bd = document.getElementById("ai-brain-instant-backdrop");
    var sub = document.getElementById("ai-instant-sub");
    if (sub) sub.textContent = (btn.innerText || "Task").trim() + " running — live status follows…";
    if (bd) bd.classList.add("active");
    if (window.__aiBrainWatch) clearInterval(window.__aiBrainWatch);
    var seenBusy = false;
    var waitTicks = 0;
    var idleTicks = 0;
    var startedAt = Date.now();
    window.__aiBrainWatch = setInterval(function () {
      var cur = document.getElementById("ai-brain-instant-backdrop");
      if (!cur) { releaseBackdrop(); return; }
      if (document.querySelector('[data-testid="stException"]')) {
        releaseBackdrop();
        return;
      }
      var hasOverlay = !!document.querySelector('.st-key-ai_task_overlay');
      var isRunning = !!document.querySelector('[data-testid="stStatusWidgetRunningIcon"], [data-test-script-state="running"]');
      if (!hasOverlay && (Date.now() - startedAt) > 180000) {
        releaseBackdrop();
        return;
      }
      var busy = hasOverlay || isRunning;
      if (busy) {
        seenBusy = true;
        idleTicks = 0;
        return;
      }
      if (seenBusy) {
        if (++idleTicks >= 4) releaseBackdrop();
      } else if (++waitTicks >= 30 || (Date.now() - startedAt) > 15000) {
        releaseBackdrop();
      }
    }, 120);
  }, true);
})();
</script>
"""


def render_instant_overlay_bridge() -> None:
    """Rendert den sofortigen JS-Klick-Blocker beim Klick auf Top-Action-Buttons."""
    with contextlib.suppress(TypeError):
        st.html(INSTANT_OVERLAY_HTML, unsafe_allow_javascript=True)


def run_with_task_overlay(
    title: str,
    initial_text: str,
    work_fn: Callable[[Callable[[float, str], None]], _T],
) -> _T:
    """Sperrt den Hintergrund per Blur-Backdrop und gibt ihn bei Ende oder Fehler garantiert wieder frei."""
    backdrop_slot = st.empty()
    overlay_slot = st.empty()
    try:
        backdrop_slot.markdown('<div class="ai-task-backdrop" aria-hidden="true"></div>', unsafe_allow_html=True)
        with overlay_slot.container(key="ai_task_overlay"):
            st.markdown(f"#### {title}")
            status_slot = st.empty()
            progress_bar = st.progress(0.05, text=initial_text)
            status_slot.info(initial_text)

            def _on_progress(fraction: float, message: str) -> None:
                progress_bar.progress(max(0.0, min(1.0, float(fraction))), text=message)
                status_slot.info(message)

            with st.spinner(initial_text):
                return work_fn(_on_progress)
    finally:
        overlay_slot.empty()
        backdrop_slot.empty()
