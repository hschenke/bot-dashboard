"""Gemeinsames Task-Overlay: grauer Blur-Backdrop im Hintergrund & Live-Status im Vordergrund."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TypeVar

import streamlit as st

_T = TypeVar("_T")

INSTANT_OVERLAY_HTML = """
<div id="ki-brain-instant-backdrop">
  <div class="ki-instant-card">
    <strong>⚙️ Running task…</strong>
    <div id="ki-instant-sub" style="font-size:0.9rem;color:#94a3b8;margin-top:0.35rem;">Please wait — locking background.</div>
  </div>
</div>
<script>
(function () {
  if (window.__kiBrainOverlayBound) return;
  window.__kiBrainOverlayBound = true;
  document.addEventListener("click", function (ev) {
    var btn = ev.target && ev.target.closest
      ? ev.target.closest('[class*="st-key-ki_top_btn_"] button, [class*="st-key-overview_top_btn_"] button')
      : null;
    if (!btn || btn.disabled) return;
    var bd = document.getElementById("ki-brain-instant-backdrop");
    var sub = document.getElementById("ki-instant-sub");
    if (sub) sub.textContent = (btn.innerText || "Task").trim() + " running — live status follows…";
    if (bd) bd.classList.add("active");
    if (window.__kiBrainWatch) clearInterval(window.__kiBrainWatch);
    var idle = 0;
    window.__kiBrainWatch = setInterval(function () {
      var cur = document.getElementById("ki-brain-instant-backdrop");
      if (!cur) { clearInterval(window.__kiBrainWatch); return; }
      var busy = document.querySelector('[data-testid="stStatusWidgetRunningIcon"], [data-test-script-state="running"], .st-key-ki_task_overlay');
      if (!busy) {
        if (++idle >= 3) { cur.classList.remove("active"); clearInterval(window.__kiBrainWatch); }
      } else { idle = 0; }
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
    """Sperrt den Hintergrund per Blur-Backdrop und hält Live-Status + Progress im Vordergrund."""
    st.markdown('<div class="ki-task-backdrop" aria-hidden="true"></div>', unsafe_allow_html=True)
    with st.container(key="ki_task_overlay"):
        st.markdown(f"#### {title}")
        status_slot = st.empty()
        progress_bar = st.progress(0.05, text=initial_text)
        status_slot.info(initial_text)

        def _on_progress(fraction: float, message: str) -> None:
            progress_bar.progress(max(0.0, min(1.0, float(fraction))), text=message)
            status_slot.info(message)

        with st.spinner(initial_text):
            return work_fn(_on_progress)
