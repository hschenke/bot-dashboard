"""Pydantic-Schemata für strukturierten Google Gemini Output im KI-View."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AppDiagnosticInsight(BaseModel):
    """Strukturierte KI-Einzelbewertung für einen der drei Bots."""

    app_id: str = Field(description="Kennung der App: news_bot, research_bot oder alert_bot")
    app_title: str = Field(description="Anzeigename der App")
    status_rating: str = Field(description="Kurzbewertung: optimal, hinweis oder kritisch")
    key_observation: str = Field(description="Wichtigste Beobachtung zu URL, Worker und Datenbank")
    recommended_action: str = Field(description="Konkrete Empfehlung oder 'Keine Aktion erforderlich'")


class FleetAIAnalysis(BaseModel):
    """Strukturierter KI-Gesamtbericht über alle 3 überwachten Streamlit-Apps."""

    fleet_health_score: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Gesundheitsscore der gesamten Bot-Flotte von 0 bis 100",
    )
    headline: str = Field(description="Prägnante Überschrift zum aktuellen Flottenzustand")
    executive_summary: str = Field(
        description="Kurze, faktenreiche Zusammenfassung des Gesamtzustands (max. 3 Sätze)"
    )
    app_insights: list[AppDiagnosticInsight] = Field(
        default_factory=list,
        description="Einzelbewertungen für News-Bot, Research-Bot und Alert-Bot",
    )
    anomalies_or_warnings: list[str] = Field(
        default_factory=list,
        description="Erkannte Auffälligkeiten, schlafende Instanzen, Fehler oder leere Tabellen",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Priorisierte nächste Schritte für den Administrator",
    )
    model_used: str = Field(default="heuristic-fallback", description="Verwendetes KI-Modell")
