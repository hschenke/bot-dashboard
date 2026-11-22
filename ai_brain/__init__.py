"""AI-Brain – Zentrales Framework für kostenfreie Modelle, Web-Tools & Tier-Sequenzen."""

from ai_brain.access import ALLOWED_CLIENT_APPS, FLEET_CLIENT_ID, FleetAccessStatus
from ai_brain.brain import AIWebsearchBrain, PipelineSummary
from ai_brain.connection import (
    CONNECTION_KEY_ENV,
    BrainConnectionConfig,
    decode_connection_key,
    encode_connection_key,
    is_brain_configured,
)
from ai_brain.discovery import DiscoverySummary, is_preview_or_experimental_model
from ai_brain.enums import BrainTier, ModelProvider, ProbeStatus, ToolCategory
from ai_brain.exceptions import (
    BrainAuthError,
    BrainError,
    BrainProviderError,
    BrainSSRFError,
    BrainStorageError,
    BrainValidationError,
)
from ai_brain.models import (
    AIModelSpec,
    AppBrainPolicy,
    BenchmarkEntry,
    BenchmarkReport,
    BrainAccessLogEntry,
    BrainCitation,
    BrainConfigSnapshot,
    BrainFetchedPage,
    BrainGenerationResult,
    BrainSearchResult,
    BrainSecrets,
    SecretStatusEntry,
    TierSequenceConfig,
    ToolSpec,
    default_app_policy,
    derive_tier_key_envs,
)
from ai_brain.pricing import (
    compute_cost_score,
    compute_value_score,
    estimate_blended_cost_per_1m,
    estimate_model_capability,
    format_model_budget_label,
    infer_model_is_free,
)
from ai_brain.quotas import (
    QuotaSnapshot,
    evaluate_model_quota,
    evaluate_tool_quota,
    format_quota_label,
    is_geo_or_maps_query,
    is_quota_limited_tool,
)
from ai_brain.storage import BrainDashboardState, BrainRepository, create_brain_repository
from ai_brain.unblock import UnblockSummary

__version__ = "0.21.0"

connect = AIWebsearchBrain.connect
is_configured = is_brain_configured

__all__ = [
    "AIModelSpec",
    "ALLOWED_CLIENT_APPS",
    "AppBrainPolicy",
    "BenchmarkEntry",
    "BenchmarkReport",
    "BrainAccessLogEntry",
    "BrainAuthError",
    "BrainCitation",
    "BrainConfigSnapshot",
    "BrainConnectionConfig",
    "BrainDashboardState",
    "BrainError",
    "BrainFetchedPage",
    "BrainGenerationResult",
    "BrainProviderError",
    "BrainRepository",
    "BrainSSRFError",
    "BrainSearchResult",
    "BrainSecrets",
    "BrainStorageError",
    "BrainTier",
    "BrainValidationError",
    "CONNECTION_KEY_ENV",
    "DiscoverySummary",
    "FLEET_CLIENT_ID",
    "FleetAccessStatus",
    "AIWebsearchBrain",
    "ModelProvider",
    "PipelineSummary",
    "ProbeStatus",
    "QuotaSnapshot",
    "SecretStatusEntry",
    "TierSequenceConfig",
    "ToolCategory",
    "ToolSpec",
    "UnblockSummary",
    "__version__",
    "compute_cost_score",
    "compute_value_score",
    "connect",
    "create_brain_repository",
    "decode_connection_key",
    "default_app_policy",
    "derive_tier_key_envs",
    "encode_connection_key",
    "estimate_blended_cost_per_1m",
    "estimate_model_capability",
    "evaluate_model_quota",
    "evaluate_tool_quota",
    "format_model_budget_label",
    "format_quota_label",
    "infer_model_is_free",
    "is_brain_configured",
    "is_configured",
    "is_geo_or_maps_query",
    "is_preview_or_experimental_model",
    "is_quota_limited_tool",
]
