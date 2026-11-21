"""KI-Brain – Zentrales Framework für kostenfreie Modelle, Web-Tools & Tier-Sequenzen."""

from ki_brain.access import ALLOWED_CLIENT_APPS, FLEET_CLIENT_ID, FleetAccessStatus
from ki_brain.brain import KIWebsearchBrain, PipelineSummary
from ki_brain.connection import (
    CONNECTION_KEY_ENV,
    BrainConnectionConfig,
    decode_connection_key,
    encode_connection_key,
    is_brain_configured,
)
from ki_brain.discovery import DiscoverySummary, is_preview_or_experimental_model
from ki_brain.enums import BrainTier, ModelProvider, ProbeStatus, ToolCategory
from ki_brain.exceptions import (
    BrainAuthError,
    BrainError,
    BrainProviderError,
    BrainSSRFError,
    BrainStorageError,
    BrainValidationError,
)
from ki_brain.models import (
    AIModelSpec,
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
    derive_tier_key_envs,
)
from ki_brain.pricing import (
    compute_cost_score,
    estimate_blended_cost_per_1m,
    format_model_budget_label,
    infer_model_is_free,
)
from ki_brain.quotas import (
    QuotaSnapshot,
    evaluate_model_quota,
    evaluate_tool_quota,
    format_quota_label,
    is_geo_or_maps_query,
    is_quota_limited_tool,
)
from ki_brain.storage import BrainDashboardState, BrainRepository, create_brain_repository
from ki_brain.unblock import UnblockSummary

__version__ = "0.18.0"

connect = KIWebsearchBrain.connect
is_configured = is_brain_configured

__all__ = [
    "AIModelSpec",
    "ALLOWED_CLIENT_APPS",
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
    "KIWebsearchBrain",
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
    "connect",
    "create_brain_repository",
    "decode_connection_key",
    "derive_tier_key_envs",
    "encode_connection_key",
    "estimate_blended_cost_per_1m",
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
