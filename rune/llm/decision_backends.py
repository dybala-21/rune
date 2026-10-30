"""Connection details for the supported decision APIs."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from rune.config.schema import DecisionRoutingConfig

MICROSOFT_PATH = "/providers/microsoft/v1/systemone"


def azure_endpoint(value: str) -> str:
    url = urlsplit(value.strip())
    if (url.scheme != "https" or not url.hostname
            or not url.hostname.endswith((".services.ai.azure.com", ".cognitiveservices.azure.com", ".openai.azure.com"))
            or url.username or url.password or url.query or url.fragment or url.port not in (None, 443)
            or url.path.rstrip("/") not in ("", MICROSOFT_PATH)):
        raise ValueError("Use the Azure resource's HTTPS endpoint or its System One URL.")
    return f"https://{url.hostname}"


@dataclass(frozen=True)
class DecisionTarget:
    backend: str
    label: str
    model: str
    billing_model: str
    key_name: str
    endpoint: str = ""
    deployment: str = ""


# Fixed versions keep alias updates from changing routing behavior.
JEV = DecisionTarget("jev", "Jev", "jev-1.13.0", "typesafe/jev-1.13.0", "TYPESAFE_API_KEY",
                     "https://api.typesafe.ai/v1/systemone", "jev-1.13.0")
MICROSOFT = DecisionTarget("microsoft", "Microsoft Decision", "microsoft-decision-1",
                           "azure-decision/microsoft-decision-1", "AZURE_DECISION_API_KEY")
BACKENDS = {target.backend: target for target in (JEV, MICROSOFT)}


def decision_target(settings: "DecisionRoutingConfig", backend: str | None = None) -> DecisionTarget:
    target = BACKENDS[backend or settings.backend]
    if target.backend == "microsoft":
        config = settings.microsoft
        return replace(target, endpoint=azure_endpoint(config.endpoint) + MICROSOFT_PATH if config.endpoint else "",
                       deployment=config.deployment)
    return target
