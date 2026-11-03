"""Save Azure credentials, deployments and selection as one validated operation."""

import hashlib
import io
import json
import os

from fastapi import HTTPException
from pydantic import Field

from rune.api.azure import AzureConnection, _credentials
from rune.config.schema import AzureDeployments, AzurePricingConfig
from rune.llm import azure
from rune.llm.context import llm_settings, model_environment
from rune.utils.logger import get_logger

log = get_logger(__name__)


class AzureSave(AzureConnection):
    revision: str = Field(min_length=1, max_length=64)
    deployments: AzureDeployments
    pricing: AzurePricingConfig | None = None
    useDeployment: str | None = Field(None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def revision() -> str:
    settings = llm_settings()
    state = [azure.endpoint(), model_environment(azure.environment_name(azure.KEY_NAMES)),
             settings.azure_deployments, settings.azure_pricing.model_dump() if settings.azure_pricing else None]
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def credential_revision() -> str:
    state = [azure.endpoint(), model_environment(azure.environment_name(azure.KEY_NAMES))]
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()


def save_settings(req: AzureSave) -> dict:
    from ruamel.yaml import YAML

    from rune.config import get_config, writer
    from rune.config.transaction import SettingsRecoveryError, replace_settings
    from rune.utils import env

    cfg = get_config()
    # Other config/env writers hold one of these locks; always acquire env first.
    with env._env_write_lock, writer._write_lock:
        if req.revision != revision():
            raise HTTPException(409, "Azure settings changed elsewhere. Reload the saved settings before trying again.")
        base, key = _credentials(req)
        if req.useDeployment and req.useDeployment not in req.deployments:
            raise HTTPException(400, "Select a deployment included in these settings.")
        active = cfg.llm.active_model if cfg.llm.active_provider == "azure" else None
        if active and active in cfg.llm.azure_deployments and active not in req.deployments and not req.useDeployment:
            raise HTTPException(400, "The active deployment was removed. Use Save and use on its replacement.")
        if req.pricing and (req.pricing.endpoint != base or set(req.pricing.deployments) - set(req.deployments)):
            raise HTTPException(400, "Pricing must match this resource and its registered deployments.")
        updates = {"llm.azureDeployments": req.deployments, "llm.azure_deployments": None,
                   "llm.azurePricing": req.pricing.model_dump() if req.pricing else None, "llm.azure_pricing": None}
        runtime = {"azure_deployments": req.deployments, "azure_pricing": req.pricing}
        if req.useDeployment:
            for name, value in {"activeProvider": "azure", "defaultProvider": "azure",
                                "activeModel": req.useDeployment, "defaultModel": req.useDeployment}.items():
                updates[f"llm.{name}"] = value
            updates.update({"llm.reasoningEffort": None, "llm.reasoning_effort": None})
            runtime.update(active_provider="azure", default_provider="azure", active_model=req.useDeployment,
                           default_model=req.useDeployment)
        next_llm = cfg.llm.model_copy(update=runtime)
        process_updates = {}
        endpoint_name = azure.environment_name(azure.ENDPOINT_NAMES)
        key_name = azure.environment_name(azure.KEY_NAMES)
        if base != azure.endpoint().rstrip("/").removesuffix("/openai/v1"):
            process_updates[endpoint_name] = base + "/openai/v1"
        if req.apiKey and req.apiKey.get_secret_value().strip():
            process_updates[key_name] = key
        try:
            files = {}
            user = env._parse_env_file(env._read_for_edit(env._user_env_path()))
            project = env._parse_env_file(env._read_for_edit(env._project_env_path()))
            env_changes = {}
            for name, value in process_updates.items():
                scope = env._effective_scope(name, user, project) or "user"
                if scope == "process":
                    raise HTTPException(409, "This value comes from the launch environment. Change it there and restart Rune.")
                path = env._user_env_path() if scope == "user" else env._project_env_path()
                env_changes.setdefault(path.resolve(), {})[name] = value
            for path, changes in env_changes.items():
                files[path] = (env._apply_env_edits(env._read_for_edit(path), changes), 0o600)
            path = writer.config_file_path().resolve()
            if path in files:
                raise HTTPException(400, "The configuration and credential files must be separate files.")
            yaml = YAML()
            yaml.preserve_quotes = True
            data = (yaml.load(path) or {}) if path.exists() else {}
            if not isinstance(data, dict):
                raise ValueError("invalid configuration")
            for name, value in updates.items():
                writer._assign(data, name, value)
            content = io.StringIO()
            yaml.dump(data, content)
            files[path] = (content.getvalue(), path.stat().st_mode & 0o777 if path.exists() else 0o600)
            replace_settings(files)
        except HTTPException:
            raise
        except SettingsRecoveryError as exc:
            log.error("azure_settings_recovery_required")
            raise HTTPException(500, "Saving and restoring settings failed. Stop using this configuration and inspect "
                                "the protected .rune-settings backups before restarting Rune.") from exc
        except Exception as exc:
            log.warning("azure_settings_save_failed", error=type(exc).__name__)
            raise HTTPException(500, "Could not save Azure settings. The original files and active settings were retained.") from exc
        os.environ.update(process_updates)
        cfg.llm = next_llm
        result = {"saved": True, "revision": revision(), "credentialRevision": credential_revision(),
                  "activeDeployment": req.useDeployment}
    from rune.llm.model_selection import _reset_llm_client
    from rune.llm.models import invalidate_cache

    invalidate_cache()
    _reset_llm_client()
    return result
