from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("keyvault_client")

_VAULT_URL_ENV_CANDIDATES = ("AZURE_KEY_VAULT_URL", "https://uc3-agents.vault.azure.net/")
_VAULT_NAME_ENV_CANDIDATES = ("AZURE_KEY_VAULT_NAME", "uc3-agents")

_lock = threading.Lock()
_secret_client = None            
_client_init_attempted = False
_cache: dict[str, Optional[str]] = {}   
_cache_source: dict[str, str] = {}      

_EXPLICIT_VAULT_NAMES: dict[str, str] = {
    "ANTHROPIC_API_KEY": "ANTHROPIC-API-KEY",
    "AZURE_OPENAI_API_KEY": "AZURE-OPENAI-API-KEY",
    "AZURE_OPENAI_API_VERSION": "AZURE-OPENAI-API-VERSION",
    "AZURE_OPENAI_ENDPOINT": "AZURE-OPENAI-ENDPOINT",
    "AZURE_OPENAI_GPT5_DEPLOYMENT": "AZURE-OPENAI-GPT5-DEPLOYMENT",
    "AZURE_STORAGE_ACCOUNT_KEY": "AZURE-STORAGE-ACCOUNT-KEY",
    "AZURE_STORAGE_CONNECTION_STRING": "AZURE-STORAGE-CONNECTION-STRING",
    "AZURE_STORAGE_ACCOUNT_KEY_MAP": "AZURE-STORAGE-ACCOUNT-KEY-MAP",
    "STORAGE_ACCOUNT_MAP": "STORAGE-ACCOUNT-MAP",
    "PERSIST_BACKEND": "PERSIST-BACKEND",
}


def _vault_url() -> Optional[str]:
    for var in _VAULT_URL_ENV_CANDIDATES:
        val = os.environ.get(var)
        if val:
            return val
    for var in _VAULT_NAME_ENV_CANDIDATES:
        name = os.environ.get(var)
        if name:
            return f"https://{name}.vault.azure.net"
    return None


def _to_kv_secret_name(env_name: str) -> str:
    """Key Vault secret names may only contain alphanumerics and hyphens."""
    if env_name in _EXPLICIT_VAULT_NAMES:
        return _EXPLICIT_VAULT_NAMES[env_name]
    return env_name.replace("_", "-")


def _get_client():
    global _secret_client, _client_init_attempted

    if _secret_client is not None:
        return _secret_client
    if _client_init_attempted:
        return None

    with _lock:
        if _client_init_attempted:
            return _secret_client
        _client_init_attempted = True

        vault_url = _vault_url()
        if not vault_url:
            logger.info(
                "AZURE_KEY_VAULT_URL / AZURE_KEY_VAULT_NAME not set - "
                "falling back to environment variables for secrets."
            )
            return None

        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError:
            logger.warning(
                "azure-keyvault-secrets is not installed. Falling back to "
                "environment variables for secrets."
            )
            return None

        try:
            credential = DefaultAzureCredential()
            _secret_client = SecretClient(vault_url=vault_url, credential=credential)
            logger.info("Connected to Azure Key Vault at %s", vault_url)
        except Exception as e:
            logger.warning("Failed to initialize Key Vault client (%s): %s", vault_url, e)
            _secret_client = None

    return _secret_client


def get_secret(name: str, default: Optional[str] = None,
                required: bool = False) -> Optional[str]:
    if name in _cache:
        value = _cache[name]
        if value is None and required and default is None:
            raise RuntimeError(f"Required secret '{name}' could not be resolved.")
        return value if value is not None else default

    value: Optional[str] = None
    source = "missing"
    client = _get_client()
    if client is not None:
        kv_name = _to_kv_secret_name(name)
        try:
            value = client.get_secret(kv_name).value
            source = "vault"
            logger.info("Secret '%s' fetched from Key Vault as '%s'.", name, kv_name)
        except Exception as e:
            logger.info(
                "Secret '%s' not found in Key Vault (%s); falling back to "
                "environment variable.", kv_name, type(e).__name__
            )

    if not value:
        value = os.environ.get(name)
        if value:
            source = "env"

    _cache[name] = value
    _cache_source[name] = source

    if not value:
        if required:
            raise RuntimeError(
                f"Required secret '{name}' not found in Key Vault or "
                f"environment variables."
            )
        return default

    return value


def secret_sources() -> dict[str, str]:
    return dict(_cache_source)


def print_secret_report(names: Optional[list[str]] = None) -> None:
    
    if names is None:
        names = list(_EXPLICIT_VAULT_NAMES.keys())

    print(f"{'SECRET':40} {'SOURCE':10} {'STATUS'}")
    print("-" * 70)
    for name in names:
        val = get_secret(name)
        source = _cache_source.get(name, "missing")
        status = "OK" if val else "NOT SET"
        label = {"vault": "Key Vault", "env": "Env Var", "missing": "MISSING"}[source]
        print(f"{name:40} {label:10} {status}")


def clear_cache() -> None:
    """Useful for tests or secret-rotation scenarios."""
    _cache.clear()