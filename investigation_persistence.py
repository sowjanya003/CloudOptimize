from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional
from keyvault_client import get_secret

CONTAINER = "investigations"
TRIGGER_PAYLOAD = "trigger_payload.json"
MANIFEST = "manifest.json"
ENRICHMENT = "enrichment.json"
PLANNER = "planner.json"
AGENTS_METADATA = "agents_metadata.json"
FINDINGS = "findings.json"
VALIDATED_FINDINGS = "validated_findings.json"
ROOT_CAUSES = "root_causes.json"
RECOMMENDATIONS = "recommendations.json"
IMPACT = "impact.json"
FINAL_REPORT = "final_report.json"
CHECKPOINTS = "checkpoints.json"


class PersistenceError(RuntimeError):
    """Raisez whenever an investigation file cannot be stored."""


class BaseDocumentStore:
    """put() writes/overwrites ONE named file inside the investigation folder."""

    async def put(self, service: str, investigation_id: str,
                  filename: str, payload) -> None:
        raise NotImplementedError

    async def list_services(self) -> list[str]:
        """Every service name that has at least one investigation folder."""
        raise NotImplementedError

    @staticmethod
    def blob_path(service: str, investigation_id: str, filename: str) -> str:
        return f"{service}/{investigation_id}/{filename}"


class BlobDocumentStore(BaseDocumentStore):
    """Writes investigations/{service}/{investigation_id}/{filename} in the
    same storage account used for telemetry reads, same auth chain."""

    def __init__(self, storage_account: str, container: str = CONTAINER):
        if not storage_account:
            raise PersistenceError(
                "BlobDocumentStore requires the telemetry storage account "
                "resolved for this subscription."
            )
        self.storage_account = storage_account
        self.container_name = container
        self._container = None

    # ---- connection --------------------

    def _connect(self):
        if self._container is not None:
            return self._container
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.core.exceptions import ResourceExistsError, ClientAuthenticationError
        except ImportError as e:
            raise PersistenceError(
                "azure-storage-blob is required for investigation persistence "
            ) from e

        account_url = f"https://{self.storage_account}.blob.core.windows.net"

        # Per-account key override: {"<storage_account>": "<key>"}
        key_map_raw = get_secret("AZURE_STORAGE_ACCOUNT_KEY_MAP")
        per_account_key = None
        if key_map_raw:
            try:
                per_account_key = json.loads(key_map_raw).get(self.storage_account)
            except json.JSONDecodeError as e:
                raise PersistenceError(
                    f"AZURE_STORAGE_ACCOUNT_KEY_MAP env var is set but is not valid JSON: {e}"
                ) from e

        conn_str = get_secret("AZURE_STORAGE_CONNECTION_STRING")
        account_key = per_account_key or get_secret("AZURE_STORAGE_ACCOUNT_KEY")

        try:
            if account_key:
                service = BlobServiceClient(account_url=account_url, credential=account_key)
            elif conn_str:
                service = BlobServiceClient.from_connection_string(conn_str)
            else:
                from azure.identity import DefaultAzureCredential
                service = BlobServiceClient(account_url=account_url,
                                            credential=DefaultAzureCredential())
            container = service.get_container_client(self.container_name)
            # The '{container}' container must exist in EVERY subscription's
            # storage account. Creates it on first touch; 
            # verify it already exists - otherwise it fails
    
            try:
                container.create_container()
            except ResourceExistsError:
                pass
            except Exception as create_err:
                try:
                    container.get_container_properties()
                except Exception:
                    raise PersistenceError(
                        f"Container '{self.container_name}' does not exist in "
                        f"storage account '{self.storage_account}' and could not "
                        f"be created: {type(create_err).__name__}: {create_err}. "
                        "Grant Blob Data Contributor or create the container manually."
                    ) from create_err
            self._container = container
            return container
        except ClientAuthenticationError as e:
            raise PersistenceError(
                f"Authentication to storage account '{self.storage_account}' failed: {e}"
            ) from e
    # ---- write ----------------------------------------------------------

    def _put_sync(self, service: str, investigation_id: str,
                  filename: str, payload) -> None:
        container = self._connect()
        path = self.blob_path(service, investigation_id, filename)
        try:
            data = json.dumps(payload, indent=2, default=str).encode("utf-8")
            container.upload_blob(name=path, data=data, overwrite=True)
        except Exception as e:
            raise PersistenceError(
                f"Failed to write investigation file "
                f"'{self.container_name}/{path}' to '{self.storage_account}': "
                f"{type(e).__name__}: {e}"
            ) from e

    async def put(self, service: str, investigation_id: str,
                  filename: str, payload) -> None:
        await asyncio.to_thread(self._put_sync, service, investigation_id, filename, payload)

    # ---- read (dashboard API) ------------------------------------------

    def _get_sync(self, service: str, investigation_id: str, filename: str):
        from azure.core.exceptions import ResourceNotFoundError
        container = self._connect()
        path = self.blob_path(service, investigation_id, filename)
        try:
            return json.loads(container.download_blob(path).readall())
        except ResourceNotFoundError as e:
            raise PersistenceError(
                f"Investigation file not found: {self.container_name}/{path} "
                f"(storage account: {self.storage_account})"
            ) from e

    async def get(self, service: str, investigation_id: str, filename: str):
        return await asyncio.to_thread(self._get_sync, service, investigation_id, filename)

    def _list_ids_sync(self, service: str) -> list[str]:
        container = self._connect()
        ids = []
        for item in container.walk_blobs(name_starts_with=f"{service}/", delimiter="/"):
            name = getattr(item, "name", "")
            if name.endswith("/"):
                ids.append(name[len(service) + 1:].strip("/"))
        return sorted(ids, reverse=True)  # inv_<timestamp>_ prefix -> newest first

    async def list_investigation_ids(self, service: str) -> list[str]:
        return await asyncio.to_thread(self._list_ids_sync, service)

    def _list_services_sync(self) -> list[str]:
        container = self._connect()
        services = []
        for item in container.walk_blobs(name_starts_with="", delimiter="/"):
            name = getattr(item, "name", "")
            if name.endswith("/"):
                services.append(name.rstrip("/"))
        return sorted(services)

    async def list_services(self) -> list[str]:
        return await asyncio.to_thread(self._list_services_sync)


class LocalDocumentStore(BaseDocumentStore):

    def __init__(self, root: str = "investigations"):
        self.root = Path(root)

    async def put(self, service: str, investigation_id: str,
                  filename: str, payload) -> None:
        path = self.root / self.blob_path(service, investigation_id, filename)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str))
        except OSError as e:
            raise PersistenceError(f"Failed to write {path}: {e}") from e

    async def get(self, service: str, investigation_id: str, filename: str):
        path = self.root / self.blob_path(service, investigation_id, filename)
        if not path.exists():
            raise PersistenceError(f"Investigation file not found: {path}")
        return json.loads(path.read_text())

    async def list_investigation_ids(self, service: str) -> list[str]:
        d = self.root / service
        if not d.exists():
            return []
        return sorted((p.name for p in d.iterdir() if p.is_dir()), reverse=True)

    async def list_services(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())


def build_document_store(telemetry_store) -> BlobDocumentStore:

    account = getattr(telemetry_store, "storage_account", None)
    if not account:
        raise PersistenceError(
            "Cannot build the investigation document store: the telemetry "
            "store has no resolved storage account."
        )
    return BlobDocumentStore(account)