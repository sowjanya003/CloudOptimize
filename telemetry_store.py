from __future__ import annotations
import asyncio
import difflib
import json
import os
import re
from typing import Any, Optional
from keyvault_client import get_secret
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
    from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, HttpResponseError
except ImportError as _e: 
    raise ImportError(
        "azure-storage-blob is required for telemetry enrichment. "
        "Install with: pip install azure-storage-blob azure-identity"
    ) from _e

RESOLVER_VERSION = "3-ordering-prefix-match"


class TelemetryFetchError(RuntimeError):
    """Raised whenever required telemetry data cannot be fetched. Never swallowed."""

def resolve_storage_account(subscription_id: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit

    mapping_raw = get_secret("STORAGE_ACCOUNT_MAP")
    if mapping_raw:
        try:
            mapping = json.loads(mapping_raw)
        except json.JSONDecodeError as e:
            raise TelemetryFetchError(
                f"STORAGE_ACCOUNT_MAP env var is set but is not valid JSON: {e}"
            ) from e
        account = mapping.get(subscription_id)
        if account:
            return account
        raise TelemetryFetchError(
            f"No storage account mapped for subscription '{subscription_id}' in "
            f"STORAGE_ACCOUNT_MAP. Known subscriptions: {sorted(mapping.keys())}"
        )

    single = os.environ.get("TELEMETRY_STORAGE_ACCOUNT")
    if single:
        return single

    raise TelemetryFetchError(
        f"Cannot resolve storage account for subscription '{subscription_id}'. "
        "Provide storage_account explicitly, or set STORAGE_ACCOUNT_MAP "
        '(JSON: {"<subscription_id>": "<account>"}) or TELEMETRY_STORAGE_ACCOUNT.'
    )


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class TelemetryStore:
    CONTAINER = "telemetry"

    def __init__(self, storage_account: str, container: str = CONTAINER):
        self.storage_account = storage_account
        self.container_name = container
        self._container: Optional[ContainerClient] = None

    # ---- connection --------------------------------------------------

    def _connect(self) -> ContainerClient:
        if self._container is not None:
            return self._container

        account_url = f"https://{self.storage_account}.blob.core.windows.net"
        # Per-account key override: {"<storage_account>": "<key>"}
        key_map_raw = get_secret("AZURE_STORAGE_ACCOUNT_KEY_MAP")
        per_account_key = None
        if key_map_raw:
            try:
                per_account_key = json.loads(key_map_raw).get(self.storage_account)
            except json.JSONDecodeError as e:
                raise TelemetryFetchError(
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
                try:
                    from azure.identity import DefaultAzureCredential
                except ImportError as e:
                    raise TelemetryFetchError(
                        "No AZURE_STORAGE_CONNECTION_STRING / AZURE_STORAGE_ACCOUNT_KEY set "
                        "and azure-identity is not installed for DefaultAzureCredential."
                    ) from e
                service = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
            self._container = service.get_container_client(self.container_name)
            return self._container
        except ClientAuthenticationError as e:
            raise TelemetryFetchError(
                f"Authentication to storage account '{self.storage_account}' failed: {e}"
            ) from e

    def _get_json_sync(self, blob_path: str) -> Any:
        container = self._connect()
        try:
            data = container.download_blob(blob_path).readall()
        except ResourceNotFoundError as e:
            raise TelemetryFetchError(
                f"Blob not found: {self.container_name}/{blob_path} "
                f"(storage account: {self.storage_account})"
            ) from e
        except HttpResponseError as e:
            raise TelemetryFetchError(
                f"Failed to download {self.container_name}/{blob_path} "
                f"from '{self.storage_account}': {e.message}"
            ) from e
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise TelemetryFetchError(
                f"Blob {self.container_name}/{blob_path} is not valid JSON: {e}"
            ) from e

    def _children_sync(self, prefix: str) -> tuple[list[str], list[str]]:
        blobs = self._list_sync(prefix)
        dirs: set[str] = set()
        files: set[str] = set()
        for b in blobs:
            rest = b.name[len(prefix):]
            if not rest:
                continue
            if "/" in rest:
                dirs.add(rest.split("/", 1)[0])
            else:
                files.add(b.name)
        files = {f for f in files if f[len(prefix):] not in dirs}
        return sorted(dirs), sorted(files)

    def _list_dirs_sync(self, prefix: str) -> list[str]:
        return self._children_sync(prefix)[0]

    def _list_sync(self, prefix: str) -> list:
        container = self._connect()
        try:
            return list(container.list_blobs(name_starts_with=prefix))
        except HttpResponseError as e:
            raise TelemetryFetchError(
                f"Failed to list blobs under '{prefix}' in "
                f"'{self.storage_account}/{self.container_name}': {e.message}"
            ) from e



    def _resolve_item_sync(self, base_dir: str, item_name: str) -> dict:
        if not isinstance(item_name, str):
            raise TelemetryFetchError(
                f"item_name must be a string, got {type(item_name).__name__}: {item_name!r} "
                f"(base_dir={base_dir})"
            )
        candidates = {item_name, item_name.replace(" ", "_"), item_name.replace("_", " ")}
        listing = self._list_sync(f"{base_dir}/")
        names = [b.name for b in listing]
        blob_modified: dict[str, object] = {
            b.name: b.last_modified for b in listing
        }

        # exact / variant file match
        for cand in candidates:
            path = f"{base_dir}/{cand}.json"
            if path in names:
                return {"kind": "file", "path": path}

        # exact / variant folder match
        for cand in candidates:
            folder_prefix = f"{base_dir}/{cand}/"
            inside = [n for n in names if n.startswith(folder_prefix) and n.endswith(".json")]
            if inside:
                return {"kind": "folder", "paths": sorted(inside)}

        # normalized match against listing
        want = _normalize(item_name)
        for n in names:
            rel = n[len(base_dir) + 1:]
            head = rel.split("/", 1)[0]
            stem = head[:-5] if head.endswith(".json") else head
            if _normalize(stem) == want:
                if "/" in rel:  # it's a folder
                    folder_prefix = f"{base_dir}/{head}/"
                    inside = [m for m in names if m.startswith(folder_prefix) and m.endswith(".json")]
                    return {"kind": "folder", "paths": sorted(inside)}
                return {"kind": "file", "path": n}

        prefix_re = re.compile(r"^\d+[a-z]?_", re.IGNORECASE)
        prefixed_hits: dict[str, str] = {}   # head -> full rel first segment
        for n in names:
            rel = n[len(base_dir) + 1:]
            head = rel.split("/", 1)[0]
            stem = head[:-5] if head.endswith(".json") else head
            if prefix_re.match(stem) and _normalize(prefix_re.sub("", stem, count=1)) == want:
                prefixed_hits[head] = rel
        if len(prefixed_hits) == 1:
            head, rel = next(iter(prefixed_hits.items()))
            if "/" in rel:  # folder
                folder_prefix = f"{base_dir}/{head}/"
                inside = [m for m in names if m.startswith(folder_prefix) and m.endswith(".json")]
                return {"kind": "folder", "paths": sorted(inside)}
            return {"kind": "file", "path": f"{base_dir}/{head}"}
        if len(prefixed_hits) > 1:
            def _latest_modified(h: str) -> object:
                rel = prefixed_hits[h]
                blob_path = f"{base_dir}/{h}" if "/" not in rel else f"{base_dir}/{h}/"
                # find the newest last_modified among all blobs under this head
                ts = max(
                    (blob_modified.get(n) for n in names if n.startswith(f"{base_dir}/{h}")),
                    default=None,
                )
                return ts

            best_head = max(prefixed_hits, key=_latest_modified)
            best_rel = prefixed_hits[best_head]
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "TelemetryStore: item %r is ambiguous under %s/%s/ — "
                "candidates %s — picking most recently modified candidate %r.",
                item_name, self.container_name, base_dir,
                sorted(prefixed_hits), best_head,
            )
            if "/" in best_rel:
                folder_prefix = f"{base_dir}/{best_head}/"
                inside = [n for n in names if n.startswith(folder_prefix) and n.endswith(".json")]
                return {"kind": "folder", "paths": sorted(inside)}
            return {"kind": "file", "path": f"{base_dir}/{best_head}"}

        stems: dict[str, str] = {}   # normalized_stem -> head (first path segment)
        for n in names:
            rel = n[len(base_dir) + 1:]
            head = rel.split("/", 1)[0]
            stem = head[:-5] if head.endswith(".json") else head
            stems.setdefault(_normalize(stem), head)

        close = difflib.get_close_matches(want, stems.keys(), n=2, cutoff=0.72)
        if len(close) == 1 or (len(close) > 1 and close[0] != close[1]
                                and difflib.SequenceMatcher(None, want, close[0]).ratio()
                                - difflib.SequenceMatcher(None, want, close[1]).ratio() > 0.08):
            head = stems[close[0]]
            is_folder = any(
                n[len(base_dir) + 1:].split("/", 1)[0] == head and "/" in n[len(base_dir) + 1:]
                for n in names
            )
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "TelemetryStore: item %r not found exactly under %s/%s/ - "
                "using closest fuzzy match %r instead. This may be a rename "
                "or typo between the baseline data and the notebook asset name.",
                item_name, self.container_name, base_dir, head,
            )
            if is_folder:
                folder_prefix = f"{base_dir}/{head}/"
                inside = [n for n in names if n.startswith(folder_prefix) and n.endswith(".json")]
                return {"kind": "folder", "paths": sorted(inside)}
            return {"kind": "file", "path": f"{base_dir}/{head}"}

        available = sorted({n[len(base_dir) + 1:].split("/", 1)[0] for n in names})
        raise TelemetryFetchError(
            f"Item '{item_name}' not found under {self.container_name}/{base_dir}/ "
            f"in storage account '{self.storage_account}'. "
            f"Tried variants: {sorted(candidates)}. Available entries: {available or '(directory empty or missing)'}"
        )

    def _put_json_sync(self, blob_path: str, payload: Any) -> None:
        container = self._connect()
        try:
            data = json.dumps(payload, indent=2, default=str).encode("utf-8")
            container.upload_blob(name=blob_path, data=data, overwrite=True)
        except HttpResponseError as e:
            raise TelemetryFetchError(
                f"Failed to write {self.container_name}/{blob_path} "
                f"to '{self.storage_account}': {e.message}"
            ) from e

    async def put_json(self, blob_path: str, payload: Any) -> None:

        await asyncio.to_thread(self._put_json_sync, blob_path, payload)

    async def get_json(self, blob_path: str) -> Any:
        return await asyncio.to_thread(self._get_json_sync, blob_path)

    async def list_dirs(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(self._list_dirs_sync, prefix)

    async def list_files(self, prefix: str) -> list[str]:
        """Blob names (files only, not sub-dirs) directly under prefix."""
        return (await asyncio.to_thread(self._children_sync, prefix))[1]

    async def list_children(self, prefix: str) -> tuple[list[str], list[str]]:
        """(sub-directories, files) directly under prefix."""
        return await asyncio.to_thread(self._children_sync, prefix)

    async def fetch_item_asset(self, service: str, resource_group: str,
                               workspace: str, item_type: str, item_name: str) -> dict:
    
        base_dir = f"{service}/{resource_group}/{workspace}/{item_type}"
        resolved = await asyncio.to_thread(self._resolve_item_sync, base_dir, item_name)
        if resolved["kind"] == "file":
            return await self.get_json(resolved["path"])
        assets: dict[str, Any] = {}
        for path in resolved["paths"]:
            stem = path.rsplit("/", 1)[-1][:-5]
            assets[stem] = await self.get_json(path)
        return {"__folder__": True, "item_name": item_name, "assets": assets}

    async def fetch_notebook(self, service: str, resource_group: str,
                             workspace: str, notebook_name: str) -> dict:
        """Notebook asset (contains extra_metadata.source_code / cells)."""
        return await self.fetch_item_asset(service, resource_group, workspace, "notebooks", notebook_name)

    async def fetch_item_runs(self, service: str, resource_group: str, workspace: str,
                              item_type: str, item_name: str, limit: int = 5) -> list[dict]:
        """Newest N run documents from item_runs/, sorted by last_modified desc."""
        base_dir = f"item_runs/{service}/{resource_group}/{workspace}/{item_type}"
        resolved = await asyncio.to_thread(self._resolve_item_sync, base_dir, item_name)
        if resolved["kind"] != "folder":
            raise TelemetryFetchError(
                f"Expected a run FOLDER under {self.container_name}/{base_dir}/{item_name}/ "
                f"but found a single file: {resolved['path']}"
            )
        folder_prefix = resolved["paths"][0].rsplit("/", 1)[0] + "/"
        blobs = await asyncio.to_thread(self._list_sync, folder_prefix)
        blobs = sorted(blobs, key=lambda b: b.last_modified or 0, reverse=True)[:limit]
        if not blobs:
            raise TelemetryFetchError(
                f"No run files found under {self.container_name}/{folder_prefix} "
                f"in '{self.storage_account}'."
            )
        return [await self.get_json(b.name) for b in blobs]

    async def fetch_activity_baselines(self, subscription_id: str, service: str,
                                       resource_group: str, workspace: str, item_type: str,
                                       item_name: str, limit: int = 5) -> list[dict]:
        
        base_dir = f"activity_baseline/{subscription_id}/{service}/{resource_group}/{workspace}/{item_type}"
        resolved = await asyncio.to_thread(self._resolve_item_sync, base_dir, item_name)
        if resolved["kind"] == "file":
            return [await self.get_json(resolved["path"])]
        folder_prefix = resolved["paths"][0].rsplit("/", 1)[0] + "/"
        blobs = await asyncio.to_thread(self._list_sync, folder_prefix)
        blobs = sorted(blobs, key=lambda b: b.last_modified or 0, reverse=True)[:limit]
        return [await self.get_json(b.name) for b in blobs]

    async def fetch_baseline(self, service: str, resource_group: str,
                             workspace: str, item_type: str, item_name: str) -> dict:
        """Item baseline: telemetry/baseline/{service}/{rg}/{ws}/{item_type}/{item_name}.json"""
        base_dir = f"baseline/{service}/{resource_group}/{workspace}/{item_type}"
        resolved = await asyncio.to_thread(self._resolve_item_sync, base_dir, item_name)
        if resolved["kind"] == "file":
            return await self.get_json(resolved["path"])
        assets: dict[str, Any] = {}
        for path in resolved["paths"]:
            stem = path.rsplit("/", 1)[-1][:-5]
            assets[stem] = await self.get_json(path)
        return {"__folder__": True, "item_name": item_name, "assets": assets}