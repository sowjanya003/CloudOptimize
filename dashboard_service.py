from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional
from keyvault_client import get_secret
from telemetry_store import TelemetryStore, TelemetryFetchError, resolve_storage_account
from investigation_persistence import (
    BlobDocumentStore, BaseDocumentStore, PersistenceError, build_document_store,
    TRIGGER_PAYLOAD, MANIFEST, ENRICHMENT, PLANNER, AGENTS_METADATA,
    FINDINGS, VALIDATED_FINDINGS, ROOT_CAUSES, RECOMMENDATIONS,
    IMPACT, FINAL_REPORT, CHECKPOINTS,
)
from datetime import datetime
import calendar

# telemetry-container dirs that are NOT services
NON_SERVICE_DIRS = {"item_runs", "baseline", "activity_baseline", "investigations"}

# finding confidence -> UI severity. This is the single derivation rule in
# this module and it is echoed inside every fault payload for transparency.
SEVERITY_RULE = "confidence>=0.85: CRITICAL | >=0.70: HIGH | >=0.55: MEDIUM | else: LOW"


def severity_for(confidence: float) -> str:
    if confidence >= 0.85:
        return "CRITICAL"
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"


MAX_STRUCTURAL_FAULTS = 5


def _headline(text: Optional[str], max_chars: int = 140) -> str:
    if not text:
        return ""
    first = text.split(". ")[0].strip()
    if len(first) > max_chars:
        cut = first[:max_chars].rsplit(" ", 1)[0]
        return cut.rstrip(",;: ") + "…"
    if not first.endswith((".", "!", "?", "…")):
        first += "."
    return first


def ui_health(health: Optional[str]) -> str:
    return {"Healthy": "Ok", "Warning": "Warning", "Severe": "High",
            "Critical": "High"}.get(health or "", "Unknown")


class DashboardService:

    def __init__(self):
        self._stores: dict[str, TelemetryStore] = {}
        self._inv_stores: dict[str, BaseDocumentStore] = {}
        # injectable factories for tests
        self.telemetry_store_factory = TelemetryStore
        self.document_store_factory = build_document_store

    # ---- per-subscription store resolution -----------------------------

    def telemetry(self, subscription_id: str) -> TelemetryStore:
        if subscription_id not in self._stores:
            account = resolve_storage_account(subscription_id)
            self._stores[subscription_id] = self.telemetry_store_factory(account)
        return self._stores[subscription_id]

    def investigations(self, subscription_id: str) -> BaseDocumentStore:
        if subscription_id not in self._inv_stores:
            self._inv_stores[subscription_id] = self.document_store_factory(
                telemetry_store=self.telemetry(subscription_id)
            )
        return self._inv_stores[subscription_id]


    def list_subscriptions(self) -> list[dict]:
        mapping_raw = get_secret("STORAGE_ACCOUNT_MAP")
        if not mapping_raw:
            single = get_secret("TELEMETRY_STORAGE_ACCOUNT")
            if single:
                return [{"subscription_id": None, "storage_account": single,
                         "note": "single-account setup (TELEMETRY_STORAGE_ACCOUNT)"}]
            raise TelemetryFetchError(
                "No subscriptions configured: set STORAGE_ACCOUNT_MAP "
                '(JSON {"<subscription_id>": "<storage_account>"}).'
            )
        try:
            mapping = json.loads(mapping_raw)
        except json.JSONDecodeError as e:
            raise TelemetryFetchError(f"STORAGE_ACCOUNT_MAP is not valid JSON: {e}") from e
        return [{"subscription_id": sub, "storage_account": acct}
                for sub, acct in sorted(mapping.items())]

    def _ab(self, subscription_id: str) -> str:
        """Root of this subscription's """
        return f"activity_baseline/{subscription_id}/"

    async def list_services(self, subscription_id: str) -> list[dict]:
        """Services = dirs under telemetry/activity_baseline/{subscription_id}/."""
        store = self.telemetry(subscription_id)
        dirs = await store.list_dirs(self._ab(subscription_id))
        if not dirs:
            raise TelemetryFetchError(
                f"Nothing found under 'telemetry/{self._ab(subscription_id)}' in "
                f"storage account '{store.storage_account}'. Check "
                "STORAGE_ACCOUNT_MAP, read permissions, and that Team A's "
                "deviation outputs exist for this subscription - an empty "
                "result here is a configuration/data problem."
            )
        return [{"service": d} for d in dirs]

    async def list_workspaces(self, subscription_id: str, service: str) -> list[dict]:
        """resource_group/workspace pairs under
        activity_baseline/{subscription_id}/{service}/."""
        store = self.telemetry(subscription_id)
        root = f"{self._ab(subscription_id)}{service}/"
        rgs = await store.list_dirs(root)
        if not rgs:
            raise TelemetryFetchError(
                f"Nothing found under 'telemetry/{root}' in storage account "
                f"'{store.storage_account}' - unknown service name or wrong path."
            )
        out = []
        for rg in rgs:
            for ws in await store.list_dirs(f"{root}{rg}/"):
                out.append({"service": service, "resource_group": rg, "workspace_name": ws})
        return out

    async def list_item_types(self, subscription_id: str, service: str,
                              resource_group: str, workspace: str) -> list[str]:
        store = self.telemetry(subscription_id)
        root = f"{self._ab(subscription_id)}{service}/{resource_group}/{workspace}/"
        types = await store.list_dirs(root)
        if not types:
            raise TelemetryFetchError(
                f"Nothing found under 'telemetry/{root}' in "
                f"'{store.storage_account}' - check the resource group / "
                "workspace names (they are case-sensitive)."
            )
        return types

    async def list_items(self, subscription_id: str, service: str,
                         resource_group: str, workspace: str,
                         item_type: Optional[str] = None) -> list[dict]:
        """every item of the workspace, merged with its
        baseline metrics (avg baseline / latest run / deviation / health)
        when a baseline exists, and the derived optimization action."""
        store = self.telemetry(subscription_id)
        base = f"{self._ab(subscription_id)}{service}/{resource_group}/{workspace}"
        types = [item_type] if item_type else await store.list_dirs(f"{base}/")

        rows: list[dict] = []
        for t in types:
            folders, files = await store.list_children(f"{base}/{t}/")
            names = sorted(
                {f.rsplit("/", 1)[-1][:-5] for f in files if f.endswith(".json")}
                | set(folders)
            )
            if not names and item_type:
                raise TelemetryFetchError(
                    f"No items found under 'telemetry/{base}/{t}/' in "
                    f"'{store.storage_account}' - unknown item type or wrong path."
                )
            for name in names:
                rows.append(await self._item_row(
                    store, subscription_id, service, resource_group, workspace, t, name))
        return rows
    
    async def _load_cost_consumption(
        self,
        store: TelemetryStore,
        service: str,
        resource_group: str,
        workspace: str,
    ) -> dict | None:
        """
        Load workspace-level cost_consumption.json.
        Returns None if the blob doesn't exist.
        """

        path = (
            f"{service}/"
            f"{resource_group}/"
            f"{workspace}/"
            "cost_consumption.json"
        )

        try:
            return await store.get_json(path)
        except TelemetryFetchError:
            return None
        
    def _estimate_monthly_cost(self, cost: dict) -> float | None:
        if not cost:
            return None

        total_cost = cost.get("total_cost")
        fetched_at = cost.get("fetched_at")

        if total_cost is None:
            return None

        if fetched_at:
            dt = datetime.fromisoformat(
                fetched_at.replace("Z", "+00:00")
            )
        else:
            dt = datetime.utcnow()

        days_elapsed = max(dt.day, 1)
        days_in_month = calendar.monthrange(
            dt.year,
            dt.month,
        )[1]

        return round(
            (total_cost / days_elapsed) * days_in_month,
            2,
        )

    async def _item_row(self, store: TelemetryStore, subscription_id: str,
                        service: str, resource_group: str, workspace: str,
                        item_type: str, item_name: str) -> dict:
        row = {
            "subscription_id": subscription_id,
            "service": service,
            "resource_group": resource_group,
            "workspace_name": workspace,
            "item_type": item_type,
            "item_name": item_name,
            "pipeline_baseline": None,
            "health_state": None,
            "activities_count": None,
            "calculated_at": None,
            
            "current_month_cost": None,
            "baseline_monthly_cost": None,
            "estimated_monthly_cost": None,
            "currency": None,
        }
        try:
            baseline = await self._latest_deviation_output(
                store, subscription_id, service, resource_group, workspace,
                item_type, item_name)
            cost = await self._load_cost_consumption(
                store,
                service,
                resource_group,
                workspace,
            )

            if cost:
                row["current_month_cost"] = cost.get("total_cost")
                row["baseline_monthly_cost"] = cost.get("baseline_monthly_cost")
                row["estimated_monthly_cost"] = self._estimate_monthly_cost(cost)
                row["currency"] = cost.get("currency")
        except TelemetryFetchError as e:
            baseline = None
            row["baseline_error"] = str(e)
        if baseline:
            pb = baseline.get("pipeline_baseline") or {}
            row.update({
                "pipeline_baseline": pb,                      
                "health_state": pb.get("health"),   
                "activities_count": len(baseline.get("activities") or []),
                "calculated_at": baseline.get("calculated_at"),
            })
        row["optimization_action"] = await self._optimization_action(
            subscription_id, service, workspace, item_type, item_name, row["health_state"])
        return row

    async def _latest_deviation_output(self, store: TelemetryStore,
                                       subscription_id: str, service: str,
                                       resource_group: str, workspace: str,
                                       item_type: str, item_name: str) -> dict:
        docs = await store.fetch_activity_baselines(
            subscription_id, service, resource_group, workspace,
            item_type, item_name, limit=1)
        if not docs:
            raise TelemetryFetchError(
                f"No deviation outputs under 'telemetry/activity_baseline/"
                f"{subscription_id}/{service}/{resource_group}/{workspace}/"
                f"{item_type}/{item_name}/'."
            )
        return docs[0]

    async def _optimization_action(self, subscription_id: str, service: str,
                                   workspace: str, item_type: str, item_name: str,
                                   health_state: str) -> dict:
        
        latest = await self.latest_investigation(
            subscription_id, service, workspace, item_type, item_name,
            completed_only=True)
        if latest and latest.get("recommendations_count"):
            return {"action": "view_fix",
                    "investigation_id": latest["investigation_id"],
                    "recommendations_count": latest["recommendations_count"]}
        if health_state in ("Warning", "High"):
            return {"action": "analyze_available", "investigation_id": None,
                    "recommendations_count": 0}
        return {"action": "no_action_needed", "investigation_id": None,
                "recommendations_count": 0}

    async def item_detail(self, subscription_id: str, service: str,
                          resource_group: str, workspace: str,
                          item_type: str, item_name: str) -> dict:
        store = self.telemetry(subscription_id)
        doc = await self._latest_deviation_output(
            store, subscription_id, service, resource_group, workspace,
            item_type, item_name)

        health_state = ui_health((doc.get("pipeline_baseline") or {}).get("health"))
        return {
            **doc,  # the whole stored JSON, untouched
            "health_state": health_state,
            "optimization_action": await self._optimization_action(
                subscription_id, service, workspace, item_type, item_name, health_state),
            "investigations": await self.investigation_history(
                subscription_id, service, workspace, item_type, item_name),
        }

    # Investigations - history / trigger / status

    async def investigation_history(self, subscription_id: str, service: str,
                                    workspace: str, item_type: str,
                                    item_name: str, limit: int = 10) -> list[dict]:
        """Past investigations for this item"""
        inv = self.investigations(subscription_id)
        out = []
        for inv_id in await inv.list_investigation_ids(service):
            if len(out) >= limit:
                break
            try:
                m = await inv.get(service, inv_id, MANIFEST)
            except PersistenceError:
                continue
            if (m.get("item_name") == item_name
                    and m.get("workspace_name") == workspace
                    and m.get("item_type") == item_type):
                out.append({
                    "investigation_id": m["investigation_id"],
                    "status": m.get("status"),
                    "error": m.get("error"),
                    "started_at": m.get("started_at"),
                    "ended_at": m.get("ended_at"),
                    "findings_count": m.get("findings_count"),
                    "validated_findings_count": m.get("validated_findings_count"),
                    "root_causes_count": m.get("root_causes_count"),
                    "recommendations_count": m.get("recommendations_count"),
                    "total_input_tokens": m.get("total_input_tokens"),
                    "total_output_tokens": m.get("total_output_tokens"),
                })
        return out

    async def latest_investigation(self, subscription_id: str, service: str,
                                   workspace: str, item_type: str, item_name: str,
                                   completed_only: bool = False) -> Optional[dict]:
        history = await self.investigation_history(
            subscription_id, service, workspace, item_type, item_name, limit=50)
        for h in history:
            if not completed_only or h["status"] == "completed":
                return h
        return None

    async def build_trigger_payload(self, subscription_id: str, service: str,
                                    resource_group: str, workspace: str,
                                    item_type: str, item_name: str) -> dict:

        store = self.telemetry(subscription_id)
        payload = await self._latest_deviation_output(
            store, subscription_id, service, resource_group, workspace,
            item_type, item_name)
        # the pipeline needs the subscription for storage resolution
        payload["subscription_id"] = subscription_id
        payload["service"] = service
        payload["resource_group"] = resource_group
        payload["workspace_name"] = workspace
        payload["item_type"] = item_type
        payload["item_name"] = item_name
        return payload

    async def investigation_status(self, subscription_id: str, service: str,
                                   investigation_id: str) -> dict:
        return await self.investigations(subscription_id).get(
            service, investigation_id, MANIFEST)
    
    async def final_report(self, subscription_id: str, service: str,
                           investigation_id: str) -> dict:
        """Same dict shape as state.final_report in PlannerAgent - exactly
        what /post-cost-investigation returns."""
        return await self.investigations(subscription_id).get(
            service, investigation_id, FINAL_REPORT)
    
    # ---- runtime vs cost investigation listing (per-subscription) -------

    @staticmethod
    def _investigation_type_of(manifest: dict) -> str:
        """manifest.investigation_type is the source of truth going forward.
        Older manifests persisted before this field existed are inferred from
        item_type=='workspace', which only cost investigations ever set."""
        it = manifest.get("investigation_type")
        if it:
            return it
        return "cost" if manifest.get("item_type") == "workspace" else "runtime"

    async def _completed_investigations_by_type(
        self, subscription_id: str, investigation_type: str,
    ) -> dict[str, dict[str, str]]:
        """{service: {item_name: investigation_id}} for the most recent
        COMPLETED (never failed) investigation of the given type, per item."""
        inv = self.investigations(subscription_id)
        services = await inv.list_services()

        result: dict[str, dict[str, str]] = {}
        for service in services:
            ids = await inv.list_investigation_ids(service)  # newest-first
            per_item: dict[str, str] = {}
            for investigation_id in ids:
                try:
                    manifest = await inv.get(service, investigation_id, MANIFEST)
                except PersistenceError:
                    continue
                if manifest.get("status") != "completed":
                    continue
                if self._investigation_type_of(manifest) != investigation_type:
                    continue
                item_name = manifest.get("item_name")
                if not item_name or item_name in per_item:
                    continue  # already have the newest completed one for this item
                per_item[item_name] = manifest.get("investigation_id", investigation_id)
            if per_item:
                result[service] = per_item
        return result

    async def list_runtime_investigations(self, subscription_id: str) -> dict[str, dict[str, str]]:
        """{service: {item_name: investigation_id}} - latest COMPLETED
        runtime-deviation investigation per pipeline/job. Cost investigations
        are never included here."""
        return await self._completed_investigations_by_type(subscription_id, "runtime")

    async def list_cost_investigations(self, subscription_id: str) -> dict[str, dict[str, str]]:
        """{service: {workspace_name: investigation_id}} - latest COMPLETED
        cost-deviation investigation per workspace. Runtime investigations
        are never included here."""
        return await self._completed_investigations_by_type(subscription_id, "cost")

    def investigation_type_of(self, manifest: dict) -> str:
        """Public wrapper around _investigation_type_of, for callers (e.g.
        main.py) that need to check a manifest's type without reaching
        into a private method."""
        return self._investigation_type_of(manifest)

    async def compose_analysis(self, subscription_id: str, service: str,
                               investigation_id: str) -> dict:
        inv = self.investigations(subscription_id)

        async def load(filename, default):
            try:
                return await inv.get(service, investigation_id, filename)
            except PersistenceError:
                return default

        manifest = await inv.get(service, investigation_id, MANIFEST)  # required
        # Load workspace cost information
        store = self.telemetry(subscription_id)

        cost = await self._load_cost_consumption(
            store,
            service,
            manifest.get("resource_group"),
            manifest.get("workspace_name"),
        )
        trigger = await load(TRIGGER_PAYLOAD, {})
        enrichment = await load(ENRICHMENT, {})
        agents_meta = await load(AGENTS_METADATA, [])
        validated = await load(VALIDATED_FINDINGS, [])
        root_causes = await load(ROOT_CAUSES, [])
        recommendations = await load(RECOMMENDATIONS, [])
        impact = await load(IMPACT, {})
        report = await load(FINAL_REPORT, {})
        checkpoints = await load(CHECKPOINTS, [])

        notebooks = (enrichment or {}).get("notebooks", {}) or {}

        estimated_monthly_cost = None
        current_month_cost = None
        baseline_monthly_cost = None
        currency = None

        if cost:
            current_month_cost = cost.get("total_cost")
            baseline_monthly_cost = cost.get("baseline_monthly_cost")
            estimated_monthly_cost = self._estimate_monthly_cost(cost)
            currency = cost.get("currency")

        # -- header ------------------------------------------------------
        header = {
            "investigation_id": investigation_id,
            "status": manifest.get("status"),
            "error": manifest.get("error"),
            "skipped_reason": manifest.get("skipped_reason") or report.get("skipped_reason"),
            "message": manifest.get("message") or report.get("message"),
            "item": {k: manifest.get(k) for k in
                     ("subscription_id", "service", "resource_group",
                      "workspace_name", "item_type", "item_name")},
            "pipeline_health": manifest.get("pipeline_health"),
            "pipeline_deviation_pct": manifest.get("pipeline_deviation_pct"),
            "pipeline_deviation_seconds": manifest.get("pipeline_deviation_seconds"),
    
            "target_code_segments": [
                {"activity_name": name,
                 "notebook_item_name": nb.get("notebook_item_name"),
                 "language": nb.get("language"),
                 "source_chars": len(nb.get("source_code") or "")}
                for name, nb in notebooks.items()
            ],
            "started_at": manifest.get("started_at"),
            "ended_at": manifest.get("ended_at"),
            "stage_timings": manifest.get("stage_timings"),
            "total_input_tokens": manifest.get("total_input_tokens"),
            "total_output_tokens": manifest.get("total_output_tokens"),

            "current_month_cost": current_month_cost,
            "baseline_monthly_cost": baseline_monthly_cost,
            "estimated_monthly_cost": estimated_monthly_cost,
            "currency": currency,
        }

        # -- "analysis pipeline telemetry" ----------------------
        pipeline_telemetry = [
            {"step": c.get("label"), "at": c.get("at"), "sequence": c.get("sequence")}
            for c in checkpoints
        ]

        # -- agents involved ----------------------------------------------
        agents = [
            {"agent": r.get("agent"), "stage": r.get("stage"),
             "model": r.get("model"), "provider": r.get("provider"),
             "latency_seconds": r.get("latency_seconds"),
             "input_tokens": r.get("input_tokens"),
             "output_tokens": r.get("output_tokens"),
             "findings_count": (r.get("outputs") or {}).get("findings_count"),
             "status": r.get("status"), "error": r.get("error")}
            for r in agents_meta
        ]

        # -- core bottleneck = highest-confidence root cause ---------------
        rcs = sorted(root_causes, key=lambda r: r.get("confidence", 0), reverse=True)
        core_bottleneck = None
        if rcs:
            top = rcs[0]
            core_bottleneck = {
                "category": top.get("category"),
                "headline": _headline(top.get("description"), max_chars=200),
                "description": top.get("description"),
                "causal_chain": top.get("causal_chain"),
                "supporting_findings": top.get("supporting_findings"),
                "confidence": top.get("confidence"),
            }

        # -- recommendation analysis ---------------------------------------
        recommendation_analysis = [
            {"title": r.get("title"), "description": r.get("description"),
             "target_activity": r.get("target_activity"),
             "impact_score": r.get("impact_score"),
             "effort_score": r.get("effort_score"),
             "risk_score": r.get("risk_score"),
             "confidence": r.get("confidence")}
            for r in recommendations
        ]

        # -- impact agent output ---------------------------------------
        
        target_optimization_forecast = impact or None

        # -- compilation adjustment one-liner --------------------------
        compilation_adjustment = report.get("compilation_adjustment")

        # -- structural faults: deduplicated root causes, not raw findings -
        
        structural_faults = []
        for rc in rcs[:MAX_STRUCTURAL_FAULTS]:
            conf = float(rc.get("confidence") or 0)
            desc = rc.get("description") or ""
            structural_faults.append({
                "title": _headline(desc, max_chars=90),
                "description": _headline(desc, max_chars=220),
                "full_description": desc,
                "category": rc.get("category"),
                "severity": severity_for(conf),
                "severity_rule": SEVERITY_RULE,
                "confidence": conf,
                "supporting_findings": rc.get("supporting_findings"),
                "causal_chain": rc.get("causal_chain"),
            })

        all_findings = []
        for f in validated:
            conf = float(f.get("confidence") or 0)
            all_findings.append({
                "title": f.get("summary"),
                "severity": severity_for(conf),
                "severity_rule": SEVERITY_RULE,
                "confidence": conf,
                "detected_by": f.get("agent"),
                "affected_activities": f.get("affected_activities"),
                "evidence": f.get("evidence"),
                "verified": f.get("verified", True),
            })
        all_findings.sort(
            key=lambda x: ["CRITICAL", "HIGH", "MEDIUM", "LOW"].index(x["severity"]))

        
        # NOTE: fix.target_activity is a free-text label that CONTAINS the
        # real activity_name (e.g. "Bronze SynapseNotebook - df.count() Call"
        # for activity_name "Bronze") - it is never an exact match, so we
        # group by substring containment instead of equality.
        fixes_by_activity: dict[str, list] = {}
        for fix in (report.get("suggested_fixes") or []):
            target = (fix or {}).get("target_activity") or ""
            matched = [name for name in notebooks if name and name in target]
            key = matched[0] if matched else (target or "__pipeline__")
            fixes_by_activity.setdefault(key, []).append(fix)
        patches_by_activity: dict[str, list] = {}
        for p in (report.get("apply_fix_payloads") or []):
            target = (p or {}).get("target_activity") or ""
            matched = [name for name in notebooks if name and name in target]
            key = matched[0] if matched else (target or "__pipeline__")
            patches_by_activity.setdefault(key, []).append(p)

        code_patches = report.get("code_patches") or {}
        code_refactoring_plan = []
        for name, nb in notebooks.items():
            patch = code_patches.get(name) or {}
            has_issues = patch.get("has_issues", bool(fixes_by_activity.get(name)))
            code_refactoring_plan.append({
                "activity_name": name,
                "language": nb.get("language"),
                "baseline_script": nb.get("source_code"),
                "suggested_fixes": fixes_by_activity.get(name) or [],
                "has_issues": has_issues,
                "auto_generated_patch": patch.get("patched_code"),
                "patch_available": bool(patch.get("patched_code")),
                "patch_explanation": patch.get("explanation") or (
                    "No code-level issues identified for this notebook based "
                    "on current evidence." if not has_issues else None
                ),
            })

        return {
            "header": header,
            "pipeline_telemetry": pipeline_telemetry,
            "agents": agents,
            "agent_recommendation_analysis": {
                "core_bottleneck": core_bottleneck,
                "compilation_adjustment": compilation_adjustment,
                "recommendations": recommendation_analysis,
                "target_optimization_forecast": target_optimization_forecast,
            },
            "structural_faults": structural_faults,
            "structural_faults_count": len(structural_faults),
            "all_findings": all_findings,
            "all_findings_count": len(all_findings),
            "code_refactoring_plan": code_refactoring_plan,
            "root_causes": rcs,
            "trigger_payload": trigger.get("payload"),
        }