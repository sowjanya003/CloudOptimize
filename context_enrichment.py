"""
it runs before the Planner dispatches any intelligence
agent. Collects everything related to the deviating item:

  enrichment["definition"]   - asset JSON of the job/pipeline itself
  enrichment["recent_runs"]  - compact newest-N run docs (status, duration,
                               per-task timings, error messages)
  enrichment["notebooks"]    - {activity_name: {source_code, language,
                               session_config, notebook_item_name}}
                               REAL code for every degraded notebook-bearing activity
  enrichment["dependencies"] - {activity_name: [upstream_activity, ...]}

Mapping rules by service:
  databricks : each task in extra_metadata.tasks carries its own notebook_path -
               the last path segment is the notebook asset name (task_key is
               NOT reliably the notebook name, e.g. task_key
               "update_Analytics_system" -> notebook_path ".../Update_Ledger") ->
               telemetry/databricks/{rg}/{ws}/notebooks/{notebook_name}.json
               (source in extra_metadata.source_code)
  synapse/adf: activity in the pipeline definition carries a
               notebook.reference_name ->
               telemetry/{service}/{rg}/{ws}/notebooks/{ref}.json
               (source in extra_metadata.cells[].source)
"""

from __future__ import annotations

from typing import Any

from schemas import InvestigationContext, ActivityDeviation
from telemetry_store import TelemetryStore, TelemetryFetchError

# activity_types whose implementation lives in a notebook asset
NOTEBOOK_BEARING_TYPES = {"notebook_task", "SynapseNotebook", "dlt_pipeline"}

# hard cap per notebook so a 10-activity job can't explode the prompt
MAX_SOURCE_CHARS = 12_000
RECENT_RUNS_LIMIT = 5
SYNTHESIS_RUNS_LIMIT = 30


def _truncate(source: str, limit: int = MAX_SOURCE_CHARS) -> str:
    if len(source) <= limit:
        return source
    return source[:limit] + f"\n... [TRUNCATED - {len(source) - limit} more chars]"


def _extract_notebook_payload(asset: dict) -> dict:
    """Normalize a notebook asset into one shape."""
    if asset.get("__folder__"):
        # notebook stored as a folder of assets - merge every asset with code
        merged, included = [], []
        for name, sub in asset.get("assets", {}).items():
            try:
                sub_payload = _extract_notebook_payload(sub)
            except TelemetryFetchError:
                continue
            merged.append(f"# ===== asset: {name} =====\n{sub_payload['source_code']}")
            included.append(name)
        if not merged:
            raise TelemetryFetchError(
                f"Notebook folder '{asset.get('item_name')}' contains no assets "
                f"with source code. Assets present: {list(asset.get('assets', {}).keys())}"
            )
        return {"source_code": _truncate("\n\n".join(merged)),
                "language": None, "session_config": {}, "assets_included": included}

    extra = asset.get("extra_metadata") or {}

    # databricks style
    source = extra.get("source_code")

    # synapse style: cells[].source
    if not source and extra.get("cells"):
        parts = []
        for i, cell in enumerate(extra["cells"]):
            cell_src = cell.get("source")
            if cell_src:
                parts.append(f"# --- cell {i} ({cell.get('language', 'unknown')}) ---\n{cell_src}")
        source = "\n\n".join(parts) if parts else None

    if not source:
        raise TelemetryFetchError(
            f"Notebook asset '{asset.get('item_name', '?')}' "
            f"(item_id={asset.get('item_id', '?')}) has no source code in "
            f"extra_metadata.source_code or extra_metadata.cells."
        )

    session_keys = ("big_data_pool", "language", "driver_memory", "driver_cores",
                    "executor_memory", "executor_cores", "num_executors")
    session_config = {k: extra[k] for k in session_keys if k in extra}

    return {
        "source_code": _truncate(source),
        "language": extra.get("language"),
        "session_config": session_config,
        "notebook_item_name": asset.get("item_name"),
        "last_modified_at": asset.get("last_modified_at"),
    }


def _flatten_activities(activities: list) -> list:
    """Recursively yield all activity defs, including those nested inside
    IfCondition (if_true_activities / if_false_activities),
    ForEach / Until (activities),
    Switch (cases[].activities / default_activities).
    """
    result = []
    for act in activities or []:
        result.append(act)
        act_type = act.get("type", "")

        if act_type == "IfCondition":
            result.extend(_flatten_activities(act.get("if_true_activities") or []))
            result.extend(_flatten_activities(act.get("if_false_activities") or []))

        elif act_type in ("ForEach", "Until"):
            result.extend(_flatten_activities(act.get("activities") or []))

        elif act_type == "Switch":
            for case in act.get("cases") or []:
                result.extend(_flatten_activities(case.get("activities") or []))
            result.extend(_flatten_activities(act.get("default_activities") or []))

        # ExecutePipeline / TryCatch / Scope containers (future-proofing)
        elif act.get("activities"):
            result.extend(_flatten_activities(act.get("activities") or []))

    return result


def _resolve_notebook_reference(service: str, item_definition: dict,
                                activity: ActivityDeviation) -> str:
    """Which notebook asset implements this activity"""
    if service == "databricks":
        # Databricks tasks declare their own notebook_path - task_key is
        # frequently NOT the notebook name (e.g. task_key "update_Analytics_system"
        # -> notebook_path ".../Update_Ledger"). Read the real path from the
        # job definition rather than assuming they match.
        extra = (item_definition or {}).get("extra_metadata") or {}
        tasks = extra.get("tasks") or []
        for task in tasks:
            if task.get("task_key") != activity.activity_name:
                continue
            path = task.get("notebook_path")
            if path:
                return path.rstrip("/").rsplit("/", 1)[-1]
            raise TelemetryFetchError(
                f"Task '{activity.activity_name}' in job definition has no "
                f"notebook_path (task_type={task.get('task_type')!r}) - it is "
                "likely a control-flow task (condition_task/for_each_task) "
                "with no notebook to enrich."
            )
        raise TelemetryFetchError(
            f"Task '{activity.activity_name}' not found in job definition's "
            "extra_metadata.tasks - cannot resolve which notebook implements it."
        )

    # synapse / adf: find the activity inside the pipeline definition and
    # read its notebook reference (search recursively through nested containers)
    extra = (item_definition or {}).get("extra_metadata") or {}
    for act_def in _flatten_activities(extra.get("activities", []) or []):
        if act_def.get("name") == activity.activity_name:
            nb = act_def.get("notebook") or {}
            ref = nb.get("reference_name") or nb.get("referenceName")
            if isinstance(ref, dict):
                ref = ref.get("value")
            if ref:
                return ref
            raise TelemetryFetchError(
                f"Activity '{activity.activity_name}' in the pipeline definition has "
                f"no notebook.reference_name (activity type: {act_def.get('type')})."
            )
    raise TelemetryFetchError(
        f"Activity '{activity.activity_name}' not found in the pipeline definition's "
        f"extra_metadata.activities - cannot resolve which notebook implements it."
    )


def _compact_runs(runs: list[dict]) -> list[dict]:
    return [
        {
            "run_id": r.get("run_id"),
            "status": r.get("status"),
            "start_time": r.get("start_time"),
            "duration_seconds": r.get("duration_seconds"),
            "error_message": r.get("error_message") or None,
            "tasks": (r.get("extra_metadata") or {}).get(
                "tasks", (r.get("extra_metadata") or {}).get("activity_runs", [])
            ),
        }
        for r in runs
    ]

_RUN_IDENTITY_KEYS = (
    "job_name", "activity_name", "task_key", "notebook_name",
    "session_name", "application_name", "name",
)


def _run_identity(run: dict) -> str:
    extra = run.get("extra_metadata") or {}
    for key in _RUN_IDENTITY_KEYS:
        val = extra.get(key)
        if val:
            return str(val)
    return run.get("item_name") or "unidentified_run"


def _synthesize_pseudo_activities(runs: list[dict]) -> list[dict]:
    
    grouped: dict[str, list[dict]] = {}
    for r in runs:
        grouped.setdefault(_run_identity(r), []).append(r)

    pseudo: list[dict] = []
    for identity, group_runs in grouped.items():
        durations = [r.get("duration_seconds") for r in group_runs
                     if r.get("duration_seconds") is not None]
        if not durations:
            continue

        runs_sorted = sorted(group_runs, key=lambda r: r.get("start_time") or "")
        latest = runs_sorted[-1]
        avg = sum(durations) / len(durations)
        latest_duration = latest.get("duration_seconds") or 0.0
        deviation_seconds = latest_duration - avg
        deviation_pct = (deviation_seconds / avg * 100) if avg else 0.0
        # job/pipeline activities
        if abs(deviation_pct) >= 30:
            health = "Severe"
        elif abs(deviation_pct) >= 10:
            health = "Warning"
        else:
            health = "Healthy"

        error_runs = [r for r in group_runs if r.get("error_message")]
        success_rate_pct = 100.0 * (len(group_runs) - len(error_runs)) / len(group_runs)

        pseudo.append({
            "activity_name": identity,
            "activity_type": "synthesized_activity",
            "avg_duration_seconds": round(avg, 2),
            "latest_duration_seconds": latest_duration,
            "deviation_seconds": round(deviation_seconds, 2),
            "deviation_pct": round(deviation_pct, 2),
            "success_rate_pct": round(success_rate_pct, 2),
            "health": health,
            "synthetic": True,
            "sample_size": len(group_runs),
        })
    return pseudo


def _extract_dependencies(item_definition: dict, recent_runs: list[dict]) -> dict:
    """depends_on graph: {activity_name: [upstream, ...]}"""
    deps: dict[str, list[str]] = {}

    extra = (item_definition or {}).get("extra_metadata") or {}
    top_level = extra.get("activities", extra.get("tasks", [])) or []
    for act_def in _flatten_activities(top_level):
        name = act_def.get("name") or act_def.get("task_key")
        if not name:
            continue
        upstream = []
        for d in act_def.get("depends_on", []) or []:
            up = d.get("activity") if isinstance(d, dict) else d
            if up:
                upstream.append(up)
        deps[name] = upstream

    # databricks: run docs carry per-task depends_on / ordering
    if not deps:
        for run in recent_runs:
            for t in (run.get("tasks") or []):
                key = t.get("task_key")
                if key and key not in deps:
                    raw = t.get("depends_on") or []
                    deps[key] = [d.get("task_key") if isinstance(d, dict) else d for d in raw]
            if deps:
                break
    return deps


class TelemetryEnricher:
    def __init__(self, store: TelemetryStore):
        self.store = store

    async def enrich(self, ctx: InvestigationContext) -> InvestigationContext:
        # 1. item (job/pipeline) definition
        definition = await self.store.fetch_item_asset(
            ctx.service, ctx.resource_group, ctx.workspace_name,
            ctx.item_type, ctx.item_name,
        )
        ctx.enrichment["definition"] = definition

        # 2. recent run documents (task timings, statuses, errors)
        runs = await self.store.fetch_item_runs(
            ctx.service, ctx.resource_group, ctx.workspace_name,
            ctx.item_type, ctx.item_name, limit=RECENT_RUNS_LIMIT,
        )
        ctx.enrichment["recent_runs"] = _compact_runs(runs)

        if not ctx.activities:
            synth_runs = runs
            if len(runs) < SYNTHESIS_RUNS_LIMIT:
                synth_runs = await self.store.fetch_item_runs(
                    ctx.service, ctx.resource_group, ctx.workspace_name,
                    ctx.item_type, ctx.item_name, limit=SYNTHESIS_RUNS_LIMIT,
                )
            pseudo = _synthesize_pseudo_activities(synth_runs)
            if pseudo:
                ctx.activities = [ActivityDeviation.from_raw(p) for p in pseudo]
                ctx.enrichment["activities_synthesized"] = True
                ctx.enrichment["activities_synthesis_note"] = (
                    f"No per-activity baseline was provided upstream for this "
                    f"item ({ctx.item_type}). These activities were derived "
                    "on-the-fly from recent run history, grouped by whatever "
                    "run-identity field was present, using a limited sample "
                    "and simple heuristic thresholds - treat with lower "
                    "confidence than baseline-derived findings."
                )

        # 3. REAL notebook source for every degraded notebook-bearing activity.
        # A single unresolvable/missing notebook must NOT abort the whole
        # investigation - record the gap and let enrichment continue, so
        # agents that don't need code (runtime, configuration, dependency)
        # can still run, and code-aware agents fail open per-activity
        # (they already refuse to guess without enriched source).
        notebooks: dict[str, dict] = {}
        enrichment_gaps: list[dict] = []
        for activity in ctx.degraded_activities():
            if activity.activity_type not in NOTEBOOK_BEARING_TYPES:
                continue
            try:
                notebook_name = _resolve_notebook_reference(ctx.service, definition, activity)
                asset = await self.store.fetch_notebook(
                    ctx.service, ctx.resource_group, ctx.workspace_name, notebook_name,
                )
                notebooks[activity.activity_name] = _extract_notebook_payload(asset)
            except TelemetryFetchError as e:
                enrichment_gaps.append({
                    "activity_name": activity.activity_name,
                    "activity_type": activity.activity_type,
                    "reason": str(e),
                })
        if notebooks:
            ctx.enrichment["notebooks"] = notebooks
        if enrichment_gaps:
            ctx.enrichment["notebook_enrichment_gaps"] = enrichment_gaps

        # 4. dependency graph
        deps = _extract_dependencies(definition, ctx.enrichment["recent_runs"])
        if deps:
            ctx.enrichment["dependencies"] = deps

        ctx.enriched = True
        return ctx