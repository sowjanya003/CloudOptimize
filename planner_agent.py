"""
Planner Agent (Orchestrator)
Model: Claude Sonnet 4
"""

from __future__ import annotations
import asyncio
from typing import Optional

from schemas import (
    InvestigationContext, InvestigationState, AgentRole, AgentFinding,
)
from telemetry_store import TelemetryStore, resolve_storage_account
from context_enrichment import TelemetryEnricher
from investigation_persistence import (
    build_document_store, BaseDocumentStore,
    TRIGGER_PAYLOAD, MANIFEST, ENRICHMENT, PLANNER, AGENTS_METADATA,
    FINDINGS, VALIDATED_FINDINGS, ROOT_CAUSES, RECOMMENDATIONS,
    IMPACT, FINAL_REPORT, CHECKPOINTS,
)
from evidence_validation_agent import MIN_CONFIDENCE_TO_VERIFY as _EV_MIN_CONST


def _EV_MIN_CONFIDENCE() -> float:
    return _EV_MIN_CONST

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from sub_agents.runtime_intelligence_agent import RuntimeIntelligenceAgent
from sub_agents.sql_intelligence_agent import SQLIntelligenceAgent
from sub_agents.python_intelligence_agent import PythonIntelligenceAgent
from sub_agents.spark_intelligence_agent import SparkIntelligenceAgent
from sub_agents.configuration_agent import ConfigurationAgent
from sub_agents.dependency_lineage_agent import DependencyLineageAgent
from sub_agents.cost_intelligence_agent import CostIntelligenceAgent
from evidence_validation_agent import EvidenceValidationAgent
from root_cause_agent import RootCauseAgent 
from recommendation_agent import RecommendationAgent
from impact_agent import ImpactAgent
from action_plan_report_agent import ActionPlanReportAgent
from azure_openai_client import azure_gpt5_available, azure_gpt5_caller

ACTIVITY_TYPE_TO_AGENTS: dict[str, list[AgentRole]] = {
    # Notebook/code activities -> all three focused code agents + runtime
    "SynapseNotebook": [
        AgentRole.RUNTIME_INTELLIGENCE,
        AgentRole.SQL_INTELLIGENCE,
        AgentRole.PYTHON_INTELLIGENCE,
        AgentRole.SPARK_INTELLIGENCE,
        AgentRole.CONFIGURATION,
    ],
    "notebook_task": [
        AgentRole.RUNTIME_INTELLIGENCE,
        AgentRole.SQL_INTELLIGENCE,
        AgentRole.PYTHON_INTELLIGENCE,
        AgentRole.SPARK_INTELLIGENCE,
        AgentRole.CONFIGURATION,
    ],
    "dlt_pipeline": [
        AgentRole.RUNTIME_INTELLIGENCE,
        AgentRole.PYTHON_INTELLIGENCE,
        AgentRole.SPARK_INTELLIGENCE,
        AgentRole.CONFIGURATION,
    ],
    # SQL-only activities
    "sql_task":   [AgentRole.SQL_INTELLIGENCE],
    "sql_script": [AgentRole.SQL_INTELLIGENCE],
    # Orchestration / control-flow activities -> runtime + dependency
    "Copy":         [AgentRole.RUNTIME_INTELLIGENCE, AgentRole.CONFIGURATION],
    "IfCondition":  [AgentRole.RUNTIME_INTELLIGENCE, AgentRole.DEPENDENCY_LINEAGE],
    "ForEach":      [AgentRole.RUNTIME_INTELLIGENCE, AgentRole.DEPENDENCY_LINEAGE],
    "Switch":       [AgentRole.RUNTIME_INTELLIGENCE, AgentRole.DEPENDENCY_LINEAGE],
    "Wait":         [AgentRole.RUNTIME_INTELLIGENCE],
    "GetMetadata":  [AgentRole.RUNTIME_INTELLIGENCE, AgentRole.CONFIGURATION],
    "SetVariable":  [AgentRole.RUNTIME_INTELLIGENCE],
}

DEFAULT_AGENTS_FOR_UNKNOWN_TYPE = [AgentRole.RUNTIME_INTELLIGENCE]

MAX_PLANNER_ROUNDS = 3  # safety-net round ceiling, per your earlier architecture


class PlannerAgent:
    

    def __init__(self, storage_account: Optional[str] = None,
                 persistence: Optional[BaseDocumentStore] = None,
                 asset_loader=None, history_loader=None, kb_search=None):
        
        self.persistence = persistence
   
        self.storage_account = storage_account
        self.asset_loader = asset_loader
        self.history_loader = history_loader
        self.kb_search = kb_search

        self.evidence_validation_agent = EvidenceValidationAgent()
        self.root_cause_agent = RootCauseAgent()
        self.recommendation_agent = RecommendationAgent()

        
        gpt5_caller = azure_gpt5_caller if azure_gpt5_available() else None
        self.impact_agent = ImpactAgent(mcp_gpt5_caller=gpt5_caller)
        self.report_agent = ActionPlanReportAgent(mcp_gpt5_caller=gpt5_caller)

        self._sub_agent_pool = {
            AgentRole.RUNTIME_INTELLIGENCE: RuntimeIntelligenceAgent(asset_loader),
            AgentRole.SQL_INTELLIGENCE:     SQLIntelligenceAgent(asset_loader),
            AgentRole.PYTHON_INTELLIGENCE:  PythonIntelligenceAgent(asset_loader),
            AgentRole.SPARK_INTELLIGENCE:   SparkIntelligenceAgent(asset_loader),
            AgentRole.CONFIGURATION:        ConfigurationAgent(asset_loader),
            AgentRole.DEPENDENCY_LINEAGE:   DependencyLineageAgent(asset_loader),
            AgentRole.COST_INTELLIGENCE:    CostIntelligenceAgent(asset_loader)
        }

    # ---- tools -----------------------------------------------------

    def load_baseline_deviation(
        self,
        payload: dict,
        investigation_type: str = "runtime",
    ) -> InvestigationContext:

        if investigation_type == "cost":
            return InvestigationContext.from_cost_payload(payload)

        return InvestigationContext.from_deviation_payload(payload)

    async def load_recent_runs(self, ctx: InvestigationContext) -> list[dict]:
        if self.history_loader is None:
            return []
        return await self.history_loader(
            service=ctx.service, resource_group=ctx.resource_group,
            workspace_name=ctx.workspace_name, item_name=ctx.item_name,
        )

    async def load_pipeline_definition(self, ctx: InvestigationContext) -> dict:
        if self.asset_loader is None:
            return {}
        return await self.asset_loader(
            service=ctx.service, resource_group=ctx.resource_group,
            workspace_name=ctx.workspace_name, item_type=ctx.item_type,
            item_name=ctx.item_name,
        )

    async def load_asset(self, ctx: InvestigationContext, asset_type: str, asset_name: str) -> dict:
        if self.asset_loader is None:
            return {}
        return await self.asset_loader(
            service=ctx.service, resource_group=ctx.resource_group,
            workspace_name=ctx.workspace_name, item_type=asset_type,
            item_name=asset_name,
        )

    async def search_knowledge_base(self, ctx: InvestigationContext, query: str) -> list[dict]:
        if self.kb_search is None:
            return []
        return await self.kb_search(query=query, service=ctx.service, item_name=ctx.item_name)

    def select_agents_to_dispatch(self, ctx: InvestigationContext) -> set[AgentRole]:
        selected: set[AgentRole] = set()

        # Cost investigation
        payload = ctx.raw_payload

        if payload.get("deviation") is not None and (
            payload.get("baseline_monthly_cost") is not None
        ):
            selected.add(AgentRole.COST_INTELLIGENCE)
            return selected

        degraded = ctx.degraded_activities()

        # If the pipeline-level health is Healthy AND no individual activities
        # are in Warning/Severe, there is genuinely nothing to investigate.
        if ctx.pipeline_health == "Healthy" and not degraded:
            return selected  # empty → triggers the skip path in run_investigation

        for activity in degraded:
            agents = ACTIVITY_TYPE_TO_AGENTS.get(
                activity.activity_type, DEFAULT_AGENTS_FOR_UNKNOWN_TYPE
            )
            selected.update(agents)

        # Pipeline-level health is Warning/Severe but no individual activities
        # were flagged (can happen with aggregate-only deviation signals) →
        # fall back to a broad runtime sweep.
        if not selected:
            selected.add(AgentRole.RUNTIME_INTELLIGENCE)

        if len(selected) > 1:
            selected.add(AgentRole.DEPENDENCY_LINEAGE)

        return selected

    async def dispatch_intelligence_agents(
        self, ctx: InvestigationContext, agents: set[AgentRole], state: InvestigationState
    ) -> list[AgentFinding]:
        coros = []
        dispatched_roles = []
        for role in agents:
            agent = self._sub_agent_pool.get(role)
            if agent is None:
                continue
            coros.append(agent.investigate(ctx))
            dispatched_roles.append(role)
            state.dispatched_agents.append(role.value)
        self._current_stage = "intelligence_agents_seconds"           # ADD
        self._active_agents = [r.value for r in dispatched_roles]
        await self.save_investigation_state(state, "agents_dispatched")  # ADD

        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
        finally:
            self._active_agents = [] 
        findings: list[AgentFinding] = []
        errors: list[str] = []
        for role, r in zip(dispatched_roles, results):
            if isinstance(r, Exception):
                errors.append(f"{role.value}: {type(r).__name__}: {r}")
                print(f"[ERROR] {role.value} raised an exception: {type(r).__name__}: {r}")
                continue
            if isinstance(r, list):
                findings.extend(r)
            elif r is not None:
                findings.append(r)

        if dispatched_roles and not findings and errors:
            raise RuntimeError(
                "All dispatched intelligence agents failed - investigation "
                "aborted. Errors:\n  " + "\n  ".join(errors)
            )
        return findings

    async def save_investigation_state(self, state: InvestigationState,
                                       checkpoint_label: str) -> None:
        
        print(f"[checkpoint:{checkpoint_label}] agents_dispatched={state.dispatched_agents}")
        self.checkpoints.append({
            "sequence": len(self.checkpoints) + 1,
            "label": checkpoint_label,
            "at": datetime.now(timezone.utc).isoformat(),
            "dispatched_agents": list(state.dispatched_agents),
            "findings_count": len(state.findings),
            "validated_findings_count": len(state.validated_findings),
            "root_causes_count": len(state.root_causes),
            "recommendations_count": len(state.recommendations),
        })
        await self._flush_live_files(state)

    async def _put(self, ctx: InvestigationContext, filename: str, payload) -> None:
        await self.document_store.put(ctx.service, self.investigation_id, filename, payload)

    async def _flush_live_files(self, state: InvestigationState) -> None:
        """Overwrites the files that grow during the run with their full
        current content."""
        ctx = state.context
        await self._put(ctx, CHECKPOINTS, self.checkpoints)
        await self._put(ctx, AGENTS_METADATA, ctx.agent_run_log)
        await self._put(ctx, PLANNER, self.planner_doc)
        await self._put(ctx, MANIFEST, self._manifest(state))

    def _manifest(self, state: InvestigationState) -> dict:
        """The index file: status, error, timings, token totals, counters,
        and the list of sibling files - read this first."""
        ctx = state.context
        runs = ctx.agent_run_log
        return {
            "investigation_id": self.investigation_id,
            "investigation_type": getattr(self, "_investigation_type", "runtime"),
            "status": self._status,
            "error": self._error,
            "skipped_reason": self._skipped_reason,
            "message": self._message,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "total_seconds_so_far": round(perf_counter() - self._t0, 3),
            "subscription_id": ctx.subscription_id,
            "service": ctx.service,
            "resource_group": ctx.resource_group,
            "workspace_name": ctx.workspace_name,
            "item_type": ctx.item_type,
            "item_name": ctx.item_name,
            "pipeline_health": ctx.pipeline_health,
            "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
            "pipeline_deviation_seconds": ctx.pipeline_deviation_seconds,
            "stage_timings": self.stage_timings,
            "current_stage": getattr(self, "_current_stage", None),
            "active_agents": list(getattr(self, "_active_agents", [])),
            "dispatched_agents": list(state.dispatched_agents),
            "agent_runs": len(runs),
            "agent_failures": [r["agent"] for r in runs if r.get("status") == "failed"],
            "total_input_tokens": sum(r.get("input_tokens") or 0 for r in runs),
            "total_output_tokens": sum(r.get("output_tokens") or 0 for r in runs),
            "findings_count": len(state.findings),
            "validated_findings_count": len(state.validated_findings),
            "root_causes_count": len(state.root_causes),
            "recommendations_count": len(state.recommendations),
            "files": [
                TRIGGER_PAYLOAD, ENRICHMENT, PLANNER, AGENTS_METADATA,
                FINDINGS, VALIDATED_FINDINGS, ROOT_CAUSES, RECOMMENDATIONS,
                IMPACT, FINAL_REPORT, CHECKPOINTS, MANIFEST,
            ],
        }

    def _stage_record(self, *, stage: str, agent, started_at: str, t0: float,
                      inputs: dict, outputs: dict, status: str = "success",
                      error: str = None, method: str = None) -> dict:
        """DETAILED metadata record for a downstream pipeline stage
        (evidence validation, root cause, recommendation, impact, report)"""
        meta = getattr(agent, "_last_call_meta", {}) or {}
        return {
            "stage": stage,
            "agent": getattr(agent, "role_name", stage),
            "model": getattr(agent, "model", None),
            "provider": method or getattr(agent, "_last_provider", "deterministic"),
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": round(perf_counter() - t0, 3),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "stop_reason": meta.get("stop_reason"),
            "system_prompt": getattr(agent, "system_prompt", None),
            "prompt": getattr(agent, "_last_prompt", None),
            "raw_response": getattr(agent, "_last_raw", None),
            "inputs": inputs,
            "outputs": outputs,
            "status": status,
            "error": error,
        }

    # ---- enrichment ---------------------------------------------------

    async def enrich_context(self, ctx: InvestigationContext) -> None:

        account = resolve_storage_account(ctx.subscription_id, self.storage_account)
        store = TelemetryStore(account)
        await TelemetryEnricher(store).enrich(ctx)

    # ---- orchestration loop -----------------------------------------

    @staticmethod
    def new_investigation_id() -> str:
        return f"inv_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    async def run_investigation(self, deviation_payload: dict,
                                investigation_id: Optional[str] = None) -> InvestigationState:
        ctx = self.load_baseline_deviation(deviation_payload)
        state = InvestigationState(context=ctx)

        # ---- investigation identity + storage --------------------------
    
        self.investigation_id = investigation_id or self.new_investigation_id()
        started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = perf_counter()
        stage_timings: dict[str, float] = {}

        account = resolve_storage_account(ctx.subscription_id, self.storage_account)
        store = TelemetryStore(account)
        if self.persistence is not None:          
            self.document_store = self.persistence
        else:
            self.document_store = build_document_store(telemetry_store=store)

        # ---- per-investigation folder: separate file per artifact ------
        # investigations/{service}/{investigation_id}/*.json
        self._status = "running"
        self._error = None
        self._started_at = started_at
        self._ended_at = None
        self._skipped_reason = None
        self._message = None
        self.stage_timings = stage_timings
        self.checkpoints: list[dict] = []
        self.planner_doc: dict = {"rounds": [], "followup_rounds": []}
        self._active_agents: list[str] = ["planner_agent"]
        self._current_stage: str = "queued"
        self._investigation_type = "runtime"

        await self._put(ctx, TRIGGER_PAYLOAD, {
            "investigation_id": self.investigation_id,
            "received_at": started_at,
            "subscription_id": ctx.subscription_id,
            "service": ctx.service,
            "resource_group": ctx.resource_group,
            "workspace_name": ctx.workspace_name,
            "item_type": ctx.item_type,
            "item_name": ctx.item_name,
            "payload": ctx.raw_payload,
        })

        await self.save_investigation_state(state, "context_loaded")

        try:
            # ---- MANDATORY ENRICHMENT (before any agent dispatch) ------
            self._current_stage = "enrichment_seconds"
            t = perf_counter()
            if ctx.investigation_type != "cost":
                await TelemetryEnricher(store).enrich(ctx)
            else:
                ctx.enriched = True
            stage_timings["enrichment_seconds"] = round(perf_counter() - t, 3)
            await self._put(ctx, ENRICHMENT, ctx.enrichment)
            await self.save_investigation_state(state, "context_enriched")

            # ---- intelligence agent rounds ------------------------------
            t = perf_counter()
            round_num = 0
            pending_agents = self.select_agents_to_dispatch(ctx)

            if not pending_agents:
                # Pipeline health is Healthy AND no activities are in Warning/Severe state
                stage_timings["intelligence_agents_seconds"] = 0.0
                message = (
                    f"'{ctx.item_name}' ({ctx.service}/{ctx.item_type}) is Healthy "
                    f"(deviation {ctx.pipeline_deviation_pct:+.2f}%, "
                    f"{ctx.pipeline_deviation_seconds:+.2f}s vs baseline) - "
                    "no degraded activities were found, so no investigation "
                    "was performed."
                )
                state.final_report = {
                    "investigation_id": self.investigation_id,
                    "skipped": True,
                    "skipped_reason": "pipeline_healthy",
                    "message": message,
                    "pipeline_health": ctx.pipeline_health,
                    "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
                    "pipeline_deviation_seconds": ctx.pipeline_deviation_seconds,
                    "suggested_fixes": [],
                    "apply_fix_payloads": [],
                }
                for filename in (FINDINGS, VALIDATED_FINDINGS, ROOT_CAUSES,
                                 RECOMMENDATIONS, IMPACT):
                    await self._put(ctx, filename, [])
                await self._put(ctx, FINAL_REPORT, state.final_report)

                self._status = "completed"
                self._skipped_reason = "pipeline_healthy"
                self._message = message
                self._ended_at = datetime.now(timezone.utc).isoformat()
                await self.save_investigation_state(state, "skipped_pipeline_healthy")
                return state

            while pending_agents and round_num < MAX_PLANNER_ROUNDS:
                round_num += 1
                self.planner_doc["rounds"].append({
                    "round": round_num,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "selected_agents": sorted(r.value for r in pending_agents),
                    "selection_basis": {
                        "degraded_activities": [
                            {"name": a.activity_name, "type": a.activity_type,
                             "deviation_pct": a.deviation_pct, "health": a.health}
                            for a in ctx.degraded_activities()
                        ],
                    },
                })
                findings = await self.dispatch_intelligence_agents(ctx, pending_agents, state)
                state.findings.extend(findings)
                await self.save_investigation_state(state, f"round_{round_num}_complete")

               
                pending_agents = self._followup_agents(findings, state.dispatched_agents)
            stage_timings["intelligence_agents_seconds"] = round(perf_counter() - t, 3)
            await self._put(ctx, FINDINGS, [f.to_dict() for f in state.findings])

            # ---- Hub: planner consolidates findings from all dispatched
            self._current_stage = "planner_synthesis_seconds"
            self._active_agents = ["planner_agent"]
            t_hub = perf_counter()
            await self.save_investigation_state(state, "planner_consolidating_findings")
            await asyncio.sleep(0.6)
            stage_timings["planner_synthesis_seconds"] = round(perf_counter() - t_hub, 3)

            # ---- Evidence Validation Agent - strict "verified" gatekeeper --
            self._current_stage = "evidence_validation_seconds"
            self._active_agents = ["evidence_validation_agent"]
            await self.save_investigation_state(state, "evidence_validation_started")
            t = perf_counter()
            stage_start = datetime.now(timezone.utc).isoformat()
            pre = [(f.agent, f.summary, f.confidence) for f in state.findings]
            state.validated_findings = await self.evidence_validation_agent.validate(ctx, state.findings)
            verified_keys = {(f.agent, f.summary) for f in state.validated_findings}
            ctx.agent_run_log.append(self._stage_record(
                stage="evidence_validation", agent=self.evidence_validation_agent,
                started_at=stage_start, t0=t, method="deterministic",
                inputs={
                    "findings_in": len(pre),
                    "findings_by_agent": {a: sum(1 for x in pre if x[0] == a)
                                          for a in {x[0] for x in pre}},
                    "min_confidence_to_verify": _EV_MIN_CONFIDENCE(),
                },
                outputs={
                    "validated_count": len(state.validated_findings),
                    "rejected_count": len(pre) - len(state.validated_findings),
                    "decisions": [
                        {"agent": a, "summary": s[:120], "confidence_in": c,
                         "verified": (a, s) in verified_keys}
                        for a, s, c in pre
                    ],
                    "validated_findings": [f.to_dict() for f in state.validated_findings],
                },
            ))
            stage_timings["evidence_validation_seconds"] = round(perf_counter() - t, 3)
            await self._put(ctx, VALIDATED_FINDINGS,
                            [f.to_dict() for f in state.validated_findings])
            await self.save_investigation_state(state, "evidence_validated")

            # ---- Root Cause Agent ------------------------------------------
            self._current_stage = "root_cause_seconds"
            self._active_agents = ["root_cause_agent"]
            await self.save_investigation_state(state, "root_cause_started")
            t = perf_counter()
            stage_start = datetime.now(timezone.utc).isoformat()
            state.root_causes = await self.root_cause_agent.analyze(ctx, state.validated_findings)
            ctx.agent_run_log.append(self._stage_record(
                stage="root_cause", agent=self.root_cause_agent,
                started_at=stage_start, t0=t,
                inputs={"validated_findings_in": len(state.validated_findings)},
                outputs={"root_causes_count": len(state.root_causes),
                         "root_causes": [asdict(rc) for rc in state.root_causes]},
            ))
            stage_timings["root_cause_seconds"] = round(perf_counter() - t, 3)
            await self._put(ctx, ROOT_CAUSES, [asdict(rc) for rc in state.root_causes])
            await self.save_investigation_state(state, "root_cause_complete")

            # ---- Recommendation & Validation Agent -------------------------
            self._current_stage = "recommendation_seconds"
            self._active_agents = ["recommendation_agent"]
            await self.save_investigation_state(state, "recommendation_started")
            t = perf_counter()
            stage_start = datetime.now(timezone.utc).isoformat()
            state.recommendations = await self.recommendation_agent.generate(ctx, state.root_causes)
            ctx.agent_run_log.append(self._stage_record(
                stage="recommendation", agent=self.recommendation_agent,
                started_at=stage_start, t0=t,
                inputs={"root_causes_in": len(state.root_causes)},
                outputs={"recommendations_count": len(state.recommendations),
                         "recommendations": [asdict(r) for r in state.recommendations],
                         "scores": [{"title": r.title,
                                     "score": self.recommendation_agent.score_recommendation(r)}
                                    for r in state.recommendations]},
            ))
            stage_timings["recommendation_seconds"] = round(perf_counter() - t, 3)
            await self._put(ctx, RECOMMENDATIONS, [asdict(r) for r in state.recommendations])
            await self.save_investigation_state(state, "recommendations_generated")

            # ---- Impact Agent ----------------------------------------------
            self._current_stage = "impact_seconds"
            self._active_agents = ["impact_agent"]
            await self.save_investigation_state(state, "impact_started")
            t = perf_counter()
            stage_start = datetime.now(timezone.utc).isoformat()
            state.impact_summary = await self.impact_agent.assess(ctx, state.recommendations)
            ctx.agent_run_log.append(self._stage_record(
                stage="impact", agent=self.impact_agent,
                started_at=stage_start, t0=t,
                method=("mcp_gpt5" if getattr(self.impact_agent, "mcp_gpt5_caller", None) else "deterministic"),
                inputs={"recommendations_in": len(state.recommendations)},
                outputs={"impact_summary": state.impact_summary},
            ))
            stage_timings["impact_seconds"] = round(perf_counter() - t, 3)
            await self._put(ctx, IMPACT, state.impact_summary)
            await self.save_investigation_state(state, "impact_assessed")

            # ---- Action Plan & Report Agent --------------------------------
            self._current_stage = "report_seconds"
            self._active_agents = ["action_plan_report_agent"]
            await self.save_investigation_state(state, "report_started")
            t = perf_counter()
            stage_start = datetime.now(timezone.utc).isoformat()
            state.final_report = await self.report_agent.generate(state)
            state.final_report["investigation_id"] = self.investigation_id
            ctx.agent_run_log.append(self._stage_record(
                stage="action_plan_report", agent=self.report_agent,
                started_at=stage_start, t0=t,
                method=("mcp_gpt5" if getattr(self.report_agent, "mcp_gpt5_caller", None) else "deterministic"),
                inputs={"root_causes_in": len(state.root_causes),
                        "recommendations_in": len(state.recommendations)},
                outputs={"suggested_fixes_count": len(state.final_report.get("suggested_fixes", [])),
                         "apply_fix_payloads_count": len(state.final_report.get("apply_fix_payloads", []))},
            ))
            stage_timings["report_seconds"] = round(perf_counter() - t, 3)
            await self._put(ctx, FINAL_REPORT, state.final_report)

            self._status = "completed"
            self._ended_at = datetime.now(timezone.utc).isoformat()
            await self.save_investigation_state(state, "report_complete")
            return state

        except Exception as e:
            self._status = "failed"
            self._error = f"{type(e).__name__}: {e}"
            self._ended_at = datetime.now(timezone.utc).isoformat()
            await self._flush_live_files(state)
            raise

    async def run_cost_investigation(
        self,
        cost_payload: dict,
        investigation_id: Optional[str] = None,
    ) -> InvestigationState:
            ctx = self.load_baseline_deviation(
                cost_payload,
                investigation_type="cost",
            )
            state = InvestigationState(context=ctx)
            print("[run_cost_investigation] investigation_id:", investigation_id)

            # ---- investigation identity + storage --------------------------
        
            self.investigation_id = investigation_id or self.new_investigation_id()
            started_at = datetime.now(timezone.utc).isoformat()
            self._t0 = perf_counter()
            stage_timings: dict[str, float] = {}

            account = resolve_storage_account(ctx.subscription_id, self.storage_account)
            store = TelemetryStore(account)
            if self.persistence is not None:          
                self.document_store = self.persistence
            else:
                self.document_store = build_document_store(telemetry_store=store)

            # ---- per-investigation folder: separate file per artifact ------
            # investigations/{service}/{investigation_id}/*.json
            self._status = "running"
            self._error = None
            self._started_at = started_at
            self._ended_at = None
            self._skipped_reason = None
            self._message = None
            self.stage_timings = stage_timings
            self.checkpoints: list[dict] = []
            self.planner_doc: dict = {"rounds": [], "followup_rounds": []}
            self._active_agents: list[str] = ["planner_agent"]
            self._current_stage: str = "queued"
            self._investigation_type = "cost"

            await self._put(ctx, TRIGGER_PAYLOAD, {
                "investigation_id": self.investigation_id,
                "received_at": started_at,
                "subscription_id": ctx.subscription_id,
                "service": ctx.service,
                "resource_group": ctx.resource_group,
                "workspace_name": ctx.workspace_name,
                "item_type": ctx.item_type,
                "item_name": ctx.item_name,
                "payload": ctx.raw_payload,
            })

            await self.save_investigation_state(state, "context_loaded")

            try:
                # ---- MANDATORY ENRICHMENT (before any agent dispatch) ------
                self._current_stage = "enrichment_seconds"
                ctx.enriched = True
                ctx.enrichment = {}

                await self._put(ctx, ENRICHMENT, {})
                await self.save_investigation_state(state, "cost_context_ready")

                # ---- intelligence agent rounds ------------------------------
                self._current_stage = "intelligence_agents_seconds"
                self._active_agents = ["cost_intelligence"]
                await self.save_investigation_state(state, "agents_dispatched")
                t = perf_counter()
                cost_agent = self._sub_agent_pool[AgentRole.COST_INTELLIGENCE]

                findings = await cost_agent.investigate(ctx)

                state.findings.extend(findings)

                state.dispatched_agents.append(
                    AgentRole.COST_INTELLIGENCE.value
                )

                await self.save_investigation_state(
                    state,
                    "cost_agent_complete",
                )
                stage_timings["intelligence_agents_seconds"] = round(perf_counter() - t, 3)
                await self._put(ctx, FINDINGS, [f.to_dict() for f in state.findings])

                # ---- Hub: planner consolidates the cost agent's findings
                # before handing off to evidence validation ----------------
                self._current_stage = "planner_synthesis_seconds"
                self._active_agents = ["planner_agent"]
                t_hub = perf_counter()
                await self.save_investigation_state(state, "planner_consolidating_findings")
                await asyncio.sleep(0.6)
                stage_timings["planner_synthesis_seconds"] = round(perf_counter() - t_hub, 3)

                # ---- Evidence Validation Agent - strict "verified" gatekeeper --
                self._current_stage = "evidence_validation_seconds"
                self._active_agents = ["evidence_validation_agent"]
                await self.save_investigation_state(state, "evidence_validation_started")
                t = perf_counter()
                stage_start = datetime.now(timezone.utc).isoformat()
                pre = [(f.agent, f.summary, f.confidence) for f in state.findings]
                state.validated_findings = await self.evidence_validation_agent.validate(ctx, state.findings)
                print("Before Evidence Validation")
                print("Findings Count :", len(state.findings))
                for f in state.findings:
                    print("--------------------------------")
                    print("Agent      :", f.agent)
                    print("Confidence :", f.confidence)
                    print("Status     :", f.status)
                    print("Affected   :", f.affected_activities)
                verified_keys = {(f.agent, f.summary) for f in state.validated_findings}
                ctx.agent_run_log.append(self._stage_record(
                    stage="evidence_validation", agent=self.evidence_validation_agent,
                    started_at=stage_start, t0=t, method="deterministic",
                    inputs={
                        "findings_in": len(pre),
                        "findings_by_agent": {a: sum(1 for x in pre if x[0] == a)
                                            for a in {x[0] for x in pre}},
                        "min_confidence_to_verify": _EV_MIN_CONFIDENCE(),
                    },
                    outputs={
                        "validated_count": len(state.validated_findings),
                        "rejected_count": len(pre) - len(state.validated_findings),
                        "decisions": [
                            {"agent": a, "summary": s[:120], "confidence_in": c,
                            "verified": (a, s) in verified_keys}
                            for a, s, c in pre
                        ],
                        "validated_findings": [f.to_dict() for f in state.validated_findings],
                    },
                ))
                stage_timings["evidence_validation_seconds"] = round(perf_counter() - t, 3)
                await self._put(ctx, VALIDATED_FINDINGS,
                                [f.to_dict() for f in state.validated_findings])
                await self.save_investigation_state(state, "evidence_validated")

                # ---- Root Cause Agent ------------------------------------------
                self._current_stage = "root_cause_seconds"
                self._active_agents = ["root_cause_agent"]
                await self.save_investigation_state(state, "root_cause_started")
                t = perf_counter()
                stage_start = datetime.now(timezone.utc).isoformat()
                state.root_causes = await self.root_cause_agent.analyze(ctx, state.validated_findings)
                ctx.agent_run_log.append(self._stage_record(
                    stage="root_cause", agent=self.root_cause_agent,
                    started_at=stage_start, t0=t,
                    inputs={"validated_findings_in": len(state.validated_findings)},
                    outputs={"root_causes_count": len(state.root_causes),
                            "root_causes": [asdict(rc) for rc in state.root_causes]},
                ))
                stage_timings["root_cause_seconds"] = round(perf_counter() - t, 3)
                await self._put(ctx, ROOT_CAUSES, [asdict(rc) for rc in state.root_causes])
                await self.save_investigation_state(state, "root_cause_complete")

                # ---- Recommendation & Validation Agent -------------------------
                self._current_stage = "recommendation_seconds"
                self._active_agents = ["recommendation_agent"]
                await self.save_investigation_state(state, "recommendation_started")
                t = perf_counter()
                stage_start = datetime.now(timezone.utc).isoformat()
                state.recommendations = await self.recommendation_agent.generate(ctx, state.root_causes)
                ctx.agent_run_log.append(self._stage_record(
                    stage="recommendation", agent=self.recommendation_agent,
                    started_at=stage_start, t0=t,
                    inputs={"root_causes_in": len(state.root_causes)},
                    outputs={"recommendations_count": len(state.recommendations),
                            "recommendations": [asdict(r) for r in state.recommendations],
                            "scores": [{"title": r.title,
                                        "score": self.recommendation_agent.score_recommendation(r)}
                                        for r in state.recommendations]},
                ))
                stage_timings["recommendation_seconds"] = round(perf_counter() - t, 3)
                await self._put(ctx, RECOMMENDATIONS, [asdict(r) for r in state.recommendations])
                await self.save_investigation_state(state, "recommendations_generated")

                # ---- Impact Agent ----------------------------------------------
                self._current_stage = "impact_seconds"
                self._active_agents = ["impact_agent"]
                await self.save_investigation_state(state, "impact_started")
                t = perf_counter()
                stage_start = datetime.now(timezone.utc).isoformat()
                state.impact_summary = await self.impact_agent.assess(ctx, state.recommendations)
                ctx.agent_run_log.append(self._stage_record(
                    stage="impact", agent=self.impact_agent,
                    started_at=stage_start, t0=t,
                    method=("mcp_gpt5" if getattr(self.impact_agent, "mcp_gpt5_caller", None) else "deterministic"),
                    inputs={"recommendations_in": len(state.recommendations)},
                    outputs={"impact_summary": state.impact_summary},
                ))
                stage_timings["impact_seconds"] = round(perf_counter() - t, 3)
                await self._put(ctx, IMPACT, state.impact_summary)
                await self.save_investigation_state(state, "impact_assessed")

                # ---- Action Plan & Report Agent --------------------------------
                self._current_stage = "report_seconds"
                self._active_agents = ["action_plan_report_agent"]
                await self.save_investigation_state(state, "report_started")
                t = perf_counter()
                stage_start = datetime.now(timezone.utc).isoformat()
                state.final_report = await self.report_agent.generate(state)
                state.final_report["investigation_id"] = self.investigation_id
                ctx.agent_run_log.append(self._stage_record(
                    stage="action_plan_report", agent=self.report_agent,
                    started_at=stage_start, t0=t,
                    method=("mcp_gpt5" if getattr(self.report_agent, "mcp_gpt5_caller", None) else "deterministic"),
                    inputs={"root_causes_in": len(state.root_causes),
                            "recommendations_in": len(state.recommendations)},
                    outputs={"suggested_fixes_count": len(state.final_report.get("suggested_fixes", [])),
                            "apply_fix_payloads_count": len(state.final_report.get("apply_fix_payloads", []))},
                ))
                stage_timings["report_seconds"] = round(perf_counter() - t, 3)
                await self._put(ctx, FINAL_REPORT, state.final_report)

                self._status = "completed"
                self._ended_at = datetime.now(timezone.utc).isoformat()
                await self.save_investigation_state(state, "report_complete")
                return state

            except Exception as e:
                self._status = "failed"
                self._error = f"{type(e).__name__}: {e}"
                self._ended_at = datetime.now(timezone.utc).isoformat()
                await self._flush_live_files(state)
                raise

    def _followup_agents(self, new_findings: list[AgentFinding], already_dispatched: list[str]) -> set[AgentRole]:
        followups: set[AgentRole] = set()
        for f in new_findings:
            needs_cfg = f.evidence.get("requires_configuration_review")
            if needs_cfg and AgentRole.CONFIGURATION.value not in already_dispatched:
                followups.add(AgentRole.CONFIGURATION)
            needs_dep = f.evidence.get("requires_dependency_review")
            if needs_dep and AgentRole.DEPENDENCY_LINEAGE.value not in already_dispatched:
                followups.add(AgentRole.DEPENDENCY_LINEAGE)
        return followups