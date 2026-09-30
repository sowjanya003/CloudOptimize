"""
Cost Intelligence Agent
Model: Claude Sonnet 4

Capability-based investigation agent responsible for analyzing
Azure workspace cost anomalies.

Responsibilities
----------------
1. Receive cost anomaly from Planner.
2. Collect additional cost evidence.
3. Build investigation context.
4. Use Claude to investigate the anomaly.
5. Return standardized AgentFinding objects.

This agent DOES NOT determine the final root cause.
It only produces evidence-backed findings.
"""

from __future__ import annotations

import json
from typing import Any

from datetime import datetime, timezone
from time import perf_counter

from schemas import (
    InvestigationContext,
    ActivityDeviation,
)

from sub_agents.base_agent import (
    BaseIntelligenceAgent,
    anthropic_ready,
)


class CostIntelligenceAgent(BaseIntelligenceAgent):

    role_name = "cost_intelligence_agent"

    model = "claude-sonnet-4-6"

    enrichment_keys = ()

    system_prompt = """
        You are an Azure FinOps Cost Investigation Agent.

        You investigate Azure workspace-level cost anomalies.

        Do not lower confidence because
        additional telemetry is unavailable.

        Confidence should reflect how well
        the supplied evidence supports the findings,
        not whether additional telemetry could improve them.

        Rules:

        1. Only use evidence provided.

        2. Never invent evidence.

        3. Never perform mathematical calculations.
        Python has already calculated deviations.

        4. Explain WHY the anomaly most likely happened.

        5. Identify the Azure resources or workloads
        that contributed to the anomaly.

        6. Mention missing evidence whenever confidence
        is reduced.

        7. Do NOT determine the final root cause.
        That responsibility belongs to the Root Cause Agent.

        8. - Never speculate.
        - Only make claims supported by the supplied evidence.
        - If evidence is insufficient, explicitly state what additional telemetry is required.
        - Do not infer Spark, SQL, Pipeline or Notebook behavior unless such evidence is provided.

        9. -Confidence represents confidence in your
            analysis of the available evidence.
            -Do not reduce confidence solely because
            additional telemetry is unavailable.
            -Only reduce confidence when the supplied
            evidence is contradictory or unreliable.

        Return ONLY valid JSON.

        Expected Output

        [
            {
                "summary": "...",
                "confidence": 0.92,
                "evidence": {},
                "affected_activities": []
            }
        ]

        "evidence": {
        "supporting_metrics": { ... },
        "evidence_gaps": [
            "Pipeline execution statistics unavailable",
            "Spark utilization unavailable",
            "SQL pool utilization unavailable"
        ]
    }
    """

    # ---------------------------------------------------------
    # Internal Evidence Sources
    # ---------------------------------------------------------

    def _load_incident(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:
        """
        Returns the original planner payload.
        """
        return dict(ctx.raw_payload)
    
    def _workspace_identifier(
        self,
        ctx: InvestigationContext,
    ) -> str:

        payload = ctx.raw_payload

        return (
            payload.get("workspace_name")
            or payload.get("factory_name")
            or ctx.workspace_name
            or ctx.item_name
            or "workspace"
        )

    def _load_workspace_cost(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:
        """
        Extracts the workspace/factory level cost information
        from the planner payload.
        """

        payload = ctx.raw_payload

        return {
            "service": payload.get("service"),

            "workspace_name":
                payload.get("workspace_name")
                or payload.get("factory_name"),

            "resource_group": payload.get("resource_group"),

            "currency": payload.get("currency"),

            "current_cost": payload.get("total_cost"),

            "baseline_cost":
                payload.get("baseline_monthly_cost"),

            "deviation":
                payload.get("deviation", {}),

            "fetched_at":
                payload.get("fetched_at"),
        }

    def _load_cost_history(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:
        """
        Historical cost evidence.
        """

        payload = ctx.raw_payload

        return {

            "last_30_days":
                payload.get("last_30_days", {}),

            "last_6_months":
                payload.get("last_6_months", {}),

        }

    def _load_meter_breakdown(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:
        """
        Azure Cost Management meter breakdown.
        """

        payload = ctx.raw_payload

        return {

            "last_30_days":

                payload.get(
                    "last_30_days",
                    {}
                ).get(
                    "cost_by_meter",
                    {}
                ),

            "last_6_months":

                payload.get(
                    "last_6_months",
                    {}
                ).get(
                    "cost_by_meter",
                    {}
                ),

        }

    def _load_limitations(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:
        """
        Returns telemetry limitations exactly as provided.

        These are passed directly to Claude so that it
        understands the boundaries of the available evidence.
        """

        payload = ctx.raw_payload

        return {
            "note": payload.get("note")
        }

    # ---------------------------------------------------------
    # Cost Investigation Context
    # ---------------------------------------------------------

    def _build_investigation_summary(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:
        """
        Compact summary of the investigation.

        This becomes the first section shown to Claude
        before the detailed evidence.
        """

        payload = ctx.raw_payload

        deviation = payload.get("deviation", {})

        return {

            "service":
                payload.get("service"),

            "workspace":
                payload.get("workspace_name")
                or payload.get("factory_name"),

            "resource_group":
                payload.get("resource_group"),

            "currency":
                payload.get("currency"),

            "current_cost":
                payload.get("total_cost"),

            "baseline_cost":
                payload.get("baseline_monthly_cost"),

            "deviation_amount":
                deviation.get("deviation_amount"),

            "deviation_percentage":
                deviation.get("deviation_pct"),

            "status":
                deviation.get("status"),

            "investigation_time":
                payload.get("fetched_at")
        }
        
    # ---------------------------------------------------------
    # Investigation Summary
    # ---------------------------------------------------------

    def _build_investigation_summary(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:

        payload = ctx.raw_payload

        deviation = payload.get("deviation", {})

        return {

            "service":
                payload.get("service"),

            "workspace":
                payload.get("workspace_name")
                or payload.get("factory_name"),

            "resource_group":
                payload.get("resource_group"),

            "current_cost":
                payload.get("total_cost"),

            "baseline_cost":
                payload.get("baseline_monthly_cost"),

            "currency":
                payload.get("currency"),

            "deviation_percentage":
                deviation.get("deviation_pct"),

            "deviation_amount":
                deviation.get("deviation_amount"),

            "status":
                deviation.get("status"),
        }
    
     # ---------------------------------------------------------
    # Cost Investigation Context
    # ---------------------------------------------------------

    def _build_cost_context(
        self,
        ctx: InvestigationContext,
    ) -> dict[str, Any]:

        return {

            "incident":
                self._load_incident(ctx),

            "workspace_cost":
                self._load_workspace_cost(ctx),

            "historical_cost":
                self._load_cost_history(ctx),

            "meter_breakdown":
                self._load_meter_breakdown(ctx),

            "limitations":
                self._load_limitations(ctx),

        }
    
    #---------------------------------------------------------
    #---------------------------------------------------------

    async def investigate(self, ctx: InvestigationContext):
        """
        Cost investigations are workspace-level, not activity-level.

        Therefore we override BaseIntelligenceAgent.investigate()
        while keeping the same output contract:
            List[AgentFinding]
        """

        if not getattr(ctx, "enriched", False):
            raise RuntimeError(
                "CostIntelligenceAgent: InvestigationContext has not been enriched."
            )

        started_at = datetime.now(timezone.utc).isoformat()
        t0 = perf_counter()

        self._last_call_meta = {}

        prompt = ""
        raw_text = ""
        provider = "unknown"

        try:
            # Build prompt from cost evidence
            prompt = self.build_prompt(ctx, [])

            # Reuse the BaseAgent enrichment block if anything exists
            prompt += self._enrichment_block(ctx, [])

        except Exception as e:
            self._record_run(
                ctx,
                prompt=prompt,
                raw_text="",
                findings=[],
                started_at=started_at,
                t0=t0,
                provider=provider,
                status="failed",
                error=f"{type(e).__name__}: {e}",
            )
            raise

        try:

            if anthropic_ready():

                provider = "anthropic"

                raw_text = await self._call_anthropic(prompt)

                truncated = (
                    self._last_call_meta.get("stop_reason") == "max_tokens"
                )

                findings = self.parse_response(
                    ctx,
                    raw_text,
                    truncated=truncated,
                )
                workspace = self._workspace_identifier(ctx)
                for finding in findings:
                    finding.affected_activities = [workspace]

                self._record_run(
                    ctx,
                    prompt=prompt,
                    raw_text=raw_text,
                    findings=findings,
                    started_at=started_at,
                    t0=t0,
                    provider=provider,
                    status="truncated" if truncated else "success",
                )

                return findings

            raise RuntimeError(
                "No Anthropic provider configured."
            )

        except Exception:

            raise
    #---------------------------------------------------------
    # Prompt Builder
    #---------------------------------------------------------


    def build_prompt(
        self,
        ctx: InvestigationContext,
        activities,
    ) -> str:
        """
        Builds the investigation prompt sent to Claude.

        The Cost Intelligence Agent investigates only
        workspace-level cost evidence.

        It never performs numerical calculations and never
        invents pipeline-level attribution.
        """

        investigation_summary = self._build_investigation_summary(ctx)

        cost_context = self._build_cost_context(ctx)

        prompt = {

            "investigation_summary":
                investigation_summary,

            "workspace_cost":
                cost_context["workspace_cost"],

            "historical_cost":
                cost_context["historical_cost"],

            "meter_breakdown":
                cost_context["meter_breakdown"],

            "limitations":
                cost_context["limitations"],
        }

        return f"""
    Investigate the following Azure Cost anomaly.

    The investigation summary is provided first,
    followed by all available supporting evidence.

    Guidelines

    - Only use the supplied evidence.

    - Never invent evidence.

    - Never perform numerical calculations.

    - Azure Cost Management currently provides
    workspace-level cost only.

    - If evidence is insufficient,
    state that explicitly.
    

    Evidence

    {json.dumps(prompt, indent=2)}

    Return ONLY a JSON array.

    Expected Schema

    [
        {{
            "summary": "...",
            "confidence": 0.92,
            "evidence": {{
                "reasoning": "...",
                "supporting_metrics": {{}}
            }},
            "affected_activities": []
        }}
    ]
    """