"""
Runtime Intelligence Agent
Model: Claude Sonnet 4

"Runtime Intelligence Agent"
to match your capability-based naming (works across Synapse pipelines,
ADF pipelines, Databricks job tasks, future Fabric items alike 

Tools: analyze_activity_flow, analyze_specific_activity, analyze_copy_activity,
       analyze_control_flow, detect_sequencing_bottlenecks,
       calculate_activity_contribution
"""

from __future__ import annotations
import json
from schemas import InvestigationContext, ActivityDeviation, AgentFinding
from sub_agents.base_agent import BaseIntelligenceAgent

CONTROL_FLOW_TYPES = {"IfCondition", "ForEach", "Switch", "Wait", "SetVariable", "GetMetadata"}
COPY_TYPES = {"Copy"}


class RuntimeIntelligenceAgent(BaseIntelligenceAgent):
    enrichment_keys = ("recent_runs",)
    role_name = "runtime_intelligence_agent"
    model = "claude-sonnet-4-6"
    system_prompt = (
        "You analyze pipeline/job orchestration behavior across Azure data "
        "services (Synapse, ADF, Databricks, Fabric). Identify which activities "
        "contribute most to runtime deviation, and whether the cause is "
        "sequencing/control-flow, copy throughput, or unexplained drift. "
        "Return ONLY a JSON array of finding objects with keys: "
        "summary, confidence (0-1), evidence (object), affected_activities (list of str)."
    )

    # --- tools -------------------------------------------------------

    def analyze_activity_flow(self, ctx: InvestigationContext) -> dict:
        return {
            "total_activities": len(ctx.activities),
            "degraded_count": len(ctx.degraded_activities()),
            "activity_types": sorted(ctx.activity_types_present()),
        }

    def analyze_specific_activity(self, activity: ActivityDeviation) -> dict:
        return {
            "name": activity.activity_name,
            "type": activity.activity_type,
            "deviation_pct": activity.deviation_pct,
            "deviation_seconds": activity.deviation_seconds,
            "success_rate_pct": activity.success_rate_pct,
        }

    def analyze_copy_activity(self, activity: ActivityDeviation) -> dict:
        if activity.activity_type not in COPY_TYPES:
            return {}
        return {
            "note": "Copy activity - check DIU usage, staging, partitioning",
            "deviation_pct": activity.deviation_pct,
        }

    def analyze_control_flow(self, activity: ActivityDeviation) -> dict:
        if activity.activity_type not in CONTROL_FLOW_TYPES:
            return {}
        return {
            "note": f"{activity.activity_type} control-flow activity",
            "deviation_pct": activity.deviation_pct,
        }

    def detect_sequencing_bottlenecks(self, activities: list[ActivityDeviation]) -> list[str]:
        # Activities whose latest duration is far above p95 of baseline
        # are likely serialization/bottleneck candidates.
        flags = []
        for a in activities:
            if a.p95_duration_seconds and a.latest_duration_seconds > a.p95_duration_seconds:
                flags.append(a.activity_name)
        return flags

    def calculate_activity_contribution(self, ctx: InvestigationContext) -> dict:
        total_dev = sum(abs(a.deviation_seconds) for a in ctx.activities) or 1.0
        return {
            a.activity_name: round(abs(a.deviation_seconds) / total_dev, 3)
            for a in ctx.activities
        }

    # --- agent contract -----------------------------------------------

    def relevant_activities(self, ctx: InvestigationContext):
        # Runtime agent looks at ALL activities for sequencing context,
        # not just degraded ones, since bottlenecks can mask upstream waits.
        return ctx.activities or ctx.degraded_activities()

    def build_prompt(self, ctx: InvestigationContext, activities: list[ActivityDeviation]) -> str:
        flow = self.analyze_activity_flow(ctx)
        bottlenecks = self.detect_sequencing_bottlenecks(ctx.activities)
        contribution = self.calculate_activity_contribution(ctx)

        payload = {
            "service": ctx.service,
            "item_type": ctx.item_type,
            "item_name": ctx.item_name,
            "pipeline_health": ctx.pipeline_health,
            "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
            "activity_flow_summary": flow,
            "sequencing_bottleneck_candidates": bottlenecks,
            "activity_contribution_to_deviation": contribution,
            "activities": [self.analyze_specific_activity(a) for a in activities],
        }
        return (
            "Investigate runtime/orchestration deviation for this item:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Return JSON array of findings."
        )