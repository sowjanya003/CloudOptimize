"""
Configuration Agent
Model: Claude Sonnet 4
Tools: analyze_activity_policies, analyze_spark_pool_config,
       analyze_dataset_and_linked_service, check_integration_runtime_and_session_config,
       validate_auto_scaling_settings
"""
from __future__ import annotations
import json
from schemas import InvestigationContext, ActivityDeviation
from sub_agents.base_agent import BaseIntelligenceAgent


class ConfigurationAgent(BaseIntelligenceAgent):
    enrichment_keys = ("definition", "notebook_session_config")
    role_name = "configuration_agent"
    model = "claude-sonnet-4-6"
    system_prompt = (
        "You analyze infrastructure and runtime configuration - Spark pool "
        "sizing/autoscale, activity retry/timeout policy, linked service and "
        "dataset settings - for misconfigurations that could explain runtime "
        "deviation. Return ONLY a JSON array of finding objects with keys: "
        "summary, confidence (0-1), evidence (object), affected_activities (list)."
    )

    async def analyze_spark_pool_config(self, ctx: InvestigationContext, pool_name: str) -> dict:
        if self.asset_loader is None or not pool_name:
            return {}
        asset = await self.asset_loader(
            service=ctx.service, resource_group=ctx.resource_group,
            workspace_name=ctx.workspace_name, item_type="spark_pool",
            item_name=pool_name,
        )
        meta = asset.get("extra_metadata", {})
        return {
            "node_count": meta.get("node_count"),
            "auto_scale_enabled": meta.get("auto_scale_enabled"),
            "auto_scale_min": meta.get("auto_scale_min"),
            "auto_scale_max": meta.get("auto_scale_max"),
        }

    def analyze_activity_policies(self, activity_raw: dict) -> dict:
        policy = activity_raw.get("policy", {})
        return {
            "timeout": policy.get("timeout"),
            "retry": policy.get("retry"),
            "retry_interval_in_seconds": policy.get("retry_interval_in_seconds"),
        }

    def validate_auto_scaling_settings(self, pool_config: dict) -> list[str]:
        flags = []
        if pool_config.get("auto_scale_min") == pool_config.get("auto_scale_max"):
            flags.append("Autoscale min==max - pool cannot scale under load")
        return flags

    def build_prompt(self, ctx: InvestigationContext, activities: list[ActivityDeviation]) -> str:
        payload = {
            "service": ctx.service,
            "item_name": ctx.item_name,
            "degraded_activities": [
                {"name": a.activity_name, "type": a.activity_type, "deviation_pct": a.deviation_pct}
                for a in activities
            ],
            "instruction": (
                "Use the item definition and per-notebook session configs in "
                "ADDITIONAL EVIDENCE below (timeouts, retries, pool sizing, "
                "driver/executor memory and cores, autoscale settings). Cite the "
                "exact config value in evidence."
            ),
        }
        return f"Investigate configuration root causes for:\n{json.dumps(payload, indent=2)}\nReturn JSON array of findings."