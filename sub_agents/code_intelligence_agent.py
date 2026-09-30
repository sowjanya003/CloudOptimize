"""
Code Intelligence Agent
Model: Claude Opus 4
Tools: load_notebook_content, analyze_notebook_code, detect_spark_anti_patterns,
       suggest_code_refactoring, analyze_data_transformation_efficiency
"""
from __future__ import annotations
import json, re
from schemas import InvestigationContext, ActivityDeviation
from sub_agents.base_agent import BaseIntelligenceAgent

CODE_BEARING_TYPES = {"SynapseNotebook", "notebook_task", "dlt_pipeline"}
SPARK_ANTI_PATTERNS = {
    r"\.collect\s*\(": "collect() pulls full dataset to driver",
    r"udf\(": "Python UDF - check for vectorized/pandas_udf alternative",
    r"\.repartition\s*\(\s*\d+": "Fixed repartition count - verify cluster sizing matches",
    r"for\s+row\s+in\s+\w+\.collect\s*\(\)": "Row-by-row Python iteration after collect() - major anti-pattern",
}


class CodeIntelligenceAgent(BaseIntelligenceAgent):
    enrichment_keys = ("notebooks",)
    role_name = "code_intelligence_agent"
    model = "claude-opus-4-6"
    system_prompt = (
        "You analyze notebook/Spark code for runtime-degrading patterns: "
        "collect(), unmanaged UDFs, skew, bad repartitioning, manual Python "
        "iteration over driver-collected data. Return ONLY a JSON array of "
        "finding objects with keys: summary, confidence (0-1), evidence (object), "
        "affected_activities (list)."
    )

    def relevant_activities(self, ctx: InvestigationContext):
        return [a for a in ctx.degraded_activities() if a.activity_type in CODE_BEARING_TYPES]

    async def load_notebook_content(self, ctx: InvestigationContext, activity: ActivityDeviation) -> dict:
        if self.asset_loader is None:
            return {}
        return await self.asset_loader(
            service=ctx.service, resource_group=ctx.resource_group,
            workspace_name=ctx.workspace_name, item_type="notebook",
            item_name=activity.activity_name,
        )

    def detect_spark_anti_patterns(self, source_code: str) -> list[str]:
        hits = []
        for pattern, label in SPARK_ANTI_PATTERNS.items():
            if re.search(pattern, source_code):
                hits.append(label)
        return hits

    def analyze_data_transformation_efficiency(self, cells_count: int, anti_pattern_count: int) -> float:
        if cells_count == 0:
            return 1.0
        return max(0.0, 1.0 - (anti_pattern_count / max(cells_count, 1)))

    def build_prompt(self, ctx: InvestigationContext, activities: list[ActivityDeviation]) -> str:
        payload = {
            "service": ctx.service,
            "item_name": ctx.item_name,
            "code_bearing_degraded_activities": [
                {"name": a.activity_name, "type": a.activity_type, "deviation_pct": a.deviation_pct}
                for a in activities
            ],
            "note": "Notebook source pulled via load_notebook_content + scanned with detect_spark_anti_patterns at runtime.",
        }
        return f"Investigate code/Spark transformation efficiency for:\n{json.dumps(payload, indent=2)}\nReturn JSON array of findings."