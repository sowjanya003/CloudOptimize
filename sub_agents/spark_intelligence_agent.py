"""
Spark Intelligence Agent
Model: claude-sonnet-4-6

Scope: Spark execution config anti-patterns — repartition, caching,
       shuffle, skew, executor sizing, autoscale, pool config.
       Does NOT look at Python logic or SQL queries — those are separate agents.
"""
from __future__ import annotations
import json
from sub_agents.base_agent import BaseIntelligenceAgent
from schemas import InvestigationContext, ActivityDeviation

SPARK_BEARING_TYPES = {
    "SynapseNotebook", "notebook_task", "dlt_pipeline",
}

SPARK_SYSTEM = """You are an Spark Intelligence Agent.
Your ONLY job: find Spark execution and configuration anti-patterns.

Look for:
- Fixed repartition count mismatched to cluster size (e.g. repartition(200) on 3-node pool)
- Missing .cache() or .persist() before repeated DataFrame reuse
- Shuffle spill caused by insufficient executor memory
- Data skew in groupBy / join keys
- Dynamic executor allocation disabled when workload varies
- Autoscale min == max (pool cannot scale under load)
- Small file problem: many tiny files causing excessive task overhead
- Default shuffle partitions (200) not tuned to actual data volume

RULES:
- Return at most 2 findings sorted by severity.
- summary: maximum 30 words.
- evidence: maximum 5 keys, short values only.
- Do NOT report Python logic issues or SQL query issues — those are other agents.
- No prose, no markdown, no text outside the JSON array.

OUTPUT FORMAT (strict):
[
  {
    "summary": "...",
    "confidence": 0.0,
    "affected_activities": ["..."],
    "evidence": {"key": "value"}
  }
]"""


class SparkIntelligenceAgent(BaseIntelligenceAgent):
    enrichment_keys = ("notebooks", "notebook_session_config")
    role_name     = "spark_intelligence_agent"
    model         = "claude-sonnet-4-6"
    system_prompt = SPARK_SYSTEM

    def relevant_activities(self, ctx: InvestigationContext):
        return [
            a for a in ctx.degraded_activities()
            if a.activity_type in SPARK_BEARING_TYPES
        ]

    def build_prompt(self, ctx: InvestigationContext, activities: list[ActivityDeviation]) -> str:
        missing = [a.activity_name for a in activities
                   if a.activity_name not in ctx.enrichment.get("notebooks", {})]
        if missing:
            raise RuntimeError(
                f"{self.role_name}: no notebook source enriched for degraded "
                f"activities {missing} - refusing to analyze code it cannot see."
            )
        payload = {
            "service":   ctx.service,
            "item_name": ctx.item_name,
            "pipeline_health": ctx.pipeline_health,
            "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
            "degraded_spark_activities": [
                {
                    "name":            a.activity_name,
                    "type":            a.activity_type,
                    "deviation_pct":   a.deviation_pct,
                    "deviation_s":     a.deviation_seconds,
                    "avg_duration_s":  a.avg_duration_seconds,
                    "latest_duration_s": a.latest_duration_seconds,
                    "p95_s":           a.p95_duration_seconds,
                    "health":          a.health,
                }
                for a in activities
            ],
            "instruction": (
                "Use the REAL notebook source and session configs in ADDITIONAL "
                "EVIDENCE below (e.g. repartition counts vs executor counts, "
                "missing cache before reuse, shuffle-heavy operations). Cite the "
                "offending line/config in evidence. Do not speculate beyond the "
                "provided evidence."
            ),
        }
        return (
            f"Investigate Spark execution/config issues:\n"
            f"{json.dumps(payload, indent=2)}\n"
            f"Return JSON array of findings."
        )