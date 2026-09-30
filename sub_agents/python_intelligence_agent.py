"""
Python Intelligence Agent
Model: claude-sonnet-4-6
"""
from __future__ import annotations
import json
from sub_agents.base_agent import BaseIntelligenceAgent
from schemas import InvestigationContext, ActivityDeviation

PYTHON_BEARING_TYPES = {
    "SynapseNotebook", "notebook_task", "dlt_pipeline",
}

PYTHON_SYSTEM = """You are an  Python Intelligence Agent.
Your ONLY job: find Python-specific anti-patterns in notebook/job code.

Look for:
- collect() pulling full datasets to the driver
- Row-by-row Python iteration over collected data (for row in df.collect())
- Python UDFs instead of native Spark/SQL functions
- pandas interop causing serialization overhead (toPandas() on large frames)
- Driver-side aggregation instead of distributed Spark aggregation
- sparkContext.parallelize() to reconstruct distributed data after collect()

RULES:
- Base findings ONLY on the notebook source provided in ADDITIONAL EVIDENCE.
- Return at most 2 findings sorted by severity.
- summary: maximum 30 words.
- evidence: maximum 5 keys - include the offending code line verbatim.
- Do NOT report SQL issues or Spark config issues - those are other agents.
- No prose, no markdown, no text outside the JSON array.

OUTPUT FORMAT (strict):
[
  {
    "summary": "...",
    "confidence": 0.0,
    "affected_activities": ["..."],
    "evidence": {"offending_pattern": "...", "line": "..."}
  }
]"""


class PythonIntelligenceAgent(BaseIntelligenceAgent):
    enrichment_keys = ("notebooks",)
    role_name     = "python_intelligence_agent"
    model         = "claude-sonnet-4-6"
    system_prompt = PYTHON_SYSTEM

    def relevant_activities(self, ctx: InvestigationContext):
        return [
            a for a in ctx.degraded_activities()
            if a.activity_type in PYTHON_BEARING_TYPES
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
            "degraded_activities": [
                {
                    "name":          a.activity_name,
                    "type":          a.activity_type,
                    "deviation_pct": a.deviation_pct,
                    "deviation_s":   a.deviation_seconds,
                }
                for a in activities
            ],
            "instruction": (
                "Analyze ONLY the notebook source provided in ADDITIONAL EVIDENCE "
                "below. Cite the offending line(s) in evidence."
            ),
        }
        return (
            f"Find Python anti-patterns in the enriched notebook code:\n"
            f"{json.dumps(payload, indent=2)}\n"
            f"Return JSON array of findings."
        )
