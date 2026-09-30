"""
SQL Intelligence Agent
Model: claude-sonnet-4-6

Scope: SQL queries, Spark SQL transformations, joins, full scans,
       missing partition pushdown, cartesian products, subquery issues.
       Does NOT look at Python code or Spark config — those are separate agents.
"""
from __future__ import annotations
import json
from sub_agents.base_agent import BaseIntelligenceAgent
from schemas import InvestigationContext, ActivityDeviation

SQL_BEARING_TYPES = {
    "SynapseNotebook", "notebook_task", "sql_task",
    "sql_script", "dlt_pipeline",
}

SQL_SYSTEM = """You are an SQL Intelligence Agent.
Your ONLY job: find SQL and Spark SQL anti-patterns causing runtime deviation.

Look for:
- SELECT * full column scans
- Missing partition filter pushdown (no WHERE on partition columns)
- Cartesian joins or cross joins
- Expensive subqueries or correlated subqueries
- Non-broadcast joins on small tables (should use broadcast hint)
- Aggregations without pre-filtering
- ORDER BY on unbounded datasets

RULES:
- Return at most 2 findings sorted by severity.
- summary: maximum 30 words.
- evidence: maximum 5 keys, short values only.
- Do NOT report Python issues or Spark config issues — those are other agents.
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


class SQLIntelligenceAgent(BaseIntelligenceAgent):
    role_name     = "sql_intelligence_agent"
    model         = "claude-sonnet-4-6"
    system_prompt = SQL_SYSTEM
    enrichment_keys = ("notebooks",)

    def relevant_activities(self, ctx: InvestigationContext):
        return [
            a for a in ctx.degraded_activities()
            if a.activity_type in SQL_BEARING_TYPES
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
            "degraded_sql_activities": [
                {
                    "name":          a.activity_name,
                    "type":          a.activity_type,
                    "deviation_pct": a.deviation_pct,
                    "deviation_s":   a.deviation_seconds,
                    "health":        a.health,
                }
                for a in activities
            ],
            "instruction": (
                "Analyze the REAL notebook source provided in ADDITIONAL EVIDENCE "
                "below for SQL / Spark SQL anti-patterns. Cite the offending "
                "query/line in evidence. Focus only on SQL issues - not Python "
                "logic or Spark config."
            ),
        }
        return (
            f"Investigate SQL performance issues:\n"
            f"{json.dumps(payload, indent=2)}\n"
            f"Return JSON array of findings."
        )