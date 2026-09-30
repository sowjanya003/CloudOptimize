"""
Dependency & Lineage Agent
Model: Claude Sonnet 4
Tools: build_dependency_graph, identify_critical_path, analyze_branching_impact,
       assess_upstream_downstream_delay, detect_complex_dependency_patterns
"""
from __future__ import annotations
import json
from schemas import InvestigationContext, ActivityDeviation
from sub_agents.base_agent import BaseIntelligenceAgent


class DependencyLineageAgent(BaseIntelligenceAgent):
    enrichment_keys = ("definition", "recent_runs", "dependencies")
    role_name = "dependency_lineage_agent"
    model = "claude-sonnet-4-6"
    
    max_tokens = 6000
    system_prompt = (
        "You analyze activity dependency chains and branching logic to find "
        "the critical path and assess upstream-to-downstream delay propagation. "
        "Return ONLY a JSON array of finding objects with keys: summary, "
        "confidence (0-1), evidence (object), affected_activities (list)."
    )

    def relevant_activities(self, ctx: InvestigationContext):
        
        return ctx.activities

    def build_dependency_graph(self, pipeline_definition: dict) -> dict:
        
        graph = {}
        for act in pipeline_definition.get("activities", pipeline_definition.get("tasks", [])):
            name = act.get("name") or act.get("task_key")
            deps = [d.get("activity") for d in act.get("depends_on", [])]
            graph[name] = deps
        return graph

    def identify_critical_path(self, graph: dict, deviations: dict[str, float]) -> list[str]:
        # Simplest useful heuristic: longest chain of degraded activities.
        degraded_nodes = {n for n, dev in deviations.items() if abs(dev) > 0}
        path = [n for n in graph if n in degraded_nodes]
        return path

    def build_prompt(self, ctx: InvestigationContext, activities: list[ActivityDeviation]) -> str:
        payload = {
            "service": ctx.service,
            "item_name": ctx.item_name,
            "activities": [
                {"name": a.activity_name, "type": a.activity_type, "deviation_pct": a.deviation_pct, "health": a.health}
                for a in activities
            ],
            "instruction": (
                "Use the 'dependencies' graph ({activity: [upstream, ...]}) and "
                "per-task timings inside recent_runs from ADDITIONAL EVIDENCE "
                "below to find critical-path and delay-propagation causes."
            ),
        }
        return f"Investigate dependency/critical-path causes for:\n{json.dumps(payload, indent=2)}\nReturn JSON array of findings."
