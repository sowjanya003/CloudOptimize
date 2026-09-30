"""
Recommendation & Validation Agent
Model: Claude Sonnet 4
Tools: generate_recommendations, validate_recommendation_feasibility, score_recommendation
"""
from __future__ import annotations
import json
import os
from schemas import InvestigationContext, RootCause, Recommendation
from sub_agents.base_agent import BaseIntelligenceAgent
from keyvault_client import get_secret
try:
    from claude_agent_sdk import ClaudeAgentOptions, query as sdk_query
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

from sub_agents.base_agent import anthropic_ready

try:
    import anthropic
except ImportError:
    anthropic = None

CATEGORY_TEMPLATES = {
    "sql": "Optimize SQL/Spark-SQL query for {target}",
    "code": "Refactor notebook transformation logic for {target}",
    "configuration": "Adjust runtime configuration for {target}",
    "dependency": "Re-sequence/parallelize dependencies around {target}",
    "orchestration": "Review activity flow/control-logic for {target}",
    "unknown": "Investigate further before remediating {target}",
}


class RecommendationAgent:
    role_name = "recommendation_validation_agent"
    model = "claude-sonnet-4-6"
    system_prompt = (
        "You turn validated root causes into concrete, implementation-ready "
        "remediation recommendations. Score each by impact/effort/risk/confidence. "
        "Return ONLY a JSON array of recommendation objects with keys: title, "
        "description, target_activity, impact_score, effort_score, risk_score, confidence "
        "(all scores 0-1)."
    )

    def validate_recommendation_feasibility(self, rec: Recommendation) -> bool:
        return rec.effort_score <= 1.0 and rec.risk_score <= 1.0

    def score_recommendation(self, rec: Recommendation) -> float:
        return round((rec.impact_score * rec.confidence) - (0.3 * rec.effort_score) - (0.3 * rec.risk_score), 3)

    async def _call_model(self, prompt: str) -> str:
        # Stashes prompt / raw response / token usage on the instance so the
        # planner can persist a DETAILED record for this pipeline stage.
        self._last_prompt = prompt
        self._last_call_meta = {}
        if anthropic_ready():
            self._last_provider = "anthropic"
            client = anthropic.AsyncAnthropic(api_key=get_secret("ANTHROPIC_API_KEY", required=True))
            response = await client.messages.create(
                model=self.model, max_tokens=2000,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(response, "usage", None)
            self._last_call_meta = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "stop_reason": getattr(response, "stop_reason", None),
            }
            raw = "".join(b.text for b in response.content if hasattr(b, "text"))
            self._last_raw = raw
            return raw
        if SDK_AVAILABLE:
            self._last_provider = "claude_agent_sdk"
            options = ClaudeAgentOptions(model=self.model, system_prompt=self.system_prompt)
            raw_text = ""
            async for message in sdk_query(prompt=prompt, options=options):
                text_piece = getattr(message, "text", None)
                if text_piece:
                    raw_text += text_piece
            self._last_raw = raw_text
            return raw_text
        self._last_provider = "none"
        self._last_raw = ""
        return ""

    async def generate(self, ctx: InvestigationContext, root_causes: list[RootCause]) -> list[Recommendation]:
        if not root_causes:
            return []

        if not anthropic_ready() and not SDK_AVAILABLE:
            raise RuntimeError(
                f"{self.role_name}: no LLM provider available (set ANTHROPIC_API_KEY). "
                "Heuristic fallbacks are disabled - failing loudly."
            )

        payload = {
            "service": ctx.service,
            "item_name": ctx.item_name,
            "root_causes": [
                {"category": rc.category, "description": rc.description, "confidence": rc.confidence}
                for rc in root_causes
            ],
        }
        prompt = f"Generate remediation recommendations for:\n{json.dumps(payload, indent=2)}\nReturn JSON array."
        raw_text = await self._call_model(prompt)
        if not raw_text:
            return self._heuristic_fallback(root_causes)

        try:
            data = json.loads(BaseIntelligenceAgent._strip_markdown_fence(raw_text))
        except (json.JSONDecodeError, TypeError):
            return self._heuristic_fallback(root_causes)

        recs = [
            Recommendation(
                title=item.get("title", ""),
                description=item.get("description", ""),
                target_activity=item.get("target_activity"),
                impact_score=float(item.get("impact_score", 0.5)),
                effort_score=float(item.get("effort_score", 0.5)),
                risk_score=float(item.get("risk_score", 0.3)),
                confidence=float(item.get("confidence", 0.5)),
            )
            for item in data
        ]
        feasible = [r for r in recs if self.validate_recommendation_feasibility(r)]
        return sorted(feasible, key=self.score_recommendation, reverse=True)

    def _heuristic_fallback(self, root_causes: list[RootCause]) -> list[Recommendation]:
        recs = []
        for rc in root_causes:
            target = rc.supporting_findings[0] if rc.supporting_findings else rc.category
            template = CATEGORY_TEMPLATES.get(rc.category, CATEGORY_TEMPLATES["unknown"])
            recs.append(Recommendation(
                title=template.format(target=target),
                description=rc.description,
                target_activity=target,
                impact_score=0.6,
                effort_score=0.4,
                risk_score=0.2,
                confidence=rc.confidence,
            ))
        return sorted(recs, key=self.score_recommendation, reverse=True)