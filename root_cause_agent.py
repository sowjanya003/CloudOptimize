"""
Root Cause Agent
Model: Claude Opus 4
Tools: perform_root_cause_correlation, generate_causal_chain,
       classify_root_cause_category
"""
from __future__ import annotations
import json
import os
from schemas import InvestigationContext, AgentFinding, RootCause
from sub_agents.base_agent import BaseIntelligenceAgent

try:
    from claude_agent_sdk import ClaudeAgentOptions, query as sdk_query
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

from sub_agents.base_agent import anthropic_ready
from keyvault_client import get_secret
try:
    import anthropic
except ImportError:
    anthropic = None

CATEGORY_KEYWORDS = {
    "sql_intelligence_agent": "sql",
    "code_intelligence_agent": "code",
    "configuration_agent": "configuration",
    "dependency_lineage_agent": "dependency",
    "runtime_intelligence_agent": "orchestration",
}


class RootCauseAgent:
    role_name = "root_cause_agent"
    model = "claude-opus-4-8"
    system_prompt = (
        "You correlate VERIFIED evidence across domains (orchestration, code, "
        "SQL, configuration, dependency) to determine the actual root cause(s) "
        "of a pipeline runtime deviation. Only use findings provided - never "
        "invent evidence. Return ONLY a JSON array of root cause objects with "
        "keys: category, description, causal_chain (list of str), "
        "supporting_findings (list of agent names), confidence (0-1)."
    )

    def classify_root_cause_category(self, finding: AgentFinding) -> str:
        return CATEGORY_KEYWORDS.get(finding.agent, "unknown")

    def perform_root_cause_correlation(self, findings: list[AgentFinding]) -> dict[str, list[AgentFinding]]:
        grouped: dict[str, list[AgentFinding]] = {}
        for f in findings:
            grouped.setdefault(self.classify_root_cause_category(f), []).append(f)
        return grouped

    def generate_causal_chain(self, findings: list[AgentFinding]) -> list[str]:
        return [f"{f.agent}: {f.summary}" for f in sorted(findings, key=lambda x: -x.confidence)]

    async def analyze(self, ctx: InvestigationContext, validated_findings: list[AgentFinding]) -> list[RootCause]:
        if not validated_findings:
            return []

        grouped = self.perform_root_cause_correlation(validated_findings)

        if not anthropic_ready() and not SDK_AVAILABLE:
            raise RuntimeError(
                f"{self.role_name}: no LLM provider available (set ANTHROPIC_API_KEY). "
                "Heuristic fallbacks are disabled - failing loudly."
            )

        payload = {
            "service": ctx.service,
            "item_name": ctx.item_name,
            "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
            "validated_findings_by_category": {
                cat: [f.to_dict() for f in fs] for cat, fs in grouped.items()
            },
        }
        prompt = (
            f"Determine root cause(s) from this VERIFIED evidence:\n"
            f"{json.dumps(payload, indent=2)}\nReturn JSON array of root cause objects."
        )
        raw_text = await self._call_model(prompt)
        if not raw_text:
            return self._heuristic_fallback(grouped)

        try:
            data = json.loads(BaseIntelligenceAgent._strip_markdown_fence(raw_text))
        except (json.JSONDecodeError, TypeError):
            return self._heuristic_fallback(grouped)

        return [
            RootCause(
                category=item.get("category", "unknown"),
                description=item.get("description", ""),
                causal_chain=item.get("causal_chain", []),
                supporting_findings=item.get("supporting_findings", []),
                confidence=float(item.get("confidence", 0.5)),
            )
            for item in data
        ]

    async def _call_model(self, prompt: str) -> str:
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

    def _heuristic_fallback(self, grouped: dict[str, list[AgentFinding]]) -> list[RootCause]:
        causes = []
        for category, findings in grouped.items():
            top = max(findings, key=lambda f: f.confidence)
            causes.append(RootCause(
                category=category,
                description=top.summary,
                causal_chain=self.generate_causal_chain(findings),
                supporting_findings=[f.agent for f in findings],
                confidence=top.confidence,
            ))
        return sorted(causes, key=lambda c: -c.confidence)