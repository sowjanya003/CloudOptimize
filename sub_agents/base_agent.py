from __future__ import annotations
import json
import os
from typing import Callable, Optional

from datetime import datetime, timezone
from time import perf_counter
from keyvault_client import get_secret
from schemas import InvestigationContext, AgentFinding

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

try:
   
    from claude_agent_sdk import ClaudeAgentOptions, query as sdk_query
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

try:
    # pip install anthropic
    import anthropic
    _ANTHROPIC_IMPORTED = True
except ImportError:
    _ANTHROPIC_IMPORTED = False


def anthropic_ready() -> bool:
    
    return _ANTHROPIC_IMPORTED and bool(get_secret("ANTHROPIC_API_KEY"))

try:
    # pip install google-genai
    from google import genai as gemini_client_lib
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


GEMINI_MODEL_MAP = {
    "claude-sonnet-4-6": "gemini-2.5-flash",
    "claude-opus-4-8": "gemini-2.5-pro",
}


class BaseIntelligenceAgent:
    role_name: str = "base_intelligence_agent"
    model: str = "claude-sonnet-4-6"
    system_prompt: str = "You are a specialized AIOps investigation agent."

    enrichment_keys: tuple = ()

    # Per-agent output budget. Agents that enumerate many activities/relationships
    # (e.g. dependency_lineage_agent) should override this with a higher value -
    # a response that gets cut off mid-JSON is worse than a slower response.
    # Can be overridden globally via LLM_MAX_TOKENS env var.
    max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

    def __init__(self, asset_loader: Optional[Callable] = None):
        self.asset_loader = asset_loader

    def _enrichment_block(self, ctx: InvestigationContext, activities: list) -> str:
        
        if not getattr(ctx, "enrichment", None) or not self.enrichment_keys:
            return ""
        act_names = {a.activity_name for a in activities}
        block: dict = {}
        for key in self.enrichment_keys:
            if key == "notebooks":
                nbs = {
                    name: nb for name, nb in ctx.enrichment.get("notebooks", {}).items()
                    if name in act_names
                }
                if nbs:
                    block["notebook_sources"] = nbs
            elif key == "notebook_session_config":
                cfgs = {
                    name: nb.get("session_config")
                    for name, nb in ctx.enrichment.get("notebooks", {}).items()
                    if nb.get("session_config")
                }
                if cfgs:
                    block["notebook_session_configs"] = cfgs
            elif ctx.enrichment.get(key):
                block[key] = ctx.enrichment[key]
        if not block:
            return ""
        return (
            "\n\nADDITIONAL EVIDENCE from the telemetry store (authoritative - "
            "base findings ONLY on evidence present here or in the deviation data; "
            "if evidence is insufficient for a claim, lower its confidence below 0.5 "
            "and say what evidence is missing):\n"
            + json.dumps(block, indent=2, default=str)
        )

    # --- override in subclasses --------------------------------------

    def relevant_activities(self, ctx: InvestigationContext) -> list:
        """Default: only the activities this agent cares about, degraded only."""
        return ctx.degraded_activities()

    def build_prompt(self, ctx: InvestigationContext, activities: list) -> str:
        raise NotImplementedError

    def parse_response(self, ctx: InvestigationContext, raw_text: str,
                       truncated: bool = False) -> list[AgentFinding]:
        
        if os.environ.get("DEBUG_RAW_RESPONSE"):
            print(f"\n--- RAW RESPONSE [{self.role_name}] ---\n{raw_text}\n--- END RAW RESPONSE ---\n")

        if not truncated:
            cleaned = self._strip_markdown_fence(raw_text)
            try:
                data = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError) as e:
                if os.environ.get("DEBUG_RAW_RESPONSE"):
                    print(f"[{self.role_name}] JSON parse failed: {e}")
                return self._parse_failure_finding(raw_text, reason=f"{type(e).__name__}: {e}")

            findings = []
            for item in data if isinstance(data, list) else [data]:
                findings.append(AgentFinding(
                    agent=self.role_name,
                    summary=item.get("summary", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    evidence=item.get("evidence", {}),
                    affected_activities=item.get("affected_activities", []),
                ))
            return findings

        if os.environ.get("DEBUG_RAW_RESPONSE"):
            print(f"[{self.role_name}] response truncated (hit max_tokens) - "
                  f"discarding as unparseable rather than fabricating a finding")
        return self._parse_failure_finding(raw_text, reason="truncated: hit max_tokens before completion")

    def _parse_failure_finding(self, raw_text: str, reason: str) -> list[AgentFinding]:
        
        return [AgentFinding(
            agent=self.role_name,
            summary=(
                f"[{self.role_name}] produced no usable structured output "
                f"({reason}). Raw response discarded from findings; see "
                "agent_run_log for the full text."
            ),
            confidence=0.0,
            evidence={
                "parse_error": True,
                "reason": reason,
                "raw_response_preview": raw_text[:500] if isinstance(raw_text, str) else None,
            },
            status="rejected",
        )]

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        
        if not isinstance(text, str):
            return text
        t = text.strip()
        if t.startswith("```"):
            t = t.split("```")[1] if t.count("```") >= 2 else t.lstrip("`")
            t = t[4:] if t.lower().startswith("json") else t
            t = t.strip()
        # Fallback: extract from first '[' or '{' to the matching last ']' or '}'
        if not (t.startswith("[") or t.startswith("{")):
            start = min((i for i in (t.find("["), t.find("{")) if i != -1), default=-1)
            if start != -1:
                end = max(t.rfind("]"), t.rfind("}"))
                if end != -1:
                    t = t[start:end + 1]
        return t

    # --- shared execution path ------------------------------------------

    def _record_run(self, ctx: InvestigationContext, *, prompt: str, raw_text: str,
                    findings: list, started_at: str, t0: float, provider: str,
                    status: str = "success", error: str = None) -> None:
        """Append this agent's execution metadata to ctx.agent_run_log.
        The planner persists each record to blob/cosmos per agent."""
        meta = getattr(self, "_last_call_meta", {}) or {}
        activities = self.relevant_activities(ctx)
        nb = ctx.enrichment.get("notebooks", {})
        ctx.agent_run_log.append({
            "stage": "intelligence",
            "agent": self.role_name,
            "agent_class": type(self).__name__,
            "model": self.model,
            "provider": provider,
            "max_tokens": self.max_tokens,
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": round(perf_counter() - t0, 3),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "stop_reason": meta.get("stop_reason"),
            "inputs": {
                "activities_investigated": [
                    {"name": a.activity_name, "type": a.activity_type,
                     "deviation_pct": a.deviation_pct,
                     "deviation_seconds": a.deviation_seconds,
                     "health": a.health}
                    for a in activities
                ],
                "enrichment_keys": list(self.enrichment_keys),
                "evidence_sizes_chars": {
                    name: len((nb.get(name) or {}).get("source_code") or "")
                    for name in nb
                },
            },
            "system_prompt": self.system_prompt,
            "prompt_chars": len(prompt or ""),
            "prompt": prompt,
            "raw_response": raw_text,
            "response_chars": len(raw_text or ""),
            "outputs": {
                "findings_count": len(findings),
                "findings": [f.to_dict() for f in findings],
            },
            "status": status,
            "error": error,
        })

    async def investigate(self, ctx: InvestigationContext) -> list[AgentFinding]:
        activities = self.relevant_activities(ctx)
        if not activities:
            return []

        if not getattr(ctx, "enriched", False):
            raise RuntimeError(
                f"{self.role_name}: InvestigationContext was not enriched from the "
                "telemetry store. Agents must never run on the bare deviation "
                "payload alone - run PlannerAgent.enrich_context() first."
            )

        started_at = datetime.now(timezone.utc).isoformat()
        t0 = perf_counter()
        self._last_call_meta = {}
        prompt, raw_text, provider = "", "", "unknown"
        try:
            prompt = self.build_prompt(ctx, activities)
            prompt += self._enrichment_block(ctx, activities)
        except Exception as e:
            self._record_run(ctx, prompt=prompt, raw_text="", findings=[],
                             started_at=started_at, t0=t0, provider=provider,
                             status="failed", error=f"{type(e).__name__}: {e}")
            raise

        try:
            if LLM_PROVIDER == "gemini":
                provider = "gemini"
                if not GEMINI_AVAILABLE or not get_secret("GEMINI_API_KEY"):
                    raise RuntimeError(
                        f"{self.role_name}: LLM_PROVIDER=gemini but google-genai is not "
                        "installed or GEMINI_API_KEY is not set."
                    )
                raw_text = await self._call_gemini(prompt)
                findings = self.parse_response(ctx, raw_text)
                self._record_run(ctx, prompt=prompt, raw_text=raw_text, findings=findings,
                                 started_at=started_at, t0=t0, provider=provider)
                return findings

            if anthropic_ready():
                provider = "anthropic"
                raw_text = await self._call_anthropic(prompt)
                truncated = self._last_call_meta.get("stop_reason") == "max_tokens"
                if os.environ.get("DEBUG_RAW_RESPONSE"):
                    print(f"\n[{self.role_name}] anthropic SDK returned {len(raw_text)} chars"
                          f"{' (TRUNCATED - hit max_tokens)' if truncated else ''}")
                findings = self.parse_response(ctx, raw_text, truncated=truncated)
                self._record_run(ctx, prompt=prompt, raw_text=raw_text, findings=findings,
                                 started_at=started_at, t0=t0, provider=provider,
                                 status="truncated" if truncated else "success")
                return findings

            if not SDK_AVAILABLE:
                raise RuntimeError(
                    f"{self.role_name}: no LLM provider available."
                )
            provider = "claude_agent_sdk"
            options = ClaudeAgentOptions(
                model=self.model,
                system_prompt=self.system_prompt,
            )
            raw_text = ""
            async for message in sdk_query(prompt=prompt, options=options):
                text_piece = getattr(message, "text", None)
                if text_piece:
                    raw_text += text_piece

            if os.environ.get("DEBUG_RAW_RESPONSE"):
                print(f"\n[{self.role_name}] sdk_query returned {len(raw_text)} chars")
                if not raw_text:
                    print(f"[{self.role_name}] WARNING: empty response - check message object shape, "
                          f"'.text' attribute may not exist on this SDK version's message type")

            findings = self.parse_response(ctx, raw_text)
            self._record_run(ctx, prompt=prompt, raw_text=raw_text, findings=findings,
                             started_at=started_at, t0=t0, provider=provider)
            return findings
        except Exception as e:
            self._record_run(ctx, prompt=prompt, raw_text=raw_text, findings=[],
                             started_at=started_at, t0=t0, provider=provider,
                             status="failed", error=f"{type(e).__name__}: {e}")
            raise

    async def _call_anthropic(self, prompt: str) -> str:
        
        client = anthropic.AsyncAnthropic(api_key=get_secret("ANTHROPIC_API_KEY", required=True))  
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(response, "usage", None)
        self._last_call_meta = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "stop_reason": getattr(response, "stop_reason", None),
        }
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    async def _call_gemini(self, prompt: str) -> str:
        
        client = gemini_client_lib.Client(api_key=get_secret("GEMINI_API_KEY", required=True))
        gemini_model = GEMINI_MODEL_MAP.get(self.model, "gemini-2.5-flash")
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        response = client.models.generate_content(model=gemini_model, contents=full_prompt)
        return response.text or ""

    def _heuristic_fallback(self, ctx: InvestigationContext, activities: list) -> list[AgentFinding]:
        
        worst = max(activities, key=lambda a: abs(a.deviation_pct))
        return [AgentFinding(
            agent=self.role_name,
            summary=(
                f"{worst.activity_name} ({worst.activity_type}) deviated "
                f"{worst.deviation_pct:+.1f}% ({worst.deviation_seconds:+.1f}s) "
                f"vs baseline, health={worst.health}."
            ),
            confidence=0.4,
            evidence={"deviation_pct": worst.deviation_pct, "raw": worst.raw},
            affected_activities=[worst.activity_name],
        )]
