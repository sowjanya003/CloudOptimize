from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum
import json


class HealthStatus(str, Enum):
    HEALTHY = "Healthy"
    WARNING = "Warning"
    SEVERE = "Severe"
    UNKNOWN = "Unknown"


class AgentRole(str, Enum):
    PLANNER = "planner_agent"
    RUNTIME_INTELLIGENCE = "runtime_intelligence_agent"   
    SQL_INTELLIGENCE = "sql_intelligence_agent"
    CODE_INTELLIGENCE = "code_intelligence_agent"
    PYTHON_INTELLIGENCE = "python_intelligence_agent"
    SPARK_INTELLIGENCE = "spark_intelligence_agent"
    COST_INTELLIGENCE = "cost_intelligence"
    CONFIGURATION = "configuration_agent"
    DEPENDENCY_LINEAGE = "dependency_lineage_agent"
    COST_ANALYSIS = "cost_analysis_agent"
    EVIDENCE_VALIDATION = "evidence_validation_agent"
    ROOT_CAUSE = "root_cause_agent"
    RECOMMENDATION = "recommendation_validation_agent"
    IMPACT = "impact_agent"
    ACTION_PLAN_REPORT = "action_plan_report_agent"


@dataclass
class ActivityDeviation:
    activity_name: str
    activity_type: str
    avg_duration_seconds: float
    latest_duration_seconds: float
    deviation_seconds: float
    deviation_pct: float
    success_rate_pct: float
    health: str
    p95_duration_seconds: Optional[float] = None
    median_duration_seconds: Optional[float] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict) -> "ActivityDeviation":
        return cls(
            activity_name=raw.get("activity_name", "unknown"),
            activity_type=raw.get("activity_type", "unknown"),
            avg_duration_seconds=raw.get("avg_duration_seconds", 0.0),
            latest_duration_seconds=raw.get("latest_duration_seconds", 0.0),
            deviation_seconds=raw.get("deviation_seconds", 0.0),
            deviation_pct=raw.get("deviation_pct", 0.0),
            success_rate_pct=raw.get("success_rate_pct", 0.0),
            health=raw.get("health", "Unknown"),
            p95_duration_seconds=raw.get("p95_duration_seconds"),
            median_duration_seconds=raw.get("median_duration_seconds"),
            raw=raw,
        )


@dataclass
class InvestigationContext:

    subscription_id: str
    service: str                 # synapse | adf | databricks | fabric | ...
    resource_group: str
    workspace_name: str
    item_type: str                # pipeline | jobs | dlt_pipeline | ...
    item_name: str
    pipeline_health: str
    pipeline_deviation_pct: float
    pipeline_deviation_seconds: float
    activities: list[ActivityDeviation]
    raw_payload: dict = field(default_factory=dict)
    investigation_type: str = "runtime" 
    enriched: bool = False
    enrichment: dict = field(default_factory=dict)


    agent_run_log: list = field(default_factory=list)

    @classmethod
    def from_deviation_payload(cls, payload: dict) -> "InvestigationContext":
    
        pb = payload.get("pipeline_baseline", payload)  # fall back to top-level
        activities_raw = payload.get("activities", [])
        activities = [ActivityDeviation.from_raw(a) for a in activities_raw]

        return cls(
            subscription_id=payload.get("subscription_id", "unknown"),
            service=payload.get("service", "unknown"),
            resource_group=payload.get("resource_group", "unknown"),
            workspace_name=payload.get("workspace_name", "unknown"),
            item_type=payload.get("item_type", pb.get("item_type", "unknown")),
            item_name=payload.get("item_name", pb.get("item_name", payload.get("pipeline_name", "unknown"))),
            pipeline_health=pb.get("health", "Unknown"),
            pipeline_deviation_pct=pb.get("deviation_pct", 0.0),
            pipeline_deviation_seconds=pb.get("deviation_seconds", 0.0),
            activities=activities,
            raw_payload=payload,
            investigation_type="runtime" ,
            )
    
    @classmethod
    def from_cost_payload(cls, payload: dict) -> "InvestigationContext":
        """
        Creates an InvestigationContext for workspace-level cost investigations.

        Unlike runtime investigations, cost investigations are not activity-based.
        """

        return cls(
            subscription_id=(
                payload.get("_subscription_id")
                or payload.get("subscription_id")
            ),

            service=payload.get("service"),

            resource_group=(
                payload.get("resource_group")
                or payload.get("_resource_group")
            ),

            workspace_name=(
                payload.get("workspace_name")
                or payload.get("factory_name")
                or payload.get("_workspace_name")
            ),

            # Cost investigation is workspace scoped
            item_type="workspace",

            item_name=(
                payload.get("workspace_name")
                or payload.get("factory_name")
                or payload.get("_workspace_name")
            ),

            # These are not used by Cost Agent
            pipeline_health="Unknown",
            pipeline_deviation_pct=0.0,
            pipeline_deviation_seconds=0.0,

            activities=[],

            raw_payload=payload,
            investigation_type="cost", 
        )
    
    @classmethod
    def from_cost_payload(cls, payload: dict) -> "InvestigationContext":
        """
        Creates an InvestigationContext for workspace-level cost investigations.

        Unlike runtime investigations, cost investigations are not activity-based.
        """

        return cls(
            subscription_id=(
                payload.get("_subscription_id")
                or payload.get("subscription_id")
            ),

            service=payload.get("service"),

            resource_group=(
                payload.get("resource_group")
                or payload.get("_resource_group")
            ),

            workspace_name=(
                payload.get("workspace_name")
                or payload.get("factory_name")
                or payload.get("_workspace_name")
            ),

            # Cost investigation is workspace scoped
            item_type="workspace",

            item_name=(
                payload.get("workspace_name")
                or payload.get("factory_name")
                or payload.get("_workspace_name")
            ),

            # These are not used by Cost Agent
            pipeline_health="Unknown",
            pipeline_deviation_pct=0.0,
            pipeline_deviation_seconds=0.0,

            activities=[],

            raw_payload=payload,
            investigation_type="cost",  # FIX: this second from_cost_payload overload previously
                                        # omitted this, so cost investigations were silently
                                        # tagged 'runtime' (the dataclass default). Restored here
                                        # without touching anything else in this method.
        )

    def degraded_activities(self) -> list[ActivityDeviation]:
        return [a for a in self.activities if a.health in ("Warning", "Severe")]

    def activity_types_present(self) -> set[str]:
        return {a.activity_type for a in self.activities}


@dataclass
class AgentFinding:
    agent: str
    summary: str
    confidence: float                 # 0.0 - 1.0, set by Evidence Validation Agent
    evidence: dict = field(default_factory=dict)
    affected_activities: list[str] = field(default_factory=list)
    status: str = "unverified"        # unverified -> verified -> rejected

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RootCause:
    category: str                      # orchestration | code | sql | configuration | dependency
    description: str
    causal_chain: list[str]
    supporting_findings: list[str]      # agent names
    confidence: float


@dataclass
class Recommendation:
    title: str
    description: str
    target_activity: Optional[str]
    impact_score: float
    effort_score: float
    risk_score: float
    confidence: float


@dataclass
class InvestigationState:
    context: InvestigationContext
    dispatched_agents: list[str] = field(default_factory=list)
    findings: list[AgentFinding] = field(default_factory=list)
    validated_findings: list[AgentFinding] = field(default_factory=list)
    root_causes: list[RootCause] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    impact_summary: dict = field(default_factory=dict)
    final_report: dict = field(default_factory=dict)

    def add_finding(self, finding: AgentFinding):
        self.findings.append(finding)

    def to_json(self) -> str:
        return json.dumps({
            "context": asdict(self.context),
            "dispatched_agents": self.dispatched_agents,
            "findings": [f.to_dict() for f in self.findings],
            "validated_findings": [f.to_dict() for f in self.validated_findings],
            "root_causes": [asdict(r) for r in self.root_causes],
            "recommendations": [asdict(r) for r in self.recommendations],
            "impact_summary": self.impact_summary,
            "final_report": self.final_report,
        }, indent=2, default=str)