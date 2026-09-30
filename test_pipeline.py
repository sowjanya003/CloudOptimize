
from __future__ import annotations
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic


# ─── SINGLE SHARED CLIENT ────────────────────────────────────────────────────

_client: Optional[anthropic.AsyncAnthropic] = None

def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


# ─── DATA MODELS ─────────────────────────────────────────────────────────────

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

    @classmethod
    def from_raw(cls, r: dict) -> "ActivityDeviation":
        return cls(
            activity_name           = r.get("activity_name", "unknown"),
            activity_type           = r.get("activity_type", "unknown"),
            avg_duration_seconds    = r.get("avg_duration_seconds", 0.0),
            latest_duration_seconds = r.get("latest_duration_seconds", 0.0),
            deviation_seconds       = r.get("deviation_seconds", 0.0),
            deviation_pct           = r.get("deviation_pct", 0.0),
            success_rate_pct        = r.get("success_rate_pct", 100.0),
            health                  = r.get("health", "Unknown"),
            p95_duration_seconds    = r.get("p95_duration_seconds"),
        )


@dataclass
class InvestigationContext:
    subscription_id: str
    service: str
    resource_group: str
    workspace_name: str
    item_type: str
    item_name: str
    pipeline_health: str
    pipeline_deviation_pct: float
    pipeline_deviation_seconds: float
    activities: list[ActivityDeviation]

    def degraded_activities(self):
        return [a for a in self.activities if a.health in ("Warning", "Severe")]

    @classmethod
    def from_payload(cls, p: dict) -> "InvestigationContext":
        pb = p.get("pipeline_baseline", p)
        return cls(
            subscription_id            = p.get("subscription_id", ""),
            service                    = p.get("service", ""),
            resource_group             = p.get("resource_group", ""),
            workspace_name             = p.get("workspace_name", ""),
            item_type                  = p.get("item_type", ""),
            item_name                  = p.get("item_name", pb.get("item_name", "")),
            pipeline_health            = pb.get("health", "Unknown"),
            pipeline_deviation_pct     = pb.get("deviation_pct", 0.0),
            pipeline_deviation_seconds = pb.get("deviation_seconds", 0.0),
            activities = [ActivityDeviation.from_raw(a) for a in p.get("activities", [])],
        )


@dataclass
class Finding:
    agent: str
    summary: str
    confidence: float
    affected_activities: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Full output from one agent - raw + parsed + metadata."""
    agent_name: str
    raw_response: str
    findings: list[Finding]
    input_tokens: int
    output_tokens: int
    stop_reason: str          # "end_turn" = good | "max_tokens" = truncated
    latency_seconds: float


@dataclass
class PlannerDecision:
    """Records why the planner selected each agent."""
    selected_agents: list[str]
    reason: str
    degraded_activities: list[str]
    activity_types_seen: list[str]


# ─── TOKEN USAGE TRACKER ─────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, inp: int, out: int):
        self.input_tokens  += inp
        self.output_tokens += out

    def print_summary(self):
        total = self.input_tokens + self.output_tokens
        cost  = (self.input_tokens / 1_000_000 * 3.0) + \
                (self.output_tokens / 1_000_000 * 15.0)
        print(f"\n{'─'*55}")
        print(f"TOKEN USAGE SUMMARY")
        print(f"  Input tokens  : {self.input_tokens:,}")
        print(f"  Output tokens : {self.output_tokens:,}")
        print(f"  Total tokens  : {total:,}")
        print(f"  Estimated cost: ${cost:.5f}  (Sonnet 4.6 rates)")
        print(f"{'─'*55}")

USAGE = TokenUsage()


# ─── SHARED HELPERS ──────────────────────────────────────────────────────────

def strip_fence(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) >= 2 else t.lstrip("`")
        t = t[4:].strip() if t.lower().startswith("json") else t.strip()
    if not (t.startswith("[") or t.startswith("{")):
        start = min((i for i in (t.find("["), t.find("{")) if i != -1), default=-1)
        if start != -1:
            end = max(t.rfind("]"), t.rfind("}"))
            if end != -1:
                t = t[start:end + 1]
    return t


def repair_truncated_json(text: str) -> str:
    """
    When stop_reason == 'max_tokens' the JSON is cut mid-stream.
    Try to recover the last complete object before the cut point.
    Recovers ~70-80% of truncated responses.
    """
    last_brace = text.rfind("}")
    if last_brace == -1:
        return text
    repaired = text[:last_brace + 1]
    # If it looked like an array, close it
    if text.lstrip().startswith("[") and not repaired.rstrip().endswith("]"):
        repaired += "]"
    return repaired


def parse_findings(agent_name: str, raw_text: str, stop_reason: str) -> list[Finding]:
    if not raw_text:
        return []

    cleaned = strip_fence(raw_text)

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        if stop_reason == "max_tokens":
            print(f"  [{agent_name}] Truncated response detected — attempting JSON repair...")
            repaired = repair_truncated_json(cleaned)
            try:
                data = json.loads(repaired)
                print(f"  [{agent_name}] JSON repair succeeded.")
            except (json.JSONDecodeError, TypeError):
                print(f"  [{agent_name}] JSON repair failed. Raw (first 400):\n  {raw_text[:400]}")
                return []
        else:
            print(f"  [{agent_name}] JSON parse failed. Raw (first 400):\n  {raw_text[:400]}")
            return []

    items = data if isinstance(data, list) else [data]
    return [
        Finding(
            agent               = agent_name,
            summary             = item.get("summary", ""),
            confidence          = float(item.get("confidence", 0.5)),
            affected_activities = item.get("affected_activities", []),
            evidence            = item.get("evidence", {}),
        )
        for item in items if isinstance(item, dict)
    ]


async def call_claude(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    label: str = "agent",
    max_tokens: int = 4096,
    timeout_seconds: int = 60,
    max_retries: int = 3,
) -> dict:
    """
    Returns dict with keys: text, input_tokens, output_tokens, stop_reason.
    Single shared client + timeout + retry + stop_reason logging.
    """
    client = get_client()

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            response = await asyncio.wait_for(
                client.messages.create(
                    model    = model,
                    max_tokens = max_tokens,
                    system   = system_prompt,
                    messages = [{"role": "user", "content": user_prompt}],
                ),
                timeout=timeout_seconds,
            )
            latency = round(time.perf_counter() - t0, 2)

            inp  = response.usage.input_tokens
            out  = response.usage.output_tokens
            stop = response.stop_reason
            USAGE.add(inp, out)

            
            status = "⚠ TRUNCATED" if stop == "max_tokens" else "✓"
            print(f"  [{label}] {status}  "
                  f"in={inp} out={out} stop={stop} latency={latency}s")

            return {
                "text":         "".join(b.text for b in response.content if hasattr(b, "text")),
                "input_tokens":  inp,
                "output_tokens": out,
                "stop_reason":   stop,
                "latency":       latency,
            }

        except asyncio.TimeoutError:
            print(f"  [{label}] Attempt {attempt}/{max_retries} timed out ({timeout_seconds}s).")
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"  [{label}] Rate limit. Waiting {wait}s... (attempt {attempt}/{max_retries})")
            await asyncio.sleep(wait)
        except anthropic.APIStatusError as e:
            print(f"  [{label}] API error attempt {attempt}/{max_retries}: {e.status_code} {e.message}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [{label}] Unexpected: {type(e).__name__}: {e}")
            raise

    return {"text": "", "input_tokens": 0, "output_tokens": 0, "stop_reason": "error", "latency": 0.0}


# ─── SAVE TO DISK ────────────────────────────────────────────────────────────

def save_result(run_dir: Path, filename: str, data: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / filename, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ─── SUB-AGENT 1: RUNTIME INTELLIGENCE ───────────────────────────────────────

RUNTIME_SYSTEM = """You are an AIOps Runtime Intelligence Agent.
Analyze pipeline/job orchestration deviation across Azure data services.

RULES:
- Return at most 2 findings sorted by impact descending.
- summary: maximum 30 words.
- evidence: maximum 5 keys, short string or number values only.
- No prose, no markdown, no explanation outside the JSON array.

OUTPUT FORMAT (strict):
[
  {
    "summary": "...",
    "confidence": 0.0,
    "affected_activities": ["..."],
    "evidence": {"key": "value"}
  }
]"""


async def runtime_intelligence_agent(ctx: InvestigationContext) -> AgentResult:
    degraded = ctx.degraded_activities()
    if not degraded:
        return AgentResult("runtime_intelligence_agent", "", [], 0, 0, "skipped", 0.0)

    total_dev = sum(abs(a.deviation_seconds) for a in ctx.activities) or 1.0
    contribution = {
        a.activity_name: round(abs(a.deviation_seconds) / total_dev * 100, 1)
        for a in ctx.activities
    }

    payload = {
        "service": ctx.service,
        "item_name": ctx.item_name,
        "pipeline_health": ctx.pipeline_health,
        "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
        "activity_contribution_pct": contribution,
        "degraded_activities": [
            {
                "name": a.activity_name,
                "type": a.activity_type,
                "deviation_pct": a.deviation_pct,
                "deviation_s": a.deviation_seconds,
                "p95_s": a.p95_duration_seconds,
                "health": a.health,
            }
            for a in degraded
        ],
    }

    print("\n  [runtime_intelligence_agent] Calling Claude...")
    resp = await call_claude(
        RUNTIME_SYSTEM,
        f"Investigate runtime deviation:\n{json.dumps(payload, indent=2)}\nReturn JSON array.",
        label="runtime_intelligence_agent",
    )

    findings = parse_findings("runtime_intelligence_agent", resp["text"], resp["stop_reason"])
    print(f"  [runtime_intelligence_agent] {len(findings)} finding(s) parsed.")

    return AgentResult(
        agent_name    = "runtime_intelligence_agent",
        raw_response  = resp["text"],
        findings      = findings,
        input_tokens  = resp["input_tokens"],
        output_tokens = resp["output_tokens"],
        stop_reason   = resp["stop_reason"],
        latency_seconds = resp["latency"],
    )


# ─── NOTEBOOK SOURCES & MAPPING (shared by python + spark agents) ────────────

ALL_NOTEBOOK_SOURCES = {
    "Customer_Sales_Aggregation": (
        'customers_df = spark.read.option("header","true").csv("/mnt/raw/customers.csv")\n'
        'orders_df    = spark.read.option("header","true").csv("/mnt/raw/orders.csv")\n'
        'orders_df    = orders_df.filter(col("OrderStatus") != "Cancelled")\n'
        'join_1 = orders_df.join(customers_df, on="CustomerID", how="inner")\n'
        'join_2 = join_1.join(order_details_df, on="OrderID", how="inner")\n'
        'join_3 = join_2.join(products_df, on="ProductID", how="inner")\n'
        'result = join_3.groupBy(...).agg(...)\n'
        'result = result.repartition(200, "CustomerID")\n'
        'result.write.mode("overwrite").parquet("/mnt/processed/customer_sales")\n'
    ),
    "Inventory_Movement_Analysis": (
        'inventory_df = spark.read.parquet("/mnt/raw/inventory_movements/")\n'
        'window_spec  = Window.partitionBy("WarehouseID","ProductID").orderBy("MovementDate")\n'
        'inventory_with_lag = inventory_df.withColumn("PrevQuantity", lag("Quantity",1,0).over(window_spec))\n'
        'recent = inventory_with_lag.filter(col("MovementDate") >= "2023-01-01")\n'
        'results = recent.collect()                 # pulls entire dataset to driver\n'
        'processed = []\n'
        'for row in results:                        # row-by-row Python iteration\n'
        '    if row.MovementType == "IN":  impact = row.Quantity * 1.05\n'
        '    else:                         impact = row.Quantity * 0.95\n'
        '    processed.append({"warehouse_id": row.WarehouseID, "movement_impact": impact})\n'
        'processed_rdd = spark.sparkContext.parallelize(processed)\n'
        'processed_df  = spark.createDataFrame(processed_rdd)\n'
        'final = processed_df.groupBy(...).agg(...).orderBy("TotalMovementImpact")\n'
        'final.show(100)\n'
    ),
}

ACTIVITY_TO_NOTEBOOK: dict[str, str] = {
    "Add_Category":    "Customer_Sales_Aggregation",
    "Validate_Schema": "Customer_Sales_Aggregation",
    "Filter_Data":     "Customer_Sales_Aggregation",
    "Validate_Source": "Customer_Sales_Aggregation",
    "Bronze":          "Inventory_Movement_Analysis",
    "Gold":            "Inventory_Movement_Analysis",
}

CODE_BEARING_TYPES = {"SynapseNotebook", "notebook_task", "dlt_pipeline"}


def _relevant_notebooks(activities: list) -> dict[str, str]:
    """Return only notebooks referenced by the given activities."""
    sources = {}
    for a in activities:
        nb = ACTIVITY_TO_NOTEBOOK.get(a.activity_name)
        if nb and nb in ALL_NOTEBOOK_SOURCES:
            sources[nb] = ALL_NOTEBOOK_SOURCES[nb]
    return sources or ALL_NOTEBOOK_SOURCES


# ─── SUB-AGENT 2: SQL INTELLIGENCE ───────────────────────────────────────────

SQL_SYSTEM = """You are an AIOps SQL Intelligence Agent.
Find SQL and Spark SQL anti-patterns ONLY.
Look for: SELECT * scans, missing partition pushdown, cartesian/cross joins,
non-broadcast joins on small tables, ORDER BY on unbounded datasets.
RULES: max 2 findings, summary max 30 words, evidence max 5 keys, no prose outside JSON.
OUTPUT: [{"summary":"...","confidence":0.0,"affected_activities":["..."],"evidence":{}}]"""


async def sql_intelligence_agent(ctx: InvestigationContext) -> AgentResult:
    sql_types = {"SynapseNotebook", "notebook_task", "sql_task", "sql_script", "dlt_pipeline"}
    relevant  = [a for a in ctx.degraded_activities() if a.activity_type in sql_types]
    if not relevant:
        return AgentResult("sql_intelligence_agent", "", [], 0, 0, "skipped", 0.0)

    payload = {
        "service": ctx.service, "item_name": ctx.item_name,
        "degraded_sql_activities": [
            {"name": a.activity_name, "type": a.activity_type, "deviation_pct": a.deviation_pct}
            for a in relevant
        ],
    }
    print("\n  [sql_intelligence_agent] Calling Claude...")
    resp = await call_claude(SQL_SYSTEM,
        f"Find SQL anti-patterns:\n{json.dumps(payload, indent=2)}\nReturn JSON array.",
        label="sql_intelligence_agent")
    findings = parse_findings("sql_intelligence_agent", resp["text"], resp["stop_reason"])
    print(f"  [sql_intelligence_agent] {len(findings)} finding(s) parsed.")
    return AgentResult("sql_intelligence_agent", resp["text"], findings,
        resp["input_tokens"], resp["output_tokens"], resp["stop_reason"], resp["latency"])


# ─── SUB-AGENT 3: PYTHON INTELLIGENCE ────────────────────────────────────────

PYTHON_SYSTEM = """You are an AIOps Python Intelligence Agent.
Find Python-specific anti-patterns ONLY in notebook source code.
Look for: collect() on large datasets, row-by-row for-loop iteration over collected data,
Python UDFs instead of native Spark functions, toPandas() on large frames,
sparkContext.parallelize() after collect() to reconstruct distributed data.
RULES: max 2 findings, summary max 30 words, evidence must include offending_line,
       max 5 evidence keys, no prose outside JSON.
OUTPUT: [{"summary":"...","confidence":0.0,"affected_activities":["..."],"evidence":{"offending_line":"..."}}]"""


async def python_intelligence_agent(ctx: InvestigationContext) -> AgentResult:
    relevant = [a for a in ctx.degraded_activities() if a.activity_type in CODE_BEARING_TYPES]
    if not relevant:
        return AgentResult("python_intelligence_agent", "", [], 0, 0, "skipped", 0.0)

    payload = {
        "service": ctx.service, "item_name": ctx.item_name,
        "degraded_activities": [
            {"name": a.activity_name, "type": a.activity_type, "deviation_pct": a.deviation_pct}
            for a in relevant
        ],
        "notebook_source_code": _relevant_notebooks(relevant),
    }
    print("\n  [python_intelligence_agent] Calling Claude...")
    resp = await call_claude(PYTHON_SYSTEM,
        f"Find Python anti-patterns:\n{json.dumps(payload, indent=2)}\nReturn JSON array.",
        label="python_intelligence_agent")
    findings = parse_findings("python_intelligence_agent", resp["text"], resp["stop_reason"])
    print(f"  [python_intelligence_agent] {len(findings)} finding(s) parsed.")
    return AgentResult("python_intelligence_agent", resp["text"], findings,
        resp["input_tokens"], resp["output_tokens"], resp["stop_reason"], resp["latency"])


# ─── SUB-AGENT 4: SPARK INTELLIGENCE ─────────────────────────────────────────

SPARK_SYSTEM = """You are an AIOps Spark Intelligence Agent.
Find Spark execution and configuration anti-patterns ONLY.
Look for: fixed repartition count mismatched to cluster size, missing cache/persist
before repeated DataFrame reuse, shuffle spill from insufficient executor memory,
data skew in groupBy/join keys, autoscale min==max, small file overhead,
default shuffle partitions (200) not tuned to data volume.
RULES: max 2 findings, summary max 30 words, evidence max 5 keys, no prose outside JSON.
OUTPUT: [{"summary":"...","confidence":0.0,"affected_activities":["..."],"evidence":{}}]"""


async def spark_intelligence_agent(ctx: InvestigationContext) -> AgentResult:
    relevant = [a for a in ctx.degraded_activities() if a.activity_type in CODE_BEARING_TYPES]
    if not relevant:
        return AgentResult("spark_intelligence_agent", "", [], 0, 0, "skipped", 0.0)

    payload = {
        "service": ctx.service, "item_name": ctx.item_name,
        "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
        "degraded_spark_activities": [
            {
                "name": a.activity_name, "type": a.activity_type,
                "deviation_pct": a.deviation_pct,
                "avg_duration_s": a.avg_duration_seconds,
                "latest_duration_s": a.latest_duration_seconds,
                "p95_s": a.p95_duration_seconds, "health": a.health,
            }
            for a in relevant
        ],
    }
    print("\n  [spark_intelligence_agent] Calling Claude...")
    resp = await call_claude(SPARK_SYSTEM,
        f"Find Spark config/execution issues:\n{json.dumps(payload, indent=2)}\nReturn JSON array.",
        label="spark_intelligence_agent")
    findings = parse_findings("spark_intelligence_agent", resp["text"], resp["stop_reason"])
    print(f"  [spark_intelligence_agent] {len(findings)} finding(s) parsed.")
    return AgentResult("spark_intelligence_agent", resp["text"], findings,
        resp["input_tokens"], resp["output_tokens"], resp["stop_reason"], resp["latency"])


# ─── MINI PLANNER ────────────────────────────────────────────────────────────

ACTIVITY_TYPE_TO_AGENTS: dict[str, list[str]] = {
    # Notebook activities → all three focused agents + runtime
    "notebook_task":   ["runtime", "sql", "python", "spark"],
    "SynapseNotebook": ["runtime", "sql", "python", "spark"],
    "dlt_pipeline":    ["runtime", "python", "spark"],
    # SQL-only
    "sql_task":    ["sql"],
    "sql_script":  ["sql"],
    # Orchestration
    "Copy":        ["runtime"],
    "IfCondition": ["runtime"],
    "ForEach":     ["runtime"],
    "Switch":      ["runtime"],
    "Wait":        ["runtime"],
    "GetMetadata": ["runtime"],
}

AGENT_FUNCTIONS = {
    "runtime": runtime_intelligence_agent,
    "sql":     sql_intelligence_agent,
    "python":  python_intelligence_agent,
    "spark":   spark_intelligence_agent,
}


async def run_mini_planner(payload: dict, run_id: str) -> None:
    run_dir = Path("investigations") / run_id
    ctx     = InvestigationContext.from_payload(payload)
    degraded = ctx.degraded_activities()

    # ── Planner header ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PLANNER  —  run_id: {run_id}")
    print(f"{'='*60}")
    print(f"  Service        : {ctx.service}")
    print(f"  Item           : {ctx.item_name} ({ctx.item_type})")
    print(f"  Pipeline Health: {ctx.pipeline_health}")
    print(f"  Deviation      : {ctx.pipeline_deviation_pct:+.1f}%  "
          f"({ctx.pipeline_deviation_seconds:+.1f}s)")
    print(f"  Total activities : {len(ctx.activities)}")
    print(f"  Degraded         : {len(degraded)}")
    for a in degraded:
        print(f"    • {a.activity_name:<28} {a.deviation_pct:+.1f}%  [{a.health}]")

    agents_needed: set[str] = set()
    activity_types_seen: list[str] = []
    for a in degraded:
        activity_types_seen.append(a.activity_type)
        for key in ACTIVITY_TYPE_TO_AGENTS.get(a.activity_type, ["runtime"]):
            agents_needed.add(key)

    if not agents_needed:
        print("  No agents needed — nothing degraded.")
        return

    reason = (
        f"Degraded activity types: {sorted(set(activity_types_seen))} "
        f"→ agents: {sorted(agents_needed)}"
    )
    decision = PlannerDecision(
        selected_agents     = sorted(agents_needed),
        reason              = reason,
        degraded_activities = [a.activity_name for a in degraded],
        activity_types_seen = sorted(set(activity_types_seen)),
    )
    print(f"\n  Selected agents: {decision.selected_agents}")
    print(f"  Reason         : {decision.reason}")
    save_result(run_dir, "planner.json", asdict(decision))

    # ── Dispatch in parallel ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("DISPATCHING AGENTS IN PARALLEL")
    print(f"{'─'*60}")

    ordered = sorted(agents_needed)
    coros   = [AGENT_FUNCTIONS[name](ctx) for name in ordered]
    results = await asyncio.gather(*coros, return_exceptions=True)

    agent_results: list[AgentResult] = []
    for name, result in zip(ordered, results):
        if isinstance(result, Exception):
            print(f"\n  [ERROR] {name}: {type(result).__name__}: {result}")
        elif isinstance(result, AgentResult):
            agent_results.append(result)
            # Save to disk
            save_result(run_dir, f"{result.agent_name}.json", {
                "agent_name":     result.agent_name,
                "stop_reason":    result.stop_reason,
                "input_tokens":   result.input_tokens,
                "output_tokens":  result.output_tokens,
                "latency_seconds": result.latency_seconds,
                "raw_response":   result.raw_response,
                "findings":       [asdict(f) for f in result.findings],
            })

    # ── Per-agent output ─────────────────────────────────────────────────────
    all_findings: list[Finding] = []
    for result in agent_results:
        print(f"\n{'='*60}")
        print(f"{result.agent_name.upper()}")
        print(f"{'='*60}")
        print(f"  Input tokens  : {result.input_tokens}")
        print(f"  Output tokens : {result.output_tokens}")
        print(f"  Stop reason   : {result.stop_reason}  "
              f"{'⚠ TRUNCATED — raise max_tokens' if result.stop_reason == 'max_tokens' else '✓ clean finish'}")
        print(f"  Latency       : {result.latency_seconds}s")
        print(f"\n  RAW RESPONSE:")
        print(f"  {result.raw_response[:600]}")
        print(f"\n  PARSED FINDINGS ({len(result.findings)}):")
        for i, f in enumerate(result.findings, 1):
            print(f"    [{i}] conf={f.confidence:.2f}  {f.affected_activities}")
            print(f"         {f.summary}")
            if f.evidence:
                print(f"         evidence: {json.dumps(f.evidence)[:150]}")
        all_findings.extend(result.findings)

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL INVESTIGATION REPORT  ({len(all_findings)} findings total)")
    print(f"{'='*60}")
    if not all_findings:
        print("  No findings. Check stop_reason above — likely max_tokens or credit issue.")
    for i, f in enumerate(sorted(all_findings, key=lambda x: -x.confidence), 1):
        print(f"\n  [{i}] {f.agent}")
        print(f"       Confidence : {f.confidence:.2f}")
        print(f"       Activities : {f.affected_activities}")
        print(f"       Summary    : {f.summary}")

    final_report = {
        "run_id":        run_id,
        "service":       ctx.service,
        "item_name":     ctx.item_name,
        "pipeline_health": ctx.pipeline_health,
        "findings":      [asdict(f) for f in all_findings],
    }
    save_result(run_dir, "final_report.json", final_report)
    print(f"\n  Saved to: investigations/{run_id}/")


# ─── SAMPLE PAYLOADS ─────────────────────────────────────────────────────────

DATABRICKS_PAYLOAD = {
    "subscription_id": "2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e",
    "service":         "databricks",
    "resource_group":  "Synapsetofabric",
    "workspace_name":  "Databrickstofabricsws",
    "item_type":       "jobs",
    "item_name":       "01_Sales_Data_Processing",
    "pipeline_baseline": {
        "avg_duration_seconds": 205.01, "deviation_seconds": 4.15,
        "deviation_pct": 2.02, "health": "Healthy",
    },
    "activities": [
        {"activity_name": "Add_Category",    "activity_type": "notebook_task",
         "avg_duration_seconds": 20.57, "latest_duration_seconds": 23.0,
         "deviation_seconds": 2.43,  "deviation_pct": 11.81,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 25.1},
        {"activity_name": "Validate_Schema", "activity_type": "notebook_task",
         "avg_duration_seconds": 18.43, "latest_duration_seconds": 23.0,
         "deviation_seconds": 4.57,  "deviation_pct": 24.8,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 23.0},
        {"activity_name": "Filter_Data",     "activity_type": "notebook_task",
         "avg_duration_seconds": 19.0,  "latest_duration_seconds": 21.0,
         "deviation_seconds": 2.0,   "deviation_pct": 10.53,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 21.0},
        {"activity_name": "Validate_Source", "activity_type": "notebook_task",
         "avg_duration_seconds": 20.0,  "latest_duration_seconds": 17.0,
         "deviation_seconds": -3.0,  "deviation_pct": -15.0,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 28.3},
        {"activity_name": "Calculate_Tax",   "activity_type": "notebook_task",
         "avg_duration_seconds": 18.14, "latest_duration_seconds": 18.0,
         "deviation_seconds": -0.14, "deviation_pct": -0.77,
         "success_rate_pct": 100.0,  "health": "Healthy", "p95_duration_seconds": 20.8},
    ],
}

SYNAPSE_PAYLOAD = {
    "subscription_id": "2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e",
    "service":         "synapse",
    "resource_group":  "Synapsetofabric",
    "workspace_name":  "synapsetofabricws1",
    "item_type":       "pipeline",
    "item_name":       "Data_Pipeline_-_Engineering",
    "pipeline_baseline": {
        "avg_duration_seconds": 2234.74, "deviation_seconds": 1135.27,
        "deviation_pct": 50.8, "health": "Severe",
    },
    "activities": [
        {"activity_name": "Bronze",             "activity_type": "SynapseNotebook",
         "avg_duration_seconds": 111.0, "latest_duration_seconds": 145.0,
         "deviation_seconds": 34.0,  "deviation_pct": 30.6,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 140.0},
        {"activity_name": "Gold",               "activity_type": "SynapseNotebook",
         "avg_duration_seconds": 112.0, "latest_duration_seconds": 160.0,
         "deviation_seconds": 48.0,  "deviation_pct": 42.9,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 150.0},
        {"activity_name": "If Condition1",      "activity_type": "IfCondition",
         "avg_duration_seconds": 620.0, "latest_duration_seconds": 700.0,
         "deviation_seconds": 80.0,  "deviation_pct": 12.9,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 690.0},
        {"activity_name": "Copy Raw to Bronze", "activity_type": "Copy",
         "avg_duration_seconds": 70.0,  "latest_duration_seconds": 82.0,
         "deviation_seconds": 12.0,  "deviation_pct": 17.1,
         "success_rate_pct": 100.0,  "health": "Warning", "p95_duration_seconds": 80.0},
    ],
}


# ─── MAIN ────────────────────────────────────────────────────────────────────

async def main():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run:  set ANTHROPIC_API_KEY=sk-ant-api03-...")
        return

    print(f"API Key  : {key[:20]}...{key[-4:]}")
    print(f"Client   : single shared AsyncAnthropic instance")
    print(f"Settings : max_tokens=4096 | timeout=60s | retries=3")

    # Databricks
    await run_mini_planner(DATABRICKS_PAYLOAD, run_id=f"run_{uuid.uuid4().hex[:8]}")

    #  Synapse 
    # await run_mini_planner(SYNAPSE_PAYLOAD, run_id=f"run_{uuid.uuid4().hex[:8]}")

    USAGE.print_summary()


if __name__ == "__main__":
    asyncio.run(main())