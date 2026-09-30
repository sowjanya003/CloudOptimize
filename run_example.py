from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from planner_agent import PlannerAgent
from schemas import InvestigationState


class RunTracker:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, inp: int, out: int):
        self.input_tokens += inp
        self.output_tokens += out

    def print_summary(self):
        elapsed = round(time.perf_counter() - self.start_time, 1)
        cost = (self.input_tokens / 1_000_000 * 3.0) + (self.output_tokens / 1_000_000 * 15.0)
        print(f"\n{'═'*70}")
        print("RUN SUMMARY")
        print(f"  Duration      : {elapsed}s")
        print(f"  Estimated Cost: ${cost:.5f}")
        print(f"{'═'*70}")


TRACKER = RunTracker()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_investigation(run_id: str, state: InvestigationState) -> Path:
    run_dir = Path("investigations") / run_id
    ctx = state.context

    save_json(run_dir / "context.json", {
        "subscription_id": ctx.subscription_id,
        "service": ctx.service,
        "resource_group": ctx.resource_group,
        "workspace_name": ctx.workspace_name,
        "item_type": ctx.item_type,
        "item_name": ctx.item_name,
        "pipeline_health": ctx.pipeline_health,
        "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
        "pipeline_deviation_seconds": ctx.pipeline_deviation_seconds,
    })

    save_json(run_dir / "root_causes.json", {"root_causes": [asdict(rc) for rc in state.root_causes]})
    save_json(run_dir / "recommendations.json", {"recommendations": [asdict(r) for r in state.recommendations]})
    save_json(run_dir / "final_report.json", state.final_report)

    return run_dir


SECTION = "═" * 70
DIVIDER = "─" * 70


def print_header(title: str, run_id: str, ctx) -> None:
    print(f"\n{SECTION}")
    print(f"  PIPELINE INVESTIGATION REPORT")
    print(f"  {title}")
    print(f"  Run ID : {run_id}")
    print(SECTION)
    print(f"  Service   : {ctx.service.upper()}")
    print(f"  Pipeline  : {ctx.item_name}")
    print(f"  Health    : {ctx.pipeline_health}")
    print(f"  Deviation : {ctx.pipeline_deviation_pct:+.1f}% ({ctx.pipeline_deviation_seconds:+.1f}s)")


async def run_investigation(payload: dict, title: str) -> None:
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    planner = PlannerAgent()

    print(f"\n{SECTION}")
    print(f"  STARTING ANALYSIS → {title}")
    print(SECTION)

    t0 = time.perf_counter()
    state = await planner.run_investigation(payload)
    elapsed = round(time.perf_counter() - t0, 1)

    ctx = state.context
    run_dir = save_investigation(run_id, state)

    print_header(title, run_id, ctx)

    print(f"\n{DIVIDER}")
    print("  EXECUTIVE SUMMARY")
    print(DIVIDER)

    print("\nROOT CAUSES:")
    for i, rc in enumerate(state.root_causes, 1):
        print(f"  {i}. [{rc.category}] (Confidence: {rc.confidence:.0f}%)")
        print(f"     {rc.description}")

    print(f"\n{DIVIDER}")
    print("  TOP RECOMMENDATIONS")
    print(DIVIDER)
    for i, r in enumerate(state.recommendations[:5], 1):
        print(f"\n  {i}. {r.title}")
        print(f"     Target     : {r.target_activity}")
        print(f"     Expected Gain : ~{r.impact_score*100:.0f}%")
        print(f"     Effort/Risk : {r.effort_score:.1f} / {r.risk_score:.1f}")
        print(f"     {r.description[:180]}...")

    print(f"\n{DIVIDER}")
    print("  IMPACT ASSESSMENT")
    print(DIVIDER)
    print(state.impact_summary.get('summary', 'Analysis completed successfully.'))

    print(f"\n{SECTION}")
    print(f"  Full report saved at: {run_dir}/final_report.json")
    print(SECTION)
    print(f"  Analysis completed in {elapsed}s")


# ─── SAMPLE PAYLOADS ─────────────────────────────────────────────────────────

PAYLOADS = {

    "databricks": (
        "Databricks",
        {
  "subscription_id": "2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e",
  "service": "databricks",
  "resource_group": "Synapsetofabric",
  "workspace_name": "Databrickstofabricsws",
  "item_type": "jobs",
  "item_name": "01_Sales_Data_Processing",
  "total_runs_available": 14,
  "baseline_window": 7,
  "pipeline_baseline": {
    "service": "databricks",
    "resource_group": "Synapsetofabric",
    "workspace_name": "Databrickstofabricsws",
    "item_type": "jobs",
    "item_name": "01_Sales_Data_Processing",
    "total_runs_available": 14,
    "baseline_window": 7,
    "avg_duration_seconds": 205.01,
    "median_duration_seconds": 206.81,
    "p95_duration_seconds": 218.35,
    "min_duration_seconds": 192.16,
    "max_duration_seconds": 220.69,
    "success_rate_pct": 100.0,
    "latest_run_duration_seconds": 209.16,
    "deviation_seconds": 4.15,
    "deviation_pct": 2.02,
    "health": "Healthy",
    "calculated_at": "2026-06-29T09:17:28.927572"
  },
  "activities": [
    {
      "activity_name": "Add_Category",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 20.57,
      "median_duration_seconds": 20.0,
      "p95_duration_seconds": 25.1,
      "min_duration_seconds": 15.0,
      "max_duration_seconds": 26.0,
      "latest_duration_seconds": 23.0,
      "deviation_seconds": 2.43,
      "deviation_pct": 11.81,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "Calculate_Tax",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 18.14,
      "median_duration_seconds": 18.0,
      "p95_duration_seconds": 20.8,
      "min_duration_seconds": 16.0,
      "max_duration_seconds": 22.0,
      "latest_duration_seconds": 18.0,
      "deviation_seconds": -0.14,
      "deviation_pct": -0.77,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Calculate_Total",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 19.43,
      "median_duration_seconds": 18.0,
      "p95_duration_seconds": 24.1,
      "min_duration_seconds": 16.0,
      "max_duration_seconds": 25.0,
      "latest_duration_seconds": 18.0,
      "deviation_seconds": -1.43,
      "deviation_pct": -7.36,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Check_Nulls",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 18.43,
      "median_duration_seconds": 18.0,
      "p95_duration_seconds": 21.7,
      "min_duration_seconds": 16.0,
      "max_duration_seconds": 22.0,
      "latest_duration_seconds": 19.0,
      "deviation_seconds": 0.57,
      "deviation_pct": 3.09,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Filter_Data",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 19.0,
      "median_duration_seconds": 20.0,
      "p95_duration_seconds": 21.0,
      "min_duration_seconds": 16.0,
      "max_duration_seconds": 21.0,
      "latest_duration_seconds": 21.0,
      "deviation_seconds": 2.0,
      "deviation_pct": 10.53,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "Generate_Report",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 18.0,
      "median_duration_seconds": 18.0,
      "p95_duration_seconds": 18.7,
      "min_duration_seconds": 17.0,
      "max_duration_seconds": 19.0,
      "latest_duration_seconds": 17.0,
      "deviation_seconds": -1.0,
      "deviation_pct": -5.56,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Record_Count",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 18.43,
      "median_duration_seconds": 17.0,
      "p95_duration_seconds": 24.6,
      "min_duration_seconds": 16.0,
      "max_duration_seconds": 27.0,
      "latest_duration_seconds": 17.0,
      "deviation_seconds": -1.43,
      "deviation_pct": -7.76,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Success_Message",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 19.29,
      "median_duration_seconds": 19.0,
      "p95_duration_seconds": 25.2,
      "min_duration_seconds": 14.0,
      "max_duration_seconds": 27.0,
      "latest_duration_seconds": 20.0,
      "deviation_seconds": 0.71,
      "deviation_pct": 3.68,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Validate_Schema",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 18.43,
      "median_duration_seconds": 17.0,
      "p95_duration_seconds": 23.0,
      "min_duration_seconds": 14.0,
      "max_duration_seconds": 23.0,
      "latest_duration_seconds": 23.0,
      "deviation_seconds": 4.57,
      "deviation_pct": 24.8,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "Validate_Source",
      "activity_type": "notebook_task",
      "total_runs_available": 7,
      "baseline_window": 7,
      "avg_duration_seconds": 20.0,
      "median_duration_seconds": 17.0,
      "p95_duration_seconds": 28.3,
      "min_duration_seconds": 16.0,
      "max_duration_seconds": 31.0,
      "latest_duration_seconds": 17.0,
      "deviation_seconds": -3.0,
      "deviation_pct": -15.0,
      "success_rate_pct": 100.0,
      "health": "Warning"
    }
  ],
  "calculated_at": "2026-06-29T09:17:28.927572"
}
    ),

    "synapse": (
        "Synapse",
        {
  "subscription_id": "2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e",
  "service": "synapse",
  "resource_group": "Synapsetofabric",
  "workspace_name": "synapsetofabricws1",
  "item_type": "pipelines",
  "item_name": "Data_Pipeline_-_Engineering",
  "total_runs_available": 15,
  "baseline_window": 5,
  "pipeline_baseline": {
    "service": "synapse",
    "resource_group": "Synapsetofabric",
    "workspace_name": "synapsetofabricws1",
    "item_type": "pipelines",
    "item_name": "Data_Pipeline_-_Engineering",
    "total_runs_available": 15,
    "baseline_window": 5,
    "avg_duration_seconds": 1088.86,
    "median_duration_seconds": 1183.1,
    "p95_duration_seconds": 1533.51,
    "min_duration_seconds": 88.58,
    "max_duration_seconds": 1556.32,
    "success_rate_pct": 80.0,
    "latest_run_duration_seconds": 88.58,
    "deviation_seconds": -1000.28,
    "deviation_pct": -91.86,
    "health": "Severe",
    "calculated_at": "2026-07-02T10:42:23.218737"
  },
  "activities": [
    {
      "activity_name": "Apply_Data_Quality_Rules",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 148.29,
      "median_duration_seconds": 138.99,
      "p95_duration_seconds": 181.08,
      "min_duration_seconds": 126.97,
      "max_duration_seconds": 188.19,
      "latest_duration_seconds": 140.75,
      "deviation_seconds": -7.54,
      "deviation_pct": -5.08,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Bronze",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 5,
      "baseline_window": 5,
      "avg_duration_seconds": 177.58,
      "median_duration_seconds": 126.16,
      "p95_duration_seconds": 331.08,
      "min_duration_seconds": 16.0,
      "max_duration_seconds": 340.53,
      "latest_duration_seconds": 16.0,
      "deviation_seconds": -161.58,
      "deviation_pct": -90.99,
      "success_rate_pct": 80.0,
      "health": "Severe"
    },
    {
      "activity_name": "Copy Raw to Bronze",
      "activity_type": "Copy",
      "total_runs_available": 5,
      "baseline_window": 5,
      "avg_duration_seconds": 80.57,
      "median_duration_seconds": 81.93,
      "p95_duration_seconds": 100.68,
      "min_duration_seconds": 60.62,
      "max_duration_seconds": 102.07,
      "latest_duration_seconds": 63.15,
      "deviation_seconds": -17.42,
      "deviation_pct": -21.62,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "Data Cleaning",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 112.71,
      "median_duration_seconds": 114.41,
      "p95_duration_seconds": 124.67,
      "min_duration_seconds": 96.17,
      "max_duration_seconds": 125.85,
      "latest_duration_seconds": 125.85,
      "deviation_seconds": 13.14,
      "deviation_pct": 11.66,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "ForEach Notebook",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 115.25,
      "median_duration_seconds": 111.4,
      "p95_duration_seconds": 124.96,
      "min_duration_seconds": 110.89,
      "max_duration_seconds": 127.31,
      "latest_duration_seconds": 111.16,
      "deviation_seconds": -4.09,
      "deviation_pct": -3.55,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "ForEach1",
      "activity_type": "ForEach",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 119.14,
      "median_duration_seconds": 115.69,
      "p95_duration_seconds": 128.65,
      "min_duration_seconds": 114.46,
      "max_duration_seconds": 130.72,
      "latest_duration_seconds": 114.46,
      "deviation_seconds": -4.68,
      "deviation_pct": -3.93,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "GM_Check_File",
      "activity_type": "GetMetadata",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 48.82,
      "median_duration_seconds": 47.09,
      "p95_duration_seconds": 92.98,
      "min_duration_seconds": 6.86,
      "max_duration_seconds": 94.25,
      "latest_duration_seconds": 6.86,
      "deviation_seconds": -41.96,
      "deviation_pct": -85.95,
      "success_rate_pct": 100.0,
      "health": "Severe"
    },
    {
      "activity_name": "Generate_Derived_Metrics",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 151.65,
      "median_duration_seconds": 149.4,
      "p95_duration_seconds": 168.45,
      "min_duration_seconds": 136.25,
      "max_duration_seconds": 171.55,
      "latest_duration_seconds": 150.89,
      "deviation_seconds": -0.76,
      "deviation_pct": -0.5,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Gold",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 135.17,
      "median_duration_seconds": 123.06,
      "p95_duration_seconds": 174.04,
      "min_duration_seconds": 112.05,
      "max_duration_seconds": 182.53,
      "latest_duration_seconds": 112.05,
      "deviation_seconds": -23.12,
      "deviation_pct": -17.1,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "IF_File_Exists",
      "activity_type": "IfCondition",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 1.97,
      "median_duration_seconds": 1.95,
      "p95_duration_seconds": 2.25,
      "min_duration_seconds": 1.69,
      "max_duration_seconds": 2.29,
      "latest_duration_seconds": 1.69,
      "deviation_seconds": -0.28,
      "deviation_pct": -14.21,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "If Condition1",
      "activity_type": "IfCondition",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 721.57,
      "median_duration_seconds": 708.41,
      "p95_duration_seconds": 760.99,
      "min_duration_seconds": 700.17,
      "max_duration_seconds": 769.28,
      "latest_duration_seconds": 700.17,
      "deviation_seconds": -21.4,
      "deviation_pct": -2.97,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Load_Curated_Layer",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 139.12,
      "median_duration_seconds": 142.21,
      "p95_duration_seconds": 143.39,
      "min_duration_seconds": 128.62,
      "max_duration_seconds": 143.45,
      "latest_duration_seconds": 141.35,
      "deviation_seconds": 2.23,
      "deviation_pct": 1.6,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Post_Load_Validation_Report",
      "activity_type": "SynapseNotebook",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 134.81,
      "median_duration_seconds": 135.34,
      "p95_duration_seconds": 141.84,
      "min_duration_seconds": 126.71,
      "max_duration_seconds": 141.87,
      "latest_duration_seconds": 141.87,
      "deviation_seconds": 7.06,
      "deviation_pct": 5.24,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    },
    {
      "activity_name": "Set_File_Found",
      "activity_type": "SetVariable",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 0.46,
      "median_duration_seconds": 0.47,
      "p95_duration_seconds": 0.53,
      "min_duration_seconds": 0.35,
      "max_duration_seconds": 0.54,
      "latest_duration_seconds": 0.35,
      "deviation_seconds": -0.11,
      "deviation_pct": -23.91,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "Switch1",
      "activity_type": "Switch",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 115.83,
      "median_duration_seconds": 117.68,
      "p95_duration_seconds": 127.79,
      "min_duration_seconds": 99.08,
      "max_duration_seconds": 128.86,
      "latest_duration_seconds": 128.86,
      "deviation_seconds": 13.03,
      "deviation_pct": 11.25,
      "success_rate_pct": 100.0,
      "health": "Warning"
    },
    {
      "activity_name": "Wait for 10 sec",
      "activity_type": "Wait",
      "total_runs_available": 4,
      "baseline_window": 5,
      "avg_duration_seconds": 10.8,
      "median_duration_seconds": 10.83,
      "p95_duration_seconds": 10.98,
      "min_duration_seconds": 10.56,
      "max_duration_seconds": 10.99,
      "latest_duration_seconds": 10.88,
      "deviation_seconds": 0.08,
      "deviation_pct": 0.74,
      "success_rate_pct": 100.0,
      "health": "Healthy"
    }
  ],
  "calculated_at": "2026-07-02T10:42:23.218817"
}
    ),

    "adf": (
        "ADF — PL_Customer_File_Process (Healthy, minor deviation)",
        {
            "subscription_id": "2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e",
            "service": "adf",
            "resource_group": "unifydcloud",
            "workspace_name": "AzureOptimization",
            "item_type": "pipeline",
            "item_name": "PL_Customer_File_Process",
            "pipeline_baseline": {
                "avg_duration_seconds": 67.42,
                "deviation_seconds": -0.3,
                "deviation_pct": -0.44,
                "health": "Healthy",
            },
            "activities": [
                {"activity_name": "Check_Source_File", "activity_type": "GetMetadata", "avg_duration_seconds": 6.4, "latest_duration_seconds": 17.7, "deviation_seconds": 11.3, "deviation_pct": 176.6, "success_rate_pct": 100.0, "health": "Warning"},
                {"activity_name": "Copy_Customers", "activity_type": "Copy", "avg_duration_seconds": 21.0, "latest_duration_seconds": 25.0, "deviation_seconds": 4.0, "deviation_pct": 19.0, "success_rate_pct": 100.0, "health": "Warning"},
                {"activity_name": "Check_Copied_File", "activity_type": "GetMetadata", "avg_duration_seconds": 17.0, "latest_duration_seconds": 18.0, "deviation_seconds": 1.0, "deviation_pct": 5.9, "success_rate_pct": 100.0, "health": "Healthy"},
            ],
        }
    ),
}


async def main():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: set ANTHROPIC_API_KEY=sk-ant-api03-...")
        return

    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "databricks"

    if arg == "all":
        for key_name, (title, payload) in PAYLOADS.items():
            await run_investigation(payload, title)
    elif arg in PAYLOADS:
        title, payload = PAYLOADS[arg]
        await run_investigation(payload, title)
    else:
        print(f"Unknown payload '{arg}'. Use: databricks | synapse | adf | all")
        return

    TRACKER.print_summary()


if __name__ == "__main__":
    asyncio.run(main())