# from __future__ import annotations
# import asyncio
# import json
# import os

# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass  # fine if you're exporting the env var manually instead

# from schemas import InvestigationContext
# from sub_agents.runtime_intelligence_agent import RuntimeIntelligenceAgent
# from sub_agents.base_agent import SDK_AVAILABLE

# # Reuse the exact databricks baseline sample you shared
# SAMPLE_PAYLOAD = {
#   "subscription_id": "2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e",
#   "service": "databricks",
#   "resource_group": "Synapsetofabric",
#   "workspace_name": "Databrickstofabricsws",
#   "item_type": "jobs",
#   "item_name": "01_Sales_Data_Processing",
#   "pipeline_baseline": {
#     "avg_duration_seconds": 205.01, "median_duration_seconds": 206.81,
#     "p95_duration_seconds": 218.35, "min_duration_seconds": 192.16,
#     "max_duration_seconds": 220.69, "success_rate_pct": 100.0,
#     "latest_run_duration_seconds": 209.16, "deviation_seconds": 4.15,
#     "deviation_pct": 2.02, "health": "Healthy"
#   },
#   "activities": [
#     {"activity_name": "Add_Category", "activity_type": "notebook_task",
#      "avg_duration_seconds": 20.57, "latest_duration_seconds": 23.0,
#      "deviation_seconds": 2.43, "deviation_pct": 11.81, "success_rate_pct": 100.0,
#      "health": "Warning", "p95_duration_seconds": 25.1, "median_duration_seconds": 20.0},
#     {"activity_name": "Validate_Schema", "activity_type": "notebook_task",
#      "avg_duration_seconds": 18.43, "latest_duration_seconds": 23.0,
#      "deviation_seconds": 4.57, "deviation_pct": 24.8, "success_rate_pct": 100.0,
#      "health": "Warning", "p95_duration_seconds": 23.0, "median_duration_seconds": 17.0},
#     {"activity_name": "Filter_Data", "activity_type": "notebook_task",
#      "avg_duration_seconds": 19.0, "latest_duration_seconds": 21.0,
#      "deviation_seconds": 2.0, "deviation_pct": 10.53, "success_rate_pct": 100.0,
#      "health": "Warning", "p95_duration_seconds": 21.0, "median_duration_seconds": 20.0},
#     {"activity_name": "Validate_Source", "activity_type": "notebook_task",
#      "avg_duration_seconds": 20.0, "latest_duration_seconds": 17.0,
#      "deviation_seconds": -3.0, "deviation_pct": -15.0, "success_rate_pct": 100.0,
#      "health": "Warning", "p95_duration_seconds": 28.3, "median_duration_seconds": 17.0},
#   ],
# }


# async def main():
#     print(f"ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
#     print(f"claude_agent_sdk importable: {SDK_AVAILABLE}\n")

#     ctx = InvestigationContext.from_deviation_payload(SAMPLE_PAYLOAD)
#     print("=== Normalized InvestigationContext ===")
#     print(f"service={ctx.service} item_name={ctx.item_name} "
#           f"pipeline_health={ctx.pipeline_health} "
#           f"degraded_activities={[a.activity_name for a in ctx.degraded_activities()]}\n")

#     agent = RuntimeIntelligenceAgent(asset_loader=None)  # no ADLS wired yet - fine for this test

#     # Inspect the exact prompt being sent, before firing it - useful to sanity
#     # check token size / content before spending real API credits.
#     prompt = agent.build_prompt(ctx, agent.relevant_activities(ctx))
#     print("=== Prompt being sent to the model ===")
#     print(prompt[:1500], "...\n" if len(prompt) > 1500 else "\n")

#     print("=== Calling RuntimeIntelligenceAgent.investigate() ===")
#     findings = await agent.investigate(ctx)

#     print(f"\n=== {len(findings)} finding(s) returned ===")
#     for f in findings:
#         print(json.dumps(f.to_dict(), indent=2))


# if __name__ == "__main__":
#     asyncio.run(main())