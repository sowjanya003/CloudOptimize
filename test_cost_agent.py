import asyncio

from schemas import InvestigationContext
from sub_agents.cost_intelligence_agent import CostIntelligenceAgent

import os
from dotenv import load_dotenv

load_dotenv()


print("Starting Cost Agent Test...")
payload = {
  "service": "synapse",
  "workspace_name": "synapseforunifydcloud",
  "resource_group": "UnifydCloud",
  "total_cost": 2140.811145,
  "currency": "INR",
  "last_30_days": {
    "total_cost": 2140.811145,
    "period_days": 30,
    "cost_by_meter": {
      "Azure Synapse Analytics": 2140.811145
    }
  },
  "last_6_months": {
    "total_cost": 24314.488738,
    "period_days": 180,
    "cost_by_meter": {
      "Azure Synapse Analytics": 24314.488737
    }
  },
  "baseline_monthly_cost": 4052.41,
  "deviation": {
    "deviation_amount": -1911.6,
    "deviation_pct": -47.17,
    "status": "below_baseline"
  },
  "fetched_at": "2026-07-08T10:18:16.402693+00:00",
  "note": "Workspace-level cost only. Pipeline/notebook/spark_pool cost is not separately tracked in Azure Cost Management.",
  "_adls_synced_at": "2026-07-08T10:18:16.402693+00:00",
  "_subscription_id": "2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e",
  "_resource_group": "UnifydCloud",
  "_workspace_name": "synapseforunifydcloud",
  "_service": "synapse"
}

ctx = InvestigationContext.from_deviation_payload(payload)

ctx.enriched = True

ctx.enrichment = {}

agent = CostIntelligenceAgent()

async def main():
    print("Inside main()")

    findings = await agent.investigate(ctx)

    for finding in findings:
        print(finding)

asyncio.run(main())