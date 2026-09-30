from __future__ import annotations
import asyncio
import os
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from dashboard_service import DashboardService
from telemetry_store import TelemetryFetchError, RESOLVER_VERSION
from investigation_persistence import PersistenceError
from planner_agent import PlannerAgent

STEP_LABELS = {
    "queued": "Starting investigation",
    "enrichment_seconds": "Gathering pipeline telemetry data",
    "intelligence_agents_seconds": "Analyzing pipeline execution",
    "planner_synthesis_seconds": "Consolidating agent findings",
    "evidence_validation_seconds": "Validating findings",
    "root_cause_seconds": "Identifying root causes",
    "recommendation_seconds": "Generating recommendations",
    "impact_seconds": "Assessing impacted assets",
    "report_seconds": "Compiling investigation report",
}
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Cloud Optimization API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("DASHBOARD_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)

svc = DashboardService()

_running_threads: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def _http(e: Exception) -> HTTPException:
    if isinstance(e, (TelemetryFetchError, PersistenceError)):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")



@app.get("/api/subscriptions")
def subscriptions():
    try:
        return svc.list_subscriptions()
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services")
async def services(sub: str):
    try:
        return await svc.list_services(sub)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/workspaces")
async def workspaces(sub: str, service: str):
    try:
        return await svc.list_workspaces(sub, service)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/item-types")
async def item_types(sub: str, service: str, rg: str, ws: str):
    try:
        return await svc.list_item_types(sub, service, rg, ws)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items")
async def items(sub: str, service: str, rg: str, ws: str,
                item_type: Optional[str] = Query(default=None)):
    try:
        return await svc.list_items(sub, service, rg, ws, item_type)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items/{item_type}/{item_name}")
async def item_detail(sub: str, service: str, rg: str, ws: str,
                      item_type: str, item_name: str):
    try:
        return await svc.item_detail(sub, service, rg, ws, item_type, item_name)
    except Exception as e:
        raise _http(e)



def _run_investigation_thread(payload: dict, investigation_id: str) -> None:
    """Thread entrypoint: runs its own private event loop for the full
    investigation lifecycle, isolated from the FastAPI request-handling loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        planner = PlannerAgent()
        loop.run_until_complete(
            planner.run_investigation(payload, investigation_id=investigation_id)
        )
    except Exception as e:
        print(f"[analyze:{investigation_id}] failed: {type(e).__name__}: {e}")
    finally:
        loop.close()
        with _lock:
            _running_threads.pop(investigation_id, None)


@app.post("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items/{item_type}/{item_name}/analyze")
async def analyze_workload(sub: str, service: str, rg: str, ws: str,
                           item_type: str, item_name: str):
    try:
        payload = await svc.build_trigger_payload(sub, service, rg, ws, item_type, item_name)
    except Exception as e:
        raise _http(e)

    investigation_id = PlannerAgent.new_investigation_id()

    thread = threading.Thread(
        target=_run_investigation_thread,
        args=(payload, investigation_id),
        name=f"investigation-{investigation_id}",
        daemon=True,   
    )
    with _lock:
        _running_threads[investigation_id] = thread
    thread.start()   

    base = f"/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}"
    return {
        "investigation_id": investigation_id,
        "status": "started",
        "item_name": item_name,
        "poll_url": base,                 # single polling endpoint:
                                          # running -> progress, completed -> full analysis
        "status_url": f"{base}/status",
        "analysis_url": f"{base}/analysis",
    }


@app.get("/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}")
async def investigation_poll(sub: str, service: str, investigation_id: str):
    with _lock:
        is_running_here = investigation_id in _running_threads

    try:
        manifest = await svc.investigation_status(sub, service, investigation_id)
    except PersistenceError as e:
        if is_running_here:
            return {"investigation_id": investigation_id, "status": "starting"}
        raise _http(e)
    except Exception as e:
        raise _http(e)

    status = manifest.get("status")
    if status == "completed":
        try:
            analysis = await svc.compose_analysis(sub, service, investigation_id)
        except Exception as e:
            raise _http(e)
        response = {"investigation_id": investigation_id, "status": "completed",
                    "analysis": analysis}
        if manifest.get("skipped_reason"):
            response["skipped_reason"] = manifest["skipped_reason"]
            response["message"] = manifest.get("message")
        return response

    if status == "failed":
        return {"investigation_id": investigation_id, "status": "failed",
                "error": manifest.get("error"), "manifest": manifest}

    current_stage_key = manifest.get("current_stage")
    current_step = STEP_LABELS.get(current_stage_key, current_stage_key) or "Starting"

    return {
        "investigation_id": investigation_id,
        "status": "running",
        "current_step": current_step,
        "elapsed_seconds": manifest.get("total_seconds_so_far"),
        "active_agents": manifest.get("active_agents", []),
        "dispatched_agents": manifest.get("dispatched_agents", []),
        "agent_runs": manifest.get("agent_runs"),
        "findings_count": manifest.get("findings_count"),
        "validated_findings_count": manifest.get("validated_findings_count"),
        "root_causes_count": manifest.get("root_causes_count"),
        "recommendations_count": manifest.get("recommendations_count"),
        "in_this_process": is_running_here,
    }


# @app.get("/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}/status")
# async def investigation_status(sub: str, service: str, investigation_id: str):
#     """Poll target: the persisted manifest (flushed at every checkpoint)."""
#     try:
#         manifest = await svc.investigation_status(sub, service, investigation_id)
#     except PersistenceError as e:
#         # id was just issued but first checkpoint not yet flushed
#         if investigation_id in _running_threads:
#             return {"investigation_id": investigation_id, "status": "starting"}
#         raise _http(e)
#     except Exception as e:
#         raise _http(e)
#     manifest["in_this_process"] = investigation_id in _running_threads
#     return manifest


# @app.get("/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}/analysis")
# async def investigation_analysis(sub: str, service: str, investigation_id: str):
#     """The full analysis screen, composed only from persisted files."""
#     try:
#         return await svc.compose_analysis(sub, service, investigation_id)
#     except Exception as e:
#         raise _http(e)


# @app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items/{item_type}/{item_name}/investigations")
# async def item_investigations(sub: str, service: str, rg: str, ws: str,
#                               item_type: str, item_name: str,
#                               limit: int = Query(default=10, le=50)):
#     try:
#         return await svc.investigation_history(sub, service, ws, item_type, item_name, limit)
#     except Exception as e:
#         raise _http(e)

@app.post("/post-cost-investigation")
async def post_cost_investigation(payload: dict):
    planner = PlannerAgent()
    state = await planner.run_cost_investigation(payload)
    return state.final_report


@app.get("/api/subscriptions/{sub}/investigations/runtime")
async def runtime_investigations(sub: str):
    try:
        return await svc.list_runtime_investigations(sub)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/investigations/cost")
async def cost_investigations(sub: str):
    try:
        return await svc.list_cost_investigations(sub)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/cost-investigations/{investigation_id}")
async def cost_investigation_detail(sub: str, service: str, investigation_id: str):
    try:
        manifest = await svc.investigation_status(sub, service, investigation_id)
    except PersistenceError as e:
        raise _http(e)
    except Exception as e:
        raise _http(e)

    if svc.investigation_type_of(manifest) != "cost":
        raise HTTPException(
            status_code=404,
            detail=f"'{investigation_id}' is not a cost investigation.",
        )

    status = manifest.get("status")
    if status == "completed":
        try:
            analysis = await svc.final_report(sub, service, investigation_id)
        except Exception as e:
            raise _http(e)
        return {"investigation_id": investigation_id, "status": "completed",
                "analysis": analysis}

    if status == "failed":
        return {"investigation_id": investigation_id, "status": "failed",
                "error": manifest.get("error"), "manifest": manifest}

    current_stage_key = manifest.get("current_stage")
    current_step = STEP_LABELS.get(current_stage_key, current_stage_key) or "Starting"
    return {
        "investigation_id": investigation_id,
        "status": "running",
        "current_step": current_step,
        "elapsed_seconds": manifest.get("total_seconds_so_far"),
        "active_agents": manifest.get("active_agents", []),
        "dispatched_agents": manifest.get("dispatched_agents", []),
        "findings_count": manifest.get("findings_count"),
        "validated_findings_count": manifest.get("validated_findings_count"),
        "root_causes_count": manifest.get("root_causes_count"),
        "recommendations_count": manifest.get("recommendations_count"),
    }


@app.get("/api/health")
def health():
    with _lock:
        running = sorted(_running_threads)
    return {"status": "ok",
            "resolver_version": RESOLVER_VERSION,
            "running_investigations": running}