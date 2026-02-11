import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager

import fastapi
import psutil
import uvicorn
from fastapi.responses import JSONResponse
from starlette.routing import Mount

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    logger.info("APP START UP")
    yield
    logger.info("APP SHUTDOWN")


app = fastapi.FastAPI(lifespan=lifespan)

requests_counter = None
requests_counter_py = 0
cpu_usaga_gauge = None


def create_metrics_recorders():
    logger.info("Create metrics recorders")
    global requests_counter, cpu_usaga_gauge
    from prometheus_client import Counter, Gauge

    requests_counter = Counter(
        "demo_requests_total", "Total demo requests", labelnames=["model"]
    )
    cpu_usaga_gauge = Gauge("cpu_usage", "CPU usage", multiprocess_mode="livesum")


@app.get("/")
async def entrypoint() -> JSONResponse:
    global requests_counter_py
    assert requests_counter is not None and cpu_usaga_gauge is not None
    requests_counter.labels(model="demo_model").inc()
    requests_counter_py += 1
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_usaga_gauge.set(cpu_percent)
    return {"msg": "Hello, World", "cpu_percent": cpu_percent}


@app.get("/json_metrics")
async def export_json_metrics() -> JSONResponse:
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_usaga_gauge.set(cpu_percent)
    return {
        "requests_counter": requests_counter_py,
        "cpu_usage": cpu_percent,
    }


prometheus_multiproc_dir = tempfile.TemporaryDirectory()
os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
logger.info(f"PROMETHEUS_MULTIPROC_DIR: {os.environ['PROMETHEUS_MULTIPROC_DIR']}")


def mount_metrics():
    logger.info("Start to mount metrics")
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry))
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
    create_metrics_recorders()


if __name__ == "__main__":
    mount_metrics()
    uvicorn.run(app)
