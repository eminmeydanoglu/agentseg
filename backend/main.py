"""
FastAPI application entry point
"""

import sys
import logging
import os
import traceback
from pathlib import Path

# Add parent directory to path so we can import core and ml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager

from backend.config import CORS_ORIGINS, API_HOST, API_PORT
from backend.api import projects, images, annotations, labels, ml, export, files, video_tracks, activity, agent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("AgentSeg API starting...")
    if os.getenv("VIDEO_CLOSE_STALE_SESSIONS") == "1":
        from backend.api.video_tracks import close_stale_sessions
        close_stale_sessions()
    yield
    # Shutdown
    logger.info("AgentSeg API shutting down...")


app = FastAPI(
    title="AgentSeg API",
    description="Image annotation API with SAM segmentation and tracking",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions with full traceback."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}:")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(images.router, prefix="/api/images", tags=["Images"])
app.include_router(annotations.router, prefix="/api/annotations", tags=["Annotations"])
app.include_router(video_tracks.router, prefix="/api/video", tags=["Video tracks"])
app.include_router(activity.router, prefix="/api/activity", tags=["Activity"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(labels.router, prefix="/api/labels", tags=["Labels"])
app.include_router(ml.router, prefix="/api/ml", tags=["ML"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agentseg-api", "ui": (FRONTEND_DIST / "index.html").is_file()}


# Built web UI (`bun run build` in frontend/). Checked per request so a build
# made after the backend started is picked up without a restart.
FRONTEND_DIST = Path(
    os.getenv("AGENTSEG_FRONTEND_DIST", str(Path(__file__).resolve().parent.parent / "frontend" / "dist"))
).resolve()


@app.get("/{path:path}", include_in_schema=False)
async def web_ui(path: str):
    """Serve the single-page web UI; unknown paths fall back to index.html."""
    if path == "api" or path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Web UI is not built; run `agentseg up` or `bun run build` in frontend/")
    target = (FRONTEND_DIST / path).resolve()
    if path and target.is_file() and target.is_relative_to(FRONTEND_DIST):
        # Vite puts a content hash in asset names, so they can be cached for good.
        cache = "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
        return FileResponse(target, headers={"Cache-Control": cache})
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
