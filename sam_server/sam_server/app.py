"""FastAPI app: thin HTTP passthrough over SamNode."""

from __future__ import annotations

import asyncio
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .node import NotFound, SamNode

logger = logging.getLogger("sam_server")
_END = object()


def _error(e: Exception) -> JSONResponse:
    if isinstance(e, NotFound):
        code = 404
    elif isinstance(e, (KeyError, ValueError, TypeError, NotImplementedError)):
        code = 400
    else:
        code = 500
    if code == 500:
        logger.error("request failed:\n%s", traceback.format_exc())
    return JSONResponse(status_code=code,
                        content={"error": type(e).__name__, "detail": str(e)})


def create_app(node: SamNode, load_on_startup: bool = True) -> FastAPI:
    app = FastAPI(title="sam_server")
    app.state.node = node

    async def run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(node.executor, fn, *args)

    @app.on_event("startup")
    async def _startup():
        if load_on_startup:
            await run(node.load)

    @app.get("/health")
    async def health():
        return {"status": "ok" if node.ready else "loading", "ready": node.ready}

    @app.get("/info")
    async def info():
        return node.info()

    @app.post("/video/{model}/request")
    async def video_request(model: str, request: Request):
        try:
            body = await request.json()
            if body.get("type") == "cancel_propagation":
                # bypasses the worker queue so it can stop a running stream
                return node.video_cancel(model, body)
            return await run(node.video_request, model, body)
        except Exception as e:
            return _error(e)

    @app.post("/video/{model}/stream_request")
    async def video_stream_request(model: str, request: Request):
        try:
            body = await request.json()
            node.get_video(model)
        except Exception as e:
            return _error(e)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        sid = body.get("session_id")
        cancel = node.register_stream(sid)

        def emit(line: str):
            loop.call_soon_threadsafe(q.put_nowait, line)

        def job():
            try:
                node.video_stream(model, body, cancel, emit)
            finally:
                node.unregister_stream(sid, cancel)
                loop.call_soon_threadsafe(q.put_nowait, _END)

        loop.run_in_executor(node.executor, job)

        async def lines():
            try:
                while True:
                    item = await q.get()
                    if item is _END:
                        break
                    yield item + "\n"
            finally:
                cancel.set()  # client went away -> stop holding the worker

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/image/set_image")
    async def image_set(request: Request):
        try:
            return await run(node.image_set, await request.json())
        except Exception as e:
            return _error(e)

    @app.post("/image/call/{method}")
    async def image_call(method: str, request: Request):
        try:
            return await run(node.image_call, method, await request.json())
        except Exception as e:
            return _error(e)

    @app.delete("/image/states/{state_id}")
    async def image_delete(state_id: str):
        try:
            return await run(node.image_delete, state_id)
        except Exception as e:
            return _error(e)

    return app
