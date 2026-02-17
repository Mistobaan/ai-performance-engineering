"""Provide a minimal FastAPI stub when fastapi is unavailable.

This keeps tests runnable in minimal environments without altering the runtime
behavior when FastAPI is installed.
"""

from __future__ import annotations

import types
import sys
import urllib.parse
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional


def _install_fastapi_stub() -> None:
    # Avoid importing asyncio at interpreter startup. Some profiling environments
    # execute sitecustomize with a restricted stdlib path and only need this stub
    # when tests exercise FastAPI endpoints.
    import asyncio

    class Request:
        def __init__(
            self,
            *,
            method: str = "GET",
            query_params: Optional[Dict[str, Any]] = None,
            json_body: Any = None,
        ) -> None:
            self.method = method
            self.query_params = query_params or {}
            self._json_body = json_body

        async def json(self) -> Any:
            if self._json_body is None:
                raise ValueError("No JSON body")
            return self._json_body

        async def is_disconnected(self) -> bool:
            return False

    class JSONResponse:
        def __init__(self, content: Any, status_code: int = 200, **_: Any) -> None:
            self.content = content
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content: Any, status_code: int = 200, **_: Any) -> None:
            self.content = content
            self.status_code = status_code

    class CORSMiddleware:
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

    class _StubRoute:
        def __init__(self, path: str, methods: set[str], endpoint: Callable[..., Any]) -> None:
            self.path = path
            self.methods = methods
            self.endpoint = endpoint

    class FastAPI:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.routes: List[_StubRoute] = []

        def add_middleware(self, *_: Any, **__: Any) -> None:
            return None

        def get(self, path: str):
            def decorator(fn: Callable[..., Any]):
                self.routes.append(_StubRoute(path, {"GET"}, fn))
                return fn

            return decorator

        def post(self, path: str):
            def decorator(fn: Callable[..., Any]):
                self.routes.append(_StubRoute(path, {"POST"}, fn))
                return fn

            return decorator

    class _Response:
        def __init__(self, status_code: int, payload: Any = None, text_chunks: Optional[List[str]] = None) -> None:
            self.status_code = status_code
            self._payload = payload
            self._text_chunks = text_chunks or []

        def json(self) -> Any:
            return self._payload

        def iter_text(self) -> Iterable[str]:
            return iter(self._text_chunks)

    class TestClient:
        def __init__(self, app: Any) -> None:
            self._app = app

        def _find_route(self, path: str, method: str) -> Optional[_StubRoute]:
            for route in getattr(self._app, "routes", []):
                if route.path == path and method in route.methods:
                    return route
            return None

        def _request(self, method: str, url: str, *, stream: bool = False) -> _Response:
            parsed = urllib.parse.urlparse(url)
            path = parsed.path
            query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
            route = self._find_route(path, method)
            if route is None:
                return _Response(404, payload={"error": "Not Found"})

            request = Request(method=method, query_params=query_params)

            async def _call_endpoint():
                return await route.endpoint(request)

            result = asyncio.run(_call_endpoint())

            if isinstance(result, StreamingResponse):
                async def _consume_async_gen(gen: Any) -> List[str]:
                    chunks: List[str] = []
                    async for chunk in gen:
                        chunks.append(str(chunk))
                    return chunks

                chunks = asyncio.run(_consume_async_gen(result.content))
                return _Response(result.status_code, text_chunks=chunks)

            if isinstance(result, JSONResponse):
                return _Response(result.status_code, payload=result.content)

            return _Response(200, payload=result)

        def get(self, url: str) -> _Response:
            return self._request("GET", url)

        @contextmanager
        def stream(self, method: str, url: str):
            response = self._request(method, url, stream=True)
            yield response

    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.FastAPI = FastAPI
    fastapi_module.Request = Request
    fastapi_module.__dict__["__all__"] = ["FastAPI", "Request"]

    middleware_module = types.ModuleType("fastapi.middleware")
    cors_module = types.ModuleType("fastapi.middleware.cors")
    cors_module.CORSMiddleware = CORSMiddleware

    responses_module = types.ModuleType("fastapi.responses")
    responses_module.JSONResponse = JSONResponse
    responses_module.StreamingResponse = StreamingResponse

    testclient_module = types.ModuleType("fastapi.testclient")
    testclient_module.TestClient = TestClient

    sys.modules.setdefault("fastapi", fastapi_module)
    sys.modules.setdefault("fastapi.middleware", middleware_module)
    sys.modules.setdefault("fastapi.middleware.cors", cors_module)
    sys.modules.setdefault("fastapi.responses", responses_module)
    sys.modules.setdefault("fastapi.testclient", testclient_module)


try:
    import fastapi  # noqa: F401
except Exception:
    _install_fastapi_stub()
