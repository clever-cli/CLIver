"""SPA static file serving routes for the admin portal."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route


def get_spa_routes(spa_dist_dir: Path) -> list:
    """Return SPA and asset serving routes."""

    async def handle_spa(request: Request):
        index_path = spa_dist_dir / "index.html"
        if not index_path.exists():
            return HTMLResponse(
                "<h1>Admin portal not built</h1><p>Run <code>make admin-build</code> to build the admin portal.</p>",
                status_code=503,
            )
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    async def handle_spa_assets(request: Request):
        file_path = request.path_params.get("path", "")
        # Vite outputs assets to an "assets/" subdirectory inside outDir.
        # The route /admin/assets/{path} consumes "assets/" from the URL,
        # so we must add it back when resolving the on-disk path.
        full_path = (spa_dist_dir / "assets" / file_path).resolve()
        if not str(full_path).startswith(str(spa_dist_dir.resolve())):
            return Response("Forbidden", status_code=403)
        if not full_path.exists() or not full_path.is_file():
            return Response("Not found", status_code=404)
        content_type = mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"
        return Response(
            full_path.read_bytes(),
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=31536000"} if "/assets/" in file_path else {},
        )

    return [
        Route("/admin/assets/{path:path}", handle_spa_assets),
        Route("/admin/{path:path}", handle_spa),
        Route("/admin", handle_spa),
        Route("/admin/", handle_spa),
    ]
