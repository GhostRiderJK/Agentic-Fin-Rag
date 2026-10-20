"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import ROUTERS
from app.core.config import get_settings
from app.core.errors import install_exception_handlers
from app.core.logging import configure_logging, get_logger, LoggingMiddleware
from app.db.base import init_db
from app.db.mongodb import client as mongo
from app.db.mongodb import runtime


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    configure_logging()
    log = get_logger(__name__)
    log.info("starting %s v%s (env=%s)", settings.app_name, __version__,
             settings.app_env)
    init_db()
    log.info("database initialized")
    # The Mongo IO loop owns the only client, so the ping goes through
    # run_sync even though we are inside the (application) event loop here.
    if not runtime.run_sync(mongo.ping()):
        raise RuntimeError(
            "MongoDB is unreachable. Start it (`docker compose up -d mongo`) "
            f"or point MONGODB_URL at a live server; refusing to serve from a "
            "data layer that cannot be read."
        )
    log.info("mongodb connected")
    yield
    runtime.shutdown()
    log.info("shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Agentic Financial Intelligence Platform â€” RAG + structured "
            "financials + multi-agent reasoning + citation verification."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    for r in ROUTERS:
        app.include_router(r)

    @app.get("/", tags=["health"])
    def root():
        return {"app": settings.app_name, "version": __version__,
                "status": "ok", "docs": "/docs"}

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    return app


app = create_app()
