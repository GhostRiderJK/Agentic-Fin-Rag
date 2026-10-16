"""API package init."""

from app.api import admin, auth, chat, documents, reports_jobs

ROUTERS = [
    auth.router,
    documents.router,
    chat.router,
    reports_jobs.router,
    reports_jobs.jobs_router,
    admin.router,
]

__all__ = ["ROUTERS"]
