"""FastAPI service for the Customer Success Agent."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import uvicorn
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)

from .app import (
    ApplicationBuildError,
    ApplicationComponents,
    build_application,
)
from .ingest import TicketIngestionError
from .metrics import ApplicationMetrics


API_TITLE = "Customer Success Agent"
API_VERSION = "1.0.0"


class ApplicationProvider:
    """Lazily construct and cache application components."""

    def __init__(
        self,
        factory: Callable[
            [],
            ApplicationComponents,
        ] = build_application,
    ) -> None:
        self.factory = factory
        self._application: (
            ApplicationComponents | None
        ) = None
        self._initialization_error: (
            Exception | None
        ) = None
        self._lock = threading.Lock()

    @property
    def initialized(self) -> bool:
        return self._application is not None

    @property
    def initialization_error(
        self,
    ) -> Exception | None:
        return self._initialization_error

    def get(self) -> ApplicationComponents:
        if self._application is not None:
            return self._application

        with self._lock:
            if self._application is not None:
                return self._application

            try:
                application = self.factory()
            except Exception as exc:
                self._initialization_error = exc
                raise
            else:
                self._application = application
                self._initialization_error = None
                return application


def create_api(
    *,
    provider: ApplicationProvider | None = None,
    application: ApplicationComponents | None = None,
    metrics: ApplicationMetrics | None = None,
) -> FastAPI:
    """Create the API with optional offline dependencies."""

    if (
        provider is not None
        and application is not None
    ):
        raise ValueError(
            "Provide either provider or application, not both."
        )

    if application is not None:
        provider = ApplicationProvider(
            factory=lambda: application
        )

    resolved_provider = (
        provider or ApplicationProvider()
    )
    resolved_metrics = (
        metrics or ApplicationMetrics()
    )

    api = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "Classifies, retrieves, routes, generates, "
            "validates, and audits customer-success tickets."
        ),
    )

    api.state.application_provider = (
        resolved_provider
    )
    api.state.metrics = resolved_metrics

    @api.middleware("http")
    async def record_http_metrics(
        request: Request,
        call_next,
    ):
        response_status = 500

        try:
            response = await call_next(request)
            response_status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = getattr(
                route,
                "path",
                "unmatched",
            )

            resolved_metrics.record_http_request(
                method=request.method,
                endpoint=endpoint,
                status_code=response_status,
            )

    @api.get(
        "/health",
        tags=["operations"],
    )
    def health() -> dict[str, Any]:
        """Liveness check that does not load models."""

        initialization_error = (
            resolved_provider.initialization_error
        )

        return {
            "status": "alive",
            "service": API_TITLE,
            "version": API_VERSION,
            "application_initialized": (
                resolved_provider.initialized
            ),
            "initialization_error_type": (
                type(initialization_error).__name__
                if initialization_error is not None
                else None
            ),
        }

    @api.get(
        "/ready",
        tags=["operations"],
    )
    def ready() -> dict[str, Any]:
        """Load dependencies and report readiness."""

        try:
            application_components = (
                resolved_provider.get()
            )
        except Exception as exc:
            resolved_metrics.set_application_ready(
                False
            )
            resolved_metrics.record_stage_failure(
                stage="api",
                error=exc,
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail={
                    "status": "not_ready",
                    "error_type": type(exc).__name__,
                },
            ) from exc

        resolved_metrics.set_application_ready(True)

        startup_summary = (
            application_components.startup_summary()
        )

        return {
            "status": "ready",
            "service": API_TITLE,
            "version": API_VERSION,
            "documentation_passage_count": (
                startup_summary.get(
                    "documentation_passage_count"
                )
            ),
            "index": startup_summary.get(
                "index"
            ),
        }

    @api.get(
        "/metrics",
        include_in_schema=False,
    )
    def prometheus_metrics() -> Response:
        """Expose Prometheus text-format metrics."""

        return Response(
            content=resolved_metrics.render(),
            headers={
                "Content-Type": (
                    resolved_metrics.content_type
                )
            },
        )

    @api.post(
        "/tickets/process",
        tags=["tickets"],
    )
    def process_ticket(
        payload: dict[str, Any],
        x_run_id: str | None = Header(
            default=None,
            alias="X-Run-ID",
        ),
    ) -> dict[str, Any]:
        """Process one ticket without delivering the response."""

        try:
            application_components = (
                resolved_provider.get()
            )
        except ApplicationBuildError as exc:
            resolved_metrics.set_application_ready(
                False
            )
            resolved_metrics.record_stage_failure(
                stage="api",
                error=exc,
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail={
                    "error": (
                        "Application dependencies are not ready."
                    ),
                    "error_type": type(exc).__name__,
                },
            ) from exc
        except Exception as exc:
            resolved_metrics.set_application_ready(
                False
            )
            resolved_metrics.record_stage_failure(
                stage="api",
                error=exc,
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail={
                    "error": (
                        "Application initialization failed."
                    ),
                    "error_type": type(exc).__name__,
                },
            ) from exc

        resolved_metrics.set_application_ready(True)

        try:
            result = (
                application_components
                .process_raw_ticket(
                    payload,
                    run_id=x_run_id,
                )
            )
        except TicketIngestionError as exc:
            resolved_metrics.record_stage_failure(
                stage="ingestion",
                error=exc,
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_422_UNPROCESSABLE_CONTENT
                ),
                detail={
                    "error": (
                        "Ticket validation failed."
                    ),
                    "ticket_id": exc.ticket_id,
                    "validation_error": exc.message,
                },
            ) from exc
        except Exception as exc:
            resolved_metrics.record_stage_failure(
                stage="api",
                error=exc,
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail={
                    "error": (
                        "Ticket processing failed."
                    ),
                    "error_type": type(exc).__name__,
                },
            ) from exc

        resolved_metrics.record_workflow_result(
            result=result,
            channel=str(
                payload.get("channel", "")
            ),
        )

        return result.as_dict()

    return api


app = create_api()


def main() -> None:
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()