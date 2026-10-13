
"""
OpenTelemetry integration for IBM watsonx Orchestrate.

Features:
- IBM Cloud IAM authentication
- Automatic bearer token renewal
- OTLP/HTTP trace export
- Safe HTTP status diagnostics
- Export success/failure logging
"""

import atexit
import json
import logging
import os
import threading
import time

from urllib.parse import urlencode
from urllib.request import Request, urlopen

import requests

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("insurance-risk-telemetry")


# --------------------------------------------------
# Environment configuration
# --------------------------------------------------

WXO_AGENT_ID = os.getenv("WXO_AGENT_ID", "")
WXO_TENANT_ID = os.getenv("WXO_TENANT_ID", "")
WXO_API_KEY = os.getenv("WXO_API_KEY", "")

OTEL_EXPORT_URL = os.getenv("OTEL_EXPORT_URL", "")

TOKEN_URL = os.getenv(
    "TOKEN_URL",
    "https://iam.cloud.ibm.com/identity/token",
)

ENVIRONMENT_NAME = os.getenv(
    "ENVIRONMENT_NAME", "draft"
)


# --------------------------------------------------
# Diagnostic HTTP session
# --------------------------------------------------

class DiagnosticSession(requests.Session):
    """
    Intercepts requests at the transport layer.

    Logs:
    - HTTP status code
    - Response body length
    - OTLP endpoint status

    Does not log:
    - Authorization headers
    - API keys
    - Bearer tokens
    - Request payloads
    """

    def send(self, request, **kwargs):

        logger.info(
            "WXO OTLP HTTP request starting"
        )

        try:
            response = super().send(
                request, **kwargs
            )

            logger.info(
                "WXO OTLP HTTP response: status=%s",
                response.status_code,
            )

            if response.status_code >= 400:
                logger.error(
                    "WXO OTLP HTTP error: "
                    "status=%s, body_length=%s",
                    response.status_code,
                    len(response.content),
                )

            elif 200 <= response.status_code < 300:
                logger.info(
                    "WXO OTLP ingestion HTTP accepted"
                )

            return response

        except requests.RequestException as exc:

            logger.error(
                "WXO OTLP transport failure: %s",
                type(exc).__name__,
            )

            raise


# --------------------------------------------------
# IBM Cloud IAM token exchange
# --------------------------------------------------

def get_iam_token():

    payload = urlencode({
        "grant_type":
            "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WXO_API_KEY,
    }).encode("utf-8")

    request = Request(
        TOKEN_URL,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type":
                "application/x-www-form-urlencoded",
        },
    )

    with urlopen(
        request, timeout=30
    ) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

    token = (
        data.get("access_token")
        or data.get("token")
    )

    if not token:
        raise RuntimeError(
            "IBM IAM did not return an access token"
        )

    expires_at = float(
        data.get(
            "expiration",
            time.time() + float(
                data.get("expires_in", 3000)
            ),
        )
    )

    logger.info(
        "IAM token obtained successfully"
    )

    return token, expires_at


# --------------------------------------------------
# Refreshing OTLP exporter
# --------------------------------------------------

class RefreshingOTLPSpanExporter(SpanExporter):

    def __init__(
        self,
        endpoint,
        tenant_id,
        agent_id,
    ):

        self.endpoint = endpoint
        self.tenant_id = tenant_id
        self.agent_id = agent_id

        self.lock = threading.Lock()

        self.delegate = None
        self.expires_at = 0
        self.closed = False

    def _get_exporter(self):

        with self.lock:

            if self.closed:
                raise RuntimeError(
                    "Exporter already shut down"
                )

            if (
                self.delegate is not None
                and time.time()
                < self.expires_at - 120
            ):
                return self.delegate

            token, expires_at = get_iam_token()

            session = DiagnosticSession()

            new_exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers={
                    "Authorization":
                        f"Bearer {token}",
                    "x-ibm-tenant-id":
                        self.tenant_id,
                    "x-ibm-agent-id":
                        self.agent_id,
                    "x-langfuse-ingestion-version":
                        "4",
                },
                session=session,
                timeout=15,
            )

            old_exporter = self.delegate

            self.delegate = new_exporter
            self.expires_at = expires_at

            if old_exporter:
                old_exporter.shutdown()

            logger.info(
                "WXO OTLP exporter "
                "initialized/refreshed"
            )

            return self.delegate

    def export(self, spans):

        logger.info(
            "WXO OTLP export started: spans=%s",
            len(spans),
        )

        try:

            exporter = self._get_exporter()

            result = exporter.export(spans)

            if result == SpanExportResult.SUCCESS:
                logger.info(
                    "WXO OTLP export result: SUCCESS"
                )

            else:
                logger.error(
                    "WXO OTLP export result: FAILURE"
                )

            return result

        except Exception:

            logger.exception(
                "Failed to export traces to wxO"
            )

            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis=30000):

        # OTLP/HTTP exports synchronously when
        # its export() method is invoked.
        # BatchSpanProcessor handles its own queue.
        return True

    def shutdown(self):

        with self.lock:

            self.closed = True

            exporter = self.delegate
            self.delegate = None

        if exporter:
            exporter.shutdown()


# --------------------------------------------------
# Configure OpenTelemetry
# --------------------------------------------------

def configure():

    resource = Resource.create({
        "service.name": "insurance-risk-demo",
        "service.version": "1.0.0",
        "agent.id": WXO_AGENT_ID,
        "tenant.id": WXO_TENANT_ID,
        "environment.name": ENVIRONMENT_NAME,
        "deployment.environment.name":
            ENVIRONMENT_NAME,
    })

    provider = TracerProvider(
        resource=resource
    )

    # Optional local debugging
    if os.getenv(
        "OTEL_CONSOLE_ENABLED", "false"
    ).lower() == "true":

        provider.add_span_processor(
            BatchSpanProcessor(
                ConsoleSpanExporter()
            )
        )

    required = {
        "WXO_AGENT_ID": WXO_AGENT_ID,
        "WXO_TENANT_ID": WXO_TENANT_ID,
        "WXO_API_KEY": WXO_API_KEY,
        "OTEL_EXPORT_URL": OTEL_EXPORT_URL,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:

        logger.warning(
            "WXO telemetry disabled. Missing: %s",
            ", ".join(missing),
        )

    else:

        exporter = RefreshingOTLPSpanExporter(
            endpoint=OTEL_EXPORT_URL,
            tenant_id=WXO_TENANT_ID,
            agent_id=WXO_AGENT_ID,
        )

        provider.add_span_processor(
            BatchSpanProcessor(
                exporter
            )
        )

        logger.info(
            "WXO telemetry exporter enabled"
        )

        logger.info(
            "WXO Agent ID: %s",
            WXO_AGENT_ID,
        )

    trace.set_tracer_provider(provider)

    atexit.register(
        provider.shutdown
    )

    return provider
