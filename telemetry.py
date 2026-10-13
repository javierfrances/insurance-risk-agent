
"""
OpenTelemetry integration for watsonx Orchestrate.

External LangGraph agent running on IBM Code Engine.
Exports OTLP traces using renewable IBM Cloud IAM tokens.
"""

import atexit
import json
import logging
import os
import threading
import time

from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurance-risk-telemetry")


# -----------------------------------------
# Environment configuration
# -----------------------------------------

WXO_AGENT_ID = os.getenv("WXO_AGENT_ID", "")
WXO_TENANT_ID = os.getenv("WXO_TENANT_ID", "")
WXO_AGENT_NAME = os.getenv(
    "WXO_AGENT_NAME",
    "insurance-risk-preassessment"
)

WXO_API_KEY = os.getenv("WXO_API_KEY", "")

TOKEN_URL = os.getenv(
    "TOKEN_URL",
    "https://iam.cloud.ibm.com/identity/token"
)

OTEL_EXPORT_URL = os.getenv("OTEL_EXPORT_URL", "")

ENVIRONMENT_NAME = os.getenv(
    "ENVIRONMENT_NAME",
    "draft"
)


# -----------------------------------------
# IBM Cloud IAM Authentication
# -----------------------------------------

def get_iam_token():
    """
    Exchange an IBM Cloud API Key for an IAM access token.

    Returns:
        token: IAM Bearer token
        expires_at: expiration timestamp
    """

    payload = urlencode({
        "grant_type": (
            "urn:ibm:params:oauth:grant-type:apikey"
        ),
        "apikey": WXO_API_KEY,
    }).encode("utf-8")

    request = Request(
        TOKEN_URL,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": (
                "application/x-www-form-urlencoded"
            ),
        },
    )

    with urlopen(request, timeout=30) as response:
        data = json.loads(
            response.read().decode("utf-8")
        )

    token = data.get("access_token") or data.get("token")

    if not token:
        raise RuntimeError(
            "IAM did not return an access token"
        )

    expires_at = float(
        data.get(
            "expiration",
            time.time() + float(
                data.get("expires_in", 3000)
            )
        )
    )

    logger.info(
        "IBM Cloud IAM token obtained successfully"
    )

    return token, expires_at


# -----------------------------------------
# Refreshing OTLP Exporter
# -----------------------------------------

class RefreshingOTLPSpanExporter(SpanExporter):
    """
    Automatically refreshes IAM tokens when required.
    """

    def __init__(self, endpoint, tenant_id, agent_id):
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
                    "OTLP exporter is shut down"
                )

            # Reuse the current token until 2 minutes
            # before its expiration.
            if (
                self.delegate is not None
                and time.time() < self.expires_at - 120
            ):
                return self.delegate

            token, expires_at = get_iam_token()

            exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-ibm-tenant-id": self.tenant_id,
                    "x-ibm-agent-id": self.agent_id,
                    "x-langfuse-ingestion-version": "4",
                },
                timeout=15,
            )

            old_exporter = self.delegate

            self.delegate = exporter
            self.expires_at = expires_at

            if old_exporter is not None:
                old_exporter.shutdown()

            logger.info(
                "watsonx Orchestrate OTLP exporter refreshed"
            )

            return self.delegate

    def export(self, spans):

        try:
            exporter = self._get_exporter()
            return exporter.export(spans)

        except Exception:
            logger.exception(
                "Failed to export traces to watsonx Orchestrate"
            )
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis=30000):
        with self.lock:
            exporter = self.delegate

        if exporter is None:
            return True

        return exporter.force_flush(timeout_millis)

    def shutdown(self):
        with self.lock:
            self.closed = True
            exporter = self.delegate
            self.delegate = None

        if exporter is not None:
            exporter.shutdown()


# -----------------------------------------
# Configure OpenTelemetry
# -----------------------------------------

def configure():

    resource = Resource.create({
        "service.name": "insurance-risk-demo",
        "service.version": "1.0.0",
        "agent.id": WXO_AGENT_ID,
        "agent.name": WXO_AGENT_NAME,
        "tenant.id": WXO_TENANT_ID,
        "environment.name": ENVIRONMENT_NAME,
        "deployment.environment.name": ENVIRONMENT_NAME,
    })

    provider = TracerProvider(resource=resource)

    # Optional local tracing for debugging.
    if os.getenv("OTEL_CONSOLE_ENABLED", "false").lower() == "true":
        provider.add_span_processor(
            BatchSpanProcessor(
                ConsoleSpanExporter()
            )
        )

    # Enable watsonx Orchestrate ingestion
    # only when all required settings exist.
    required_values = {
        "OTEL_EXPORT_URL": OTEL_EXPORT_URL,
        "WXO_API_KEY": WXO_API_KEY,
        "WXO_AGENT_ID": WXO_AGENT_ID,
        "WXO_TENANT_ID": WXO_TENANT_ID,
    }

    missing = [
        key for key, value in required_values.items()
        if not value
    ]

    if missing:
        logger.warning(
            "wxO telemetry disabled. Missing: %s",
            ", ".join(missing)
        )

    else:
        exporter = RefreshingOTLPSpanExporter(
            endpoint=OTEL_EXPORT_URL,
            tenant_id=WXO_TENANT_ID,
            agent_id=WXO_AGENT_ID,
        )

        provider.add_span_processor(
            BatchSpanProcessor(exporter)
        )

        logger.info(
            "watsonx Orchestrate telemetry enabled"
        )

        logger.info(
            "External Agent ID: %s",
            WXO_AGENT_ID
        )

    trace.set_tracer_provider(provider)

    atexit.register(provider.shutdown)

    return provider
