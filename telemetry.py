
"""
OpenTelemetry exporter for watsonx Orchestrate.
Includes IAM token renewal and HTTP error diagnostics.
"""

import atexit
import json
import logging
import os
import re
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurance-risk-telemetry")


# -----------------------------------------
# Configuration
# -----------------------------------------

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


# -----------------------------------------
# Safe HTTP diagnostics
# -----------------------------------------

class DiagnosticSession(requests.Session):
    """
    Log the HTTP status and a sanitized error response.
    Never log Authorization headers or request payloads.
    """

    def post(self, url, *args, **kwargs):
        response = super().post(url, *args, **kwargs)

        if response.status_code >= 400:
            body = response.text[:1500]

            # Mask known credentials if echoed in errors.
            for secret in (
                WXO_API_KEY,
                getattr(self, "iam_token", ""),
            ):
                if secret:
                    body = body.replace(secret, "[REDACTED]")

            body = re.sub(
                r"(?i)Bearer\s+[A-Za-z0-9._~+/-]+",
                "Bearer [REDACTED]",
                body,
            )

            body = re.sub(
                r'(?i)(["\'](?:access_token|'
                r'api_key|apikey|authorization)["\']'
                r'\s*:\s*["\'])[^"\']+',
                r'\1[REDACTED]',
                body,
            )

            logger.error(
                "WXO OTLP ingestion rejected request. "
                "HTTP status=%s; response=%s",
                response.status_code,
                body or "<empty response>",
            )

        elif response.status_code < 300:
            logger.info(
                "WXO OTLP HTTP accepted: status=%s",
                response.status_code,
            )

        return response


# -----------------------------------------
# IBM Cloud IAM
# -----------------------------------------

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
            ),
        )
    )

    logger.info("IAM token obtained successfully")

    return token, expires_at


# -----------------------------------------
# Auto-refreshing OTLP exporter
# -----------------------------------------

class RefreshingOTLPSpanExporter(SpanExporter):

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
                raise RuntimeError("Exporter shut down")

            if (
                self.delegate is not None
                and time.time() < self.expires_at - 120
            ):
                return self.delegate

            token, expires_at = get_iam_token()

            session = DiagnosticSession()
            session.iam_token = token

            new_exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-ibm-tenant-id": self.tenant_id,
                    "x-ibm-agent-id": self.agent_id,
                    "x-langfuse-ingestion-version": "4",
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
                "WXO OTLP exporter initialized/refreshed"
            )

            return self.delegate

    def export(self, spans):

        try:
            exporter = self._get_exporter()
            return exporter.export(spans)

        except Exception:
            logger.exception(
                "Failed to export traces to wxO"
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

        if exporter:
            exporter.shutdown()


# -----------------------------------------
# OpenTelemetry initialization
# -----------------------------------------

def configure():

    resource = Resource.create({
        "service.name": "insurance-risk-demo",
        "service.version": "1.0.0",
        "agent.id": WXO_AGENT_ID,
        "tenant.id": WXO_TENANT_ID,
        "environment.name": ENVIRONMENT_NAME,
        "deployment.environment": ENVIRONMENT_NAME,
    })

    provider = TracerProvider(resource=resource)

    # Optional console debugging.
    if os.getenv(
        "OTEL_CONSOLE_ENABLED", "false"
    ).lower() == "true":

        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

    required = {
        "WXO_AGENT_ID": WXO_AGENT_ID,
        "WXO_TENANT_ID": WXO_TENANT_ID,
        "WXO_API_KEY": WXO_API_KEY,
        "OTEL_EXPORT_URL": OTEL_EXPORT_URL,
    }

    missing = [
        key for key, value in required.items()
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
            BatchSpanProcessor(exporter)
        )

        logger.info("WXO telemetry exporter enabled")
        logger.info("WXO Agent ID: %s", WXO_AGENT_ID)

    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)

    return provider
