"""Optional OTLP tracing; WXO URL and identity configured later after registration."""
import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

log = logging.getLogger(__name__)

def configure():
    provider = TracerProvider(resource=Resource.create({
        'service.name': 'insurance-risk-demo',
        'deployment.environment.name': os.getenv('ENVIRONMENT_NAME', 'draft'),
    }))
    # Console fallback: see spans immediately in Code Engine logs.
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    endpoint = os.getenv('OTEL_EXPORT_URL')
    if endpoint:
        headers = {}
        if os.getenv('OTEL_AUTH_TOKEN'):
            headers['Authorization'] = 'Bearer ' + os.environ['OTEL_AUTH_TOKEN']
        if os.getenv('WXO_TENANT_ID'):
            headers['x-ibm-tenant-id'] = os.environ['WXO_TENANT_ID']
        if os.getenv('WXO_AGENT_ID'):
            headers['x-ibm-agent-id'] = os.environ['WXO_AGENT_ID']
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=10)))
        log.info('OTLP exporter enabled')
    else:
        log.info('OTEL_EXPORT_URL absent: traces exported to logs only')
    trace.set_tracer_provider(provider)
