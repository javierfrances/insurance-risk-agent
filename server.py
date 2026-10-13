
import os
import json
import uuid
import secrets
import logging
import asyncio
import contextvars
import uvicorn

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, Status, StatusCode

from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
)
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentInterface,
    AgentSkill,
)
from a2a.helpers import new_text_message

from telemetry import configure

# Initialize telemetry before importing the agent
configure()

from agent import ask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurance-risk-server")

tracer = trace.get_tracer("insurance-risk-a2a-server")

A2A_API_KEY = os.environ["A2A_API_KEY"]
API_KEY_HEADER = "X-Agent-API-Key"

WXO_AGENT_ID = os.environ["WXO_AGENT_ID"]
WXO_AGENT_NAME = os.getenv(
    "WXO_AGENT_NAME",
    "insurance-risk-preassessment",
)
ENVIRONMENT_NAME = os.getenv(
    "ENVIRONMENT_NAME", "draft"
)


# -----------------------------------------
# JSON helpers
# -----------------------------------------

def decode_json(data):
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


def extract_text(parts):
    if not isinstance(parts, list):
        return ""

    values = []

    for part in parts:
        if not isinstance(part, dict):
            continue

        value = part.get("text")

        if value is None:
            value = part.get("root", {}).get("text") \
                if isinstance(part.get("root"), dict) \
                else None

        if isinstance(value, str):
            values.append(value)

    return "\n".join(values)


def response_text(payload):
    """
    Supports A2A Message and Task responses.
    """

    if not isinstance(payload, dict):
        return ""

    result = payload.get("result", payload)

    if not isinstance(result, dict):
        return ""

    text = extract_text(result.get("parts"))

    if text:
        return text

    status = result.get("status", {})
    if isinstance(status, dict):
        message = status.get("message", {})
        if isinstance(message, dict):
            text = extract_text(message.get("parts"))
            if text:
                return text

    artifacts = result.get("artifacts", [])

    if isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, dict):
                text = extract_text(
                    artifact.get("parts")
                )
                if text:
                    return text

    return ""


def request_metadata(payload):
    params = payload.get("params", {})
    if not isinstance(params, dict):
        params = {}

    message = params.get("message", {})
    if not isinstance(message, dict):
        message = {}

    message_id = (
        message.get("messageId")
        or message.get("message_id")
        or str(uuid.uuid4())
    )

    thread_id = (
        message.get("contextId")
        or message.get("context_id")
        or params.get("contextId")
        or str(uuid.uuid4())
    )

    user_text = extract_text(
        message.get("parts", [])
    )

    return (
        str(message_id),
        str(thread_id),
        user_text,
    )


# -----------------------------------------
# LangGraph Agent Executor
# -----------------------------------------

class RiskExecutor(AgentExecutor):

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:

        question = context.get_user_input()

        # The root span is now created by the
        # outer ASGI middleware. Do not create
        # another invoke_agent span here.
        ctx = contextvars.copy_context()

        try:
            answer = await asyncio.to_thread(
                ctx.run,
                ask,
                question,
            )

        except Exception:
            logger.exception(
                "Agent invocation failed"
            )

            # Propagate to the A2A handler so
            # the request is not falsely marked
            # successful.
            raise

        await event_queue.enqueue_event(
            new_text_message(answer)
        )

        logger.info(
            "Risk agent completed A2A request"
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:

        raise NotImplementedError(
            "Cancellation not supported"
        )


# -----------------------------------------
# Root conversation tracing middleware
# -----------------------------------------

class A2ATraceMiddleware:
    """
    Creates a root invoke_agent span around
    the entire A2A HTTP request.

    Replays the request body to the A2A SDK
    and captures the final JSON-RPC response.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(
        self,
        scope,
        receive,
        send,
    ):

        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/"
        ):
            await self.app(
                scope, receive, send
            )
            return

        # Read complete ASGI request body.
        body_chunks = []

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            if message["type"] == "http.request":
                body_chunks.append(
                    message.get("body", b"")
                )

                if not message.get(
                    "more_body", False
                ):
                    break

        request_body = b"".join(body_chunks)
        request_json = decode_json(request_body)

        message_id, thread_id, user_text = (
            request_metadata(request_json)
        )

        # Replay the body to Starlette/A2A.
        body_replayed = False

        async def replay_receive():
            nonlocal body_replayed

            if not body_replayed:
                body_replayed = True
                return {
                    "type": "http.request",
                    "body": request_body,
                    "more_body": False,
                }

            return await receive()

        response_chunks = []
        http_status = 500

        async def capture_send(message):
            nonlocal http_status

            if message["type"] == "http.response.start":
                http_status = message.get(
                    "status", 500
                )

            elif message["type"] == "http.response.body":
                response_chunks.append(
                    message.get("body", b"")
                )

            await send(message)

        span_name = (
            f"invoke_agent {WXO_AGENT_NAME}"
        )

        # Start a new trace instead of making
        # invoke_agent a child of an A2A span.
        with tracer.start_as_current_span(
            span_name,
            context=Context(),
            kind=SpanKind.SERVER,
        ) as span:

            span.set_attribute(
                "agent.id", WXO_AGENT_ID
            )
            span.set_attribute(
                "environment.name",
                ENVIRONMENT_NAME,
            )
            span.set_attribute(
                "message.id", message_id
            )
            span.set_attribute(
                "thread.id", thread_id
            )
            span.set_attribute(
                "conversation.id", thread_id
            )
            span.set_attribute(
                "langfuse.session.id", thread_id
            )
            span.set_attribute(
                "langfuse.user.id",
                "insurance-risk-demo-user",
            )
            span.set_attribute(
                "input", user_text
            )

            try:
                await self.app(
                    scope,
                    replay_receive,
                    capture_send,
                )

            except Exception as exc:
                span.record_exception(exc)
                span.set_attribute(
                    "agent.execution.status",
                    "error",
                )
                span.set_attribute(
                    "agent.business_outcome",
                    "failed",
                )
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        str(exc)[:1000],
                    )
                )
                raise

            finally:
                response_json = decode_json(
                    b"".join(response_chunks)
                )

                span.set_attribute(
                    "http.response.status_code",
                    http_status,
                )

                output = response_text(
                    response_json
                )

                if output:
                    span.set_attribute(
                        "output", output
                    )

                rpc_error = response_json.get(
                    "error"
                ) if isinstance(
                    response_json, dict
                ) else None

                if (
                    http_status >= 400
                    or rpc_error is not None
                ):
                    span.set_attribute(
                        "agent.execution.status",
                        "error",
                    )
                    span.set_attribute(
                        "agent.business_outcome",
                        "failed",
                    )
                    span.set_status(
                        Status(StatusCode.ERROR)
                    )

                elif span.is_recording():
                    span.set_attribute(
                        "agent.execution.status",
                        "success",
                    )
                    span.set_attribute(
                        "agent.business_outcome",
                        "success",
                    )
                    span.set_status(
                        Status(StatusCode.OK)
                    )

                logger.info(
                    "A2A root trace completed: "
                    "message_id=%s thread_id=%s "
                    "http_status=%s",
                    message_id,
                    thread_id,
                    http_status,
                )


# -----------------------------------------
# Health endpoint
# -----------------------------------------

async def health(request):
    return JSONResponse({
        "status": "ok",
        "app": "insurance-risk-demo",
    })


# -----------------------------------------
# Public URL
# -----------------------------------------

url = os.getenv(
    "PUBLIC_URL",
    "http://localhost:8080",
).rstrip("/")


# -----------------------------------------
# A2A Agent Card
# -----------------------------------------

card = AgentCard(
    name="Insurance Risk Preassessment Demo",

    description=(
        "Fictional life insurance risk assessment "
        "for EU AI Act governance testing. "
        "Human review mandatory."
    ),

    version="1.0.0",

    capabilities=AgentCapabilities(
        streaming=False
    ),

    default_input_modes=["text"],
    default_output_modes=["text"],

    supported_interfaces=[
        AgentInterface(
            url=url + "/",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        ),
        AgentInterface(
            url=url + "/",
            protocol_binding="JSONRPC",
            protocol_version="0.3",
        ),
    ],

    skills=[
        AgentSkill(
            id="insurance-risk-demo",
            name="Synthetic life-insurance risk score",
            description=(
                "Evaluates fictional life insurance "
                "risk data and requires human review."
            ),
            tags=[
                "insurance",
                "risk",
                "demo",
            ],
            examples=[
                "Tengo 43 años, no fumo y solicito "
                "un seguro de vida de 150000 euros."
            ],
        )
    ],
)


# -----------------------------------------
# A2A Handler
# -----------------------------------------

handler = DefaultRequestHandler(
    agent_executor=RiskExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=card,
)


# -----------------------------------------
# Starlette application
# -----------------------------------------

starlette_app = Starlette(
    routes=[
        Route("/health", health),

        *create_agent_card_routes(card),

        *create_jsonrpc_routes(
            handler,
            "/",
            enable_v0_3_compat=True,
        ),
    ]
)


# -----------------------------------------
# A2A API Key authentication
# -----------------------------------------

@starlette_app.middleware("http")
async def validate_a2a_api_key(
    request: Request,
    call_next,
):

    if request.method == "POST":

        received_key = request.headers.get(
            API_KEY_HEADER, ""
        )

        if (
            not received_key
            or not secrets.compare_digest(
                received_key,
                A2A_API_KEY,
            )
        ):
            logger.warning(
                "Unauthorized A2A request"
            )

            return JSONResponse(
                {
                    "error": "Unauthorized",
                    "message":
                        "Invalid or missing API Key",
                },
                status_code=401,
            )

    return await call_next(request)


# -----------------------------------------
# Wrap the complete A2A application
# -----------------------------------------

app = A2ATraceMiddleware(
    starlette_app
)


# -----------------------------------------
# Start server
# -----------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8080")
        ),
    )
