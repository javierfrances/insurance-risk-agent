
import os
import secrets
import logging
import asyncio
import contextvars
import uuid
import uvicorn

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from a2a.server.agent_execution import AgentExecutor, RequestContext
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

configure()

from agent import ask

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("insurance-risk-server")

tracer = trace.get_tracer("insurance-risk-a2a-server")

A2A_API_KEY = os.environ["A2A_API_KEY"]
API_KEY_HEADER = "X-Agent-API-Key"

WXO_AGENT_ID = os.getenv("WXO_AGENT_ID", "")
WXO_AGENT_NAME = os.getenv(
    "WXO_AGENT_NAME", "insurance-risk-preassessment"
)
ENVIRONMENT_NAME = os.getenv("ENVIRONMENT_NAME", "draft")


# -----------------------------------------
# Conversation metadata
# -----------------------------------------

def get_conversation_metadata(context):
    """
    Extract A2A message and conversation IDs.

    Reuse contextId when provided, otherwise generate
    a unique ID for this invocation.
    """

    message = getattr(context, "message", None)

    message_id = (
        getattr(message, "message_id", None)
        or str(uuid.uuid4())
    )

    thread_id = (
        getattr(message, "context_id", None)
        or getattr(context, "context_id", None)
        or str(uuid.uuid4())
    )

    return str(message_id), str(thread_id)


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

        message_id, thread_id = get_conversation_metadata(
            context
        )

        span_name = f"invoke_agent {WXO_AGENT_NAME}"

        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.SERVER
        ) as span:

            # Agent identification
            span.set_attribute(
                "agent.id", WXO_AGENT_ID
            )

            span.set_attribute(
                "environment.name", ENVIRONMENT_NAME
            )

            # Conversation correlation
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
                "langfuse.user.id", "insurance-risk-demo-user"
            )

            # Input
            span.set_attribute(
                "input", question
            )

            ctx = contextvars.copy_context()

            try:

                answer = await asyncio.to_thread(
                    ctx.run,
                    ask,
                    question
                )

                span.set_attribute(
                    "output", str(answer)
                )

                span.set_attribute(
                    "agent.execution.status", "success"
                )

                span.set_attribute(
                    "agent.business_outcome", "success"
                )

                span.set_status(
                    Status(StatusCode.OK)
                )

            except Exception as exc:

                logger.exception(
                    "Agent invocation failed"
                )

                span.record_exception(exc)

                span.set_attribute(
                    "agent.execution.status", "error"
                )

                span.set_attribute(
                    "agent.business_outcome", "failed"
                )

                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )

                answer = (
                    "Error procesando la solicitud ficticia. "
                    "Revisa configuración y logs."
                )

                span.set_attribute(
                    "output", answer
                )

            await event_queue.enqueue_event(
                new_text_message(answer)
            )

            logger.info(
                "A2A conversation completed: "
                "thread_id=%s, message_id=%s",
                thread_id,
                message_id
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:

        raise NotImplementedError(
            "Cancellation not supported by this demo"
        )


# -----------------------------------------
# Health endpoint
# -----------------------------------------

async def health(request):

    return JSONResponse({
        "status": "ok",
        "app": "insurance-risk-demo"
    })


# -----------------------------------------
# Public URL
# -----------------------------------------

url = os.environ.get(
    "PUBLIC_URL",
    "http://localhost:8080"
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
            protocol_version="1.0"
        ),

        AgentInterface(
            url=url + "/",
            protocol_binding="JSONRPC",
            protocol_version="0.3"
        )

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
                "demo"
            ],

            examples=[
                "Tengo 43 años, no fumo y solicito "
                "un seguro de vida de 150000 euros."
            ]

        )

    ]
)


# -----------------------------------------
# A2A Handler
# -----------------------------------------

handler = DefaultRequestHandler(
    agent_executor=RiskExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=card
)


# -----------------------------------------
# Starlette application
# -----------------------------------------

app = Starlette(

    routes=[

        Route(
            "/health",
            health
        ),

        *create_agent_card_routes(card),

        *create_jsonrpc_routes(
            handler,
            "/",
            enable_v0_3_compat=True
        )

    ]
)


# -----------------------------------------
# API Key Authentication Middleware
# -----------------------------------------

@app.middleware("http")
async def validate_a2a_api_key(
    request: Request,
    call_next
):

    if request.method == "POST":

        received_key = request.headers.get(
            API_KEY_HEADER,
            ""
        )

        if (
            not received_key
            or not secrets.compare_digest(
                received_key,
                A2A_API_KEY
            )
        ):

            logger.warning(
                "Unauthorized A2A request"
            )

            return JSONResponse(
                {
                    "error": "Unauthorized",
                    "message": "Invalid or missing API Key"
                },
                status_code=401
            )

    return await call_next(request)


# -----------------------------------------
# Start Server
# -----------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080"))
    )
