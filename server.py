import os
import uuid
import logging
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from opentelemetry import trace

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentInterface, AgentSkill
from a2a.helpers import new_text_message

from telemetry import configure
configure()
from agent import ask

logging.basicConfig(level=logging.INFO)
tracer = trace.get_tracer('insurance-risk-a2a-server')

class RiskExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        import asyncio
        question = context.get_user_input()
        with tracer.start_as_current_span('invoke_agent') as span:
            span.set_attribute('agent.name', 'insurance-risk-demo')
            # LangChain model invoke is synchronous; run in thread (context propagated below).
            import contextvars
            ctx = contextvars.copy_context()
            try:
                answer = await asyncio.to_thread(ctx.run, ask, question)
            except Exception:
                logging.exception('Agent invocation failed')
                answer = 'Error procesando la solicitud ficticia. Revisa configuración y logs.'
            await event_queue.enqueue_event(new_text_message(answer))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError('Cancellation not supported by this demo')

async def health(request):
    return JSONResponse({'status':'ok', 'app':'insurance-risk-demo'})

url = os.environ.get('PUBLIC_URL', 'http://localhost:8080').rstrip('/')
card = AgentCard(
    name='Insurance Risk Preassessment Demo',
    description='Fictional life insurance risk assessment for EU AI Act governance testing. Human review mandatory.',
    version='1.0.0',
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=['text'],
    default_output_modes=['text'],
    supported_interfaces=[AgentInterface(url=url+'/', protocol_binding='JSONRPC')],
    skills=[AgentSkill(id='insurance-risk-demo', name='Synthetic life-insurance risk score',
        description='Evaluates fictional life insurance risk data and requires human review.',
        tags=['insurance','risk','demo'],
        examples=['Tengo 43 años, no fumo y solicito un seguro de vida de 150000 euros.'])],
)
handler = DefaultRequestHandler(agent_executor=RiskExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
app = Starlette(routes=[Route('/health', health), *create_agent_card_routes(card), *create_jsonrpc_routes(handler, '/')])

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8080')))
