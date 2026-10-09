"""Synthetic life-insurance underwriting demo. Never use for real decisions."""
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ibm import ChatWatsonx
from opentelemetry import trace

tracer = trace.get_tracer('insurance-risk-agent')

@tool
def calculate_demo_risk(age: int, smoker: bool, insured_amount_eur: int) -> dict:
    """Calculate a SYNTHETIC demo score for life insurance from age, smoking status and coverage. No real underwriting, no binding decisions."""
    with tracer.start_as_current_span('tool.calculate_demo_risk') as span:
        if not 18 <= age <= 85 or not 10000 <= insured_amount_eur <= 1000000:
            return {'error': 'Valid demo ranges: 18-85 years; €10,000-€1,000,000 insured.'}
        score = min(100, 10 + max(0, age - 30) * 1.2 + (27 if smoker else 0) + (insured_amount_eur / 1000000) * 12)
        bucket = 'low' if score < 35 else 'medium' if score < 65 else 'high'
        span.set_attribute('demo.risk_bucket', bucket)
        span.set_attribute('demo.score', round(score, 1))
        return {'synthetic_score': round(score, 1), 'synthetic_risk_tier': bucket,
                'status': 'HUMAN_REVIEW_REQUIRED', 'policy': 'Demo only. No insurance decision or real tariff.'}

SYSTEM = '''You are InsuranceRisk Demo, a Spanish-language LIFE INSURANCE pre-assessment assistant.
This is a fictional high-risk-AI governance exercise, NOT a real underwriting system.
If user gives age, smoking status and insurance amount, ALWAYS call calculate_demo_risk.
Never invent scores or premiums. Never issue accept/deny decisions, premiums or medical advice.
Ask for missing input, give the synthetic score and explain that a human underwriter must review.
Do not solicit names, identification numbers, medical details, or other real personal data.
Clearly say that the figures are illustrative and not valid for real insurance decisions.'''

def make_agent():
    llm = ChatWatsonx(
        model_id=os.environ['WATSONX_MODEL_ID'],
        url=os.environ.get('WATSONX_URL', 'https://eu-de.ml.cloud.ibm.com'),
        apikey=os.environ['WATSONX_API_KEY'],
        project_id=os.environ['WATSONX_PROJECT_ID'],
        params={'temperature': 0, 'max_tokens': 700},
    )
    return create_agent(model=llm, tools=[calculate_demo_risk], system_prompt=SYSTEM)

_agent = None

def ask(question: str) -> str:
    global _agent
    if _agent is None:
        _agent = make_agent()
    with tracer.start_as_current_span('agent.langgraph') as span:
        span.set_attribute('gen_ai.agent.name', 'insurance-risk-demo')
        result = _agent.invoke({'messages': [{'role':'user','content':question}]})
        last = result['messages'][-1].content
        return last if isinstance(last, str) else str(last)
