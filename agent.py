
"""Synthetic life-insurance underwriting demo.
Never use for real decisions.
"""

import os
import logging

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ibm import ChatWatsonx
from langchain_core.messages import AIMessage

from opentelemetry import trace
from opentelemetry.trace import SpanKind

logger = logging.getLogger("insurance-risk-agent")
tracer = trace.get_tracer("insurance-risk-agent")

MODEL_ID = os.environ["WATSONX_MODEL_ID"]


# -----------------------------------------
# Synthetic risk calculation tool
# -----------------------------------------

@tool
def calculate_demo_risk(
    age: int,
    smoker: bool,
    insured_amount_eur: int
) -> dict:
    """Calculate a SYNTHETIC demo score for life insurance from age, smoking status and coverage. No real underwriting, no binding decisions."""

    with tracer.start_as_current_span(
        "tool.calculate_demo_risk"
    ) as span:

        if (
            not 18 <= age <= 85
            or not 10000 <= insured_amount_eur <= 1000000
        ):
            return {
                "error": (
                    "Valid demo ranges: 18-85 years; "
                    "€10,000-€1,000,000 insured."
                )
            }

        score = min(
            100,
            10
            + max(0, age - 30) * 1.2
            + (27 if smoker else 0)
            + (insured_amount_eur / 1000000) * 12
        )

        bucket = (
            "low" if score < 35
            else "medium" if score < 65
            else "high"
        )

        span.set_attribute(
            "demo.risk_bucket", bucket
        )

        span.set_attribute(
            "demo.score", round(score, 1)
        )

        return {
            "synthetic_score": round(score, 1),
            "synthetic_risk_tier": bucket,
            "status": "HUMAN_REVIEW_REQUIRED",
            "policy": (
                "Demo only. No insurance decision "
                "or real tariff."
            ),
        }


# -----------------------------------------
# System prompt
# -----------------------------------------

SYSTEM = """You are InsuranceRisk Demo, a Spanish-language LIFE INSURANCE pre-assessment assistant.
This is a fictional high-risk-AI governance exercise, NOT a real underwriting system.
If user gives age, smoking status and insurance amount, ALWAYS call calculate_demo_risk.
Never invent scores or premiums. Never issue accept/deny decisions, premiums or medical advice.
Ask for missing input, give the synthetic score and explain that a human underwriter must review.
Do not solicit names, identification numbers, medical details, or other real personal data.
Clearly say that the figures are illustrative and not valid for real insurance decisions."""


# -----------------------------------------
# LangGraph agent
# -----------------------------------------

def make_agent():

    llm = ChatWatsonx(
        model_id=MODEL_ID,
        url=os.environ.get(
            "WATSONX_URL",
            "https://eu-de.ml.cloud.ibm.com"
        ),
        apikey=os.environ["WATSONX_API_KEY"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
        params={
            "temperature": 0,
            "max_tokens": 700,
        },
    )

    return create_agent(
        model=llm,
        tools=[calculate_demo_risk],
        system_prompt=SYSTEM,
    )


_agent = None


# -----------------------------------------
# Real token usage instrumentation
# -----------------------------------------

def record_token_usage(messages, agent_span):
    """
    Extract real token counts reported by
    ChatWatsonx through LangChain AIMessage.

    Never estimate tokens from text length.
    Never invent missing usage values.

    Creates one usage-reporting span per AIMessage
    with valid token usage metadata.
    """

    total_input = 0
    total_output = 0
    total_tokens = 0
    reported_calls = 0
    missing_calls = 0

    for message in messages:

        if not isinstance(message, AIMessage):
            continue

        usage = getattr(
            message,
            "usage_metadata",
            None
        )

        if not isinstance(usage, dict):
            missing_calls += 1
            logger.warning(
                "LLM response without usage_metadata"
            )
            continue

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        if (
            type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens < 0
            or output_tokens < 0
        ):
            missing_calls += 1
            logger.warning(
                "LLM response missing valid token counts"
            )
            continue

        provider_total = usage.get("total_tokens")

        if (
            type(provider_total) is int
            and provider_total >= 0
        ):
            call_total = provider_total
        else:
            call_total = input_tokens + output_tokens

        reported_calls += 1

        total_input += input_tokens
        total_output += output_tokens
        total_tokens += call_total

        # This span reports completed model usage.
        # It is not intended to measure model latency.
        with tracer.start_as_current_span(
            f"chat {MODEL_ID}",
            kind=SpanKind.INTERNAL,
        ) as span:

            span.set_attribute(
                "gen_ai.operation.name", "chat"
            )

            span.set_attribute(
                "gen_ai.provider.name", "ibm.watsonx.ai"
            )

            span.set_attribute(
                "gen_ai.system", "watsonx.ai"
            )

            span.set_attribute(
                "gen_ai.request.model", MODEL_ID
            )

            span.set_attribute(
                "gen_ai.response.model", MODEL_ID
            )

            span.set_attribute(
                "gen_ai.usage.input_tokens",
                input_tokens
            )

            span.set_attribute(
                "gen_ai.usage.output_tokens",
                output_tokens
            )

            span.set_attribute(
                "gen_ai.usage.total_tokens",
                call_total
            )

            span.set_attribute(
                "insurance.telemetry.usage_report",
                True
            )

            span.set_attribute(
                "insurance.llm.call_index",
                reported_calls
            )

        logger.info(
            "LLM usage call=%s input_tokens=%s "
            "output_tokens=%s total_tokens=%s",
            reported_calls,
            input_tokens,
            output_tokens,
            call_total,
        )

    # Custom summary attributes on agent.langgraph,
    # avoiding duplicate standard GenAI usage counts.
    agent_span.set_attribute(
        "insurance.llm.reported_calls",
        reported_calls
    )

    agent_span.set_attribute(
        "insurance.llm.missing_usage_calls",
        missing_calls
    )

    if reported_calls:

        agent_span.set_attribute(
            "insurance.tokens.input_total",
            total_input
        )

        agent_span.set_attribute(
            "insurance.tokens.output_total",
            total_output
        )

        agent_span.set_attribute(
            "insurance.tokens.total",
            total_tokens
        )

    agent_span.set_attribute(
        "insurance.tokens.complete",
        missing_calls == 0 and reported_calls > 0
    )

    logger.info(
        "Agent token usage: input=%s output=%s "
        "total=%s reported_calls=%s missing_calls=%s",
        total_input,
        total_output,
        total_tokens,
        reported_calls,
        missing_calls,
    )


# -----------------------------------------
# Agent execution
# -----------------------------------------

def ask(question: str) -> str:

    global _agent

    if _agent is None:
        _agent = make_agent()

    with tracer.start_as_current_span(
        "agent.langgraph"
    ) as span:

        span.set_attribute(
            "gen_ai.agent.name",
            "insurance-risk-demo"
        )

        result = _agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        })

        messages = result["messages"]

        # Read actual provider-reported token counts.
        record_token_usage(messages, span)

        last = messages[-1].content

        return (
            last
            if isinstance(last, str)
            else str(last)
        )
