# Insurance Risk Demo — LangGraph + A2A + OpenTelemetry

**PoC ONLY. Fictional input. No decisions, no real insurance tariff, no production use.**

## Files
- `agent.py`: LangChain agent (internally LangGraph), watsonx.ai LLM and deterministic demo-risk tool.
- `server.py`: A2A 1.0 JSON-RPC server and Agent Card.
- `telemetry.py`: OTEL spans to Code Engine logs, and optional OTLP endpoint.
- `Dockerfile`: Code Engine deployment.

## Before deployment
Choose a watsonx.ai deployed model that supports **tool calling**. Replace placeholders.
Make an IBM Code Engine project, install IBM Cloud CLI + Code Engine plugin, log in, then:

```sh
ibmcloud target -r eu-de
ibmcloud ce project select --name YOUR_PROJECT
ibmcloud ce application create --name insurance-risk-demo --build-source . --port 8080 \
  --env WATSONX_URL=https://eu-de.ml.cloud.ibm.com \
  --env WATSONX_PROJECT_ID=YOUR_PROJECT_ID \
  --env WATSONX_MODEL_ID=YOUR_TOOL_CAPABLE_MODEL_ID \
  --env WATSONX_API_KEY=YOUR_API_KEY
ibmcloud ce application get --name insurance-risk-demo
```

**Security:** CLI `--env` can expose secrets in shell history. For real deployment create a Code Engine Secret and bind via `--env-from-secret`, not `--env`. Do not put API keys in source control. For this PoC prefer creating secrets in the console and deploying using `--env-from-secret`.

After retrieving the public URL, set `PUBLIC_URL` to its exact value in the application config and redeploy/restart. This ensures the Agent Card advertises the correct address.

```sh
curl https://YOUR_APP/health
curl https://YOUR_APP/.well-known/agent-card.json
```

Invoke from PowerShell/Postman/curl:

```sh
curl -X POST https://YOUR_APP/ -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{"kind":"message","messageId":"test-1","role":"user","parts":[{"kind":"text","text":"Tengo 43 años, no fumo y solicito 150000 euros de seguro de vida"}]}}}'
```

## OTEL
Traces are produced locally in the container logs from day one (`invoke_agent`, `agent.langgraph`, `tool.calculate_demo_risk`). **Automatic nested LLM spans are not guaranteed** by this skeleton; they require model instrumentation. Later configure `WXO_AGENT_ID`, `WXO_TENANT_ID`, `OTEL_EXPORT_URL`, and dynamic IBM IAM bearer-token management according to your Orchestrate tenant documentation. Static `OTEL_AUTH_TOKEN` is ONLY for a brief validation; it expires. Configuring the endpoint is not proof of compatibility with Orchestrate's span schema.

**Important:** Public A2A endpoint in this demo is unauthenticated. Do not expose real data or leave it publicly deployed unattended. For a customer or production environment, add A2A authentication, rate-limiting, secret management, hardened storage, and auditability.

The demo score is invented; Annex III 5(c) of EU AI Act covers AI for life/health insurance risk assessment and pricing concerning natural persons, with applicable legal qualifications. The PoC deliberately requires human review.
