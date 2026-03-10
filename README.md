# ☁️ CloudSage
### Autonomous Cloud Reliability, Security & FinOps Intelligence Platform
> *From Alert Fatigue to Closed-Loop Autonomous Remediation*

---

## What It Does

CloudSage is a production-grade, multi-agent AI platform built entirely on Azure that autonomously monitors, diagnoses, and remediates cloud infrastructure incidents — closing the loop from telemetry to automated action in under 10 seconds.

```
Azure Monitor / App Insights
        ↓
Azure Event Hubs  →  Multi-Agent AI Core (Azure AI Foundry)  →  Decision Engine  →  AKS Automation
Azure Service Bus        ↓                                              ↓
                  3-Model Consensus                              Azure Cosmos DB
                  (P1 incidents only)                            Azure AI Search (RAG)
```

---

## Architecture

### Azure Services

| Layer | Azure Service | Purpose |
|-------|--------------|---------|
| **Observability** | Azure Monitor · Application Insights · Log Analytics | Collect telemetry, metrics, and traces |
| **Ingestion** | Azure Event Hubs | Real-time high-throughput telemetry stream |
| **Messaging** | Azure Service Bus | Reliable incident routing between services |
| **AI / LLM** | Azure AI Foundry — GPT-4o | Agent reasoning, consensus, RCA, runbook generation |
| **Embeddings** | Azure OpenAI — text-embedding-3-small (1536d) | RAG knowledge base vectorisation |
| **Vector Search** | Azure AI Search (hybrid vector + keyword) | Historical runbook and postmortem retrieval |
| **Storage** | Azure Cosmos DB (NoSQL) | Incident persistence, audit trail, feedback loop |
| **Blob Storage** | Azure Blob Storage | AI Search index backing, vector index snapshots |
| **Compute** | Azure Kubernetes Service (AKS) | Automated restart, scale-out, rolling rollback |
| **Automation** | Azure Functions (Service Bus triggered) | Serverless cloud action dispatch |
| **Security** | Microsoft Defender for Cloud | Threat detection, policy violations, compliance |
| **Chaos** | Azure Chaos Studio | Healing validation experiments |
| **Dashboard** | Azure Static Web Apps + Next.js 14 | Real-time executive and SRE dashboard |
| **Notifications** | Microsoft Teams Webhooks | SRE alert and approval-request routing |
| **IaC** | Azure Bicep | Repeatable, auditable environment provisioning |

### Agent Pipeline (per incident, ~7–12 seconds end-to-end)

```
Step 0:  Adaptive Threshold Evaluation  → anomaly detection (Isolation Forest)
Step 1:  IncidentResponseAgent          → severity classification + recommended action
         └─ P1 only: 3-model consensus  → 3× independent GPT-4o votes required
Step 2:  Causal Inference Engine        → Granger causality chain across services
Step 3:  Blast Radius Predictor         → downstream services at risk + impact score
Step 4:  SLO Error Budget Check         → tier: standard / conservative / restricted / freeze
Step 5:  RCAAgent + PredictiveAgent     → parallel: root cause + failure probability
Step 6:  PolicyEngine                   → APPROVED / APPROVAL_REQUIRED / DENIED
Step 7:  ActionMapper → Azure Functions → restart / scale_up / rollback / alert_only
Step 8:  Economic Impact Model          → $/min revenue at risk + automation savings
Step 9:  Cosmos DB Persist              → full enriched incident document
Step 10: Feedback Loop                  → score outcome, update reliability index
Step 11: Runbook Generator              → auto-generate + index into Azure AI Search
```

### SLO Policy Tiers (Google SRE burn-rate model)

| Tier | Error Budget | Behaviour |
|------|-------------|-----------|
| `standard` | > 50% remaining | Full automation active |
| `conservative` | 20–50% remaining | Automation with tighter monitoring |
| `restricted` | < 20% remaining · burn rate > 14× | **Human approval required for all actions** |
| `freeze` | Budget exhausted | All automation halted · deployments blocked |

### 5 Specialised AI Agents

| Agent | Trigger | Output |
|-------|---------|--------|
| 🚨 **IncidentResponseAgent** | Metric anomaly / Event Hub message | Severity P1–P4, recommended action, 3-model consensus on P1 |
| 🔎 **RCAAgent** | Incident confirmed | Root cause + remediation steps (RAG-augmented via AI Search) |
| 🔮 **PredictiveFailureAgent** | Time-series data | Failure probability + preemptive action recommendations |
| 💰 **FinOpsAgent** | Cost spike / budget alert | Savings recommendations + cost efficiency index |
| 🔐 **SecurityAgent** | Defender for Cloud alert | Threat report + compliance summary + risk score |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Azure subscription with the following provisioned:
  - Azure AI Foundry (GPT-4o deployment + text-embedding-3-small)
  - Azure AI Search (Standard tier, semantic ranking enabled)
  - Azure Cosmos DB (NoSQL, serverless or provisioned)
  - Azure Event Hubs (Standard tier, 1 hub)
  - Azure Kubernetes Service (for live automation)
  - Microsoft Defender for Cloud (P2 for SecurityAgent)

### Configuration

Edit `config/config.json` with your Azure credentials:

```json
{
  "llm_provider": "openai",
  "openai": {
    "api_key": "YOUR_AZURE_OPENAI_KEY",
    "base_url": "https://YOUR_FOUNDRY_ENDPOINT.openai.azure.com/",
    "primary_model": "gpt-4o",
    "embedding_model": "text-embedding-3-small"
  },
  "cosmos_db": {
    "endpoint": "https://YOUR_COSMOS.documents.azure.com:443/",
    "key": "YOUR_COSMOS_KEY"
  },
  "azure": {
    "subscription_id": "YOUR_SUB_ID",
    "resource_group": "cloudsage-rg",
    "location": "eastus"
  }
}
```

Or export environment variables — `config/loader.py` resolves them automatically:

```bash
export OPENAI_API_KEY="..."
export COSMOS_DB_ENDPOINT="https://..."
export COSMOS_DB_KEY="..."
export AZURE_SUBSCRIPTION_ID="..."
```

### Provision Infrastructure

```bash
az deployment group create \
  --resource-group cloudsage-rg \
  --template-file infra/main.bicep
```

### Run Backend

```bash
# Start Event Hub consumer (production) or HTTP ingestion server (local dev)
python main.py consumer
# → Connects to Event Hubs in production; falls back to HTTP on port 8080 locally
```

### Run Dashboard

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:3000
```

### Run Pipeline Demo

```bash
# Fire a live P1 incident through the full pipeline → Cosmos DB
python main.py demo
```

### Other Commands

```bash
# Index runbooks and postmortems into Azure AI Search
python main.py index --dir /path/to/runbooks

# View reliability metrics for a service
python main.py metrics --service payment-api --days 30

# Run a Chaos Studio healing validation experiment
python main.py chaos --type pod_failure
```

---

## Directory Structure

```
cloudsage/
├── config/
│   ├── config.json              # All credentials + model + SLO config
│   └── loader.py                # Env var resolution + provider switching
├── agents/                      # 5 AI agents + multi-model consensus
│   ├── incident_agent.py
│   ├── rca_agent.py
│   ├── predictive_agent.py
│   ├── finops_agent.py
│   ├── security_agent.py
│   ├── consensus.py             # 3-model voting with confidence scoring
│   └── orchestrator.py          # 12-step pipeline with step timing + Cosmos persist
├── analytics/
│   ├── adaptive_thresholds.py   # Isolation Forest anomaly detection
│   ├── causal_engine.py         # Granger causality + causal path analysis
│   ├── blast_radius.py          # Downstream service impact prediction
│   ├── slo_tracker.py           # Google SRE burn-rate error budget model
│   ├── economic_impact.py       # Revenue-at-risk + automation savings model
│   ├── reliability_score.py     # 0–100 composite reliability score
│   └── mttr_calculator.py       # Mean Time To Resolve analytics
├── decision_engine/
│   ├── policy_engine.py         # SLO-gated approval/deny/auto logic
│   └── action_mapper.py         # Routes decisions to AKS / Functions / Teams
├── automation/
│   ├── restart_service.py       # AKS rolling restart
│   ├── scale_cluster.py         # AKS horizontal scale-out
│   ├── rollback_deployment.py   # AKS deployment rollback
│   ├── chaos_studio.py          # Chaos Studio healing validation
│   ├── notify_teams.py          # Microsoft Teams webhook alerts
│   └── deployment_risk_scorer.py
├── rag/
│   ├── rag_pipeline.py          # Azure AI Search hybrid retrieval
│   ├── embeddings.py            # Azure OpenAI text-embedding-3-small
│   └── search_client.py         # AI Search vector + keyword client
├── database/
│   ├── cosmos_client.py         # Azure Cosmos DB NoSQL client
│   └── models.py                # IncidentRecord + AIDecisionRecord dataclasses
├── streaming/
│   ├── eventhub_consumer.py     # Azure Event Hubs consumer
│   └── message_router.py        # Routes events to the correct agent pipeline
├── functions/
│   └── cloud_action_function/   # Azure Functions Service Bus trigger
├── dashboard/
│   ├── pages/index.jsx          # Main dashboard — 6 tabs, 5-second live polling
│   ├── pages/api/               # Next.js API routes — read Cosmos DB directly
│   └── pages/lib/cosmosClient.js # Shared Cosmos credential resolution
├── tests/                       # 106 pytest unit + integration tests
├── infra/                       # Azure Bicep IaC templates
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Executive KPIs (live from Cosmos DB)

| Metric | Description | Target |
|--------|-------------|--------|
| **Reliability Score** | 0–100 composite: uptime, MTTR, P1 frequency, automation rate | ≥ 90 |
| **MTTR** | Mean Time To Resolve — automated vs 45-min human P50 baseline | < 5 min |
| **Cost Efficiency Index** | Automation rate × MTTR efficiency vs human benchmark | ≥ 0.80 |
| **Security Risk Score** | 0–100 derived from Defender for Cloud alert severity | ≤ 30 |
| **Revenue Saved** | Cumulative: (human P50 MTTR − actual MTTR) × $/min at risk | — |

---

## Design Decisions

**Why Azure AI Foundry (GPT-4o)?** Enterprise-grade model with Azure's compliance envelope, private networking, and no data residency concerns. GPT-4o's reasoning quality is essential for accurate P1 root cause diagnosis — false positives on automation cost more than the incident itself.

**Why 3-model consensus only on P1?** Running three independent LLM votes on every P4 alert wastes token budget and adds latency. Consensus is reserved for high-stakes decisions where a wrong automated action (restarting a healthy service, rolling back a good deployment) causes more damage than the incident.

**Why Azure AI Search over pure vector search?** Hybrid retrieval (BM25 + vector + semantic reranking) outperforms pure embedding similarity for runbook retrieval. Keyword matches on service names and error codes are high-signal and would be lost in a pure embedding approach.

**Why Cosmos DB NoSQL?** Schema-free documents let the orchestrator persist heterogeneous pipeline outputs — causal chains, blast radius graphs, economic models — without schema migrations. The `/service` partition key gives single-partition read performance for per-service dashboard queries.

**Why SLO burn rate over raw uptime?** A service at 99.9% uptime with all downtime concentrated in the last 48 hours is far more dangerous than one at 99.8% spread evenly across a month. Burn rate captures the current rate of budget consumption and drives proportional automation policy — the same model Google SRE uses in production.

---

## Testing

```bash
pytest tests/ -v --cov=agents --cov=decision_engine --cov=analytics
# 106 tests covering agents, policy engine, SLO tracker, economic model,
# embeddings, causal engine, and end-to-end pipeline integration
```

---

## Hackathon Pitch

**Problem:** SRE teams spend 60%+ of their time on reactive incident response. P1 MTTR averages 45+ minutes. Alert fatigue causes mis-classification. Cost waste from idle resources goes undetected for weeks.

**Solution — 4 Layers:**
1. 🚨 **Intelligent Triage** — 5 specialised AI agents with RAG-augmented reasoning + 3-model GPT-4o consensus for every P1 decision
2. ⚖️ **Safe Autonomous Action** — SLO-gated policy engine prevents automation from amplifying outages; high burn-rate incidents require human sign-off
3. 📊 **Real-Time Intelligence** — Causal graphs, blast radius prediction, economic impact modelling, and error budget tracking — all live from Cosmos DB
4. 🔄 **Closed-Loop Learning** — Every automated action is scored and fed back; the reliability index and runbook knowledge base improve with each incident

**Azure:** AI Foundry · AI Search · Event Hubs · Service Bus · Cosmos DB · AKS · Functions · Defender for Cloud · Chaos Studio · Blob Storage · Monitor · Static Web Apps · Bicep IaC
