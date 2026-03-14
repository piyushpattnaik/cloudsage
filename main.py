"""
CloudSage — Main Orchestrator Entry Point

Commands:
  consumer              Run Event Hub consumer (real-time production mode)
  demo                  Simulate a P1 incident end-to-end (no Azure required)
  index --dir DIR       Index knowledge base documents into Azure AI Search
  metrics --service S   Print reliability metrics for a service
  chaos --type TYPE     Run a Chaos Studio healing validation experiment
"""

import argparse
import json
import pathlib
import sys

# FIXED: ensure logs/ directory exists BEFORE configuring FileHandler.
# Previously FileHandler was configured at module level but the mkdir()
# only ran inside main(), causing FileNotFoundError on import.
pathlib.Path("logs").mkdir(exist_ok=True)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/cloudsage.log"),
    ],
)
logger = logging.getLogger("CloudSage.Main")


def run_consumer():
    logger.info("Starting CloudSage HTTP ingestion server (free-tier: replaces Event Hub)...")
    from streaming.eventhub_consumer import HttpIngestionServer
    import os
    host = os.getenv("INGESTION_HOST", "0.0.0.0")
    port = int(os.getenv("INGESTION_PORT", "8080"))
    HttpIngestionServer(host=host, port=port).run()


def run_demo():
    """
    Run a real end-to-end P1 incident through the full CloudSage pipeline.
    Uses the same AgentOrchestrator as production — LLM calls, Cosmos DB save,
    feedback loop, runbook generation — everything live, nothing mocked.
    """
    import uuid
    import time

    logger.info("=" * 60)
    logger.info("CloudSage Demo — Live P1 Incident Pipeline")
    logger.info("=" * 60)

    incident_id = f"DEMO-{str(uuid.uuid4())[:8].upper()}"

    payload = {
        "event_type": "incident_alert",
        "incident_id": incident_id,
        "service": "payment-api",
        "alert_description": (
            "CPU utilization exceeded 95% for 5 consecutive minutes. "
            "Error rate spiked to 8.3%. Request latency p99 at 4200ms."
        ),
        "metrics": {
            "cpu_percent": 96,
            "memory_percent": 71,
            "error_rate": 8.3,
            "request_latency_ms": 4200,
            "active_connections": 450,
        },
        "namespace": "production",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"\n{'='*60}")
    print(f"  CloudSage — Live Pipeline Demo")
    print(f"  Incident ID: {incident_id}")
    print(f"{'='*60}")
    print(f"\n📡 Event payload:")
    print(json.dumps(payload, indent=2))
    print(f"\n🚀 Starting AgentOrchestrator pipeline...")
    print(f"   IncidentAgent → Consensus → Causal → Blast Radius → SLO")
    print(f"   → RCA + Predictive (parallel) → PolicyEngine → Cosmos DB\n")

    start = time.time()

    try:
        from streaming.message_router import MessageRouter
        router = MessageRouter()
        result = router.route_sync("incident_alert", payload)

        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
        if result:
            print(f"  Severity:  {result.get('severity', 'N/A')}")
            print(f"  Action:    {result.get('action', 'N/A')}")
            print(f"  Status:    {result.get('status', 'N/A')}")
            mttr = result.get('mttr_minutes')
            if mttr:
                print(f"  MTTR:      {mttr}m")
            savings = result.get('revenue_saved_usd')
            if savings:
                print(f"  Savings:   ${savings:,.2f}")
            print(f"  Cosmos ID: {result.get('id', incident_id)}")
        print(f"{'='*60}\n")

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Demo pipeline failed after {elapsed:.1f}s: {e}", exc_info=True)
        print(f"\n❌ Pipeline error: {e}")
        print("   Check logs/cloudsage.log for details.")
        sys.exit(1)

    logger.info("Demo complete")


def run_index(docs_dir: str):
    from rag.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    docs_path = pathlib.Path(docs_dir)
    if not docs_path.exists():
        logger.error(f"Directory not found: {docs_dir}")
        sys.exit(1)

    documents = []
    for file_path in docs_path.glob("**/*.md"):
        content = file_path.read_text(encoding="utf-8")
        parts = file_path.parts
        category = (
            "runbook" if any("runbook" in p.lower() for p in parts) else
            "postmortem" if any("postmortem" in p.lower() for p in parts) else
            "sop" if any("sop" in p.lower() for p in parts) else
            "architecture" if any("arch" in p.lower() for p in parts) else
            "general"
        )
        documents.append({"content": content, "source": str(file_path), "category": category})

    if not documents:
        logger.warning(f"No .md files found in {docs_dir}")
        return

    logger.info(f"Indexing {len(documents)} documents from {docs_dir} (with chunking)...")
    total_chunks = pipeline.index_documents_bulk(documents)
    logger.info(f"Indexing complete: {total_chunks} chunks from {len(documents)} documents.")


def run_metrics(service: str, days: int):
    from analytics.reliability_score import ReliabilityScoreCalculator
    from analytics.mttr_calculator import MTTRCalculator

    print(f"\n📊 CloudSage Metrics — service='{service}' period={days}d")
    print("=" * 60)

    score = ReliabilityScoreCalculator().compute(service=service, days=days)
    print("Reliability Score:")
    print(json.dumps(score, indent=2))

    mttr = MTTRCalculator().compute_mttr(service=service, days=days)
    print("\nMTTR Analysis:")
    print(json.dumps(mttr, indent=2))


def run_chaos(experiment_type: str):
    from automation.chaos_studio import ChaosStudioClient
    client = ChaosStudioClient()
    result = client.run_healing_validation(experiment_type=experiment_type)
    label = "Chaos Studio Validation" if result.get("status") != "unavailable" else "Chaos Studio (Free-Tier)"
    print(f"\n🔥 {label}:")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="CloudSage — Autonomous Cloud Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("consumer", help="Start real-time Event Hub consumer")
    sub.add_parser("demo", help="Run simulated P1 incident demo (no Azure needed)")

    idx = sub.add_parser("index", help="Index knowledge base documents")
    idx.add_argument("--dir", required=True, help="Path to directory containing .md files")

    met = sub.add_parser("metrics", help="Print service reliability metrics")
    met.add_argument("--service", required=True)
    met.add_argument("--days", type=int, default=30)

    cha = sub.add_parser("chaos", help="Run Chaos Studio healing validation")
    cha.add_argument(
        "--type", default="pod_failure",
        choices=["pod_failure", "cpu_pressure", "network_latency"],
    )

    args = parser.parse_args()

    dispatch = {
        "consumer": lambda: run_consumer(),
        "demo": lambda: run_demo(),
        "index": lambda: run_index(args.dir),
        "metrics": lambda: run_metrics(args.service, args.days),
        "chaos": lambda: run_chaos(args.type),
    }

    fn = dispatch.get(args.command)
    if fn:
        fn()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
