"""
CloudSage — Cosmos DB Client
Handles all persistence: incidents, decisions, metrics, MTTR data.
Uses DefaultAzureCredential in production; key-based auth in dev.
"""

import uuid
import logging
from datetime import datetime, timedelta, timezone
from azure.cosmos import CosmosClient, exceptions
from azure.identity import DefaultAzureCredential

from config.loader import load_config
from database.models import IncidentRecord

logger = logging.getLogger("CosmosDBClient")


def _build_cosmos_client(config: dict) -> CosmosClient:
    endpoint = config["cosmos_db"]["endpoint"]
    key = config["cosmos_db"].get("key")
    if key:
        return CosmosClient(url=endpoint, credential=key)
    return CosmosClient(url=endpoint, credential=DefaultAzureCredential())


class CosmosDBClient:
    """
    Azure Cosmos DB client for CloudSage operational data.
    Database: cloudsage | Container: incidents | Partition key: /service
    """

    def __init__(self):
        config = load_config()
        db_cfg = config["cosmos_db"]
        self._client = _build_cosmos_client(config)
        self._db = self._client.get_database_client(db_cfg["database"])
        self._container = self._db.get_container_client(db_cfg["container"])

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def save_incident(self, incident: dict) -> dict:
        """Upsert an incident/feedback/decision record."""
        doc = {
            "id": incident.get("id") or str(uuid.uuid4()),
            "timestamp": incident.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "ttl": 90 * 24 * 3600,
            **incident,
        }
        if "service" not in doc:
            doc["service"] = "unknown"
        try:
            result = self._container.upsert_item(doc)
            logger.info(f"Saved: id={doc['id']} service={doc.get('service')}")
            return result
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Cosmos upsert failed: {e}")
            raise

    def resolve_incident(self, incident_id: str, service: str, resolution_note: str = "") -> dict:
        """Mark an incident as resolved, compute MTTR, and persist."""
        existing = self.get_incident(incident_id, service)
        if not existing:
            logger.warning(f"Cannot resolve — incident not found: {incident_id}")
            return {}

        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(IncidentRecord)}
        record = IncidentRecord(**{k: v for k, v in existing.items() if k in valid_fields})
        record.resolve(resolution_note)

        patch_ops = [
            {"op": "replace", "path": "/status", "value": "resolved"},
            {"op": "add", "path": "/resolved_at", "value": record.resolved_at},
            {"op": "add", "path": "/mttr_minutes", "value": record.mttr_minutes},
            # FIXED: was op:"replace" which requires the field to already exist (HTTP 422).
            # updated_at is not in IncidentRecord's initial save, so it never exists.
            # op:"add" creates-or-replaces — correct for optional fields.
            {"op": "add", "path": "/updated_at", "value": datetime.now(timezone.utc).isoformat()},
        ]
        try:
            result = self._container.patch_item(
                item=incident_id, partition_key=service, patch_operations=patch_ops
            )
            logger.info(f"Resolved incident {incident_id}: MTTR={record.mttr_minutes}m")
            return result
        except Exception as e:
            logger.error(f"Failed to resolve incident {incident_id}: {e}")
            return {}

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_incident(self, incident_id: str, service: str) -> dict:
        try:
            return self._container.read_item(item=incident_id, partition_key=service)
        except exceptions.CosmosResourceNotFoundError:
            return None

    def query_incidents(
        self,
        service: str = None,
        severity: str = None,
        limit: int = 100,
        days_back: int = 30,
    ) -> list:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()[:10]

        conditions = ["c.timestamp >= @cutoff"]
        params = [{"name": "@cutoff", "value": cutoff}]

        if service:
            # FIXED: use parameterized query instead of f-string interpolation
            conditions.append("c.service = @service")
            params.append({"name": "@service", "value": service})
        if severity:
            conditions.append("c.severity = @severity")
            params.append({"name": "@severity", "value": severity})

        where_clause = " AND ".join(conditions)
        query = f"SELECT TOP {limit} * FROM c WHERE {where_clause} ORDER BY c.timestamp DESC"

        try:
            return list(self._container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True,
            ))
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_mttr_data(self, service: str = None, days: int = 30) -> list:
        """Retrieve resolved incidents for MTTR calculation."""
        # FIXED: use parameterized query — service was previously f-string interpolated
        conditions = ["c.status = 'resolved'", "IS_DEFINED(c.resolved_at)"]
        params = []

        if service:
            conditions.append("c.service = @service")
            params.append({"name": "@service", "value": service})

        where_clause = " AND ".join(conditions)
        query = (
            f"SELECT c.id, c.service, c.severity, c.timestamp, c.resolved_at, "
            f"c.mttr_minutes, c.automation_result "
            f"FROM c WHERE {where_clause} ORDER BY c.timestamp DESC"
        )
        try:
            return list(self._container.query_items(
                query=query,
                parameters=params if params else None,
                enable_cross_partition_query=True,
            ))
        except Exception as e:
            logger.error(f"MTTR query failed: {e}")
            return []

    def get_incidents_by_service(self, service: str, limit: int = 200) -> list:
        """Retrieve recent incidents for a specific service (used by SLO tracker)."""
        query = (
            # BUG 3 FIX: c.tags was omitted from SELECT list.
            # SLO tracker and compute_cumulative_savings both read
            # inc.get("tags", {}).get("error_rate") — without c.tags in the
            # projection, error_rate always came back None → 5% fallback every time.
            "SELECT c.id, c.service, c.severity, c.timestamp, c.resolved_at, "
            "c.mttr_minutes, c.status, c.action, c.tags "
            "FROM c WHERE c.service = @service ORDER BY c.timestamp DESC "
            f"OFFSET 0 LIMIT {min(limit, 500)}"
        )
        params = [{"name": "@service", "value": service}]
        try:
            return list(self._container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=False,  # service is partition key
            ))
        except Exception as e:
            logger.error(f"get_incidents_by_service failed for {service}: {e}")
            return []

    def get_feedback_trend(self, service: str = None, days: int = 30) -> list:
        """Retrieve feedback scores for trend analysis."""
        # FIXED: parameterized query
        conditions = ["c.type = 'feedback'"]
        params = []

        if service:
            conditions.append("c.service = @service")
            params.append({"name": "@service", "value": service})

        where_clause = " AND ".join(conditions)
        query = f"SELECT * FROM c WHERE {where_clause} ORDER BY c.timestamp DESC"

        try:
            return list(self._container.query_items(
                query=query,
                parameters=params if params else None,
                enable_cross_partition_query=True,
            ))
        except Exception as e:
            logger.error(f"Feedback query failed: {e}")
            return []
