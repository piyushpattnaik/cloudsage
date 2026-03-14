"""
CloudSage — Microsoft Teams Notification
Sends rich adaptive card notifications to a Teams channel via webhook.
"""

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone

from config.loader import load_config

logger = logging.getLogger("NotifyTeams")

SEVERITY_COLORS = {
    "P1": "FF0000",
    "P2": "FF8C00",
    "P3": "FFD700",
    "P4": "00B050",
}


def notify_teams(context: dict) -> dict:
    """
    Send an incident/action notification to Microsoft Teams.

    Context keys:
        service, severity, summary, action, incident_id
    """
    config = load_config()
    webhook_url = config["teams"].get("webhook_url", "")

    if not webhook_url:
        logger.warning("TEAMS_WEBHOOK_URL not configured — skipping Teams notification.")
        return {"status": "skipped", "reason": "webhook_url not configured"}

    service = context.get("service", "unknown")
    severity = context.get("severity", "P4")
    summary = context.get("summary", "CloudSage detected an issue.")
    action_taken = context.get("action", "alert_only")
    incident_id = context.get("incident_id", "N/A")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    color = SEVERITY_COLORS.get(severity, "808080")

    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": color,
        "summary": f"CloudSage {severity} Incident: {service}",
        "sections": [
            {
                "activityTitle": f"🚨 CloudSage Incident Alert — {severity}",
                "activitySubtitle": f"Service: **{service}** | {timestamp}",
                "facts": [
                    {"name": "Incident ID", "value": incident_id},
                    {"name": "Severity", "value": severity},
                    {"name": "Service", "value": service},
                    {"name": "Action Taken", "value": action_taken.replace("_", " ").title()},
                    {"name": "Time", "value": timestamp},
                ],
                "markdown": True,
            },
            {"activityTitle": "📋 Summary", "text": summary},
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": "View in CloudSage Dashboard",
                "targets": [{"os": "default", "uri": f"https://cloudsage.azurestaticapps.net/incidents/{incident_id}"}],
            }
        ],
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            response_body = resp.read().decode()

        logger.info(f"Teams notification sent: incident={incident_id} severity={severity}")
        return {
            "status": "success",
            "message": f"Teams notification sent for incident {incident_id}.",
            "teams_response": response_body,
        }

    except urllib.error.URLError as e:
        logger.error(f"Teams webhook failed: {e}")
        return {"status": "error", "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error sending Teams notification: {e}")
        return {"status": "error", "error": str(e)}
