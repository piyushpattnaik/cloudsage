"""
CloudSage — Security & Compliance Agent
Integrates with Microsoft Defender for Cloud and Azure Policy to detect threats.
"""

import json
from agents.base_agent import BaseAgent

SYSTEM_PROMPT = """
You are the CloudSage Security & Compliance Agent — a senior cloud security engineer.
Analyze the provided Defender for Cloud alerts, Azure Policy violations, and access logs.

Respond ONLY in valid JSON (no markdown fences):
{
  "security_risk_score": 72,
  "critical_threats": [
    {"id": "...", "type": "...", "resource": "...", "severity": "High", "action_required": "..."}
  ],
  "policy_violations": [
    {"policy": "...", "resource": "...", "compliance_state": "NonCompliant", "remediation": "..."}
  ],
  "public_exposure_risks": ["..."],
  "suspicious_activity": [
    {"type": "anomalous_login", "user": "...", "detail": "...", "risk": "High"}
  ],
  "immediate_actions": ["..."],
  "compliance_summary": {
    "total_resources": 0, "compliant": 0, "non_compliant": 0, "compliance_percentage": 0.0
  }
}
"""


class SecurityAgent(BaseAgent):
    def __init__(self):
        super().__init__("SecurityAgent")

    def _score_risk(self, alerts: list) -> int:
        if not alerts:
            return 10
        severity_weights = {"High": 30, "Medium": 15, "Low": 5, "Informational": 1}
        total = sum(severity_weights.get(a.get("severity", "Low"), 5) for a in alerts)
        return min(100, total)

    def run(self, payload: dict) -> dict:
        defender_alerts = payload.get("defender_alerts", [])
        policy_states = payload.get("policy_states", [])
        access_logs = payload.get("access_logs", [])
        network_security_groups = payload.get("network_security_groups", [])

        risk_score = self._score_risk(defender_alerts)

        public_exposure = []
        for nsg in network_security_groups:
            for rule in nsg.get("rules", []):
                if (
                    rule.get("direction") == "Inbound"
                    and rule.get("access") == "Allow"
                    and rule.get("sourceAddressPrefix") in ("*", "Internet", "0.0.0.0/0")
                ):
                    public_exposure.append(
                        f"NSG {nsg.get('name')}: port {rule.get('destinationPortRange')} open to Internet"
                    )

        user_prompt = f"""
Pre-computed Risk Score: {risk_score}/100

Defender for Cloud Alerts ({len(defender_alerts)} total):
{json.dumps(defender_alerts[:10], indent=2)}

Azure Policy States ({len(policy_states)} resources evaluated):
{json.dumps(policy_states[:10], indent=2)}

Public Exposure Detected: {public_exposure}

Recent Access Log Sample (last 10):
{json.dumps(access_logs[-10:] if access_logs else [], indent=2)}

Analyze threats and produce security report in JSON only.
"""
        response_text = self.reason(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
        report = self.safe_parse_json(response_text, {
            "security_risk_score": risk_score,
            "critical_threats": [],
            "policy_violations": [],
            "public_exposure_risks": public_exposure,
            "suspicious_activity": [],
            "immediate_actions": [],
            "compliance_summary": {},
        })

        report["status"] = "success"
        return report
