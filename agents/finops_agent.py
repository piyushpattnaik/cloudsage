"""
CloudSage — FinOps Optimization Agent
Analyzes cloud spend, identifies idle resources, and recommends cost optimizations.
"""

import json
from agents.base_agent import BaseAgent

SYSTEM_PROMPT = """
You are the CloudSage FinOps Optimization Agent — a cloud cost intelligence specialist.
Analyze the provided Azure cost data, resource utilization, and Azure Advisor recommendations.

Respond ONLY in valid JSON (no markdown fences):
{
  "cost_efficiency_index": 0.72,
  "monthly_savings_potential_usd": 4200,
  "cost_spike_detected": true,
  "spike_cause": "...",
  "idle_resources": [
    {"resource": "vm-dev-01", "type": "VirtualMachine", "monthly_cost_usd": 180, "utilization_percent": 3, "action": "shutdown"}
  ],
  "rightsizing_candidates": [
    {"resource": "aks-node-pool", "current_sku": "D8s_v3", "recommended_sku": "D4s_v3", "saving_usd": 300}
  ],
  "quick_wins": ["..."],
  "strategic_recommendations": ["..."],
  "reserved_instance_opportunity": {"service": "...", "annual_saving_usd": 0}
}
"""


class FinOpsAgent(BaseAgent):
    def __init__(self):
        super().__init__("FinOpsAgent")

    def _detect_cost_spike(self, cost_history: list) -> dict:
        if len(cost_history) < 7:
            return {"spike_detected": False}
        costs = [d.get("cost_usd", 0) for d in cost_history]
        baseline = sum(costs[:-1]) / len(costs[:-1])
        latest = costs[-1]
        threshold_pct = self.config["thresholds"].get("cost_spike_percent", 20)
        change_pct = ((latest - baseline) / baseline * 100) if baseline > 0 else 0
        return {
            "spike_detected": change_pct >= threshold_pct,
            "change_percent": round(change_pct, 2),
            "baseline_daily_usd": round(baseline, 2),
            "latest_daily_usd": round(latest, 2),
        }

    def run(self, payload: dict) -> dict:
        cost_history = payload.get("cost_history", [])
        resource_utilization = payload.get("resource_utilization", [])
        advisor_recommendations = payload.get("advisor_recommendations", [])
        current_month_spend = payload.get("current_month_spend_usd", 0)
        budget_usd = payload.get("budget_usd", 0)

        spike_analysis = self._detect_cost_spike(cost_history)

        user_prompt = f"""
Current Month Spend: ${current_month_spend:,.2f}
Monthly Budget: ${budget_usd:,.2f}
Budget Utilization: {round(current_month_spend/budget_usd*100, 1) if budget_usd else 'N/A'}%

Cost Spike Analysis: {json.dumps(spike_analysis)}

Resource Utilization (sample):
{json.dumps(resource_utilization[:10], indent=2)}

Azure Advisor Recommendations:
{json.dumps(advisor_recommendations[:5], indent=2)}

Analyze costs and provide optimization recommendations in JSON only.
"""
        response_text = self.reason(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
        analysis = self.safe_parse_json(response_text, {
            "cost_efficiency_index": 0.5,
            "monthly_savings_potential_usd": 0,
            "cost_spike_detected": spike_analysis.get("spike_detected", False),
            "spike_cause": "Analysis unavailable",
            "idle_resources": [],
            "rightsizing_candidates": [],
            "quick_wins": [],
            "strategic_recommendations": [],
            "reserved_instance_opportunity": {},
        })

        analysis["spike_analysis"] = spike_analysis
        analysis["current_month_spend_usd"] = current_month_spend
        analysis["status"] = "success"
        return analysis
