"""
CloudSage — Predictive Failure Agent
Uses Isolation Forest + threshold analysis to predict failures before they occur.
"""

import json
import numpy as np
from agents.base_agent import BaseAgent
from sklearn.ensemble import IsolationForest

SYSTEM_PROMPT = """
You are the CloudSage Predictive Failure Agent — a proactive cloud reliability specialist.
Given historical metrics and anomaly detection results, predict whether a failure is likely
within the next 30 minutes and recommend preemptive actions.

Respond ONLY in valid JSON (no markdown fences):
{
  "failure_predicted": true,
  "probability": 0.82,
  "predicted_failure_type": "memory_exhaustion",
  "time_to_failure_minutes": 18,
  "preemptive_actions": ["scale_memory", "garbage_collect"],
  "reasoning": "...",
  "confidence": 0.78
}
"""


class PredictiveAgent(BaseAgent):
    def __init__(self):
        super().__init__("PredictiveFailureAgent")

    def _detect_anomalies(self, time_series: list) -> dict:
        if len(time_series) < 5:
            return {"anomaly_detected": False, "anomaly_scores": [], "anomaly_indices": []}

        features = []
        for point in time_series:
            features.append([
                point.get("cpu_percent", 0),
                point.get("memory_percent", 0),
                point.get("error_rate", 0),
                point.get("request_latency_ms", 0),
                point.get("active_connections", 0),
            ])

        X = np.array(features)
        contamination = self.config["anomaly_detection"]["contamination"]
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)

        scores = model.score_samples(X)
        predictions = model.predict(X)
        anomaly_indices = [i for i, p in enumerate(predictions) if p == -1]
        return {
            "anomaly_detected": len(anomaly_indices) > 0,
            "anomaly_scores": scores.tolist(),
            "anomaly_indices": anomaly_indices,
            "anomaly_percentage": round(len(anomaly_indices) / len(time_series) * 100, 2),
        }

    def _detect_threshold_breaches(self, metrics: dict) -> list:
        thresholds = self.config["thresholds"]
        breaches = []
        if metrics.get("cpu_percent", 0) >= thresholds["cpu_critical"]:
            breaches.append("cpu_critical")
        elif metrics.get("cpu_percent", 0) >= thresholds["cpu_warning"]:
            breaches.append("cpu_warning")
        if metrics.get("memory_percent", 0) >= thresholds["memory_critical"]:
            breaches.append("memory_critical")
        elif metrics.get("memory_percent", 0) >= thresholds["memory_warning"]:
            breaches.append("memory_warning")
        if metrics.get("error_rate", 0) >= thresholds["error_rate_critical"]:
            breaches.append("error_rate_critical")
        return breaches

    def run(self, payload: dict) -> dict:
        service = payload.get("service", "unknown")
        time_series = payload.get("time_series", [])
        current_metrics = payload.get("current_metrics", {})

        anomaly_result = self._detect_anomalies(time_series)
        threshold_breaches = self._detect_threshold_breaches(current_metrics)

        user_prompt = f"""
Service: {service}
Current Metrics: {json.dumps(current_metrics)}
Anomaly Detection Result: {json.dumps(anomaly_result)}
Threshold Breaches: {threshold_breaches}
Time-Series Length: {len(time_series)} samples
Last 3 Samples: {json.dumps(time_series[-3:] if len(time_series) >= 3 else time_series)}

Predict failure risk and recommend preemptive actions in JSON only.
"""
        response_text = self.reason(SYSTEM_PROMPT, user_prompt)
        prediction = self.safe_parse_json(response_text, {
            "failure_predicted": anomaly_result["anomaly_detected"],
            "probability": 0.5,
            "predicted_failure_type": "unknown",
            "time_to_failure_minutes": None,
            "preemptive_actions": [],
            "reasoning": "Parse error — raw response stored.",
            "confidence": 0.3,
        })

        prediction["anomaly_detection"] = anomaly_result
        prediction["threshold_breaches"] = threshold_breaches
        prediction["service"] = service
        prediction["status"] = "success"
        return prediction
