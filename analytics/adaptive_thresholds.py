"""
CloudSage — Adaptive Threshold Engine
=====================================================
Eliminates the #1 cause of alert fatigue: static thresholds.

CPU > 90% is the right threshold for a payment API at 3am.
It's the wrong threshold for the same API at Black Friday peak.

This engine maintains per-service, per-metric, per-hour-of-day
dynamic baselines using Exponentially Weighted Moving Average (EWMA).
Alerts fire on *deviation from expected behaviour* rather than
absolute values.

Algorithm:
  - EWMA mean: μₜ = α × xₜ + (1-α) × μₜ₋₁
  - EWMA variance: σ²ₜ = α × (xₜ - μₜ₋₁)² + (1-α) × σ²ₜ₋₁
  - Alert when: xₜ > μₜ + k × σₜ  (default k=3.0, ~99.7th percentile)

Time-of-day model:
  Maintains a separate EWMA for each hour-of-day (0-23), so the engine
  learns that "CPU 85% at 2pm Monday is normal, CPU 85% at 4am is anomalous."
"""

import logging
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("CloudSage.AdaptiveThresholds")


@dataclass
class MetricBaseline:
    """EWMA baseline for a single service+metric+hour combination."""
    service: str
    metric: str
    hour_of_day: int

    mean: float = 0.0
    variance: float = 100.0    # start with high uncertainty
    n_samples: int = 0
    last_updated: str = ""

    # EWMA smoothing factor (higher = faster adaptation, less stability)
    alpha: float = 0.05


@dataclass
class ThresholdResult:
    service: str
    metric: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    is_anomalous: bool
    anomaly_severity: str      # "normal" | "warning" | "critical"
    dynamic_threshold: float
    static_threshold: float
    using_adaptive: bool
    confidence: float          # How confident we are in the baseline (0-1)


class AdaptiveThresholdEngine:
    """
    Maintains per-service, per-metric, time-aware dynamic thresholds.
    """

    # Standard deviations from mean to trigger alerts
    WARNING_SIGMA = 2.0    # ~95th percentile
    CRITICAL_SIGMA = 3.0   # ~99.7th percentile

    # Minimum samples before we trust the adaptive threshold
    MIN_SAMPLES_FOR_ADAPTIVE = 30

    # Static fallback thresholds (used until enough data is collected)
    STATIC_THRESHOLDS = {
        "cpu_percent":          {"warning": 75, "critical": 90},
        "memory_percent":       {"warning": 70, "critical": 85},
        "error_rate":           {"warning": 2.0, "critical": 5.0},
        "request_latency_ms":   {"warning": 500, "critical": 2000},
        "active_connections":   {"warning": 800, "critical": 1000},
    }

    def __init__(self):
        # key: (service, metric, hour_of_day) -> MetricBaseline
        self._baselines: dict = {}

    def _key(self, service: str, metric: str, hour: int) -> tuple:
        return (service, metric, hour)

    def _get_or_create(self, service: str, metric: str, hour: int) -> MetricBaseline:
        key = self._key(service, metric, hour)
        if key not in self._baselines:
            # Seed with static threshold midpoint so we don't start at 0
            static = self.STATIC_THRESHOLDS.get(metric, {})
            seed_mean = static.get("warning", 50) * 0.7  # assume 70% of warning is normal
            self._baselines[key] = MetricBaseline(
                service=service,
                metric=metric,
                hour_of_day=hour,
                mean=seed_mean,
                variance=max(seed_mean * 0.2, 1.0) ** 2,
            )
        return self._baselines[key]

    def update(self, service: str, metric: str, value: float, timestamp: datetime = None):
        """
        Ingest a new metric observation and update the EWMA baseline.
        Call this on every telemetry tick (every 30s in production).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour
        baseline = self._get_or_create(service, metric, hour)
        alpha = baseline.alpha

        # EWMA update
        prev_mean = baseline.mean
        baseline.mean = alpha * value + (1 - alpha) * prev_mean
        baseline.variance = alpha * (value - prev_mean) ** 2 + (1 - alpha) * baseline.variance
        baseline.n_samples += 1
        baseline.last_updated = timestamp.isoformat()

    def evaluate(
        self,
        service: str,
        metric: str,
        value: float,
        timestamp: datetime = None,
    ) -> ThresholdResult:
        """
        Evaluate a metric value against the adaptive baseline.
        Returns a ThresholdResult with anomaly classification.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour
        baseline = self._get_or_create(service, metric, hour)
        std = math.sqrt(max(baseline.variance, 0.01))

        z_score = (value - baseline.mean) / std if std > 0 else 0.0
        dynamic_threshold = baseline.mean + self.CRITICAL_SIGMA * std

        static = self.STATIC_THRESHOLDS.get(metric, {})
        static_critical = static.get("critical", float("inf"))
        static_warning = static.get("warning", float("inf"))

        # Confidence in adaptive threshold (based on sample count)
        confidence = min(1.0, baseline.n_samples / self.MIN_SAMPLES_FOR_ADAPTIVE)
        using_adaptive = confidence >= 0.5

        # Blend adaptive and static thresholds during warm-up
        if using_adaptive:
            effective_threshold = dynamic_threshold
            is_anomalous = z_score > self.CRITICAL_SIGMA
            is_warning = z_score > self.WARNING_SIGMA
        else:
            effective_threshold = static_critical
            is_anomalous = value >= static_critical
            is_warning = value >= static_warning

        if is_anomalous:
            severity = "critical"
        elif is_warning:
            severity = "warning"
        else:
            severity = "normal"

        return ThresholdResult(
            service=service,
            metric=metric,
            value=round(value, 3),
            baseline_mean=round(baseline.mean, 3),
            baseline_std=round(std, 3),
            z_score=round(z_score, 3),
            is_anomalous=is_anomalous or is_warning,
            anomaly_severity=severity,
            dynamic_threshold=round(effective_threshold, 3),
            static_threshold=static_critical,
            using_adaptive=using_adaptive,
            confidence=round(confidence, 3),
        )

    def evaluate_all(self, service: str, metrics: dict, timestamp: datetime = None) -> dict:
        """
        Evaluate all metrics for a service at once.
        Returns dict of metric -> ThresholdResult as dict.
        """
        results = {}
        for metric, value in metrics.items():
            if metric in self.STATIC_THRESHOLDS:
                # BUG 4 FIX: value may be None when telemetry is partially populated.
                # float(None) raises TypeError, crashing Step 0 and halting the pipeline.
                if value is None:
                    continue
                try:
                    fvalue = float(value)
                except (TypeError, ValueError):
                    continue
                self.update(service, metric, fvalue, timestamp)
                result = self.evaluate(service, metric, fvalue, timestamp)
                results[metric] = {
                    "value": result.value,
                    "baseline_mean": result.baseline_mean,
                    "baseline_std": result.baseline_std,
                    "z_score": result.z_score,
                    "anomaly_severity": result.anomaly_severity,
                    "dynamic_threshold": result.dynamic_threshold,
                    "using_adaptive": result.using_adaptive,
                    "confidence": result.confidence,
                }
        return results

    def get_summary(self, service: str) -> dict:
        """Return all baselines for a service."""
        baselines = {}
        for (svc, metric, hour), baseline in self._baselines.items():
            if svc != service:
                continue
            key = f"{metric}:h{hour:02d}"
            baselines[key] = {
                "mean": round(baseline.mean, 3),
                "std": round(math.sqrt(max(baseline.variance, 0)), 3),
                "samples": baseline.n_samples,
                "dynamic_critical_threshold": round(
                    baseline.mean + self.CRITICAL_SIGMA * math.sqrt(max(baseline.variance, 0.01)), 3
                ),
            }
        return {"service": service, "baselines": baselines}


# Shared singleton
_shared_engine = AdaptiveThresholdEngine()


def get_adaptive_threshold_engine() -> AdaptiveThresholdEngine:
    return _shared_engine
