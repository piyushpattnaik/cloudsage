"""
CloudSage — Pre-Deployment Risk Scorer
=====================================================
Shifts reliability LEFT into CI/CD — flags risky PRs before they merge.

Called as a GitHub Actions step. Scores a pull request for reliability
risk based on:
  1. Which services are touched (high-incident services = higher risk)
  2. Historical incident rate for those services in the last 90 days
  3. Files changed and diff complexity (large diffs = higher risk)
  4. Whether the changed service has runbooks in the RAG knowledge base
  5. Whether similar changes caused incidents previously

Output:
  - Risk score 0-100
  - Risk grade A-F
  - Specific warnings with runbook references
  - GitHub PR comment with full report
  - Exit code 1 if score >= BLOCK_THRESHOLD (to fail the CI check)

Usage (GitHub Actions):
  - name: CloudSage Pre-Deployment Risk Check
    run: python -m automation.deployment_risk_scorer
    env:
      PR_CHANGED_FILES: ${{ steps.changed-files.outputs.all_changed_files }}
      COSMOS_DB_ENDPOINT: ${{ secrets.COSMOS_DB_ENDPOINT }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      GITHUB_REPOSITORY: ${{ github.repository }}
      GITHUB_PR_NUMBER: ${{ github.event.pull_request.number }}
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("CloudSage.DeploymentRisk")

# Risk score threshold that blocks the PR (exits with code 1)
BLOCK_THRESHOLD = 75

# Service-to-file patterns (which files belong to which service)
SERVICE_PATTERNS = {
    "payment-api":      ["payment", "checkout/payment", "billing"],
    "checkout-service": ["checkout", "cart", "order"],
    "auth-service":     ["auth", "login", "session", "token"],
    "inventory-api":    ["inventory", "stock", "product"],
    "notification-svc": ["notification", "email", "sms"],
}

# Risk multipliers per service (based on historical incident severity)
SERVICE_RISK_WEIGHTS = {
    "payment-api":      2.0,   # highest — revenue critical
    "auth-service":     1.8,   # authentication failure = all services down
    "checkout-service": 1.5,
    "inventory-api":    1.2,
    "default":          1.0,
}


@dataclass
class RiskFactor:
    name: str
    score: float     # 0-100 contribution
    weight: float    # 0-1
    detail: str
    runbook_ref: Optional[str] = None


@dataclass
class DeploymentRiskReport:
    overall_score: float        # 0-100
    grade: str                  # A-F
    services_affected: list
    risk_factors: list          # list[RiskFactor]
    warnings: list              # human-readable warning strings
    runbook_references: list    # relevant runbooks from RAG
    block_deployment: bool
    summary: str


class DeploymentRiskScorer:
    """
    Scores a PR's reliability risk before deployment.
    """

    def __init__(self, cosmos_client=None, rag_pipeline=None):
        self.cosmos = cosmos_client
        self.rag = rag_pipeline

    def score(
        self,
        changed_files: list,
        pr_title: str = "",
        pr_description: str = "",
        author: str = "",
    ) -> DeploymentRiskReport:
        """
        Score a deployment's reliability risk.

        Args:
            changed_files: List of changed file paths from the PR
            pr_title: PR title (used for semantic search)
            pr_description: PR description
            author: PR author login

        Returns:
            DeploymentRiskReport
        """
        services = self._identify_services(changed_files)
        factors = []
        warnings = []
        runbook_refs = []

        # ── Factor 1: Diff volume ─────────────────────────────────────
        file_count = len(changed_files)
        volume_score = min(100, file_count * 4)
        factors.append(RiskFactor(
            name="Change Volume",
            score=volume_score,
            weight=0.20,
            detail=f"{file_count} files changed — larger diffs correlate with higher incident rate",
        ))
        if file_count > 20:
            warnings.append(f"⚠️ Large change: {file_count} files. Consider splitting into smaller PRs.")

        # ── Factor 2: Service criticality ─────────────────────────────
        max_weight = max(
            (SERVICE_RISK_WEIGHTS.get(s, 1.0) for s in services),
            default=1.0,
        )
        criticality_score = min(100, (max_weight - 1.0) / 1.0 * 60 + 20)
        affected_critical = [s for s in services if SERVICE_RISK_WEIGHTS.get(s, 1.0) >= 1.5]
        factors.append(RiskFactor(
            name="Service Criticality",
            score=criticality_score,
            weight=0.30,
            detail=f"Critical services affected: {affected_critical or 'none'}",
        ))
        for s in affected_critical:
            warnings.append(f"🔴 High-criticality service modified: {s}")

        # ── Factor 3: Historical incident rate ────────────────────────
        historical_score = 0
        if self.cosmos and services:
            try:
                for service in services:
                    incidents = self.cosmos.get_incidents_by_service(service, limit=50)
                    p1_count = sum(1 for i in incidents if i.get("severity") == "P1")
                    p2_count = sum(1 for i in incidents if i.get("severity") == "P2")
                    svc_score = min(100, p1_count * 15 + p2_count * 5)
                    historical_score = max(historical_score, svc_score)
                    if p1_count > 0:
                        warnings.append(
                            f"📊 {service} had {p1_count} P1 incident(s) in the last 90 days"
                        )
            except Exception as e:
                logger.warning(f"Could not fetch historical incidents: {e}")
                historical_score = 30  # assume moderate risk if can't check

        factors.append(RiskFactor(
            name="Historical Incident Rate",
            score=historical_score,
            weight=0.30,
            detail="Based on P1/P2 incidents in the last 90 days for affected services",
        ))

        # ── Factor 4: Config/infrastructure changes ────────────────────
        infra_files = [f for f in changed_files if any(
            k in f.lower() for k in ["config", "bicep", "yaml", "yml", "terraform", "helm", "k8s", "kubernetes"]
        )]
        infra_score = min(100, len(infra_files) * 20)
        factors.append(RiskFactor(
            name="Infrastructure Changes",
            score=infra_score,
            weight=0.15,
            detail=f"{len(infra_files)} infrastructure/config files changed",
        ))
        if infra_files:
            warnings.append(f"⚙️ Infrastructure files changed: {', '.join(infra_files[:3])}")

        # ── Factor 5: Missing runbooks ────────────────────────────────
        missing_runbook_score = 0
        if self.rag and services:
            try:
                for service in services:
                    results = self.rag.retrieve(
                        f"runbook {service} deployment incident", top_k=3
                    )
                    if results:
                        for r in results:
                            ref = r.get("source", "")
                            if ref and ref not in runbook_refs:
                                runbook_refs.append(ref)
                    else:
                        missing_runbook_score = max(missing_runbook_score, 40)
                        warnings.append(
                            f"📖 No runbook found for {service}. "
                            "Consider adding one before deployment."
                        )
            except Exception as e:
                logger.warning(f"RAG lookup failed: {e}")

        factors.append(RiskFactor(
            name="Runbook Coverage",
            score=missing_runbook_score,
            weight=0.05,
            detail="Measures whether affected services have operational runbooks",
            runbook_ref=runbook_refs[0] if runbook_refs else None,
        ))

        # ── Weighted aggregate score ───────────────────────────────────
        total_score = sum(f.score * f.weight for f in factors)
        # Apply service criticality multiplier
        total_score = min(100, total_score * (max_weight ** 0.3))

        grade = (
            "A" if total_score < 20 else
            "B" if total_score < 40 else
            "C" if total_score < 60 else
            "D" if total_score < 75 else
            "F"
        )

        block = total_score >= BLOCK_THRESHOLD

        summary = (
            f"Risk Score: {total_score:.0f}/100 (Grade {grade}). "
            f"Services: {', '.join(services) or 'general'}. "
            f"{'⛔ DEPLOYMENT BLOCKED — high risk. Human review required.' if block else '✅ Deployment approved.'}"
        )

        if block:
            warnings.insert(0, f"⛔ Risk score {total_score:.0f} exceeds threshold {BLOCK_THRESHOLD}. Deployment requires team lead sign-off.")

        return DeploymentRiskReport(
            overall_score=round(total_score, 1),
            grade=grade,
            services_affected=services,
            risk_factors=[self._factor_to_dict(f) for f in factors],
            warnings=warnings,
            runbook_references=runbook_refs,
            block_deployment=block,
            summary=summary,
        )

    @staticmethod
    def _identify_services(changed_files: list) -> list:
        """Map changed files to service names."""
        services = set()
        for f in changed_files:
            f_lower = f.lower()
            for service, patterns in SERVICE_PATTERNS.items():
                if any(p in f_lower for p in patterns):
                    services.add(service)
        return sorted(services)

    @staticmethod
    def _factor_to_dict(f: RiskFactor) -> dict:
        return {
            "name": f.name,
            "score": round(f.score, 1),
            "weight": f.weight,
            "weighted_contribution": round(f.score * f.weight, 1),
            "detail": f.detail,
            "runbook_ref": f.runbook_ref,
        }

    @staticmethod
    def to_dict(report: DeploymentRiskReport) -> dict:
        return {
            "overall_score": report.overall_score,
            "grade": report.grade,
            "services_affected": report.services_affected,
            "risk_factors": report.risk_factors,
            "warnings": report.warnings,
            "runbook_references": report.runbook_references,
            "block_deployment": report.block_deployment,
            "summary": report.summary,
        }

    def format_pr_comment(self, report: DeploymentRiskReport) -> str:
        """Format the risk report as a GitHub PR comment in Markdown."""
        grade_emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"}.get(report.grade, "❓")
        lines = [
            "## ☁ CloudSage Pre-Deployment Risk Analysis",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Risk Score | **{report.overall_score}/100** |",
            f"| Grade | {grade_emoji} **{report.grade}** |",
            f"| Services | {', '.join(report.services_affected) or 'general'} |",
            f"| Decision | {'⛔ **BLOCKED**' if report.block_deployment else '✅ **Approved**'} |",
            "",
            "### Risk Breakdown",
            "",
            "| Factor | Score | Weight | Contribution |",
            "|--------|-------|--------|--------------|",
        ]
        for f in report.risk_factors:
            lines.append(
                f"| {f['name']} | {f['score']:.0f}/100 | {f['weight']*100:.0f}% | {f['weighted_contribution']:.1f} |"
            )

        if report.warnings:
            lines += ["", "### ⚠️ Warnings", ""]
            lines += [f"- {w}" for w in report.warnings]

        if report.runbook_references:
            lines += ["", "### 📖 Relevant Runbooks", ""]
            lines += [f"- `{r}`" for r in report.runbook_references]

        lines += [
            "",
            "---",
            f"*Generated by [CloudSage](https://github.com/your-org/cloudsage) · "
            f"Autonomous Cloud Intelligence Platform*",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point (called from GitHub Actions)
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    changed_files_raw = os.environ.get("PR_CHANGED_FILES", "")
    changed_files = [f.strip() for f in changed_files_raw.split() if f.strip()]
    pr_title = os.environ.get("PR_TITLE", "")
    pr_number = os.environ.get("GITHUB_PR_NUMBER", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")

    if not changed_files:
        print("No changed files detected — skipping risk assessment.")
        sys.exit(0)

    # Initialise scorer (cosmos + rag optional in CI — gracefully degrade)
    cosmos_client = None
    rag = None
    try:
        from database.cosmos_client import CosmosDBClient
        cosmos_client = CosmosDBClient()
    except Exception as e:
        logger.warning(f"Cosmos not available in CI: {e}")

    try:
        from rag.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
    except Exception as e:
        logger.warning(f"RAG not available in CI: {e}")

    scorer = DeploymentRiskScorer(cosmos_client=cosmos_client, rag_pipeline=rag)
    report = scorer.score(changed_files, pr_title=pr_title)
    report_dict = DeploymentRiskScorer.to_dict(report)

    print(json.dumps(report_dict, indent=2))
    print()
    print(report.summary)

    # Post PR comment if GitHub token available
    if github_token and pr_number and repo:
        try:
            import urllib.request
            comment_body = scorer.format_pr_comment(report)
            data = json.dumps({"body": comment_body}).encode()
            req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
                data=data,
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/vnd.github.v3+json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                logger.info(f"PR comment posted: {resp.status}")
        except Exception as e:
            logger.warning(f"Could not post PR comment: {e}")

    # Block deployment if risk too high
    if report.block_deployment:
        print(f"\n⛔ Deployment BLOCKED: risk score {report.overall_score} >= threshold {BLOCK_THRESHOLD}")
        sys.exit(1)

    print(f"\n✅ Deployment approved: risk score {report.overall_score}")
    sys.exit(0)


if __name__ == "__main__":
    main()
