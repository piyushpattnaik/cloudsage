"""
CloudSage — Action Mapper
Routes approved decisions to the correct automation module.
"""

import logging
from typing import Callable

logger = logging.getLogger("ActionMapper")


class ActionMapper:
    """
    Maps a policy-approved action string to the corresponding automation function.
    Supports dynamic registration of custom action handlers.
    """

    def __init__(self):
        # Import here to avoid circular issues at module load time
        from automation.restart_service import restart_service
        from automation.scale_cluster import scale_cluster
        from automation.rollback_deployment import rollback_deployment
        from automation.notify_teams import notify_teams

        self._registry: dict[str, Callable] = {
            "restart_service": restart_service,
            "scale_up": lambda ctx: scale_cluster({**ctx, "direction": "up"}),
            "scale_down": lambda ctx: scale_cluster({**ctx, "direction": "down"}),
            "rollback_deployment": rollback_deployment,
            "notify_teams": notify_teams,
            "enforce_policy": self._enforce_policy,
            "alert_only": self._alert_only,
        }

    def register(self, action_name: str, handler: Callable):
        """Register a custom action handler."""
        self._registry[action_name] = handler
        logger.info(f"Registered custom action handler: {action_name}")

    def dispatch(self, action: str, context: dict) -> dict:
        """
        Dispatch an approved action to its handler.

        Args:
            action: Action string (e.g. 'restart_service')
            context: Full decision context passed to the handler

        Returns:
            dict with execution result
        """
        handler = self._registry.get(action)
        if not handler:
            logger.error(f"No handler registered for action: {action}")
            return {"status": "error", "error": f"Unknown action: {action}"}

        logger.info(f"Dispatching action: {action} | service={context.get('service')}")
        try:
            result = handler(context)
            result["action"] = action
            return result
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return {"status": "error", "action": action, "error": str(e)}

    # ------------------------------------------------------------------
    # Built-in no-op handlers
    # ------------------------------------------------------------------
    @staticmethod
    def _alert_only(context: dict) -> dict:
        logger.info(f"Alert-only action for service={context.get('service')} severity={context.get('severity')}")
        return {"status": "success", "message": "Alert recorded. No automated action taken."}

    @staticmethod
    def _enforce_policy(context: dict) -> dict:
        """Placeholder — real implementation calls Azure Policy REST API."""
        logger.info(f"Enforce policy triggered for resource={context.get('resource')}")
        return {"status": "success", "message": f"Policy enforcement initiated for {context.get('resource')}."}
