from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Difficulty = Literal["easy", "medium", "hard"]


ACTION_MAP = {
    "check_cpu": "Check current CPU and top process list on the affected node",
    "review_logs": "Review recent deployment logs for abnormal worker count or config changes",
    "scale_out_api": "Scale out API replicas by one and verify latency recovers",
    "inspect_db_metrics": "Inspect application metrics for active DB connections versus configured pool size",
    "identify_long_queries": "Identify long-running queries from database monitoring and capture offending endpoints",
    "raise_pool_size": "Temporarily raise connection pool size within safe database limits",
    "throttle_endpoint": "Apply request throttling for the heaviest endpoint and confirm error rate drops",
    "create_optimization_ticket": "Create follow-up ticket for query optimization with captured evidence",
    "declare_incident": "Declare incident severity and freeze non-essential deployments",
    "verify_standby": "Verify standby cluster health and capacity in secondary region",
    "promote_db_replica": "Promote replicated database read-write role in secondary region",
    "update_routing": "Update global traffic routing to direct production traffic to secondary region",
    "run_synthetic_checks": "Run synthetic checks for API, auth, and billing critical paths",
    "publish_status_update": "Confirm error budget stabilization and publish stakeholder status update",
    "open_recovery_workstream": "Open recovery workstream for controlled failback planning",
    "restart_service": "Restart the affected service and monitor recovery",
    "rollback_deploy": "Rollback the most recent deployment to the previous stable version",
}


class Task(BaseModel):
    id: str
    name: str
    description: str
    difficulty: Difficulty
    steps: list[str] = Field(min_length=1)
    allowed_actions: list[str] = Field(min_length=1)
    max_steps: int = Field(gt=0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("id", "name", "description")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("steps", "allowed_actions")
    @classmethod
    def validate_non_empty_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("list items must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_step_rules(self) -> Task:
        missing_actions = [step for step in self.steps if step not in self.allowed_actions]
        if missing_actions:
            raise ValueError(
                "all steps must be present in allowed_actions; missing: "
                + ", ".join(missing_actions)
            )
        if self.max_steps < len(self.steps):
            raise ValueError("max_steps must be greater than or equal to the number of steps")
        return self


TASK_REGISTRY: dict[str, Task] = {
    "cpu_spike_easy": Task.model_validate({
        "id": "cpu_spike_easy",
        "name": "Investigate CPU Spike on API Node",
        "description": (
            "A single API node reports sustained 95-99% CPU usage after a deployment. "
            "Service remains reachable but latency is elevated."
        ),
        "difficulty": "easy",
        "steps": ["check_cpu", "review_logs", "scale_out_api"],
        "allowed_actions": [
            "check_cpu", "review_logs", "scale_out_api",
            "restart_service", "rollback_deploy",
        ],
        "max_steps": 3,
    }),
    "db_connection_pool_medium": Task.model_validate({
        "id": "db_connection_pool_medium",
        "name": "Stabilize Database Connection Pool Exhaustion",
        "description": (
            "The primary service is returning intermittent 500 errors due to exhausted "
            "database connections during peak traffic."
        ),
        "difficulty": "medium",
        "steps": [
            "inspect_db_metrics",
            "identify_long_queries",
            "raise_pool_size",
            "throttle_endpoint",
            "create_optimization_ticket",
        ],
        "allowed_actions": [
            "inspect_db_metrics",
            "identify_long_queries",
            "raise_pool_size",
            "throttle_endpoint",
            "create_optimization_ticket",
            "check_cpu",
            "restart_service",
            "review_logs",
        ],
        "max_steps": 5,
    }),
    "k8s_region_outage_hard": Task.model_validate({
        "id": "k8s_region_outage_hard",
        "name": "Handle Regional Kubernetes Control Plane Outage",
        "description": (
            "A regional control plane outage prevents scheduling in the primary cluster. "
            "Customer traffic must be shifted with minimal downtime."
        ),
        "difficulty": "hard",
        "steps": [
            "declare_incident",
            "verify_standby",
            "promote_db_replica",
            "update_routing",
            "run_synthetic_checks",
            "publish_status_update",
            "open_recovery_workstream",
        ],
        "allowed_actions": [
            "declare_incident",
            "verify_standby",
            "promote_db_replica",
            "update_routing",
            "run_synthetic_checks",
            "publish_status_update",
            "open_recovery_workstream",
            "scale_out_api",
            "throttle_endpoint",
        ],
        "max_steps": 7,
    }),
}


def get_task(task_id: str) -> Task:
    return TASK_REGISTRY[task_id]


def list_tasks() -> list[Task]:
    return [TASK_REGISTRY[task_id] for task_id in sorted(TASK_REGISTRY)]
