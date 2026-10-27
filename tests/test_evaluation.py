"""Tests for evaluation datasets, schema integrity, and evaluation metrics."""

import json
from pathlib import Path
import pytest

from asterline.config.settings import Settings
from evaluation.evaluators import (
    RoutingEvaluator,
    MemoryEvaluator,
)
from asterline.memory.store import SQLiteStore
from asterline.memory.service import MemoryService


@pytest.fixture
def settings():
    return Settings()


def test_research_dataset_schema(settings):
    dataset_path = settings.resolve_path(Path("evaluation/datasets/research.jsonl"))
    assert dataset_path.exists(), "research.jsonl dataset must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 20, f"Expected at least 20 research cases, found {len(lines)}"

    required_keys = {"id", "query", "expected_chunk_ids", "category", "rationale", "should_abstain"}
    for idx, line in enumerate(lines):
        data = json.loads(line)
        missing = required_keys - set(data.keys())
        assert not missing, f"Line {idx} missing required keys: {missing}"
        assert isinstance(data["id"], str)
        assert isinstance(data["query"], str)
        assert len(data["query"]) > 10
        assert isinstance(data["expected_chunk_ids"], list)
        assert isinstance(data["should_abstain"], bool)


def test_routing_dataset_schema(settings):
    dataset_path = settings.resolve_path(Path("evaluation/datasets/routing.jsonl"))
    assert dataset_path.exists(), "routing.jsonl dataset must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 20, f"Expected at least 20 routing cases, found {len(lines)}"

    valid_routes = {"conversational", "clarification", "memory_ops", "factual", "investigation"}
    for idx, line in enumerate(lines):
        data = json.loads(line)
        assert "id" in data
        assert "query" in data
        assert "expected_route" in data
        assert data["expected_route"] in valid_routes, f"Invalid route: {data['expected_route']}"
        assert "rationale" in data


def test_memory_dataset_schema(settings):
    dataset_path = settings.resolve_path(Path("evaluation/datasets/memory.jsonl"))
    assert dataset_path.exists(), "memory.jsonl dataset must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 10, f"Expected at least 10 memory scenarios, found {len(lines)}"

    valid_actions = {"create", "forget", "update"}
    for idx, line in enumerate(lines):
        data = json.loads(line)
        assert "id" in data
        assert "name" in data
        assert "user_id" in data
        assert "action" in data
        assert data["action"] in valid_actions


def test_routing_evaluator_deterministic():
    evaluator = RoutingEvaluator()
    sample_cases = [
        {"id": "r1", "query": "hello there!", "expected_route": "conversational"},
        {"id": "r2", "query": "?", "expected_route": "clarification"},
        {"id": "r3", "query": "remember that I use BigQuery", "expected_route": "memory_ops"},
        {"id": "r4", "query": "What is the query timeout for AQL?", "expected_route": "factual"},
        {"id": "r5", "query": "compare Launch and Team pricing", "expected_route": "investigation"},
    ]
    results = evaluator.evaluate(sample_cases)
    assert results["correct_routing_rate"] == 1.0
    for r in ("conversational", "clarification", "memory_ops", "factual", "investigation"):
        assert results["per_route_accuracy"][r] == 1.0


def test_memory_evaluator_isolation_and_forget(tmp_path):
    store = SQLiteStore(tmp_path / "test_eval_memory.db")
    service = MemoryService(store)
    evaluator = MemoryEvaluator(service)

    test_scenarios = [
        {
            "id": "iso_01",
            "name": "create_test",
            "user_id": "alice",
            "action": "create",
            "content": "I prefer dark mode in UI",
            "category": "preference",
            "expected_status": "accepted",
            "isolation_check_users": ["bob"],
        },
        {
            "id": "iso_02",
            "name": "forget_test",
            "user_id": "alice",
            "action": "forget",
            "target_content": "dark mode",
            "expected_status": "purged",
        },
    ]

    report = evaluator.evaluate(test_scenarios)
    assert report["memory_operation_success_rate"] == 1.0
    assert report["cross_user_isolation_violations"] == 0
    assert report["forget_purge_rate"] == 1.0
