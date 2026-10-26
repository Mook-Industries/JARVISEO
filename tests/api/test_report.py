"""성능 리포트(/api/report) — 목업 숫자 대신 팀 기록을 돌려준다."""

import pytest

from jarviseo.api.report import KPIS, LATENCY_STAGES, RESULTS, report
from jarviseo.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(database_url=f"sqlite:///{tmp_path / 'r.db'}")
    s.init_schema()
    return s


def test_empty_db_reports_zero_turns_and_team_results(store):
    r = report(store)
    assert r["turns"] == 0 and r["eval_runs"] == []
    assert [x["stage"] for x in r["latency"]] == [s for s, _, _ in LATENCY_STAGES]
    assert all(x["count"] == 0 for x in r["latency"])
    assert len(r["results"]) == len(RESULTS) and len(r["kpis"]) == len(KPIS)
    assert {"model", "owner", "metric", "now", "goal", "ok", "source"} <= set(r["results"][0])


def test_eval_runs_come_from_team_table(store):
    from jarviseo.memory.models import EvalRun

    with store.session() as db:
        db.add(EvalRun(variant="v2", sample_count=62, metrics={"accuracy": 0.21}, note="R0"))
    runs = report(store)["eval_runs"]
    assert runs[0]["variant"] == "v2" and runs[0]["metrics"] == {"accuracy": 0.21}


def test_api_report_route(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("langgraph")
    from fastapi.testclient import TestClient

    from jarviseo.api.assistant_web import WebAssistant
    from jarviseo.api.main import create_app
    from jarviseo.vlm.client import FakeVLMClient
    from tests.api.test_assistant_web import NoObjects, _NoChecker

    s = MemoryStore(database_url=f"sqlite:///{tmp_path / 'a.db'}")
    s.init_schema()
    user = s.ensure_user()
    web = WebAssistant(
        s, None, user, vlm=FakeVLMClient(["-"]), detector=NoObjects(), photo_dir=None
    )
    with TestClient(create_app(store=s, checker=_NoChecker(), assistant=web)) as c:
        body = c.get("/api/report").json()
    assert body["turns"] == 0 and body["results"]
