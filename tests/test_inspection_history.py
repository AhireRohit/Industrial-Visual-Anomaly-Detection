"""Fast, offline tests for inspection_history.py (no GPU, model, or dataset)."""

from hashlib import sha256

import pytest

import inspection_history as history


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture(autouse=True)
def temporary_history_db(tmp_path, monkeypatch):
    """Never touch the dashboard's real inspection database."""
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "inspection_history.sqlite3")


def save(entries, files, **kwargs):
    return history.save_inspection_run(
        entries=entries,
        files=files,
        elapsed_seconds=0.25,
        mode="single" if len(entries) == 1 else "batch",
        image_threshold=12.0,
        pixel_threshold=10.0,
        **kwargs,
    )


def test_initialize_database_creates_empty_history():
    db = history.initialize_database()
    assert db.exists()
    assert history.get_recent_inspections() == []
    history.initialize_database()  # Idempotent


def test_single_inspection_roundtrip_with_sha256():
    payload = b"example image bytes"
    run_id = save(
        [{"decision": "Anomalous", "score": 14.5}],
        [FakeUpload("example.png", payload)],
        category="cable",
    )

    records = history.get_recent_inspections()
    assert len(records) == 1
    row = records[0]
    assert row["filename"] == "example.png"
    assert row["category"] == "cable"
    assert row["decision"] == "Anomalous"
    assert row["anomaly_score"] == 14.5
    assert row["image_threshold"] == 12.0
    assert row["pixel_threshold"] == 10.0
    assert row["mode"] == "single"
    assert row["timestamp_utc"]

    with history.connect() as connection:
        saved_run_id, image_hash = connection.execute(
            "SELECT run_id, image_hash FROM inspections"
        ).fetchone()
        assert saved_run_id == run_id
        assert image_hash == sha256(payload).hexdigest()
        columns = [column[1] for column in connection.execute("PRAGMA table_info(inspections)")]
        assert "image" not in columns  # Store metadata, not raw uploaded images


def test_batch_history_recent_first_and_limit():
    save(
        [{"decision": "Normal", "score": 4}, {"decision": "Anomalous", "score": 16}],
        [FakeUpload("a.png", b"a"), FakeUpload("b.png", b"b")],
        category="tile",
    )
    save(
        [{"decision": "Normal", "score": 5}],
        [FakeUpload("c.png", b"c")],
        category="metal_nut",
    )

    assert [r["filename"] for r in history.get_recent_inspections()] == ["c.png", "b.png", "a.png"]
    assert [r["filename"] for r in history.get_recent_inspections(limit=2)] == ["c.png", "b.png"]
    assert history.get_recent_inspections(limit=0) == []


@pytest.mark.parametrize(
    "entries, files, extra, message",
    [
        ([{"decision": "Normal", "score": 1}], [FakeUpload("a", b"a")], {"mode": "wrong"}, "Invalid inspection mode"),
        ([{"decision": "Normal", "score": 1}], [FakeUpload("a", b"a")], {"category": "unknown"}, "Unsupported category"),
        ([], [], {}, "Inspection entries/files mismatch"),
        ([{"decision": "Normal", "score": 1}], [], {}, "Inspection entries/files mismatch"),
    ],
)
def test_reject_invalid_requests(entries, files, extra, message):
    arguments = dict(
        entries=entries,
        files=files,
        elapsed_seconds=0.25,
        mode="single",
        image_threshold=12.0,
        pixel_threshold=10.0,
    )
    arguments.update(extra)
    with pytest.raises(ValueError, match=message):
        history.save_inspection_run(**arguments)
    assert not history.DB_PATH.exists()  # Invalid requests must not write records
