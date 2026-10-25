"""Persistent inspection metadata for the industrial anomaly detection dashboard."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import sqlite3
import uuid

DB_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "inspection_history.sqlite3"
)


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspection_runs (
                run_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                mode TEXT NOT NULL,
                model_version TEXT NOT NULL,
                category TEXT NOT NULL,
                total_images INTEGER NOT NULL,
                elapsed_seconds REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                decision TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                image_threshold REAL NOT NULL,
                pixel_threshold REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES inspection_runs(run_id)
            )
        """)
    return DB_PATH


def save_inspection_run(
    entries,
    files,
    elapsed_seconds,
    mode,
    image_threshold,
    pixel_threshold,
    model_version="PatchCore V1",
    category="metal_nut",
):
    if mode not in ("single", "batch"):
        raise ValueError("Invalid inspection mode")
    if category not in ("metal_nut", "cable", "tile"):
        raise ValueError(f"Unsupported category: {category}")
    if not entries or len(entries) != len(files):
        raise ValueError("Inspection entries/files mismatch")

    initialize_database()
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO inspection_runs
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                mode,
                model_version,
                category,
                len(entries),
                float(elapsed_seconds),
            ),
        )
        for entry, file in zip(entries, files):
            image_hash = hashlib.sha256(file.getvalue()).hexdigest()
            conn.execute(
                """
                INSERT INTO inspections (
                    run_id, filename, image_hash,
                    decision, anomaly_score,
                    image_threshold, pixel_threshold
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    file.name,
                    image_hash,
                    entry["decision"],
                    float(entry["score"]),
                    float(image_threshold),
                    float(pixel_threshold),
                ),
            )
    return run_id


def get_recent_inspections(limit=100):
    initialize_database()
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                r.timestamp_utc,
                r.mode,
                r.category,
                r.model_version,
                r.elapsed_seconds,
                i.filename,
                i.decision,
                i.anomaly_score,
                i.image_threshold,
                i.pixel_threshold
            FROM inspections i
            JOIN inspection_runs r ON i.run_id = r.run_id
            ORDER BY i.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    database = initialize_database()
    print("Database initialized successfully!")
    print("Location:", database)
    print("Existing inspection records:", len(get_recent_inspections()))
