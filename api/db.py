import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/predictions.db")


def init_db() -> None:
    """Tạo file DB + bảng nếu chưa có. Gọi 1 lần lúc server khởi động."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(predicted_class: str, confidence: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, predicted_class, confidence) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), predicted_class, confidence),
    )
    conn.commit()
    conn.close()
