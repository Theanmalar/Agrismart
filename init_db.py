import sqlite3
import os
from datetime import datetime

DB_PATH = "backend/database.db"

def init_db():
    os.makedirs("backend", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create farmers table
    c.execute("""
    CREATE TABLE IF NOT EXISTS farmers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT
    )
    """)

    # Create predictions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT,
        image_path TEXT,
        prediction TEXT,
        confidence REAL,
        recommendations TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized at", DB_PATH)

if __name__ == "_main_":
    init_db()