"""
broker_memory.py — Broker Lead Memory System
=============================================
SQLite-backed storage for lead history, outreach attempts, and offer tracking.

Tables:
  leads        — one row per lead (domain + prospect contact)
  outreach_log — one row per email sent/generated
  offer_log    — one row per offer or counteroffer

All changes are ADDITIVE — existing API contracts are preserved.
Falls back gracefully if the DB cannot be initialised.

Usage:
    from broker_memory import MemoryDB
    db = MemoryDB()                          # init / open
    lead_id = db.upsert_lead("LondonPlumber.com", "john@example.com", "John")
    db.log_outreach(lead_id, "cold_outreach", "Subject line", "Email body")
    db.log_offer(lead_id, 1500, "sent", notes="Initial ask")
    summary = db.lead_summary(lead_id)       # inject into prompt
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

DB_FILE = Path(__file__).parent / "broker_memory.db"


class MemoryDB:
    """
    Thin wrapper around SQLite for broker lead memory.
    Thread-safe for FastAPI's async handlers via check_same_thread=False.
    All public methods return plain Python dicts/lists — no SQLite types leak out.
    """

    def __init__(self, db_path: Path = DB_FILE):
        self._path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init(self) -> None:
        """Create tables if they don't exist. Safe to call multiple times."""
        try:
            self._conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_tables()
            self._conn.commit()
        except Exception as e:
            print(f"[MemoryDB] Failed to initialise: {e}")
            self._conn = None

    def _create_tables(self) -> None:
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                domain        TEXT NOT NULL,
                prospect_email TEXT,
                prospect_name  TEXT,
                notes          TEXT,
                stage          TEXT DEFAULT 'new',
                created_at     REAL NOT NULL,
                updated_at     REAL NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS outreach_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id    INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
                preset     TEXT,
                subject    TEXT,
                body       TEXT,
                sent_at    REAL NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS offer_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id    INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
                amount     REAL NOT NULL,
                direction  TEXT NOT NULL CHECK(direction IN ('sent','received')),
                notes      TEXT,
                offered_at REAL NOT NULL
            )
        """)
        # Indexes for common lookups
        c.execute("CREATE INDEX IF NOT EXISTS idx_leads_domain ON leads(domain)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_outreach_lead ON outreach_log(lead_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_offer_lead ON offer_log(lead_id)")

    @property
    def available(self) -> bool:
        return self._conn is not None

    def _ts(self) -> float:
        return time.time()

    # ── Lead CRUD ─────────────────────────────────────────────────────────────

    def upsert_lead(
        self,
        domain: str,
        prospect_email: Optional[str] = None,
        prospect_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[int]:
        """
        Create a new lead or return the existing lead_id if same domain+email combo.
        Returns lead_id on success, None if DB unavailable.
        """
        if not self.available:
            return None
        try:
            now = self._ts()
            # Check for existing lead with same domain + email
            row = self._conn.execute(
                "SELECT id FROM leads WHERE domain=? AND (prospect_email=? OR prospect_email IS NULL)",
                (domain, prospect_email),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE leads SET updated_at=?, notes=COALESCE(?,notes) WHERE id=?",
                    (now, notes, row["id"]),
                )
                self._conn.commit()
                return row["id"]
            cur = self._conn.execute(
                "INSERT INTO leads (domain, prospect_email, prospect_name, notes, stage, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'new', ?, ?)",
                (domain, prospect_email, prospect_name, notes, now, now),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] upsert_lead error: {e}")
            return None

    def get_lead(self, lead_id: int) -> Optional[dict]:
        """Return lead dict or None."""
        if not self.available:
            return None
        try:
            row = self._conn.execute(
                "SELECT * FROM leads WHERE id=?", (lead_id,)
            ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            print(f"[MemoryDB] get_lead error: {e}")
            return None

    def list_leads(self, domain: Optional[str] = None) -> list[dict]:
        """List all leads, optionally filtered by domain substring."""
        if not self.available:
            return []
        try:
            if domain:
                rows = self._conn.execute(
                    "SELECT * FROM leads WHERE domain LIKE ? ORDER BY updated_at DESC",
                    (f"%{domain}%",),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM leads ORDER BY updated_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] list_leads error: {e}")
            return []

    def update_lead_stage(self, lead_id: int, stage: str) -> bool:
        """Update the conversation stage of a lead."""
        if not self.available:
            return False
        try:
            self._conn.execute(
                "UPDATE leads SET stage=?, updated_at=? WHERE id=?",
                (stage, self._ts(), lead_id),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] update_lead_stage error: {e}")
            return False

    def delete_lead(self, lead_id: int) -> bool:
        """Delete lead and all associated logs (CASCADE)."""
        if not self.available:
            return False
        try:
            self._conn.execute("DELETE FROM leads WHERE id=?", (lead_id,))
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] delete_lead error: {e}")
            return False

    # ── Outreach log ──────────────────────────────────────────────────────────

    def log_outreach(
        self,
        lead_id: int,
        preset: Optional[str] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
    ) -> bool:
        """Record an outreach attempt for a lead."""
        if not self.available:
            return False
        try:
            self._conn.execute(
                "INSERT INTO outreach_log (lead_id, preset, subject, body, sent_at) VALUES (?,?,?,?,?)",
                (lead_id, preset, subject, body, self._ts()),
            )
            self._conn.execute(
                "UPDATE leads SET updated_at=?, stage='contacted' WHERE id=?",
                (self._ts(), lead_id),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] log_outreach error: {e}")
            return False

    def get_outreach_history(self, lead_id: int) -> list[dict]:
        """Return all outreach attempts for a lead, newest first."""
        if not self.available:
            return []
        try:
            rows = self._conn.execute(
                "SELECT * FROM outreach_log WHERE lead_id=? ORDER BY sent_at DESC",
                (lead_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] get_outreach_history error: {e}")
            return []

    # ── Offer log ─────────────────────────────────────────────────────────────

    def log_offer(
        self,
        lead_id: int,
        amount: float,
        direction: str,           # 'sent' (your ask) | 'received' (their offer)
        notes: Optional[str] = None,
    ) -> bool:
        """Record an offer or counteroffer."""
        if not self.available:
            return False
        if direction not in ("sent", "received"):
            return False
        try:
            self._conn.execute(
                "INSERT INTO offer_log (lead_id, amount, direction, notes, offered_at) VALUES (?,?,?,?,?)",
                (lead_id, amount, direction, notes, self._ts()),
            )
            self._conn.execute(
                "UPDATE leads SET updated_at=?, stage='negotiating' WHERE id=?",
                (self._ts(), lead_id),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] log_offer error: {e}")
            return False

    def get_offer_history(self, lead_id: int) -> list[dict]:
        """Return all offers for a lead, newest first."""
        if not self.available:
            return []
        try:
            rows = self._conn.execute(
                "SELECT * FROM offer_log WHERE lead_id=? ORDER BY offered_at DESC",
                (lead_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] get_offer_history error: {e}")
            return []

    # ── Prompt context builder ─────────────────────────────────────────────────

    def lead_summary(self, lead_id: int) -> Optional[str]:
        """
        Build a plain-text summary of a lead's history — ready to inject
        into a prompt as 'LEAD HISTORY'.

        Returns None if lead not found or DB unavailable.
        """
        if not self.available:
            return None
        lead = self.get_lead(lead_id)
        if not lead:
            return None

        import datetime

        def _fmt(ts: float) -> str:
            return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y")

        lines = [
            f"Domain: {lead['domain']}",
            f"Prospect: {lead.get('prospect_name') or 'Unknown'}"
            + (f" <{lead['prospect_email']}>" if lead.get("prospect_email") else ""),
            f"Stage: {lead.get('stage', 'new')}",
        ]
        if lead.get("notes"):
            lines.append(f"Notes: {lead['notes']}")

        outreach = self.get_outreach_history(lead_id)
        if outreach:
            lines.append(f"\nOutreach attempts ({len(outreach)} total):")
            for o in outreach[:3]:   # show last 3 max
                preset_label = o.get("preset") or "general"
                lines.append(
                    f"  - {_fmt(o['sent_at'])}: {preset_label}"
                    + (f" — \"{o['subject']}\"" if o.get("subject") else "")
                )
            if len(outreach) > 3:
                lines.append(f"  ... and {len(outreach) - 3} earlier attempts")

        offers = self.get_offer_history(lead_id)
        if offers:
            lines.append(f"\nOffer history ({len(offers)} total):")
            for o in offers[:5]:   # show last 5
                direction = "You asked" if o["direction"] == "sent" else "They offered"
                amount = f"${o['amount']:,.0f}" if o["amount"] >= 1 else f"${o['amount']:.2f}"
                note = f" ({o['notes']})" if o.get("notes") else ""
                lines.append(f"  - {_fmt(o['offered_at'])}: {direction} {amount}{note}")

        return "\n".join(lines)

    def full_history(self, lead_id: int) -> dict:
        """Return full structured history for API responses."""
        lead = self.get_lead(lead_id) or {}
        return {
            "lead":     lead,
            "outreach": self.get_outreach_history(lead_id),
            "offers":   self.get_offer_history(lead_id),
        }


# Module-level singleton — imported by main.py
memory_db = MemoryDB()
