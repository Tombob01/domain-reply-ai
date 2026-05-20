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
        # ── Phase 2 tables — angle / objection / prospect reply memory ──────────
        # All additive. Existing tables and methods are untouched.
        # Rollback: DROP TABLE angle_log; DROP TABLE objection_log;
        #           DROP TABLE prospect_reply_log.  Nothing else changes.

        c.execute("""
            CREATE TABLE IF NOT EXISTS angle_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,

                -- which lead and outreach this angle belongs to
                lead_id          INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
                outreach_id      INTEGER REFERENCES outreach_log(id) ON DELETE SET NULL,

                -- controlled-vocabulary angle identifier (from _ANGLE_REGISTRY in angle_memory.py)
                -- e.g. "seo_benefit", "competitor_risk", "brand_protection"
                angle_id         TEXT    NOT NULL,

                -- which numbered outreach was this? 1-based counter for easy ordering
                -- allows "angle X was used in email #2" without joining outreach_log
                outreach_seq     INTEGER NOT NULL DEFAULT 1,

                -- how prominently was this angle used in the email?
                -- "primary"   = the email's main persuasion hook
                -- "secondary" = mentioned but not the lead argument
                -- "mentioned" = incidentally referenced (e.g. answering a question)
                pitched_as       TEXT    NOT NULL DEFAULT 'primary'
                                     CHECK(pitched_as IN ('primary','secondary','mentioned')),

                -- did the prospect send a reply after the email that contained this angle?
                -- NULL = unknown (reply not yet logged or not tracked)
                -- populated by Phase 2 when prospect_reply_log is written
                prospect_replied INTEGER,          -- 0/1/NULL boolean

                -- aggregated sentiment of the prospect's reply, if known
                -- NULL = no reply yet logged
                reply_sentiment  TEXT
                                     CHECK(reply_sentiment IN
                                           ('positive','neutral','negative','no_reply', NULL)),

                logged_at        REAL    NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS objection_log (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,

                lead_id               INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,

                -- controlled vocabulary — from _OBJECTION_REGISTRY in angle_memory.py
                -- e.g. "price_too_high", "have_website", "not_now", "trust_concern"
                objection_type        TEXT    NOT NULL,

                -- raw snippet (max 200 chars) of the prospect's message that triggered this
                -- used for debugging and as context for Phase 2's reply instruction
                source_snippet        TEXT,

                -- has this objection been addressed in a subsequent reply?
                -- starts FALSE; Phase 2 sets TRUE when an addressing reply is logged
                addressed             INTEGER NOT NULL DEFAULT 0,   -- 0=false, 1=true
                addressed_at          REAL,                          -- timestamp when addressed
                addressed_outreach_id INTEGER REFERENCES outreach_log(id) ON DELETE SET NULL,

                detected_at           REAL    NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS prospect_reply_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,

                lead_id           INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,

                -- which of our outreach emails is this prospect replying to?
                -- NULL if reply arrived outside tracked outreach context
                in_reply_to_outreach_id INTEGER REFERENCES outreach_log(id) ON DELETE SET NULL,

                -- raw prospect message body
                body              TEXT    NOT NULL,

                -- lightweight engagement signal — proxy for interest level
                word_count        INTEGER NOT NULL DEFAULT 0,

                -- did the prospect ask at least one question in this reply?
                -- TRUE = prospect wants more information (positive engagement)
                has_questions     INTEGER NOT NULL DEFAULT 0,   -- 0/1 boolean

                -- controlled-vocabulary sentiment label
                -- populated by caller; never inferred automatically in Phase 1
                -- "positive"  = expressed interest, asked to proceed, positive language
                -- "neutral"   = factual / non-committal
                -- "negative"  = objection, rejection, frustration
                -- "no_reply"  = sentinel for when we log that no reply came (future use)
                sentiment         TEXT    CHECK(sentiment IN
                                                ('positive','neutral','negative','no_reply', NULL)),

                received_at       REAL    NOT NULL
            )
        """)

        # Indexes for common lookups
        c.execute("CREATE INDEX IF NOT EXISTS idx_leads_domain ON leads(domain)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_outreach_lead ON outreach_log(lead_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_offer_lead ON offer_log(lead_id)")
        # Phase 2 indexes — angle and objection lookups will be hot paths
        c.execute("CREATE INDEX IF NOT EXISTS idx_angle_lead     ON angle_log(lead_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_angle_id       ON angle_log(angle_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_objection_lead ON objection_log(lead_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_objection_type ON objection_log(objection_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reply_lead     ON prospect_reply_log(lead_id)")

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


    # ── Phase 2 methods — angle / objection / prospect reply ─────────────────
    # All additive. All non-blocking: return False / [] on failure rather than
    # raising. Pattern matches the existing log_outreach / log_offer methods.

    # ── angle_log ─────────────────────────────────────────────────────────────

    def log_angle(
        self,
        lead_id:      int,
        angle_id:     str,
        outreach_seq: int            = 1,
        pitched_as:   str            = "primary",   # primary | secondary | mentioned
        outreach_id:  Optional[int]  = None,
    ) -> bool:
        """
        Record which angle (value proposition) was used in an outreach email.

        Call once per distinct angle per outreach.  For an email that leads with
        SEO and also mentions competitor risk as a secondary point, call twice:
            log_angle(lead_id, "seo_benefit",    outreach_seq=2, pitched_as="primary")
            log_angle(lead_id, "competitor_risk", outreach_seq=2, pitched_as="secondary")

        outreach_seq — 1-based counter for this lead.  Pass len(outreach_history)+1
                       at the moment of logging.  Used by Phase 2 to order angles
                       without joining outreach_log.
        outreach_id  — optional FK to outreach_log.id; lets Phase 2 cross-reference
                       the full email body when needed.

        Returns True on success, False if DB unavailable or insert fails.
        Rollback: this method and its table can be removed with no other changes.
        """
        if not self.available:
            return False
        if pitched_as not in ("primary", "secondary", "mentioned"):
            pitched_as = "primary"
        try:
            self._conn.execute(
                """
                INSERT INTO angle_log
                    (lead_id, outreach_id, angle_id, outreach_seq, pitched_as, logged_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (lead_id, outreach_id, angle_id, outreach_seq, pitched_as, self._ts()),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] log_angle error: {e}")
            return False

    def get_angle_history(self, lead_id: int) -> list[dict]:
        """
        Return all angle records for a lead, oldest first.

        Each record contains:
          angle_id, outreach_seq, pitched_as, prospect_replied, reply_sentiment, logged_at

        Phase 2 uses this to build AngleInventory: which angles have been used,
        how many times each was used, and which generated a positive reply.
        Returns [] if DB unavailable or lead not found.
        """
        if not self.available:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT id, lead_id, outreach_id, angle_id, outreach_seq,
                       pitched_as, prospect_replied, reply_sentiment, logged_at
                FROM   angle_log
                WHERE  lead_id = ?
                ORDER  BY outreach_seq ASC, logged_at ASC
                """,
                (lead_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] get_angle_history error: {e}")
            return []

    # ── objection_log ─────────────────────────────────────────────────────────

    def log_objection(
        self,
        lead_id:        int,
        objection_type: str,
        source_snippet: Optional[str] = None,
    ) -> Optional[int]:
        """
        Record a structured objection detected in a prospect's message.

        objection_type  — controlled vocabulary from _OBJECTION_REGISTRY:
                          "price_too_high" | "have_website" | "not_now" |
                          "no_budget" | "trust_concern" | "need_approval" |
                          "competitor_preferred" | "not_relevant"
        source_snippet  — raw text extract (max 200 chars) that triggered the detection.
                          Stored as-is; never used for prompt injection in Phase 1.

        Returns the new objection_log.id on success, None on failure.
        The id is useful if the caller later wants to mark the objection as addressed
        via mark_objection_addressed().

        Does NOT update lead stage — objections are informational in Phase 1.
        Rollback: this method and its table can be removed with no other changes.
        """
        if not self.available:
            return None
        # Truncate snippet to avoid runaway storage
        if source_snippet and len(source_snippet) > 200:
            source_snippet = source_snippet[:197] + "..."
        try:
            cur = self._conn.execute(
                """
                INSERT INTO objection_log
                    (lead_id, objection_type, source_snippet, addressed, detected_at)
                VALUES (?, ?, ?, 0, ?)
                """,
                (lead_id, objection_type, source_snippet, self._ts()),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] log_objection error: {e}")
            return None

    def get_objection_history(self, lead_id: int, unresolved_only: bool = False) -> list[dict]:
        """
        Return objection records for a lead, newest first.

        unresolved_only=True  → only objections where addressed=0.
                                 Used by Phase 2 to surface active objections.
        unresolved_only=False → full objection history including addressed ones.

        Returns [] if DB unavailable or no objections recorded.
        """
        if not self.available:
            return []
        try:
            query = """
                SELECT id, lead_id, objection_type, source_snippet,
                       addressed, addressed_at, addressed_outreach_id, detected_at
                FROM   objection_log
                WHERE  lead_id = ?
            """
            params: tuple = (lead_id,)
            if unresolved_only:
                query += " AND addressed = 0"
            query += " ORDER BY detected_at DESC"
            rows = self._conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] get_objection_history error: {e}")
            return []

    def mark_objection_addressed(
        self,
        objection_id:  int,
        outreach_id:   Optional[int] = None,
    ) -> bool:
        """
        Mark a previously logged objection as addressed.

        Called by Phase 2 after an outreach email is generated that specifically
        responds to the objection.  In Phase 1 this is never called — it is
        provided here so the schema is complete and the method signature is stable.

        outreach_id — optional FK to the outreach that handled this objection.
        Returns True on success, False on failure.
        """
        if not self.available:
            return False
        try:
            self._conn.execute(
                """
                UPDATE objection_log
                SET    addressed = 1,
                       addressed_at = ?,
                       addressed_outreach_id = ?
                WHERE  id = ?
                """,
                (self._ts(), outreach_id, objection_id),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] mark_objection_addressed error: {e}")
            return False

    # ── prospect_reply_log ────────────────────────────────────────────────────

    def log_prospect_reply(
        self,
        lead_id:                  int,
        body:                     str,
        sentiment:                Optional[str]  = None,   # positive|neutral|negative|no_reply
        has_questions:            bool           = False,
        in_reply_to_outreach_id:  Optional[int]  = None,
    ) -> Optional[int]:
        """
        Record a prospect's inbound reply message.

        This is the critical missing link in Phase 1 — the system previously stored
        what we *sent* but not what the prospect *said back*.  This record enables
        Phase 2 to answer: "did the SEO angle generate engagement?"

        body                     — raw prospect message text.
        sentiment                — caller-supplied sentiment label.  In Phase 1 this is
                                   passed in from the endpoint when the broker describes
                                   the situation (e.g. "they seemed interested" → "positive").
                                   Phase 2 will infer sentiment from intent detection.
        has_questions            — True if the prospect's message contains questions.
                                   Caller computes this; in Phase 1 it can be
                                   approximated as "?" in body.
        in_reply_to_outreach_id  — FK to outreach_log.id; links this reply to the
                                   specific email it is responding to.

        Returns new prospect_reply_log.id on success, None on failure.
        Rollback: this method and its table can be removed with no other changes.
        """
        if not self.available:
            return None
        if sentiment and sentiment not in ("positive", "neutral", "negative", "no_reply"):
            sentiment = None
        word_count = len(body.split()) if body else 0
        has_q_int  = 1 if has_questions else 0
        try:
            cur = self._conn.execute(
                """
                INSERT INTO prospect_reply_log
                    (lead_id, in_reply_to_outreach_id, body, word_count,
                     has_questions, sentiment, received_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (lead_id, in_reply_to_outreach_id, body,
                 word_count, has_q_int, sentiment, self._ts()),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] log_prospect_reply error: {e}")
            return None

    def get_prospect_replies(self, lead_id: int) -> list[dict]:
        """
        Return all prospect reply records for a lead, newest first.

        Each record includes body, word_count, has_questions, sentiment,
        in_reply_to_outreach_id, and received_at.

        Phase 2 uses this to:
        - compute per-angle reply rates (join with angle_log on outreach_id)
        - surface prospect questions that were never answered
        - determine whether the lead has been engaging at all

        Returns [] if DB unavailable or no replies recorded.
        """
        if not self.available:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT id, lead_id, in_reply_to_outreach_id, body,
                       word_count, has_questions, sentiment, received_at
                FROM   prospect_reply_log
                WHERE  lead_id = ?
                ORDER  BY received_at DESC
                """,
                (lead_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] get_prospect_replies error: {e}")
            return []


# Module-level singleton — imported by main.py
memory_db = MemoryDB()
