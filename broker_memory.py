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
            self._migrate()
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

        # ── Phase 3a tables — outcome capture ─────────────────────────────────
        # All CREATE TABLE IF NOT EXISTS — safe to add to existing DB files.
        # Rollback: DROP TABLE strategy_outcome_log; DROP TABLE
        #           conversion_event_log; DROP TABLE subject_effectiveness_log.
        # No existing tables or columns are altered.

        c.execute("""
            CREATE TABLE IF NOT EXISTS strategy_outcome_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,

                lead_id          INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,

                -- FK to outreach_log — NULL in Phase 3a because the generate-reply
                -- flow does not have access to outreach_log.id at generation time.
                -- Populated in a future phase when the frontend back-links the ID.
                outreach_id      INTEGER REFERENCES outreach_log(id) ON DELETE SET NULL,

                -- Strategy decisions recorded at generation time.
                -- All TEXT/INTEGER — no computed values, no aggregates.
                selected_angle   TEXT,
                goal             TEXT,
                buyer_state      TEXT,
                cta_style        TEXT,
                tone_posture     TEXT,
                reply_length     TEXT,
                persuasion_level INTEGER,
                urgency_level    INTEGER,
                outreach_seq     INTEGER,

                -- Outcome fields — written by backfill after reply arrives.
                -- NULL = outcome not yet known.
                got_reply        INTEGER,          -- 0/1 boolean
                reply_sentiment  TEXT,             -- positive|neutral|negative|no_reply
                reply_latency_s  REAL,             -- seconds between sent and reply

                decided_at       REAL NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS conversion_event_log (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,

                lead_id              INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,

                -- Terminal outcome label.
                -- accepted | rejected | unsubscribed | stalled_closed
                event_type           TEXT NOT NULL
                                         CHECK(event_type IN
                                               ('accepted','rejected',
                                                'unsubscribed','stalled_closed')),

                -- Snapshot of lead at the moment of conversion.
                final_stage          TEXT,
                total_outreach_count INTEGER,
                total_offer_count    INTEGER,

                -- Seconds from first outreach to this event.
                -- NULL if first outreach timestamp unavailable.
                time_to_resolution   REAL,

                notes                TEXT,
                recorded_at          REAL NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS subject_effectiveness_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,

                lead_id         INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
                outreach_id     INTEGER REFERENCES outreach_log(id) ON DELETE SET NULL,

                -- SHA1[:8] of normalised (lowercase, stripped) subject.
                -- Groups similar subjects without storing verbatim text.
                subject_hash    TEXT NOT NULL,

                -- First 60 chars of the actual subject — for debugging only.
                subject_preview TEXT,

                -- Outcome — NULL until reply status is known.
                -- Populated by backfill_angle_reply_data() or direct update.
                got_reply       INTEGER,    -- 0/1/NULL boolean
                reply_sentiment TEXT,

                sent_at         REAL NOT NULL
            )
        """)

        # ── Operational Reliability table ────────────────────────────────────
        # analytics_snapshot_log — point-in-time snapshots of analytics state.
        # Used for baseline establishment, regression detection, and audit trail.
        # Rollback: DROP TABLE analytics_snapshot_log.  Nothing else changes.
        c.execute("""
            CREATE TABLE IF NOT EXISTS analytics_snapshot_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Type label for the snapshot.
                -- "angle_performance" | "cta_performance" | "conversion_funnel"
                -- | "attribution_integrity" | "score_safety" | "full_system"
                snapshot_type    TEXT    NOT NULL,

                -- Full JSON blob of the analytics state at capture time.
                -- Stored verbatim — never modified after write.
                snapshot_json    TEXT    NOT NULL,

                -- Schema version tag so future readers know how to parse
                -- the JSON (format may evolve between phases).
                snapshot_version TEXT    NOT NULL DEFAULT 'v1',

                -- Human-readable label for the snapshot (e.g. "pre-Phase-3b baseline")
                label            TEXT,

                -- Who triggered the snapshot: "operator" | "automated" | "test"
                triggered_by     TEXT    DEFAULT 'operator',

                created_at       REAL    NOT NULL
            )
        """)

        # ── analyst_correction_log — audit trail for human corrections ────────
        # Every operator correction is logged here before being applied.
        # Corrections are never silent — they are always traceable.
        c.execute("""
            CREATE TABLE IF NOT EXISTS analyst_correction_log (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,

                -- What type of correction was made.
                -- "relink_reply"       — changed in_reply_to_outreach_id
                -- "override_confidence"— changed attribution_confidence
                -- "invalidate_mapping" — marked a reply or outcome as invalid
                -- "merge_reply_chains" — merged two reply chains into one
                correction_type      TEXT    NOT NULL,

                -- Target table and row that was corrected
                target_table         TEXT    NOT NULL,
                target_row_id        INTEGER NOT NULL,

                -- JSON snapshot of the row BEFORE correction (immutable record)
                before_json          TEXT    NOT NULL,

                -- JSON snapshot of the row AFTER correction
                after_json           TEXT    NOT NULL,

                -- Reason supplied by the operator
                reason               TEXT,

                -- The event_id of the row that was corrected (for cross-referencing)
                target_event_id      TEXT,

                corrected_at         REAL    NOT NULL
            )
        """)

        # ── frontend_idempotency_log — deduplication registry ─────────────────
        # Prevents duplicate outreach/reply submissions from the frontend.
        # Keyed on client-generated idempotency_key.
        c.execute("""
            CREATE TABLE IF NOT EXISTS frontend_idempotency_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Client-supplied idempotency key (UUID recommended)
                idempotency_key  TEXT    NOT NULL UNIQUE,

                -- What was deduplicated
                operation_type   TEXT    NOT NULL,   -- "outreach" | "reply" | "offer"
                lead_id          INTEGER,
                result_row_id    INTEGER,            -- the ID of the row that was created

                -- Is this key still valid or has it been invalidated?
                invalidated      INTEGER NOT NULL DEFAULT 0,   -- 0/1

                created_at       REAL    NOT NULL
            )
        """)

        # Phase 3a indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_outcome_lead    ON strategy_outcome_log(lead_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_outcome_angle   ON strategy_outcome_log(selected_angle)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_conversion_lead ON conversion_event_log(lead_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_subject_hash    ON subject_effectiveness_log(subject_hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_subject_lead    ON subject_effectiveness_log(lead_id)")

        # Operational Reliability indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_type   ON analytics_snapshot_log(snapshot_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_ts     ON analytics_snapshot_log(created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_correction_type ON analyst_correction_log(correction_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_correction_row  ON analyst_correction_log(target_row_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_idempotency_key ON frontend_idempotency_log(idempotency_key)")

    def _migrate(self) -> None:
        """
        Additive schema migrations for existing database files.
        Uses ALTER TABLE ADD COLUMN — safe for SQLite; silently ignored if
        column already exists (caught by the broad except).

        Phase QC migrations:
          outreach_log           → event_id TEXT
          prospect_reply_log     → event_id TEXT, attribution_confidence REAL
          strategy_outcome_log   → event_id TEXT, attribution_confidence REAL

        Rollback: columns are nullable with no defaults — dropping them is not
        possible in SQLite but they impose zero cost when NULL.
        """
        migrations = [
            # (table, column, definition)
            ("outreach_log",         "event_id",               "TEXT"),
            ("prospect_reply_log",   "event_id",               "TEXT"),
            ("prospect_reply_log",   "attribution_confidence", "REAL"),
            ("strategy_outcome_log", "event_id",               "TEXT"),
            ("strategy_outcome_log", "attribution_confidence", "REAL"),
        ]
        for table, col, defn in migrations:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
            except Exception:
                pass  # column already exists — safe to ignore
        self._conn.commit()

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
            import uuid as _uuid
            _event_id = str(_uuid.uuid4())
            # attribution_confidence: 1.0 when outreach_id supplied (direct link),
            # 0.5 when inferred (same lead, no explicit outreach link),
            # NULL when completely unlinked.
            _attr_conf: Optional[float] = None
            if in_reply_to_outreach_id is not None:
                _attr_conf = 1.0
            elif lead_id is not None:
                _attr_conf = 0.5
            cur = self._conn.execute(
                """
                INSERT INTO prospect_reply_log
                    (lead_id, in_reply_to_outreach_id, body, word_count,
                     has_questions, sentiment, received_at,
                     event_id, attribution_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (lead_id, in_reply_to_outreach_id, body,
                 word_count, has_q_int, sentiment, self._ts(),
                 _event_id, _attr_conf),
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


    # ── Phase 3a methods — outcome capture ───────────────────────────────────
    # All additive. All non-blocking. All follow the existing pattern:
    #   try/except, return False/None on failure, print bracketed prefix.
    # Rollback: delete these methods; drop the three Phase 3a tables.
    # No existing method is modified.

    # ── log_outreach_with_id ──────────────────────────────────────────────────

    def log_outreach_with_id(
        self,
        lead_id: int,
        preset:  Optional[str] = None,
        subject: Optional[str] = None,
        body:    Optional[str] = None,
    ) -> Optional[int]:
        """
        Identical to log_outreach() but returns the new outreach_log.id
        instead of a bool.  Used by Phase 3a to link strategy_outcome_log
        and subject_effectiveness_log to the specific outreach row.

        The original log_outreach() is UNCHANGED — existing call sites keep
        working without modification.  This is a new method, not a replacement.

        Returns outreach_log.id on success, None if DB unavailable or insert fails.
        """
        if not self.available:
            return None
        try:
            import uuid as _uuid
            now      = self._ts()
            event_id = str(_uuid.uuid4())
            cur = self._conn.execute(
                "INSERT INTO outreach_log (lead_id, preset, subject, body, sent_at, event_id) VALUES (?,?,?,?,?,?)",
                (lead_id, preset, subject, body, now, event_id),
            )
            self._conn.execute(
                "UPDATE leads SET updated_at=?, stage='contacted' WHERE id=?",
                (now, lead_id),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] log_outreach_with_id error: {e}")
            return None

    # ── log_strategy_outcome ──────────────────────────────────────────────────

    def log_strategy_outcome(
        self,
        lead_id:         int,
        goal:            str,
        buyer_state:     str,
        cta_style:       str,
        tone_posture:    str,
        reply_length:    str,
        persuasion_level: int,
        urgency_level:   int,
        outreach_seq:    int,
        selected_angle:  Optional[str] = None,
        outreach_id:     Optional[int] = None,   # NULL in Phase 3a; back-linked later
    ) -> Optional[int]:
        """
        Record the strategy decisions made for a single reply generation.

        One row per build_strategy() call that produces an outreach.  The
        outcome fields (got_reply, reply_sentiment, reply_latency_s) start
        NULL and are populated by backfill_angle_reply_data() when a prospect
        reply arrives.

        outreach_id — NULL in Phase 3a.  The generate-reply flow does not have
                      access to outreach_log.id at generation time (the frontend
                      calls /leads/{id}/outreach *after* generation).  A future
                      phase will back-link this FK.

        Returns strategy_outcome_log.id on success, None on failure.
        """
        if not self.available:
            return None
        try:
            import uuid as _uuid
            _event_id = str(_uuid.uuid4())
            # attribution_confidence: 1.0 when outreach_id links to actual outreach row,
            # 0.7 when lead_id only (outreach not yet logged by frontend).
            _attr_conf: Optional[float] = 1.0 if outreach_id else 0.7
            cur = self._conn.execute(
                """
                INSERT INTO strategy_outcome_log
                    (lead_id, outreach_id, selected_angle, goal, buyer_state,
                     cta_style, tone_posture, reply_length, persuasion_level,
                     urgency_level, outreach_seq, decided_at,
                     event_id, attribution_confidence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (lead_id, outreach_id, selected_angle, goal, buyer_state,
                 cta_style, tone_posture, reply_length, persuasion_level,
                 urgency_level, outreach_seq, self._ts(),
                 _event_id, _attr_conf),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] log_strategy_outcome error: {e}")
            return None

    # ── log_conversion_event ──────────────────────────────────────────────────

    def log_conversion_event(
        self,
        lead_id:    int,
        event_type: str,             # accepted|rejected|unsubscribed|stalled_closed
        notes:      Optional[str] = None,
    ) -> Optional[int]:
        """
        Record the terminal outcome for a lead.

        Called when update_lead_stage() is called with a terminal stage value
        ('accepted' or 'rejected' or 'unsubscribed').  Computes time-to-resolution
        from the first outreach row for this lead.

        event_type must be one of:
            'accepted'       — deal completed
            'rejected'       — prospect hard-rejected
            'unsubscribed'   — prospect asked to be removed
            'stalled_closed' — manually closed after prolonged inactivity

        Returns conversion_event_log.id on success, None on failure.
        """
        if not self.available:
            return None
        if event_type not in ("accepted", "rejected", "unsubscribed", "stalled_closed"):
            return None
        try:
            # Snapshot current lead state for attribution context
            lead = self.get_lead(lead_id) or {}
            outreach_rows = self.get_outreach_history(lead_id)
            offer_rows    = self.get_offer_history(lead_id)

            total_outreach = len(outreach_rows)
            total_offers   = len(offer_rows)

            # Time to resolution = now - first outreach sent_at (if any)
            ttr: Optional[float] = None
            if outreach_rows:
                # get_outreach_history returns newest-first; last item is oldest
                first_sent = outreach_rows[-1].get("sent_at")
                if first_sent:
                    ttr = self._ts() - first_sent

            cur = self._conn.execute(
                """
                INSERT INTO conversion_event_log
                    (lead_id, event_type, final_stage, total_outreach_count,
                     total_offer_count, time_to_resolution, notes, recorded_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (lead_id, event_type, lead.get("stage"), total_outreach,
                 total_offers, ttr, notes, self._ts()),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] log_conversion_event error: {e}")
            return None

    # ── log_subject_effectiveness ─────────────────────────────────────────────

    def log_subject_effectiveness(
        self,
        lead_id:    int,
        subject:    str,
        outreach_id: Optional[int] = None,   # NULL in Phase 3a
    ) -> Optional[int]:
        """
        Record a subject line for effectiveness tracking.

        The subject is hashed (SHA1[:8] of lowercased, stripped text) so that
        structurally similar subjects can be grouped without storing verbatim
        text at volume.

        got_reply and reply_sentiment start NULL.  They are populated by
        backfill_angle_reply_data() when a prospect reply is linked to the
        outreach that used this subject.

        outreach_id — NULL in Phase 3a.  The same constraint as log_strategy_outcome:
                      the ID is not available at generation time.

        Returns subject_effectiveness_log.id on success, None on failure.
        """
        if not self.available:
            return None
        if not subject:
            return None
        try:
            import hashlib
            normalised   = subject.lower().strip()
            subject_hash = hashlib.sha1(normalised.encode()).hexdigest()[:8]
            preview      = subject[:60]

            cur = self._conn.execute(
                """
                INSERT INTO subject_effectiveness_log
                    (lead_id, outreach_id, subject_hash, subject_preview, sent_at)
                VALUES (?,?,?,?,?)
                """,
                (lead_id, outreach_id, subject_hash, preview, self._ts()),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] log_subject_effectiveness error: {e}")
            return None

    # ── backfill_angle_reply_data ─────────────────────────────────────────────

    def backfill_angle_reply_data(
        self,
        outreach_id:    int,
        got_reply:      bool,
        reply_sentiment: Optional[str] = None,  # positive|neutral|negative|no_reply
        reply_latency_s: Optional[float] = None,
    ) -> bool:
        """
        Back-fill reply outcome data into angle_log and strategy_outcome_log
        rows that are linked to a specific outreach.

        Called after log_prospect_reply() when in_reply_to_outreach_id is known.

        Updates:
          angle_log          — sets prospect_replied and reply_sentiment
                               WHERE outreach_id = ?
          strategy_outcome_log — sets got_reply, reply_sentiment, reply_latency_s
                               WHERE outreach_id = ?

        Also updates subject_effectiveness_log:
          subject_effectiveness_log — sets got_reply, reply_sentiment
                               WHERE outreach_id = ?

        Phase 3a: this function is defined and wired but will produce zero
        updates until the frontend starts passing in_reply_to_outreach_id
        when logging prospect replies (both angle_log.outreach_id and
        strategy_outcome_log.outreach_id are currently NULL in all rows).

        Returns True if at least one table was updated, False on total failure.
        """
        if not self.available:
            return False
        if reply_sentiment and reply_sentiment not in (
            "positive", "neutral", "negative", "no_reply"
        ):
            reply_sentiment = None

        replied_int = 1 if got_reply else 0
        now         = self._ts()
        any_ok      = False

        # Update angle_log rows for this outreach
        try:
            self._conn.execute(
                """
                UPDATE angle_log
                SET    prospect_replied = ?,
                       reply_sentiment  = ?
                WHERE  outreach_id = ?
                """,
                (replied_int, reply_sentiment, outreach_id),
            )
            self._conn.commit()
            any_ok = True
        except Exception as e:
            print(f"[MemoryDB] backfill_angle_reply_data angle_log error: {e}")

        # Update strategy_outcome_log rows for this outreach
        try:
            self._conn.execute(
                """
                UPDATE strategy_outcome_log
                SET    got_reply              = ?,
                       reply_sentiment        = ?,
                       reply_latency_s        = ?,
                       attribution_confidence = 1.0
                WHERE  outreach_id = ?
                """,
                (replied_int, reply_sentiment, reply_latency_s, outreach_id),
            )
            self._conn.commit()
            any_ok = True
        except Exception as e:
            print(f"[MemoryDB] backfill_angle_reply_data strategy_outcome error: {e}")

        # Update subject_effectiveness_log rows for this outreach
        try:
            self._conn.execute(
                """
                UPDATE subject_effectiveness_log
                SET    got_reply       = ?,
                       reply_sentiment = ?
                WHERE  outreach_id = ?
                """,
                (replied_int, reply_sentiment, outreach_id),
            )
            self._conn.commit()
            any_ok = True
        except Exception as e:
            print(f"[MemoryDB] backfill_angle_reply_data subject_log error: {e}")

        return any_ok


    # ── Operational Reliability methods ──────────────────────────────────────
    # All additive. All non-blocking. Rollback: remove these methods and
    # drop the three Operational Reliability tables.

    # ── Snapshot methods ──────────────────────────────────────────────────────

    def save_snapshot(
        self,
        snapshot_type: str,
        snapshot_json: str,
        label:         Optional[str] = None,
        triggered_by:  str           = "operator",
        version:       str           = "v1",
    ) -> Optional[int]:
        """
        Persist a point-in-time analytics snapshot.

        snapshot_type  — one of: angle_performance, cta_performance,
                          conversion_funnel, attribution_integrity,
                          score_safety, full_system
        snapshot_json  — the full JSON string from the analytics function
        label          — human label, e.g. "pre-Phase-3b baseline"
        triggered_by   — operator | automated | test

        Returns snapshot_log.id on success, None on failure.
        """
        if not self.available:
            return None
        try:
            cur = self._conn.execute(
                """
                INSERT INTO analytics_snapshot_log
                    (snapshot_type, snapshot_json, snapshot_version,
                     label, triggered_by, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (snapshot_type, snapshot_json, version,
                 label, triggered_by, self._ts()),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] save_snapshot error: {e}")
            return None

    def list_snapshots(
        self,
        snapshot_type: Optional[str] = None,
        limit:         int            = 50,
    ) -> list[dict]:
        """
        List snapshots, newest first.  Optionally filter by type.
        Returns metadata only — snapshot_json is excluded for performance.
        """
        if not self.available:
            return []
        try:
            if snapshot_type:
                rows = self._conn.execute(
                    """
                    SELECT id, snapshot_type, snapshot_version, label,
                           triggered_by, created_at
                    FROM   analytics_snapshot_log
                    WHERE  snapshot_type = ?
                    ORDER  BY created_at DESC
                    LIMIT  ?
                    """,
                    (snapshot_type, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT id, snapshot_type, snapshot_version, label,
                           triggered_by, created_at
                    FROM   analytics_snapshot_log
                    ORDER  BY created_at DESC
                    LIMIT  ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] list_snapshots error: {e}")
            return []

    def get_snapshot(self, snapshot_id: int) -> Optional[dict]:
        """
        Return a full snapshot including snapshot_json.
        Returns None if not found or DB unavailable.
        """
        if not self.available:
            return None
        try:
            row = self._conn.execute(
                "SELECT * FROM analytics_snapshot_log WHERE id=?",
                (snapshot_id,)
            ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            print(f"[MemoryDB] get_snapshot error: {e}")
            return None

    # ── Analyst correction methods ────────────────────────────────────────────

    def log_correction(
        self,
        correction_type: str,
        target_table:    str,
        target_row_id:   int,
        before_json:     str,
        after_json:      str,
        reason:          Optional[str] = None,
        target_event_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Log an analyst correction to the immutable audit trail.
        Must be called BEFORE applying the actual change.

        Returns analyst_correction_log.id on success, None on failure.
        """
        if not self.available:
            return None
        try:
            cur = self._conn.execute(
                """
                INSERT INTO analyst_correction_log
                    (correction_type, target_table, target_row_id,
                     before_json, after_json, reason,
                     target_event_id, corrected_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (correction_type, target_table, target_row_id,
                 before_json, after_json, reason,
                 target_event_id, self._ts()),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"[MemoryDB] log_correction error: {e}")
            return None

    def get_correction_history(
        self,
        target_table:    Optional[str] = None,
        target_row_id:   Optional[int] = None,
        limit:           int           = 100,
    ) -> list[dict]:
        """Return correction log entries, newest first."""
        if not self.available:
            return []
        try:
            if target_table and target_row_id is not None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM analyst_correction_log
                    WHERE  target_table=? AND target_row_id=?
                    ORDER  BY corrected_at DESC LIMIT ?
                    """,
                    (target_table, target_row_id, limit),
                ).fetchall()
            elif target_table:
                rows = self._conn.execute(
                    """
                    SELECT * FROM analyst_correction_log
                    WHERE  target_table=?
                    ORDER  BY corrected_at DESC LIMIT ?
                    """,
                    (target_table, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM analyst_correction_log ORDER BY corrected_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryDB] get_correction_history error: {e}")
            return []

    # ── Analyst correction operations ─────────────────────────────────────────

    def relink_reply(
        self,
        reply_id:            int,
        new_outreach_id:     Optional[int],
        reason:              str,
    ) -> bool:
        """
        Relink a prospect reply to a different (or no) outreach.
        Logs the correction before applying it.

        Recalculates attribution_confidence after relinking:
          1.0 if new_outreach_id is not None and outreach exists
          0.5 if new_outreach_id is None (lead-level link only)

        Returns True on success.
        """
        if not self.available:
            return False
        try:
            import json
            row = self._conn.execute(
                "SELECT * FROM prospect_reply_log WHERE id=?", (reply_id,)
            ).fetchone()
            if not row:
                return False
            before = dict(row)

            # Validate new_outreach_id exists (if not None)
            new_conf = 0.5
            if new_outreach_id is not None:
                exists = self._conn.execute(
                    "SELECT id FROM outreach_log WHERE id=?", (new_outreach_id,)
                ).fetchone()
                if not exists:
                    print(f"[MemoryDB] relink_reply: outreach {new_outreach_id} not found")
                    return False
                new_conf = 1.0

            self._conn.execute(
                """
                UPDATE prospect_reply_log
                SET    in_reply_to_outreach_id = ?,
                       attribution_confidence  = ?
                WHERE  id = ?
                """,
                (new_outreach_id, new_conf, reply_id),
            )
            after = dict(self._conn.execute(
                "SELECT * FROM prospect_reply_log WHERE id=?", (reply_id,)
            ).fetchone())

            self.log_correction(
                correction_type = "relink_reply",
                target_table    = "prospect_reply_log",
                target_row_id   = reply_id,
                before_json     = json.dumps(before),
                after_json      = json.dumps(after),
                reason          = reason,
                target_event_id = before.get("event_id"),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] relink_reply error: {e}")
            return False

    def override_attribution_confidence(
        self,
        table:      str,
        row_id:     int,
        new_conf:   float,
        reason:     str,
    ) -> bool:
        """
        Override the attribution_confidence on a specific row.

        table must be one of: prospect_reply_log, strategy_outcome_log.
        new_conf must be in [0.0, 1.0].
        Logs the correction before applying.

        Returns True on success.
        """
        if not self.available:
            return False
        if table not in ("prospect_reply_log", "strategy_outcome_log"):
            return False
        if not 0.0 <= new_conf <= 1.0:
            return False
        try:
            import json
            row = self._conn.execute(
                f"SELECT * FROM {table} WHERE id=?", (row_id,)
            ).fetchone()
            if not row:
                return False
            before = dict(row)

            self._conn.execute(
                f"UPDATE {table} SET attribution_confidence=? WHERE id=?",
                (new_conf, row_id),
            )
            after = dict(self._conn.execute(
                f"SELECT * FROM {table} WHERE id=?", (row_id,)
            ).fetchone())

            self.log_correction(
                correction_type = "override_confidence",
                target_table    = table,
                target_row_id   = row_id,
                before_json     = json.dumps(before),
                after_json      = json.dumps(after),
                reason          = reason,
                target_event_id = before.get("event_id"),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] override_attribution_confidence error: {e}")
            return False

    def invalidate_mapping(
        self,
        table:    str,
        row_id:   int,
        reason:   str,
    ) -> bool:
        """
        Mark a reply or outcome row as invalidated by setting
        attribution_confidence to 0.0.

        This preserves the row data (never deletes) but signals to Phase 3b
        score materialization that this row should be excluded from score
        computation.

        table must be one of: prospect_reply_log, strategy_outcome_log,
                               angle_log.
        Logs the correction before applying.

        Returns True on success.
        """
        if not self.available:
            return False
        allowed = ("prospect_reply_log", "strategy_outcome_log", "angle_log")
        if table not in allowed:
            return False
        try:
            import json
            row = self._conn.execute(
                f"SELECT * FROM {table} WHERE id=?", (row_id,)
            ).fetchone()
            if not row:
                return False
            before = dict(row)

            # For angle_log, set prospect_replied=0 as the invalidation signal
            # (it has no attribution_confidence column)
            if table == "angle_log":
                self._conn.execute(
                    "UPDATE angle_log SET prospect_replied=0, reply_sentiment='no_reply' WHERE id=?",
                    (row_id,),
                )
            else:
                self._conn.execute(
                    f"UPDATE {table} SET attribution_confidence=0.0 WHERE id=?",
                    (row_id,),
                )

            after = dict(self._conn.execute(
                f"SELECT * FROM {table} WHERE id=?", (row_id,)
            ).fetchone())

            self.log_correction(
                correction_type = "invalidate_mapping",
                target_table    = table,
                target_row_id   = row_id,
                before_json     = json.dumps(before),
                after_json      = json.dumps(after),
                reason          = reason,
                target_event_id = before.get("event_id"),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] invalidate_mapping error: {e}")
            return False

    def merge_reply_chains(
        self,
        keep_reply_id:    int,
        discard_reply_id: int,
        reason:           str,
    ) -> bool:
        """
        Merge two reply records by relinking the discard reply's outreach
        attribution to the keep reply and invalidating the discard reply.

        Use when the same prospect sent two messages that both reply to the
        same outreach (e.g. a follow-up message immediately after), and
        only one canonical reply should be attributed.

        Logs two corrections: one invalidation and one relink.
        Returns True on success.
        """
        if not self.available:
            return False
        try:
            import json
            keep = self._conn.execute(
                "SELECT * FROM prospect_reply_log WHERE id=?", (keep_reply_id,)
            ).fetchone()
            discard = self._conn.execute(
                "SELECT * FROM prospect_reply_log WHERE id=?", (discard_reply_id,)
            ).fetchone()
            if not keep or not discard:
                return False

            keep_d    = dict(keep)
            discard_d = dict(discard)

            # If discard has an outreach link and keep doesn't, transfer it
            if (discard_d.get("in_reply_to_outreach_id") and
                    not keep_d.get("in_reply_to_outreach_id")):
                self._conn.execute(
                    """
                    UPDATE prospect_reply_log
                    SET    in_reply_to_outreach_id = ?,
                           attribution_confidence  = 1.0
                    WHERE  id = ?
                    """,
                    (discard_d["in_reply_to_outreach_id"], keep_reply_id),
                )

            # Invalidate discard reply
            self._conn.execute(
                "UPDATE prospect_reply_log SET attribution_confidence=0.0 WHERE id=?",
                (discard_reply_id,),
            )

            after_keep    = dict(self._conn.execute(
                "SELECT * FROM prospect_reply_log WHERE id=?", (keep_reply_id,)
            ).fetchone())
            after_discard = dict(self._conn.execute(
                "SELECT * FROM prospect_reply_log WHERE id=?", (discard_reply_id,)
            ).fetchone())

            self.log_correction(
                correction_type = "merge_reply_chains",
                target_table    = "prospect_reply_log",
                target_row_id   = discard_reply_id,
                before_json     = json.dumps(discard_d),
                after_json      = json.dumps(after_discard),
                reason          = f"merged into reply {keep_reply_id}: {reason}",
                target_event_id = discard_d.get("event_id"),
            )
            self.log_correction(
                correction_type = "merge_reply_chains",
                target_table    = "prospect_reply_log",
                target_row_id   = keep_reply_id,
                before_json     = json.dumps(keep_d),
                after_json      = json.dumps(after_keep),
                reason          = f"kept as canonical (merged from {discard_reply_id}): {reason}",
                target_event_id = keep_d.get("event_id"),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] merge_reply_chains error: {e}")
            return False

    # ── Frontend idempotency ──────────────────────────────────────────────────

    def check_idempotency(self, idempotency_key: str) -> Optional[dict]:
        """
        Check whether an idempotency key has been used before.

        Returns the existing log entry if the key exists and is not invalidated,
        None if the key is new (safe to proceed).
        Used by frontend submission endpoints to prevent duplicate logging.
        """
        if not self.available:
            return None
        try:
            row = self._conn.execute(
                """
                SELECT * FROM frontend_idempotency_log
                WHERE  idempotency_key = ? AND invalidated = 0
                """,
                (idempotency_key,),
            ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            print(f"[MemoryDB] check_idempotency error: {e}")
            return None

    def register_idempotency(
        self,
        idempotency_key: str,
        operation_type:  str,
        lead_id:         Optional[int] = None,
        result_row_id:   Optional[int] = None,
    ) -> bool:
        """
        Register an idempotency key after a successful operation.
        Subsequent calls with the same key will be detected by check_idempotency().

        Returns True on success, False if key already registered or DB error.
        """
        if not self.available:
            return False
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO frontend_idempotency_log
                    (idempotency_key, operation_type, lead_id,
                     result_row_id, created_at)
                VALUES (?,?,?,?,?)
                """,
                (idempotency_key, operation_type, lead_id,
                 result_row_id, self._ts()),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] register_idempotency error: {e}")
            return False

    def invalidate_idempotency_key(self, idempotency_key: str) -> bool:
        """
        Invalidate an idempotency key so it can be re-used.
        Used when a submission was logged but the downstream action failed.
        """
        if not self.available:
            return False
        try:
            self._conn.execute(
                "UPDATE frontend_idempotency_log SET invalidated=1 WHERE idempotency_key=?",
                (idempotency_key,),
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryDB] invalidate_idempotency_key error: {e}")
            return False


# Module-level singleton — imported by main.py
memory_db = MemoryDB()
