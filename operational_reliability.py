"""
operational_reliability.py — Operational Reliability Layer
===========================================================
Pre-Phase-3b tooling for:
  1. Adaptive safety configuration (flags + thresholds)
  2. Deterministic replay of lead history through strategy logic
  3. QC dashboard metrics (attribution confidence, mapping health, volatility)
  4. Snapshot management helpers
  5. Frontend integrity safeguard helpers

DESIGN CONTRACT
---------------
- Zero imports from main.py (no circular dependency)
- Zero writes except via broker_memory methods (never raw SQL)
- All functions return plain dicts — JSON-serialisable directly
- Every function is independently callable with just (db,) signature
- Graceful degradation when db is None or unavailable

ROLLBACK
--------
Delete this file and remove the /qc/operational/* endpoint registrations
from main.py. No other changes required.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── 1. Adaptive Safety Configuration ─────────────────────────────────────────

@dataclass
class AdaptiveSafetyConfig:
    """
    Runtime configuration flags that gate adaptive learning behaviour.

    All flags default to the SAFE (non-adaptive) state.  No flag activates
    any adaptive behaviour in Phase QC or Phase Operational Reliability.
    They are defined here so Phase 3b can read them from a single source
    of truth without hardcoding values in main.py.

    CHANGING THESE FLAGS DOES NOT YET AFFECT STRATEGY SELECTION.
    They are read-only configuration placeholders until Phase 3b wires
    them into _select_next_angle() and build_strategy().

    All values can be overridden via environment variables.
    """

    # Master switch — when False, all score-based overrides are suppressed
    # regardless of all other flags.  Phase 3b must check this first.
    ENABLE_ADAPTIVE_SELECTION: bool  = False

    # When True, forces static position-order selection even if
    # ENABLE_ADAPTIVE_SELECTION is True.  Emergency override for operators.
    FORCE_STATIC_SELECTION:    bool  = True

    # Minimum primary-use count per angle before its score is trusted.
    # Mirrors MIN_SAMPLE_SIZE in analytics_qc.py — single source here.
    MIN_SAMPLE_SIZE:           int   = 5

    # When True, scores decay toward 0.5 (neutral) over time using
    # exponential decay weighted by score age.  Prevents stale early
    # data from permanently distorting selection.
    SCORE_DECAY_ENABLED:       bool  = False

    # Half-life for score decay in seconds (default: 30 days).
    # Only active when SCORE_DECAY_ENABLED is True.
    SCORE_DECAY_HALF_LIFE_S:   float = 30 * 86_400.0

    # Minimum attribution_confidence required for a row to be included
    # in score materialization.  Rows below this threshold are excluded.
    MIN_ATTRIBUTION_CONFIDENCE: float = 0.5

    # Maximum staleness (seconds) before a materialised score is considered
    # stale and suppressed.  Mirrors SCORE_STALE_THRESHOLD_S in analytics_qc.
    SCORE_STALE_THRESHOLD_S:   float = 86_400.0

    # When True, logs every angle selection decision to the console.
    # Useful for debugging without changing behaviour.
    DEBUG_SELECTION_LOGGING:   bool  = False

    def load_from_env(self) -> "AdaptiveSafetyConfig":
        """
        Override values from environment variables.
        Returns self for chaining.

        Environment variables:
          ENABLE_ADAPTIVE_SELECTION   — "true" / "false"
          FORCE_STATIC_SELECTION      — "true" / "false"
          MIN_SAMPLE_SIZE             — integer
          SCORE_DECAY_ENABLED         — "true" / "false"
          MIN_ATTRIBUTION_CONFIDENCE  — float 0.0–1.0
          SCORE_STALE_THRESHOLD_S     — float (seconds)
          DEBUG_SELECTION_LOGGING     — "true" / "false"
        """
        import os

        def _bool(key: str, default: bool) -> bool:
            v = os.getenv(key, "").lower()
            if v == "true":  return True
            if v == "false": return False
            return default

        def _int(key: str, default: int) -> int:
            try:    return int(os.getenv(key, str(default)))
            except: return default

        def _float(key: str, default: float) -> float:
            try:    return float(os.getenv(key, str(default)))
            except: return default

        self.ENABLE_ADAPTIVE_SELECTION  = _bool("ENABLE_ADAPTIVE_SELECTION",  self.ENABLE_ADAPTIVE_SELECTION)
        self.FORCE_STATIC_SELECTION     = _bool("FORCE_STATIC_SELECTION",     self.FORCE_STATIC_SELECTION)
        self.MIN_SAMPLE_SIZE            = _int( "MIN_SAMPLE_SIZE",             self.MIN_SAMPLE_SIZE)
        self.SCORE_DECAY_ENABLED        = _bool("SCORE_DECAY_ENABLED",        self.SCORE_DECAY_ENABLED)
        self.MIN_ATTRIBUTION_CONFIDENCE = _float("MIN_ATTRIBUTION_CONFIDENCE", self.MIN_ATTRIBUTION_CONFIDENCE)
        self.SCORE_STALE_THRESHOLD_S    = _float("SCORE_STALE_THRESHOLD_S",    self.SCORE_STALE_THRESHOLD_S)
        self.DEBUG_SELECTION_LOGGING    = _bool("DEBUG_SELECTION_LOGGING",     self.DEBUG_SELECTION_LOGGING)
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def safety_status(self) -> dict:
        """
        Human-readable status of the safety configuration.
        Returns a verdict and a plain-English explanation of each flag.
        """
        adaptive_active = (
            self.ENABLE_ADAPTIVE_SELECTION and
            not self.FORCE_STATIC_SELECTION
        )
        return {
            "adaptive_selection_active": adaptive_active,
            "flags": self.to_dict(),
            "verdict": (
                "STATIC (safe)"
                if not adaptive_active else
                "ADAPTIVE (score-driven selection active)"
            ),
            "explanation": {
                "ENABLE_ADAPTIVE_SELECTION": (
                    "Master switch. Currently "
                    + ("ON — score-based overrides enabled."
                       if self.ENABLE_ADAPTIVE_SELECTION else
                       "OFF — all selection uses static position order.")
                ),
                "FORCE_STATIC_SELECTION": (
                    "Emergency override. Currently "
                    + ("ON — static selection forced regardless of other flags."
                       if self.FORCE_STATIC_SELECTION else
                       "OFF — does not override ENABLE_ADAPTIVE_SELECTION.")
                ),
                "MIN_SAMPLE_SIZE": (
                    f"{self.MIN_SAMPLE_SIZE} primary uses required per angle "
                    "before its score is trusted."
                ),
                "SCORE_DECAY_ENABLED": (
                    "Score decay is "
                    + ("enabled — scores decay toward neutral over time."
                       if self.SCORE_DECAY_ENABLED else
                       "disabled — scores retain full weight indefinitely.")
                ),
                "MIN_ATTRIBUTION_CONFIDENCE": (
                    f"Rows with attribution_confidence < {self.MIN_ATTRIBUTION_CONFIDENCE} "
                    "are excluded from score computation."
                ),
            },
        }


# Module-level singleton — loaded once at import, overridable from env
safety_config = AdaptiveSafetyConfig().load_from_env()


# ── 2. Deterministic Replay ────────────────────────────────────────────────────

def replay_lead_history(
    db,
    lead_id:       int,
    config:        Optional[AdaptiveSafetyConfig] = None,
) -> dict:
    """
    Replay a lead's full outreach history through the current strategy logic.

    For each outreach in sequence, reconstructs the StrategySignals that
    would have been used at that point in time, runs build_strategy(), and
    records:
      - the strategy that WOULD be selected NOW for that context
      - the strategy that WAS recorded in strategy_outcome_log at that time
      - any drift between the two (angle, CTA, goal, tone)

    This is a READ-ONLY operation. It does not modify any rows.
    It does not affect current reply generation.

    Returns a drift report showing historical vs current strategy decisions,
    which is the authoritative input for deciding whether Phase 3b is safe
    to activate.
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    cfg = config or safety_config

    try:
        from reply_strategy import build_strategy, StrategySignals
        from angle_memory   import build_angle_inventory, _select_next_angle
    except ImportError as e:
        return {"error": f"Strategy modules unavailable: {e}"}

    conn = db._conn

    lead_row = conn.execute(
        "SELECT * FROM leads WHERE id=?", (lead_id,)
    ).fetchone()
    if not lead_row:
        return {"error": f"lead {lead_id} not found"}

    lead = dict(lead_row)

    outreach_rows = conn.execute(
        "SELECT * FROM outreach_log WHERE lead_id=? ORDER BY sent_at ASC",
        (lead_id,)
    ).fetchall()

    replay_chain = []
    total_drift  = 0
    drift_fields = []

    for seq, o in enumerate(outreach_rows, 1):
        od = dict(o)

        # Historical outcome (what was actually decided at the time)
        hist = conn.execute(
            """
            SELECT * FROM strategy_outcome_log
            WHERE lead_id=? AND (outreach_id=? OR outreach_seq=?)
            LIMIT 1
            """,
            (lead_id, od["id"], seq),
        ).fetchone()
        historical = dict(hist) if hist else None

        # Reconstruct StrategySignals for this point in the sequence.
        # We use the outreach preset as intent (best available proxy),
        # and prior outreach bodies for suppression context.
        prior_bodies = [
            dict(r)["body"] or ""
            for r in conn.execute(
                "SELECT body FROM outreach_log WHERE lead_id=? AND sent_at<? ORDER BY sent_at ASC",
                (lead_id, od["sent_at"]),
            ).fetchall()
        ]

        # Offers at this point in time
        offers_at_time = conn.execute(
            "SELECT * FROM offer_log WHERE lead_id=? AND offered_at<=? ORDER BY offered_at DESC",
            (lead_id, od["sent_at"]),
        ).fetchall()
        last_offer = dict(offers_at_time[0]) if offers_at_time else None

        intent = od.get("preset") or "follow_up"

        # Build minimal StrategySignals for replay
        sig_kwargs = dict(
            intent                = intent,
            message               = "",
            stage                 = lead.get("stage", "unknown"),
            neg_state             = "none",
            response_frame        = "inferred_reply",
            tone_requested        = "professional and persuasive",
            outreach_count        = seq - 1,
            prior_outreach_bodies = prior_bodies,
        )

        # Add offer context if available
        if last_offer:
            sig_kwargs["last_offer_amount"]     = last_offer.get("amount")
            sig_kwargs["last_offer_direction"]  = last_offer.get("direction")

        # Build angle inventory up to this point in the sequence
        try:
            # Get only angle rows up to this outreach's timestamp
            angle_rows_at_time = conn.execute(
                """
                SELECT * FROM angle_log
                WHERE lead_id=? AND logged_at<=?
                ORDER BY outreach_seq ASC, logged_at ASC
                """,
                (lead_id, od["sent_at"]),
            ).fetchall()

            if angle_rows_at_time:
                from angle_memory import AngleInventory, AngleRecord, get_angle_labels
                records = [AngleRecord.from_db_row(dict(r)) for r in angle_rows_at_time]
                all_a   = get_angle_labels()

                used_ids      = {r.angle_id for r in records}
                primary_count = {}
                last_used     = {}
                for r in records:
                    if r.pitched_as == "primary":
                        primary_count[r.angle_id] = primary_count.get(r.angle_id, 0) + 1
                    if r.logged_at > last_used.get(r.angle_id, 0.0):
                        last_used[r.angle_id] = r.logged_at

                threshold_map = {
                    a_id: entry.get("exhaustion_threshold", 2)
                    for a_id, entry in __import__('angle_memory')._ANGLE_REGISTRY.items()
                }
                exhausted = [
                    a for a in used_ids
                    if primary_count.get(a, 0) >= threshold_map.get(a, 2)
                ]
                available = [a for a in all_a if a not in used_ids and a not in exhausted]

                inv = AngleInventory(
                    lead_id          = lead_id,
                    all_angles       = all_a,
                    used_angles      = records,
                    exhausted_angles = exhausted,
                    available_angles = available,
                    primary_use_count = primary_count,
                    last_used_at     = last_used,
                )
                sig_kwargs["angle_inventory"] = inv
        except Exception as _inv_err:
            pass  # replay without inventory if it fails

        try:
            sig      = StrategySignals(**sig_kwargs)
            strategy = build_strategy(sig)
        except Exception as _strat_err:
            replay_chain.append({
                "seq":         seq,
                "outreach_id": od["id"],
                "error":       str(_strat_err),
            })
            continue

        # Compare current replay vs historical record
        current = {
            "goal":           strategy.primary_goal,
            "buyer_state":    strategy.buyer_state,
            "cta_style":      strategy.cta_style,
            "tone_posture":   strategy.tone_posture,
            "selected_angle": strategy.selected_angle,
            "persuasion_level": strategy.persuasion_level,
        }

        diffs = {}
        if historical:
            for field_name, curr_val in current.items():
                hist_val = historical.get(field_name)
                if hist_val and curr_val != hist_val:
                    diffs[field_name] = {"historical": hist_val, "current": curr_val}

        if diffs:
            total_drift += len(diffs)
            drift_fields.extend(diffs.keys())

        replay_chain.append({
            "seq":                seq,
            "outreach_id":        od["id"],
            "preset":             od.get("preset"),
            "sent_at":            od.get("sent_at"),
            "historical_outcome": historical,
            "current_replay": {
                "goal":             strategy.primary_goal,
                "buyer_state":      strategy.buyer_state,
                "cta_style":        strategy.cta_style,
                "tone_posture":     strategy.tone_posture,
                "selected_angle":   strategy.selected_angle,
                "persuasion_level": strategy.persuasion_level,
            },
            "drift":      diffs,
            "has_drift":  bool(diffs),
        })

    # Drift summary
    from collections import Counter
    field_counts = Counter(drift_fields)

    drift_severity = (
        "none"     if total_drift == 0 else
        "low"      if total_drift <= 2 else
        "medium"   if total_drift <= 6 else
        "high"
    )

    return {
        "lead_id":       lead_id,
        "domain":        lead.get("domain"),
        "total_outreach": len(outreach_rows),
        "replay_chain":  replay_chain,
        "drift_summary": {
            "total_field_drifts": total_drift,
            "drift_severity":     drift_severity,
            "most_drifted_fields": dict(field_counts.most_common(5)),
            "outreach_with_drift": sum(1 for r in replay_chain if r.get("has_drift")),
        },
        "phase3b_safety": {
            "safe_to_activate": drift_severity in ("none", "low"),
            "note": (
                "Drift is within acceptable bounds — adaptive selection "
                "would not produce substantially different outcomes."
                if drift_severity in ("none", "low") else
                f"Drift is {drift_severity} — review before activating Phase 3b."
            ),
        },
    }


# ── 3. QC Dashboard Metrics ────────────────────────────────────────────────────

def get_dashboard_metrics(db) -> dict:
    """
    Consolidated QC dashboard metrics for the operations panel.

    Returns all key health indicators in a single call:
      attribution_confidence_distribution
      unresolved_mapping_count
      stale_analytics_detection
      low_confidence_outcome_percentage
      score_volatility_monitoring
      correction_activity
      idempotency_health
      overall_health_score (0–100)
      overall_verdict

    All metrics are read-only. Zero writes.
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    # ── Attribution confidence distribution ───────────────────────────────────
    conf_buckets = {
        "high (≥0.9)":   0,
        "medium (0.5–0.9)": 0,
        "low (0.0–0.5)": 0,
        "zero (invalid)": 0,
        "null (unknown)": 0,
    }
    for table in ("prospect_reply_log", "strategy_outcome_log"):
        try:
            rows = conn.execute(
                f"SELECT attribution_confidence FROM {table}"
            ).fetchall()
            for (c,) in rows:
                if c is None:
                    conf_buckets["null (unknown)"] += 1
                elif c >= 0.9:
                    conf_buckets["high (≥0.9)"] += 1
                elif c >= 0.5:
                    conf_buckets["medium (0.5–0.9)"] += 1
                elif c > 0.0:
                    conf_buckets["low (0.0–0.5)"] += 1
                else:
                    conf_buckets["zero (invalid)"] += 1
        except Exception:
            pass

    total_conf_rows = sum(conf_buckets.values())
    high_conf_pct   = round(
        conf_buckets["high (≥0.9)"] / total_conf_rows * 100, 1
    ) if total_conf_rows else 0
    low_conf_pct    = round(
        (conf_buckets["low (0.0–0.5)"] + conf_buckets["zero (invalid)"]) /
        total_conf_rows * 100, 1
    ) if total_conf_rows else 0

    # ── Unresolved mapping count ──────────────────────────────────────────────
    unlinked_replies = conn.execute(
        "SELECT COUNT(*) FROM prospect_reply_log WHERE in_reply_to_outreach_id IS NULL"
    ).fetchone()[0]

    orphaned_outcomes = conn.execute("""
        SELECT COUNT(*) FROM strategy_outcome_log
        WHERE outreach_id IS NULL
    """).fetchone()[0]

    cross_lead = conn.execute("""
        SELECT COUNT(*) FROM prospect_reply_log r
        JOIN outreach_log o ON o.id = r.in_reply_to_outreach_id
        WHERE r.lead_id != o.lead_id
    """).fetchone()[0]

    duplicate_mappings = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT in_reply_to_outreach_id FROM prospect_reply_log
            WHERE in_reply_to_outreach_id IS NOT NULL
            GROUP BY in_reply_to_outreach_id HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    # ── Stale analytics detection ─────────────────────────────────────────────
    latest_outcome_ts = conn.execute(
        "SELECT MAX(decided_at) FROM strategy_outcome_log"
    ).fetchone()[0]
    latest_reply_ts = conn.execute(
        "SELECT MAX(received_at) FROM prospect_reply_log"
    ).fetchone()[0]

    now = time.time()
    outcome_age_h = round((now - latest_outcome_ts) / 3600, 1) if latest_outcome_ts else None
    reply_age_h   = round((now - latest_reply_ts)   / 3600, 1) if latest_reply_ts   else None

    outcome_stale = outcome_age_h is not None and outcome_age_h > 24
    reply_stale   = reply_age_h   is not None and reply_age_h   > 168  # 7 days

    # ── Score volatility monitoring ────────────────────────────────────────────
    # Compare the two most recent full_system snapshots if they exist
    volatility_report = {"status": "no_snapshots", "max_delta": None, "note": ""}
    try:
        snaps = conn.execute(
            """
            SELECT id, snapshot_json FROM analytics_snapshot_log
            WHERE snapshot_type='full_system'
            ORDER BY created_at DESC LIMIT 2
            """,
        ).fetchall()
        if len(snaps) >= 2:
            snap1 = json.loads(snaps[0][1])
            snap2 = json.loads(snaps[1][1])
            s1_angles = {a["angle_id"]: a.get("weighted_score")
                         for a in snap1.get("angle_performance", {}).get("angles", [])}
            s2_angles = {a["angle_id"]: a.get("weighted_score")
                         for a in snap2.get("angle_performance", {}).get("angles", [])}
            deltas = []
            for a_id in set(s1_angles) | set(s2_angles):
                v1 = s1_angles.get(a_id) or 0.0
                v2 = s2_angles.get(a_id) or 0.0
                if v1 is not None and v2 is not None:
                    deltas.append(abs(v1 - v2))
            max_delta = max(deltas) if deltas else 0.0
            volatility_report = {
                "status":    "volatile" if max_delta > 0.15 else "stable",
                "max_delta": round(max_delta, 3),
                "note": (
                    f"Max score delta {max_delta:.3f} between last two snapshots — "
                    + ("high volatility, review before Phase 3b."
                       if max_delta > 0.15 else "stable.")
                ),
            }
        elif len(snaps) == 1:
            volatility_report = {
                "status": "single_snapshot",
                "max_delta": None,
                "note": "Only one snapshot — take another to enable volatility monitoring.",
            }
    except Exception:
        pass

    # ── Correction activity ───────────────────────────────────────────────────
    corrections_total = conn.execute(
        "SELECT COUNT(*) FROM analyst_correction_log"
    ).fetchone()[0]
    corrections_last_7d = conn.execute(
        "SELECT COUNT(*) FROM analyst_correction_log WHERE corrected_at > ?",
        (now - 7 * 86_400,),
    ).fetchone()[0]
    correction_types = {
        r[0]: r[1] for r in conn.execute(
            "SELECT correction_type, COUNT(*) FROM analyst_correction_log GROUP BY correction_type"
        ).fetchall()
    }

    # ── Idempotency health ────────────────────────────────────────────────────
    idempotency_total = conn.execute(
        "SELECT COUNT(*) FROM frontend_idempotency_log"
    ).fetchone()[0]
    idempotency_invalid = conn.execute(
        "SELECT COUNT(*) FROM frontend_idempotency_log WHERE invalidated=1"
    ).fetchone()[0]

    # ── Overall health score 0–100 ────────────────────────────────────────────
    # Scoring heuristic:
    #   Start at 100
    #   -20 if cross-lead mismatches exist
    #   -15 if duplicate mappings exist
    #   -10 for each 10% of low-confidence rows (max -30)
    #   -10 if outcome data is stale (>24h)
    #   -10 if score volatility is high
    #   -5  if corrections > 5 in last 7 days

    score = 100
    if cross_lead > 0:                 score -= 20
    if duplicate_mappings > 0:         score -= 15
    score -= min(30, int(low_conf_pct / 10) * 10)
    if outcome_stale:                  score -= 10
    if volatility_report.get("status") == "volatile": score -= 10
    if corrections_last_7d > 5:        score -= 5
    score = max(0, score)

    verdict = (
        "healthy"   if score >= 80 else
        "degraded"  if score >= 55 else
        "critical"
    )

    return {
        "attribution_confidence": {
            "distribution":    conf_buckets,
            "total_rows":      total_conf_rows,
            "high_conf_pct":   high_conf_pct,
            "low_conf_pct":    low_conf_pct,
        },
        "unresolved_mappings": {
            "unlinked_replies":    unlinked_replies,
            "orphaned_outcomes":   orphaned_outcomes,
            "cross_lead_errors":   cross_lead,
            "duplicate_mappings":  duplicate_mappings,
            "total_issues":        cross_lead + duplicate_mappings,
        },
        "staleness": {
            "latest_outcome_age_h":  outcome_age_h,
            "latest_reply_age_h":    reply_age_h,
            "outcome_stale":         outcome_stale,
            "reply_stale":           reply_stale,
        },
        "score_volatility":     volatility_report,
        "correction_activity": {
            "total_corrections":     corrections_total,
            "last_7d":               corrections_last_7d,
            "by_type":               correction_types,
        },
        "idempotency_health": {
            "total_keys":      idempotency_total,
            "invalidated_keys": idempotency_invalid,
        },
        "overall_health_score": score,
        "overall_verdict":      verdict,
        "phase3b_gate": {
            "recommended": score >= 80 and not cross_lead and not duplicate_mappings,
            "note": (
                "System health sufficient for Phase 3b consideration."
                if score >= 80 and not cross_lead and not duplicate_mappings else
                f"Health score {score}/100 — resolve issues before activating Phase 3b."
            ),
        },
    }


# ── 4. Snapshot management helpers ────────────────────────────────────────────

def create_full_system_snapshot(db, label: Optional[str] = None) -> dict:
    """
    Capture a full system analytics snapshot and persist it.

    Calls all analytics_qc functions, bundles the results into a single
    JSON blob, and saves it via memory_db.save_snapshot().

    This is the canonical "baseline" operation before activating Phase 3b.
    Returns the snapshot metadata dict on success.
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    try:
        import analytics_qc as aqc
        snapshot_data = {
            "captured_at":           time.time(),
            "snapshot_type":         "full_system",
            "angle_performance":     aqc.get_angle_performance(db),
            "cta_performance":       aqc.get_cta_performance(db),
            "conversion_funnel":     aqc.get_conversion_funnel(db),
            "attribution_integrity": aqc.check_attribution_integrity(db),
            "score_safety":          aqc.score_safety_check(db),
            "dashboard":             get_dashboard_metrics(db),
        }
        snapshot_json = json.dumps(snapshot_data, default=str)
        snap_id = db.save_snapshot(
            snapshot_type = "full_system",
            snapshot_json = snapshot_json,
            label         = label or f"full_system snapshot {time.strftime('%Y-%m-%d %H:%M')}",
            triggered_by  = "operator",
        )
        if snap_id is None:
            return {"error": "snapshot save failed"}
        return {
            "snapshot_id":   snap_id,
            "snapshot_type": "full_system",
            "label":         label,
            "captured_at":   snapshot_data["captured_at"],
            "size_bytes":    len(snapshot_json),
            "dashboard_verdict": snapshot_data["dashboard"].get("overall_verdict"),
            "phase3b_gate":  snapshot_data["dashboard"].get("phase3b_gate"),
        }
    except Exception as e:
        return {"error": str(e)}


# ── 5. Frontend integrity safeguard helpers ───────────────────────────────────

def check_and_register_idempotency(
    db,
    idempotency_key: str,
    operation_type:  str,
    lead_id:         Optional[int] = None,
) -> dict:
    """
    Check whether an operation with this idempotency_key has already been
    performed.  If yes, return the original result.  If no, signal the caller
    to proceed and register the key after completion.

    Returns:
      {"duplicate": True,  "result_row_id": int,  "message": "already processed"}
      {"duplicate": False, "message": "proceed"}

    The caller must call db.register_idempotency(...) with the result_row_id
    after successfully completing the operation.
    """
    if not db or not db.available:
        return {"duplicate": False, "message": "proceed (db unavailable)"}

    existing = db.check_idempotency(idempotency_key)
    if existing:
        return {
            "duplicate":    True,
            "result_row_id": existing.get("result_row_id"),
            "operation_type": existing.get("operation_type"),
            "message":      "duplicate request — original result returned",
            "original_at":  existing.get("created_at"),
        }
    return {"duplicate": False, "message": "proceed"}


def validate_outreach_threading(
    db,
    lead_id:         int,
    outreach_id:     Optional[int],
    event_id:        Optional[str],
) -> dict:
    """
    Validate that an outreach_id / event_id pair is consistent with the
    given lead_id before accepting a frontend submission.

    Prevents a retry from accidentally relinking to the wrong outreach.

    Returns:
      {"valid": True}
      {"valid": False, "reason": str}
    """
    if not db or not db.available:
        return {"valid": True, "note": "db unavailable — validation skipped"}

    conn = db._conn

    if outreach_id is None and event_id is None:
        return {"valid": True, "note": "no outreach link supplied"}

    if outreach_id is not None:
        row = conn.execute(
            "SELECT lead_id, event_id FROM outreach_log WHERE id=?",
            (outreach_id,)
        ).fetchone()
        if not row:
            return {"valid": False, "reason": f"outreach {outreach_id} not found"}
        if row[0] != lead_id:
            return {
                "valid":  False,
                "reason": (
                    f"outreach {outreach_id} belongs to lead {row[0]}, "
                    f"not lead {lead_id} — cross-lead threading error"
                ),
            }
        if event_id and row[1] and row[1] != event_id:
            return {
                "valid":  False,
                "reason": (
                    f"event_id mismatch: supplied {event_id!r} "
                    f"but outreach has {row[1]!r}"
                ),
            }

    return {"valid": True}
