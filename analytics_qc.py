"""
analytics_qc.py — Attribution Diagnostics & Observability Layer
================================================================
Phase QC: read-only analytics, integrity checks, and score safety guards.

This module contains ALL query logic for the QC endpoints in main.py.
It imports broker_memory but never modifies any table.

Design principles:
  - Read-only: zero INSERT/UPDATE/DELETE
  - No strategy modifications
  - No prompt modifications
  - No adaptive scoring activations
  - All functions return plain dicts — safe to JSON-serialise directly
  - Graceful degradation: every function handles missing data / NULL fields

Rollback: delete this file and remove the /qc/* endpoint registrations
          from main.py. Zero other changes required.

Sections:
  1. Attribution integrity checks
  2. Lead attribution report
  3. Angle performance analytics
  4. CTA performance analytics
  5. Conversion funnel analytics
  6. Reply latency analytics
  7. Score safety guards (dry-run preview)
  8. Analytics materialization verification
"""

from __future__ import annotations

import time
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum outcome rows before a metric is considered trustworthy
MIN_SAMPLE_SIZE: int = 5

# Attribution confidence thresholds
CONF_HIGH:   float = 0.9   # outreach_id linked + backfilled
CONF_MEDIUM: float = 0.5   # lead-only link, no outreach_id
CONF_LOW:    float = 0.0   # completely unlinked

# Score staleness threshold (seconds) — 24 hours
SCORE_STALE_THRESHOLD_S: float = 86_400.0


# ── 1. Attribution integrity checks ──────────────────────────────────────────

def check_attribution_integrity(db) -> dict:
    """
    Run a full attribution integrity audit against broker_memory tables.

    Returns a structured report with:
      orphaned_outreach        — outreach rows with no strategy_outcome row
      duplicate_reply_mappings — reply rows sharing the same in_reply_to_outreach_id
      cross_lead_mismatches    — replies whose lead_id differs from their linked
                                  outreach's lead_id (corrupted mapping)
      missing_outcome_rows     — leads with outreach but zero strategy_outcome rows
      unlinked_reply_rows      — prospect_reply rows with NULL outreach link
      null_event_ids           — rows missing event_id (pre-QC phase rows; expected)
      summary                  — counts and health verdict
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    # ── orphaned outreach: in outreach_log but no strategy_outcome_log row ────
    orphaned = conn.execute("""
        SELECT o.id, o.lead_id, o.sent_at, o.subject
        FROM   outreach_log o
        LEFT JOIN strategy_outcome_log s ON s.outreach_id = o.id
        WHERE  s.id IS NULL
        ORDER  BY o.sent_at DESC
        LIMIT  100
    """).fetchall()

    orphaned_rows = [
        {"outreach_id": r[0], "lead_id": r[1],
         "sent_at": r[2], "subject_preview": (r[3] or "")[:60]}
        for r in orphaned
    ]

    # Classify: pre-Phase-3a rows (no event_id) vs genuinely missing
    pre_phase3a_count = conn.execute("""
        SELECT COUNT(*) FROM outreach_log o
        LEFT JOIN strategy_outcome_log s ON s.outreach_id = o.id
        WHERE s.id IS NULL
    """).fetchone()[0]

    # ── duplicate reply mappings: two replies → same outreach ─────────────────
    dupes = conn.execute("""
        SELECT in_reply_to_outreach_id, COUNT(*) as cnt
        FROM   prospect_reply_log
        WHERE  in_reply_to_outreach_id IS NOT NULL
        GROUP  BY in_reply_to_outreach_id
        HAVING cnt > 1
        ORDER  BY cnt DESC
        LIMIT  50
    """).fetchall()

    duplicate_mappings = [
        {"outreach_id": r[0], "reply_count": r[1]} for r in dupes
    ]

    # ── cross-lead mismatches: reply.lead_id ≠ its linked outreach.lead_id ────
    mismatches = conn.execute("""
        SELECT r.id, r.lead_id, o.lead_id as outreach_lead_id,
               r.in_reply_to_outreach_id
        FROM   prospect_reply_log r
        JOIN   outreach_log o ON o.id = r.in_reply_to_outreach_id
        WHERE  r.lead_id != o.lead_id
        LIMIT  50
    """).fetchall()

    cross_lead = [
        {"reply_id": r[0], "reply_lead_id": r[1],
         "outreach_lead_id": r[2], "outreach_id": r[3]}
        for r in mismatches
    ]

    # ── leads with outreach but zero strategy_outcome rows ────────────────────
    missing_outcomes = conn.execute("""
        SELECT l.id, l.domain, COUNT(o.id) as outreach_count
        FROM   leads l
        JOIN   outreach_log o ON o.lead_id = l.id
        LEFT JOIN strategy_outcome_log s ON s.lead_id = l.id
        WHERE  s.id IS NULL
        GROUP  BY l.id
        ORDER  BY outreach_count DESC
        LIMIT  50
    """).fetchall()

    missing_outcome_rows = [
        {"lead_id": r[0], "domain": r[1], "outreach_count": r[2]}
        for r in missing_outcomes
    ]

    # ── unlinked replies: no in_reply_to_outreach_id ──────────────────────────
    unlinked = conn.execute("""
        SELECT COUNT(*) FROM prospect_reply_log
        WHERE  in_reply_to_outreach_id IS NULL
    """).fetchone()[0]

    total_replies = conn.execute(
        "SELECT COUNT(*) FROM prospect_reply_log"
    ).fetchone()[0]

    # ── null event_ids (pre-QC phase rows) ───────────────────────────────────
    null_event_ids = {}
    for table in ("outreach_log", "prospect_reply_log", "strategy_outcome_log"):
        try:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE event_id IS NULL"
            ).fetchone()[0]
            null_event_ids[table] = n
        except Exception:
            null_event_ids[table] = "column_absent"

    # ── summary ───────────────────────────────────────────────────────────────
    integrity_issues = (
        len(cross_lead) +
        len(duplicate_mappings)
    )
    health = (
        "healthy"   if integrity_issues == 0 else
        "degraded"  if integrity_issues <= 3 else
        "corrupted"
    )

    return {
        "orphaned_outreach":         orphaned_rows,
        "orphaned_count":            len(orphaned_rows),
        "pre_phase3a_orphan_count":  pre_phase3a_count,
        "duplicate_reply_mappings":  duplicate_mappings,
        "cross_lead_mismatches":     cross_lead,
        "missing_outcome_rows":      missing_outcome_rows,
        "unlinked_replies": {
            "count":        unlinked,
            "total_replies": total_replies,
            "pct":          round(unlinked / total_replies * 100, 1) if total_replies else 0,
        },
        "null_event_ids":            null_event_ids,
        "integrity_issues_count":    integrity_issues,
        "health":                    health,
        "note": (
            "orphaned_outreach rows without event_id are pre-Phase-3a rows — "
            "expected and not indicative of corruption."
        ),
    }


# ── 2. Lead attribution report ────────────────────────────────────────────────

def get_lead_attribution(db, lead_id: int) -> dict:
    """
    Full attribution chain for a single lead.

    Returns every outreach, its linked strategy outcome, angles used,
    objections detected, prospect replies, and the attribution confidence
    of each link. Suitable for manually verifying that the data pipeline
    captured events correctly.
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    lead_row = conn.execute(
        "SELECT * FROM leads WHERE id=?", (lead_id,)
    ).fetchone()
    if not lead_row:
        return {"error": f"lead {lead_id} not found"}

    lead = dict(lead_row)

    # Outreach chain — each outreach with its linked outcome + angles + replies
    outreach_rows = conn.execute(
        "SELECT * FROM outreach_log WHERE lead_id=? ORDER BY sent_at ASC",
        (lead_id,)
    ).fetchall()

    chain = []
    for o in outreach_rows:
        od = dict(o)

        # Strategy outcome for this outreach
        outcome = conn.execute(
            "SELECT * FROM strategy_outcome_log WHERE outreach_id=? OR "
            "(outreach_id IS NULL AND lead_id=? AND outreach_seq=?) LIMIT 1",
            (od["id"], lead_id, len(chain) + 1)
        ).fetchone()

        # Angles logged for this outreach_id
        angles = conn.execute(
            "SELECT angle_id, pitched_as, prospect_replied, reply_sentiment "
            "FROM angle_log WHERE outreach_id=? OR "
            "(outreach_id IS NULL AND lead_id=? AND outreach_seq=?)",
            (od["id"], lead_id, len(chain) + 1)
        ).fetchall()

        # Subject effectiveness
        subj = conn.execute(
            "SELECT subject_hash, subject_preview, got_reply, reply_sentiment "
            "FROM subject_effectiveness_log WHERE outreach_id=? OR "
            "(outreach_id IS NULL AND lead_id=? AND rowid IN "
            "(SELECT rowid FROM subject_effectiveness_log WHERE lead_id=? "
            "ORDER BY sent_at LIMIT 1 OFFSET ?))",
            (od["id"], lead_id, lead_id, len(chain))
        ).fetchone()

        # Prospect replies linked to this outreach
        replies = conn.execute(
            "SELECT id, word_count, has_questions, sentiment, "
            "attribution_confidence, received_at "
            "FROM prospect_reply_log WHERE in_reply_to_outreach_id=?",
            (od["id"],)
        ).fetchall()

        chain.append({
            "outreach_id":    od["id"],
            "event_id":       od.get("event_id"),
            "seq":            len(chain) + 1,
            "preset":         od.get("preset"),
            "subject":        od.get("subject"),
            "sent_at":        od.get("sent_at"),
            "strategy_outcome": dict(outcome) if outcome else None,
            "angles": [
                {"angle_id": a[0], "pitched_as": a[1],
                 "prospect_replied": bool(a[2]) if a[2] is not None else None,
                 "reply_sentiment": a[3]}
                for a in angles
            ],
            "subject_effectiveness": dict(subj) if subj else None,
            "prospect_replies": [
                {"reply_id": r[0], "word_count": r[1],
                 "has_questions": bool(r[2]), "sentiment": r[3],
                 "attribution_confidence": r[4], "received_at": r[5]}
                for r in replies
            ],
        })

    # Objections for this lead
    objections = conn.execute(
        "SELECT objection_type, addressed, detected_at, source_snippet "
        "FROM objection_log WHERE lead_id=? ORDER BY detected_at ASC",
        (lead_id,)
    ).fetchall()

    # Offers
    offers = conn.execute(
        "SELECT amount, direction, notes, offered_at "
        "FROM offer_log WHERE lead_id=? ORDER BY offered_at ASC",
        (lead_id,)
    ).fetchall()

    # Conversion events
    conversions = conn.execute(
        "SELECT event_type, final_stage, total_outreach_count, "
        "time_to_resolution, recorded_at "
        "FROM conversion_event_log WHERE lead_id=? ORDER BY recorded_at ASC",
        (lead_id,)
    ).fetchall()

    # Attribution health for this lead
    has_outreach_id_links = any(
        row["strategy_outcome"] and row["strategy_outcome"].get("outreach_id")
        for row in chain
    )
    any_replies = sum(len(row["prospect_replies"]) for row in chain)
    attribution_health = (
        "full"    if has_outreach_id_links and any_replies else
        "partial" if (chain or any_replies) else
        "none"
    )

    return {
        "lead":       lead,
        "outreach_chain": chain,
        "objections": [
            {"type": r[0], "addressed": bool(r[1]),
             "detected_at": r[2], "snippet": r[3]}
            for r in objections
        ],
        "offers": [
            {"amount": r[0], "direction": r[1],
             "notes": r[2], "offered_at": r[3]}
            for r in offers
        ],
        "conversion_events": [
            {"event_type": r[0], "final_stage": r[1],
             "total_outreach": r[2], "time_to_resolution_h": (
                 round(r[3] / 3600, 1) if r[3] else None
             ), "recorded_at": r[4]}
            for r in conversions
        ],
        "attribution_summary": {
            "total_outreach":         len(chain),
            "total_prospect_replies": any_replies,
            "has_outreach_id_links":  has_outreach_id_links,
            "attribution_health":     attribution_health,
        },
    }


# ── 3. Angle performance analytics ───────────────────────────────────────────

def get_angle_performance(db, min_samples: int = MIN_SAMPLE_SIZE) -> dict:
    """
    Aggregate angle effectiveness across all leads.

    For each angle_id computes:
      total_uses         — all angle_log rows for this angle
      primary_uses       — pitched_as='primary' rows only
      reply_rate         — primary uses that got a reply / primary uses
      positive_rate      — primary uses with positive sentiment / primary uses
      negative_rate      — primary uses with negative sentiment / primary uses
      no_reply_rate      — primary uses with no_reply or prospect_replied=0
      avg_word_count     — average prospect reply word count (quality proxy)
      trusted           — whether primary_uses >= min_samples
      confidence_note    — human-readable confidence explanation

    Returns scores sorted by weighted_score (reply_rate*0.4 + positive_rate*0.5
    - negative_rate*0.1), with a safety guard suppressing scores below
    min_samples.
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    rows = conn.execute("""
        SELECT
            a.angle_id,
            COUNT(*)                                            AS total_uses,
            SUM(CASE WHEN a.pitched_as='primary' THEN 1 ELSE 0 END)
                                                                AS primary_uses,
            SUM(CASE WHEN a.pitched_as='primary'
                          AND a.prospect_replied=1 THEN 1 ELSE 0 END)
                                                                AS got_reply,
            SUM(CASE WHEN a.pitched_as='primary'
                          AND a.reply_sentiment='positive' THEN 1 ELSE 0 END)
                                                                AS positive,
            SUM(CASE WHEN a.pitched_as='primary'
                          AND a.reply_sentiment='negative' THEN 1 ELSE 0 END)
                                                                AS negative,
            SUM(CASE WHEN a.pitched_as='primary'
                          AND (a.reply_sentiment='no_reply'
                               OR a.prospect_replied=0) THEN 1 ELSE 0 END)
                                                                AS no_reply_count
        FROM angle_log a
        GROUP BY a.angle_id
        ORDER BY a.angle_id
    """).fetchall()

    # Average word count from joined replies (via outreach_id)
    wc_map = {}
    try:
        wc_rows = conn.execute("""
            SELECT a.angle_id, AVG(r.word_count) as avg_wc
            FROM   angle_log a
            JOIN   prospect_reply_log r ON r.in_reply_to_outreach_id = a.outreach_id
            WHERE  a.pitched_as = 'primary'
            GROUP  BY a.angle_id
        """).fetchall()
        wc_map = {r[0]: round(r[1], 1) if r[1] else None for r in wc_rows}
    except Exception:
        pass

    angles = []
    for r in rows:
        angle_id, total, primary, got_reply, pos, neg, no_rep = r
        primary = primary or 0
        trusted = primary >= min_samples

        # Guard: suppress rates if below min_samples
        if primary > 0:
            reply_rate    = round((got_reply or 0) / primary, 3)
            positive_rate = round((pos or 0)       / primary, 3)
            negative_rate = round((neg or 0)       / primary, 3)
            no_reply_rate = round((no_rep or 0)    / primary, 3)
            weighted      = round(
                reply_rate * 0.4 + positive_rate * 0.5 - negative_rate * 0.1, 3
            )
        else:
            reply_rate = positive_rate = negative_rate = no_reply_rate = None
            weighted   = None

        angles.append({
            "angle_id":         angle_id,
            "total_uses":       total,
            "primary_uses":     primary,
            "got_reply":        got_reply or 0,
            "reply_rate":       reply_rate    if trusted else None,
            "positive_rate":    positive_rate if trusted else None,
            "negative_rate":    negative_rate if trusted else None,
            "no_reply_rate":    no_reply_rate if trusted else None,
            "weighted_score":   weighted      if trusted else None,
            "avg_reply_words":  wc_map.get(angle_id),
            "trusted":          trusted,
            "confidence_note":  (
                f"trusted ({primary} primary uses)"
                if trusted else
                f"insufficient data ({primary}/{min_samples} required primary uses)"
            ),
        })

    # Sort trusted angles by weighted_score desc, untrusted last
    trusted_angles   = sorted(
        [a for a in angles if a["trusted"]],
        key=lambda x: x["weighted_score"] or 0, reverse=True
    )
    untrusted_angles = sorted(
        [a for a in angles if not a["trusted"]],
        key=lambda x: x["primary_uses"], reverse=True
    )

    total_primary_uses = sum(a["primary_uses"] for a in angles)
    trusted_count      = len(trusted_angles)

    return {
        "angles":              trusted_angles + untrusted_angles,
        "total_primary_uses":  total_primary_uses,
        "trusted_angle_count": trusted_count,
        "min_sample_size":     min_samples,
        "adaptation_ready":    trusted_count >= 3,
        "adaptation_note": (
            f"{trusted_count} angle(s) have sufficient data for adaptive selection. "
            f"Minimum {min_samples} primary uses required per angle."
        ),
    }


# ── 4. CTA performance analytics ─────────────────────────────────────────────

def get_cta_performance(db, min_samples: int = MIN_SAMPLE_SIZE) -> dict:
    """
    Aggregate CTA style effectiveness from strategy_outcome_log.

    Groups by cta_style × goal and computes:
      total_uses, got_reply, reply_rate, positive_rate, weighted_score, trusted
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    rows = conn.execute("""
        SELECT
            cta_style,
            goal,
            COUNT(*)                                            AS total,
            SUM(CASE WHEN got_reply=1 THEN 1 ELSE 0 END)       AS replies,
            SUM(CASE WHEN reply_sentiment='positive' THEN 1 ELSE 0 END) AS pos,
            SUM(CASE WHEN reply_sentiment='negative' THEN 1 ELSE 0 END) AS neg,
            AVG(CASE WHEN got_reply IS NOT NULL
                     THEN attribution_confidence END)           AS avg_conf
        FROM strategy_outcome_log
        WHERE cta_style IS NOT NULL AND goal IS NOT NULL
        GROUP BY cta_style, goal
        ORDER BY cta_style, goal
    """).fetchall()

    cta_entries = []
    for r in rows:
        cta, goal, total, replies, pos, neg, avg_conf = r
        trusted = total >= min_samples
        outcome_known = conn.execute(
            "SELECT COUNT(*) FROM strategy_outcome_log "
            "WHERE cta_style=? AND goal=? AND got_reply IS NOT NULL",
            (cta, goal)
        ).fetchone()[0]

        if outcome_known > 0:
            rr  = round((replies or 0) / outcome_known, 3)
            pr  = round((pos or 0)     / outcome_known, 3)
            nr  = round((neg or 0)     / outcome_known, 3)
            ws  = round(rr * 0.4 + pr * 0.5 - nr * 0.1, 3)
        else:
            rr = pr = nr = ws = None

        cta_entries.append({
            "cta_style":        cta,
            "goal":             goal,
            "total_uses":       total,
            "outcomes_known":   outcome_known,
            "reply_rate":       rr  if trusted else None,
            "positive_rate":    pr  if trusted else None,
            "negative_rate":    nr  if trusted else None,
            "weighted_score":   ws  if trusted else None,
            "avg_attribution_confidence": round(avg_conf, 2) if avg_conf else None,
            "trusted":          trusted,
            "confidence_note": (
                f"trusted ({total} uses, {outcome_known} with known outcome)"
                if trusted else
                f"insufficient data ({total}/{min_samples} required)"
            ),
        })

    trusted_entries = sorted(
        [e for e in cta_entries if e["trusted"]],
        key=lambda x: x["weighted_score"] or 0, reverse=True
    )
    untrusted_entries = [e for e in cta_entries if not e["trusted"]]

    return {
        "cta_performance":         trusted_entries + untrusted_entries,
        "trusted_combinations":    len(trusted_entries),
        "min_sample_size":         min_samples,
        "adaptation_ready":        len(trusted_entries) >= 2,
        "adaptation_note": (
            f"{len(trusted_entries)} cta×goal combination(s) have sufficient data. "
            f"Minimum {min_samples} uses required per combination."
        ),
    }


# ── 5. Conversion funnel analytics ───────────────────────────────────────────

def get_conversion_funnel(db) -> dict:
    """
    High-level conversion funnel from first outreach to terminal outcome.

    Reports:
      stage_distribution       — count of leads at each stage
      conversion_rates         — accepted / total leads with ≥1 outreach
      avg_time_to_resolution_h — by event_type
      outreach_to_conversion   — avg outreach count before conversion
      angle_to_conversion      — which angles appear most in converted leads
      funnel_health            — data quality assessment
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    # Stage distribution
    stages = conn.execute(
        "SELECT stage, COUNT(*) FROM leads GROUP BY stage ORDER BY COUNT(*) DESC"
    ).fetchall()

    # Conversion event stats
    conv_stats = conn.execute("""
        SELECT event_type,
               COUNT(*)                          AS count,
               AVG(time_to_resolution)            AS avg_ttr_s,
               MIN(time_to_resolution)            AS min_ttr_s,
               MAX(time_to_resolution)            AS max_ttr_s,
               AVG(total_outreach_count)          AS avg_outreach
        FROM conversion_event_log
        GROUP BY event_type
        ORDER BY count DESC
    """).fetchall()

    total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    leads_with_outreach = conn.execute(
        "SELECT COUNT(DISTINCT lead_id) FROM outreach_log"
    ).fetchone()[0]
    total_accepted = conn.execute(
        "SELECT COUNT(*) FROM conversion_event_log WHERE event_type='accepted'"
    ).fetchone()[0]

    conversion_rate = (
        round(total_accepted / leads_with_outreach * 100, 1)
        if leads_with_outreach else 0
    )

    # Angles appearing in converted leads
    top_angles = conn.execute("""
        SELECT a.angle_id, COUNT(*) AS cnt
        FROM   angle_log a
        JOIN   conversion_event_log c ON c.lead_id = a.lead_id
        WHERE  c.event_type = 'accepted'
          AND  a.pitched_as = 'primary'
        GROUP  BY a.angle_id
        ORDER  BY cnt DESC
        LIMIT  5
    """).fetchall()

    # Funnel health: do we have enough conversion events to be meaningful?
    total_conversions = conn.execute(
        "SELECT COUNT(*) FROM conversion_event_log"
    ).fetchone()[0]

    funnel_health = (
        "rich"        if total_conversions >= 20 else
        "developing"  if total_conversions >= 5  else
        "sparse"
    )

    return {
        "totals": {
            "total_leads":           total_leads,
            "leads_with_outreach":   leads_with_outreach,
            "total_conversions":     total_conversions,
            "total_accepted":        total_accepted,
            "conversion_rate_pct":   conversion_rate,
        },
        "stage_distribution": [
            {"stage": r[0], "count": r[1]} for r in stages
        ],
        "conversion_events": [
            {
                "event_type":        r[0],
                "count":             r[1],
                "avg_time_h":        round(r[2] / 3600, 1) if r[2] else None,
                "min_time_h":        round(r[3] / 3600, 1) if r[3] else None,
                "max_time_h":        round(r[4] / 3600, 1) if r[4] else None,
                "avg_outreach_count": round(r[5], 1) if r[5] else None,
            }
            for r in conv_stats
        ],
        "top_angles_in_conversions": [
            {"angle_id": r[0], "conversion_count": r[1]} for r in top_angles
        ],
        "funnel_health":  funnel_health,
        "funnel_note": (
            "Conversion data is sparse — funnel metrics will become meaningful "
            f"after ~20 conversion events (currently {total_conversions})."
            if funnel_health == "sparse" else
            f"Funnel data is {funnel_health} ({total_conversions} conversion events)."
        ),
    }


# ── 6. Reply latency analytics ────────────────────────────────────────────────

def get_reply_latency(db) -> dict:
    """
    Distribution of prospect reply latencies (time from outreach to reply).

    Uses strategy_outcome_log.reply_latency_s populated by backfill.
    Buckets: <1h, 1–6h, 6–24h, 1–3d, >3d.

    Also reports:
      median_latency_h          — median response time in hours
      pct_replied               — percentage of tracked outreach that got a reply
      latency_by_angle          — median latency per selected_angle
      latency_by_goal           — median latency per primary goal
      attribution_coverage      — % of latency values with high confidence
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    latency_rows = conn.execute("""
        SELECT reply_latency_s, selected_angle, goal, attribution_confidence
        FROM   strategy_outcome_log
        WHERE  reply_latency_s IS NOT NULL
        ORDER  BY reply_latency_s ASC
    """).fetchall()

    total_outcomes = conn.execute(
        "SELECT COUNT(*) FROM strategy_outcome_log"
    ).fetchone()[0]

    if not latency_rows:
        return {
            "message": "No latency data yet — reply backfill has not run.",
            "total_outcomes": total_outcomes,
            "latency_records": 0,
        }

    latencies_s = [r[0] for r in latency_rows]
    n = len(latencies_s)

    # Percentiles
    def pct(lst, p):
        idx = max(0, int(len(lst) * p / 100) - 1)
        return round(lst[idx] / 3600, 2)

    # Buckets (seconds thresholds)
    buckets = [
        ("< 1 hour",   0,        3_600),
        ("1–6 hours",  3_600,    21_600),
        ("6–24 hours", 21_600,   86_400),
        ("1–3 days",   86_400,   259_200),
        ("> 3 days",   259_200,  float("inf")),
    ]
    bucket_counts = {}
    for label, lo, hi in buckets:
        bucket_counts[label] = sum(1 for s in latencies_s if lo <= s < hi)

    # By angle
    from collections import defaultdict
    angle_latencies: dict[str, list] = defaultdict(list)
    goal_latencies:  dict[str, list] = defaultdict(list)
    high_conf_count = 0
    for s, angle, goal, conf in latency_rows:
        if angle:
            angle_latencies[angle].append(s)
        if goal:
            goal_latencies[goal].append(s)
        if conf and conf >= CONF_HIGH:
            high_conf_count += 1

    def median_h(lst):
        if not lst: return None
        sl = sorted(lst)
        mid = len(sl) // 2
        m = sl[mid] if len(sl) % 2 else (sl[mid-1] + sl[mid]) / 2
        return round(m / 3600, 2)

    by_angle = {
        a: {"median_h": median_h(v), "count": len(v)}
        for a, v in sorted(angle_latencies.items())
    }
    by_goal = {
        g: {"median_h": median_h(v), "count": len(v)}
        for g, v in sorted(goal_latencies.items())
    }

    return {
        "latency_records":   n,
        "total_outcomes":    total_outcomes,
        "pct_with_latency":  round(n / total_outcomes * 100, 1) if total_outcomes else 0,
        "percentiles_h": {
            "p25":    pct(latencies_s, 25),
            "p50":    pct(latencies_s, 50),
            "p75":    pct(latencies_s, 75),
            "p90":    pct(latencies_s, 90),
        },
        "bucket_distribution": bucket_counts,
        "by_angle":  by_angle,
        "by_goal":   by_goal,
        "attribution_coverage": {
            "high_confidence_count": high_conf_count,
            "high_confidence_pct":   round(high_conf_count / n * 100, 1) if n else 0,
            "note": f"High confidence = attribution_confidence >= {CONF_HIGH}",
        },
    }


# ── 7. Score safety guards ────────────────────────────────────────────────────

def score_safety_check(db, min_samples: int = MIN_SAMPLE_SIZE) -> dict:
    """
    Dry-run: preview which scores would be trusted by Phase 3b/3c.

    Returns a safety report with:
      angles_ready           — angles with >= min_samples primary uses
      angles_not_ready       — angles below threshold
      cta_combinations_ready — cta×goal combos with >= min_samples
      stale_warning          — True if all scores are from > 24h ago
      suppression_summary    — what Phase 3b would suppress
      recommendation         — go / hold / caution verdict

    This is a READ-ONLY preview. Calling this endpoint does not activate
    adaptive scoring.
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    # Per-angle primary use counts
    angle_counts = conn.execute("""
        SELECT angle_id,
               SUM(CASE WHEN pitched_as='primary' THEN 1 ELSE 0 END) AS primary_uses
        FROM   angle_log
        GROUP  BY angle_id
    """).fetchall()

    angles_ready     = []
    angles_not_ready = []
    for angle_id, primary in angle_counts:
        (angles_ready if primary >= min_samples else angles_not_ready).append({
            "angle_id":     angle_id,
            "primary_uses": primary,
            "gap":          max(0, min_samples - primary),
        })

    # CTA combinations
    cta_counts = conn.execute("""
        SELECT cta_style, goal, COUNT(*) AS total
        FROM   strategy_outcome_log
        WHERE  cta_style IS NOT NULL AND goal IS NOT NULL
        GROUP  BY cta_style, goal
    """).fetchall()

    cta_ready     = []
    cta_not_ready = []
    for cta, goal, total in cta_counts:
        (cta_ready if total >= min_samples else cta_not_ready).append({
            "cta_style": cta, "goal": goal, "uses": total,
            "gap": max(0, min_samples - total),
        })

    # Staleness check — when was the most recent strategy_outcome row?
    latest_ts = conn.execute(
        "SELECT MAX(decided_at) FROM strategy_outcome_log"
    ).fetchone()[0]
    stale = False
    staleness_note = "no data"
    if latest_ts:
        age_s = time.time() - latest_ts
        stale = age_s > SCORE_STALE_THRESHOLD_S
        staleness_note = (
            f"last outcome {round(age_s/3600, 1)}h ago — "
            + ("STALE (>24h)" if stale else "fresh")
        )

    # Recommendation
    if len(angles_ready) >= 5 and not stale:
        verdict = "go"
        verdict_note = (
            f"{len(angles_ready)} angles ready. Data is fresh. "
            "Phase 3b score materialization can proceed."
        )
    elif len(angles_ready) >= 3 and not stale:
        verdict = "caution"
        verdict_note = (
            f"Only {len(angles_ready)} angles ready (5 recommended). "
            "Phase 3b can run with reduced coverage."
        )
    else:
        verdict = "hold"
        verdict_note = (
            f"Only {len(angles_ready)} angles have sufficient data. "
            f"Need {min_samples} primary uses each. Continue data collection."
        )

    return {
        "min_sample_size":     min_samples,
        "angles_ready":        sorted(angles_ready, key=lambda x: -x["primary_uses"]),
        "angles_not_ready":    sorted(angles_not_ready, key=lambda x: x["gap"]),
        "cta_ready":           sorted(cta_ready, key=lambda x: -x["uses"]),
        "cta_not_ready":       sorted(cta_not_ready, key=lambda x: x["gap"]),
        "staleness_check":     {"stale": stale, "note": staleness_note},
        "suppression_summary": {
            "angles_that_would_be_suppressed": len(angles_not_ready),
            "angles_that_would_be_active":     len(angles_ready),
            "cta_combos_that_would_be_active": len(cta_ready),
        },
        "recommendation":      verdict,
        "recommendation_note": verdict_note,
    }


# ── 8. Analytics materialization verification ─────────────────────────────────

def verify_materialization_readiness(
    db,
    min_samples:  int   = MIN_SAMPLE_SIZE,
    dry_run:      bool  = True,
    regression_baseline: Optional[dict] = None,
) -> dict:
    """
    Verify that the data in broker_memory is consistent enough to support
    Phase 3b score materialization.

    dry_run=True  — report only; no writes (always true in Phase QC)
    dry_run=False — reserved for Phase 3b; this function always treats
                    Phase QC calls as dry_run=True regardless of the flag.

    regression_baseline — optional dict from a previous call to this function.
                          If supplied, reports whether scores would have changed
                          since the baseline. Used to detect data drift.

    Returns:
      data_quality_checks  — row counts, NULL rates, orphan rates
      score_preview        — what scores Phase 3b would compute (without writing)
      regression_report    — comparison to baseline (if supplied)
      readiness_verdict    — pass / warn / fail
    """
    if not db or not db.available:
        return {"error": "broker_memory unavailable"}

    conn = db._conn

    # ── Data quality checks ───────────────────────────────────────────────────
    def null_rate(table, col):
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            nulls = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
            ).fetchone()[0]
            return {"total": total, "nulls": nulls,
                    "null_pct": round(nulls/total*100, 1) if total else 0}
        except Exception:
            return {"total": 0, "nulls": 0, "null_pct": 0, "error": "column absent"}

    quality = {
        "outreach_log": {
            "event_id":  null_rate("outreach_log", "event_id"),
        },
        "prospect_reply_log": {
            "event_id":               null_rate("prospect_reply_log", "event_id"),
            "attribution_confidence": null_rate("prospect_reply_log", "attribution_confidence"),
            "in_reply_to_outreach_id": null_rate("prospect_reply_log", "in_reply_to_outreach_id"),
        },
        "strategy_outcome_log": {
            "event_id":               null_rate("strategy_outcome_log", "event_id"),
            "attribution_confidence": null_rate("strategy_outcome_log", "attribution_confidence"),
            "got_reply":              null_rate("strategy_outcome_log", "got_reply"),
            "outreach_id":            null_rate("strategy_outcome_log", "outreach_id"),
        },
        "angle_log": {
            "prospect_replied": null_rate("angle_log", "prospect_replied"),
            "reply_sentiment":  null_rate("angle_log", "reply_sentiment"),
            "outreach_id":      null_rate("angle_log", "outreach_id"),
        },
    }

    # ── Score preview (dry-run computation) ───────────────────────────────────
    # Compute what scores WOULD look like — without writing them anywhere.
    angle_preview = []
    angle_rows = conn.execute("""
        SELECT angle_id,
               SUM(CASE WHEN pitched_as='primary' THEN 1 ELSE 0 END)    AS pu,
               SUM(CASE WHEN pitched_as='primary' AND prospect_replied=1 THEN 1 ELSE 0 END) AS rep,
               SUM(CASE WHEN pitched_as='primary' AND reply_sentiment='positive' THEN 1 ELSE 0 END) AS pos,
               SUM(CASE WHEN pitched_as='primary' AND reply_sentiment='negative' THEN 1 ELSE 0 END) AS neg
        FROM angle_log GROUP BY angle_id
    """).fetchall()

    for a_id, pu, rep, pos, neg in angle_rows:
        pu = pu or 0
        if pu >= min_samples:
            rr = round((rep or 0) / pu, 3)
            pr = round((pos or 0) / pu, 3)
            nr = round((neg or 0) / pu, 3)
            ws = round(rr*0.4 + pr*0.5 - nr*0.1, 3)
            angle_preview.append({
                "angle_id": a_id, "primary_uses": pu,
                "reply_rate": rr, "positive_rate": pr,
                "negative_rate": nr, "weighted_score": ws, "trusted": True,
            })
        else:
            angle_preview.append({
                "angle_id": a_id, "primary_uses": pu,
                "trusted": False, "weighted_score": None,
            })

    # ── Regression comparison ─────────────────────────────────────────────────
    regression_report = None
    if regression_baseline and "score_preview" in regression_baseline:
        prev_scores = {
            r["angle_id"]: r.get("weighted_score")
            for r in regression_baseline["score_preview"]
        }
        curr_scores = {
            r["angle_id"]: r.get("weighted_score")
            for r in angle_preview
        }
        changed = []
        for a_id in set(prev_scores) | set(curr_scores):
            prev = prev_scores.get(a_id)
            curr = curr_scores.get(a_id)
            if prev != curr:
                changed.append({
                    "angle_id": a_id,
                    "previous": prev,
                    "current":  curr,
                    "delta":    round((curr or 0) - (prev or 0), 3),
                })
        regression_report = {
            "angles_changed": len(changed),
            "changes":        sorted(changed, key=lambda x: abs(x["delta"]), reverse=True),
            "drift_detected": len(changed) > 0,
        }

    # ── Readiness verdict ─────────────────────────────────────────────────────
    trusted_count       = sum(1 for a in angle_preview if a["trusted"])
    got_reply_null_pct  = quality["strategy_outcome_log"]["got_reply"]["null_pct"]
    event_id_null_pct   = quality["outreach_log"]["event_id"]["null_pct"]

    if trusted_count >= 5 and got_reply_null_pct < 50:
        verdict = "pass"
        note    = "Data quality sufficient for Phase 3b materialization."
    elif trusted_count >= 2 and got_reply_null_pct < 80:
        verdict = "warn"
        note    = (
            f"Partial readiness: {trusted_count} trusted angles, "
            f"{got_reply_null_pct}% of outcome rows still NULL. "
            "Continue data collection or run /leads/{id}/reply backfill."
        )
    else:
        verdict = "fail"
        note    = (
            f"Not ready: {trusted_count} trusted angles (need 5), "
            f"{got_reply_null_pct}% outcome NULLs. "
            "Phase 3b should not activate yet."
        )

    return {
        "dry_run":              True,  # always True in Phase QC
        "min_sample_size":      min_samples,
        "data_quality_checks":  quality,
        "score_preview":        sorted(angle_preview,
                                       key=lambda x: x.get("weighted_score") or -1,
                                       reverse=True),
        "regression_report":    regression_report,
        "trusted_angle_count":  trusted_count,
        "readiness_verdict":    verdict,
        "readiness_note":       note,
        "event_id_coverage": {
            "outreach_null_pct": event_id_null_pct,
            "note": (
                "NULL event_ids are pre-Phase-QC rows — expected. "
                "New rows written after Phase QC will have event_ids."
            ),
        },
    }
