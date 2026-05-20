"""
strategy_eval.py — Strategy Evaluation & Calibration Layer
===========================================================
Phase 3 of the reasoning-first architecture.

PURPOSE
-------
After a reply is generated, evaluate whether it actually followed the
intended ReplyStrategy. This is the feedback loop that makes the strategy
layer self-correcting over time.

DESIGN PRINCIPLES
-----------------
- Zero model calls. All evaluation is heuristic and deterministic.
- Non-blocking. Evaluation failures never affect reply delivery.
- Additive only. No changes to generation, humanization, or QC systems.
- Strategy-aware. Uses the typed ReplyStrategy object, not raw prompts.

ROLLBACK
--------
Remove the import from main.py and the two call sites in
generate_variations_ai() and /qc/replay-strategy. Everything else is
untouched.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from reply_strategy import ReplyStrategy

# ── Constants ─────────────────────────────────────────────────────────────────

ANALYTICS_DB = Path(__file__).parent / "strategy_analytics.db"

ProgressionVerdict = Literal["progressed", "neutral", "regressed"]
ConfidenceMismatch = Literal["overconfident", "underconfident", "aligned", "unchecked"]


# ── CTA detection patterns ─────────────────────────────────────────────────────
# Maps cta_style → signals that indicate the CTA style was followed

_CTA_SIGNALS: dict[str, list[str]] = {
    "soft_question":    [r"\?$", r"would (you|this|it)", r"does (this|that) (sound|seem|work)",
                         r"is (this|that) (something|of interest)", r"thoughts\?"],
    "forward_question": [r"would you (like|be interested|want)", r"shall (we|i)",
                         r"ready to (take|move|discuss|proceed)", r"want me to"],
    "specific_counter": [r"\$[\d,]+", r"£[\d,]+", r"€[\d,]+",
                         r"counter(ing)?( offer)?", r"(my|our) (price|offer|counter) is",
                         r"willing to (accept|take|do)"],
    "decision_prompt":  [r"ready (whenever|when)", r"(your|the) (call|decision|choice)",
                         r"let me know (how|when|if) you('d| would) like to proceed"],
    "transaction":      [r"(purchase|buy|secure|get) (it |this |the domain )?(here|now|directly)",
                         r"www\.", r"http", r"payment", r"escrow"],
    "exit_open_door":   [r"(no (pressure|rush|obligation))", r"whenever (you'?re|the time is)",
                         r"(feel free to|happy to) (reach out|reconnect|revisit)",
                         r"(wishing you|all the best|take care)"],
    "none":             [],  # informational — no CTA expected
}

# Signals that indicate progression (reply moved conversation forward)
_PROGRESSION_SIGNALS: list[str] = [
    r"to (address|answer|clarify) your (question|concern|point)",
    r"(based on|given) what you (said|mentioned|shared)",
    r"(i understand|i hear|i see) (that|why|your)",
    r"specifically (for|to|in)",
    r"(next step|move forward|proceed)",
    r"(happy to|can) (set up|arrange|schedule|send)",
    r"(escrow|transfer|payment) (process|link|details)",
    r"(let me|allow me to) (clarify|explain|address)",
]

# Signals that indicate regression (reply restarted pitch from scratch)
_REGRESSION_SIGNALS: list[str] = [
    r"i (am|'m) (reaching out|writing) (today )?to (inform|let you know|introduce)",
    r"(imagine|picture|consider) (having|owning|capturing)",
    r"i (came across|noticed|found) your (business|company|website)",
    r"(this domain|the domain) (could|can|will) (help|drive|boost|transform)",
    r"i (wanted|want) to (reach out|follow up|touch base|introduce)",
]

# Assertive language that signals overconfidence
_ASSERTIVE_SIGNALS: list[str] = [
    r"(you (must|need to|should) (act|decide|buy|secure))",
    r"(this (won't|will not) (last|be available))",
    r"(final (offer|price|chance))",
    r"(act (now|fast|quickly|today))",
    r"(don't (miss|wait|delay))",
    r"(last (chance|opportunity))",
]

# Weak/hedging language that signals underconfidence in negotiation
_HEDGING_SIGNALS: list[str] = [
    r"(maybe|perhaps|possibly) (we|i|you) (could|can|might)",
    r"(i'?m not sure|not certain|hard to say)",
    r"(whatever (works|you think|you prefer))",
    r"(any (price|offer|amount) (works|is fine|is okay))",
    r"(i understand if (you|that) (is|seems|sounds|feels) too (much|high|expensive))",
    r"(feel free to (offer|suggest|propose) (what|whatever))",
]

# Import topic signals from reply_strategy for repetition detection
from reply_strategy import _VALUE_TOPIC_SIGNALS


# ── Adherence dimension evaluators ────────────────────────────────────────────

def _check_cta_adherence(reply: str, cta_style: str) -> tuple[bool, str]:
    """
    Check whether the reply contains a CTA matching the intended style.
    Returns (passed, explanation).
    """
    if cta_style == "none":
        # No CTA expected — check it doesn't have a pushy one
        has_pushy = any(
            re.search(p, reply, re.IGNORECASE)
            for p in _CTA_SIGNALS.get("decision_prompt", []) +
                     _CTA_SIGNALS.get("transaction", [])
        )
        return (not has_pushy), ("unexpected pushy CTA in informational reply" if has_pushy else "ok")

    signals = _CTA_SIGNALS.get(cta_style, [])
    if not signals:
        return True, "no signals defined for this CTA style"

    matched = any(re.search(p, reply, re.IGNORECASE) for p in signals)
    if matched:
        return True, "ok"

    # Partial credit: any question mark in a reply that wanted a question
    if cta_style in ("soft_question", "forward_question") and "?" in reply:
        return True, "question present (loose match)"

    return False, f"reply does not appear to follow '{cta_style}' CTA style"


def _check_prohibited_adherence(reply: str, prohibited_topics: list[str]) -> list[str]:
    """
    Check which prohibited topics appear to be violated in the reply.
    Returns list of violated prohibition labels.
    """
    violations: list[str] = []
    reply_low = reply.lower()

    # Map prohibition text patterns → detection keywords
    prohibition_detectors: list[tuple[str, list[str]]] = [
        ("no pitch content",        ["domain", "seo", "traffic", "rank", "brand", "purchase"]),
        ("no value propositions",   ["seo", "traffic", "brand", "visibility", "rank"]),
        ("re-sell",                 ["domain can help", "domain will", "perfect for", "ideal for"]),
        ("no guilt language",       ["still waiting", "haven't heard", "reaching out again",
                                     "just following up", "touching base"]),
        ("do not lower the price",  ["reduced", "lower", "discount", "special price", "knock off"]),
        ("no counter-pitch",        ["however", "that said", "but consider", "actually"]),
        ("two sentences maximum",   None),   # checked by length
        ("do not repeat",           None),   # handled by repetition detector
    ]

    for prohibition in prohibited_topics:
        p_low = prohibition.lower()
        for label, keywords in prohibition_detectors:
            if label in p_low and keywords:
                if any(kw in reply_low for kw in keywords):
                    violations.append(prohibition[:60])
                    break
            elif label in p_low and keywords is None:
                if label == "two sentences maximum":
                    sentences = re.split(r'(?<=[.!?])\s+', reply.strip())
                    real = [s for s in sentences if len(s.split()) > 3]
                    if len(real) > 3:
                        violations.append("reply exceeds length prohibition")
    return violations


def _check_persuasion_calibration(reply: str, persuasion_level: int) -> tuple[bool, str]:
    """
    Check whether the reply's persuasion level matches the strategy intent.
    Returns (passed, explanation).
    """
    reply_low = reply.lower()

    # Count value-pitch phrases
    pitch_signals = [
        "seo", "rank", "traffic", "brand", "visibility", "capture",
        "imagine", "potential", "opportunity", "benefit", "advantage",
    ]
    pitch_count = sum(1 for s in pitch_signals if s in reply_low)

    if persuasion_level == 0 and pitch_count >= 3:
        return False, f"over-persuasive for persuasion_level=0 ({pitch_count} pitch signals)"
    if persuasion_level == 3 and pitch_count == 0:
        return False, "no value content despite persuasion_level=3"
    return True, "ok"


# ── Repetition violation detector ─────────────────────────────────────────────

def detect_repetition_violations(reply: str, suppressed_topics: list[str]) -> list[str]:
    """
    Detect whether the reply re-explains value topics already covered in
    prior outreach (as identified by the strategy's suppressed_topics list).

    Returns list of violated topic labels.
    """
    if not suppressed_topics:
        return []

    reply_low = reply.lower()
    violations: list[str] = []

    for topic in suppressed_topics:
        signals = _VALUE_TOPIC_SIGNALS.get(topic, [])
        if any(signal in reply_low for signal in signals):
            violations.append(topic)

    return violations


# ── Progression evaluator ─────────────────────────────────────────────────────

def evaluate_progression(
    reply:           str,
    progression_goal: str,
    primary_goal:    str,
    outreach_count:  int,
) -> tuple[ProgressionVerdict, str]:
    """
    Evaluate whether the reply moved the conversation forward.
    Returns (verdict, explanation).

    progressed → reply contains new, relevant engagement signals
    neutral    → reply is acceptable but doesn't clearly advance
    regressed  → reply restarts the pitch or repeats prior content
    """
    reply_low = reply.lower()

    # Check regression first — hardest failure
    regression_hits = sum(
        1 for p in _REGRESSION_SIGNALS
        if re.search(p, reply_low)
    )
    if regression_hits >= 2:
        return "regressed", f"reply contains {regression_hits} re-pitch signals (restarted original pitch)"
    if regression_hits == 1 and outreach_count >= 2:
        return "regressed", "re-pitch signal detected in later-stage follow-up"

    # Check progression signals
    progression_hits = sum(
        1 for p in _PROGRESSION_SIGNALS
        if re.search(p, reply_low)
    )

    # Goal-specific progression checks
    if primary_goal == "counter_offer":
        has_number = bool(re.search(r"[\$£€][\d,]+", reply))
        if has_number:
            return "progressed", "specific counter-figure present"
        return "neutral", "counter_offer goal but no specific price figure found"

    if primary_goal in ("close", "confirm_next_step"):
        has_transaction = any(
            re.search(p, reply_low)
            for p in _CTA_SIGNALS.get("transaction", [])
        )
        if has_transaction:
            return "progressed", "transaction signal present for close/confirm goal"

    if primary_goal == "defuse":
        # Short and no pitch = progressed for defuse
        word_count = len(reply.split())
        has_pitch  = any(s in reply_low for s in ["seo", "traffic", "rank", "domain can"])
        if word_count < 80 and not has_pitch:
            return "progressed", "brief and clean defuse response"
        if has_pitch:
            return "regressed", "pitch content in defuse reply"

    if progression_hits >= 2:
        return "progressed", f"{progression_hits} progression signals detected"
    if progression_hits == 1:
        return "neutral", "one progression signal — reply acceptable but not strongly advancing"

    # Default: neutral for first outreach (no prior to compare against)
    if outreach_count == 0:
        return "neutral", "first outreach — no regression baseline"

    return "neutral", "no clear progression or regression signals"


# ── Confidence mismatch detector ──────────────────────────────────────────────

def detect_confidence_mismatch(
    reply:    str,
    strategy: "ReplyStrategy",
) -> tuple[ConfidenceMismatch, str]:
    """
    Check whether the reply's assertiveness matches the strategy's confidence level.

    overconfident  → low confidence strategy but assertive reply
    underconfident → high confidence negotiation but weak/hedging reply
    aligned        → assertiveness matches confidence
    unchecked      → not enough signal to judge
    """
    overall_conf = min(
        strategy.stage_confidence,
        strategy.buyer_confidence,
        strategy.goal_confidence,
    )
    reply_low = reply.lower()

    assertive_count = sum(
        1 for p in _ASSERTIVE_SIGNALS
        if re.search(p, reply_low)
    )
    hedging_count = sum(
        1 for p in _HEDGING_SIGNALS
        if re.search(p, reply_low)
    )

    # Low confidence + assertive language = overconfident generation
    if overall_conf < 0.60 and assertive_count >= 1:
        return (
            "overconfident",
            f"strategy confidence={overall_conf:.2f} but reply contains "
            f"{assertive_count} assertive signal(s) — tone more certain than evidence warrants"
        )

    # High confidence negotiation goal + hedging language = underconfident
    is_negotiation_goal = strategy.primary_goal in ("counter_offer", "hold_position", "close")
    if (overall_conf >= 0.80 and is_negotiation_goal and hedging_count >= 2):
        return (
            "underconfident",
            f"high-confidence {strategy.primary_goal} strategy but reply contains "
            f"{hedging_count} hedging signal(s) — may undermine negotiation position"
        )

    if assertive_count == 0 and hedging_count == 0:
        return "unchecked", "no assertive or hedging signals detected"

    return "aligned", f"assertive={assertive_count} hedging={hedging_count} conf={overall_conf:.2f}"


# ── Master adherence evaluator ────────────────────────────────────────────────

def evaluate_adherence(
    reply:    str,
    strategy: "ReplyStrategy",
) -> dict:
    """
    Evaluate whether the generated reply followed the intended ReplyStrategy.
    Zero model calls. All heuristic.

    Returns:
    {
        "adherence_score":       int 0-100,
        "failed_dimensions":     list[str],
        "passed_dimensions":     list[str],
        "explanation":           str,
        "dimension_results":     dict,
    }
    """
    failed:  list[str] = []
    passed:  list[str] = []
    details: dict      = {}

    # ── 1. CTA adherence ─────────────────────────────────────────────────────
    cta_ok, cta_note = _check_cta_adherence(reply, strategy.cta_style)
    details["cta"] = {"passed": cta_ok, "note": cta_note}
    (passed if cta_ok else failed).append("cta_style")

    # ── 2. Prohibited topics ──────────────────────────────────────────────────
    proh_violations = _check_prohibited_adherence(reply, strategy.prohibited_topics)
    proh_ok = len(proh_violations) == 0
    details["prohibited_topics"] = {"passed": proh_ok, "violations": proh_violations}
    (passed if proh_ok else failed).append("prohibited_topics")

    # ── 3. Repetition suppression ─────────────────────────────────────────────
    rep_violations = detect_repetition_violations(reply, strategy.suppressed_topics)
    rep_ok = len(rep_violations) == 0
    details["repetition_suppression"] = {"passed": rep_ok, "violations": rep_violations}
    (passed if rep_ok else failed).append("repetition_suppression")

    # ── 4. Persuasion calibration ─────────────────────────────────────────────
    pers_ok, pers_note = _check_persuasion_calibration(reply, strategy.persuasion_level)
    details["persuasion_calibration"] = {"passed": pers_ok, "note": pers_note}
    (passed if pers_ok else failed).append("persuasion_calibration")

    # ── 5. Progression ────────────────────────────────────────────────────────
    prog_verdict, prog_note = evaluate_progression(
        reply, strategy.progression_goal, strategy.primary_goal,
        outreach_count=len(strategy.suppressed_topics),  # proxy for prior contact depth
    )
    prog_ok = prog_verdict != "regressed"
    details["progression"] = {
        "passed":  prog_ok,
        "verdict": prog_verdict,
        "note":    prog_note,
    }
    (passed if prog_ok else failed).append("progression")

    # ── 6. Confidence alignment ───────────────────────────────────────────────
    mismatch, mismatch_note = detect_confidence_mismatch(reply, strategy)
    conf_ok = mismatch in ("aligned", "unchecked")
    details["confidence_alignment"] = {
        "passed":   conf_ok,
        "verdict":  mismatch,
        "note":     mismatch_note,
    }
    (passed if conf_ok else failed).append("confidence_alignment")

    # ── Score ─────────────────────────────────────────────────────────────────
    # Weights: CTA=20, prohibited=20, repetition=20, persuasion=15, progression=15, confidence=10
    weights = {
        "cta_style":            20,
        "prohibited_topics":    20,
        "repetition_suppression": 20,
        "persuasion_calibration": 15,
        "progression":          15,
        "confidence_alignment": 10,
    }
    score = sum(weights[d] for d in passed if d in weights)

    # ── Explanation ───────────────────────────────────────────────────────────
    if not failed:
        explanation = f"Reply followed all {len(passed)} strategy dimensions. Score: {score}/100."
    else:
        issues = "; ".join(
            details[d].get("note") or details[d].get("violations", ["?"])[0]
            for d in failed
            if d in details
        )
        explanation = f"Score {score}/100. Failed: {', '.join(failed)}. Issues: {issues}"

    return {
        "adherence_score":   score,
        "failed_dimensions": failed,
        "passed_dimensions": passed,
        "explanation":       explanation,
        "dimension_results": details,
    }


# ── Top-level composite evaluator ─────────────────────────────────────────────

def evaluate_strategy_adherence(
    reply:    str,
    strategy: "ReplyStrategy",
) -> dict:
    """
    Run the full evaluation suite and return a single composite report
    ready to be merged into quality_report.

    This is the function called from generate_variations_ai() and
    /qc/replay-strategy.

    Returns:
    {
        "strategy_adherence":    { adherence_score, failed_dimensions, ... },
        "repetition_violations": list[str],
        "progression_result":    { verdict, note },
        "confidence_alignment":  { verdict, note },
    }
    """
    try:
        adherence = evaluate_adherence(reply, strategy)
        rep_viol  = detect_repetition_violations(reply, strategy.suppressed_topics)

        prog_verdict, prog_note = evaluate_progression(
            reply, strategy.progression_goal, strategy.primary_goal,
            outreach_count=len(strategy.suppressed_topics),
        )
        mismatch, mismatch_note = detect_confidence_mismatch(reply, strategy)

        return {
            "strategy_adherence":   adherence,
            "repetition_violations": rep_viol,
            "progression_result":   {"verdict": prog_verdict, "note": prog_note},
            "confidence_alignment": {"verdict": mismatch,     "note": mismatch_note},
        }
    except Exception as e:
        # Never block reply delivery
        print(f"[STRATEGY_EVAL] evaluation failed (non-blocking): {e}")
        return {}


# ── Learning analytics ────────────────────────────────────────────────────────

class StrategyAnalytics:
    """
    Lightweight SQLite-backed analytics store.
    Tracks strategy outcomes to identify patterns over time.
    No ML — simple counters and averages only.
    Thread-safe via check_same_thread=False.
    """

    def __init__(self, db_path: Path = ANALYTICS_DB):
        self._path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init()

    def _init(self) -> None:
        try:
            self._conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_outcomes (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts               REAL    NOT NULL,
                    primary_goal     TEXT    NOT NULL,
                    buyer_state      TEXT    NOT NULL,
                    email_preset     TEXT,
                    stage            TEXT,
                    adherence_score  INTEGER NOT NULL,
                    progression      TEXT    NOT NULL,
                    conf_mismatch    TEXT    NOT NULL,
                    rep_violations   INTEGER NOT NULL DEFAULT 0,
                    failed_dims      TEXT    NOT NULL DEFAULT ''
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_so_goal
                ON strategy_outcomes(primary_goal)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_so_buyer
                ON strategy_outcomes(buyer_state)
            """)
            self._conn.commit()
        except Exception as e:
            print(f"[StrategyAnalytics] init failed (non-blocking): {e}")
            self._conn = None

    @property
    def available(self) -> bool:
        return self._conn is not None

    def record(
        self,
        strategy:        "ReplyStrategy",
        eval_result:     dict,
        email_preset:    Optional[str] = None,
    ) -> None:
        """Record one strategy outcome. Safe to call — never raises."""
        if not self.available or not eval_result:
            return
        try:
            adherence   = eval_result.get("strategy_adherence", {})
            progression = eval_result.get("progression_result", {})
            conf_align  = eval_result.get("confidence_alignment", {})
            rep_viol    = eval_result.get("repetition_violations", [])

            self._conn.execute("""
                INSERT INTO strategy_outcomes
                (ts, primary_goal, buyer_state, email_preset, stage,
                 adherence_score, progression, conf_mismatch, rep_violations, failed_dims)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                time.time(),
                strategy.primary_goal,
                strategy.buyer_state,
                email_preset,
                strategy.reasoning_trace.get("goal_source", ""),
                adherence.get("adherence_score", 0),
                progression.get("verdict", "neutral"),
                conf_align.get("verdict",  "unchecked"),
                len(rep_viol),
                ",".join(adherence.get("failed_dimensions", [])),
            ))
            self._conn.commit()
        except Exception as e:
            print(f"[StrategyAnalytics] record failed (non-blocking): {e}")

    def get_summary(self, limit_days: int = 30) -> dict:
        """
        Return aggregated statistics for the last N days.
        Safe to call — returns empty dict on failure.
        """
        if not self.available:
            return {"available": False}
        try:
            cutoff = time.time() - (limit_days * 86400)
            rows = self._conn.execute("""
                SELECT primary_goal, buyer_state, email_preset,
                       AVG(adherence_score) as avg_score,
                       COUNT(*) as count,
                       SUM(CASE WHEN progression='regressed' THEN 1 ELSE 0 END) as regressions,
                       SUM(rep_violations) as total_rep_violations,
                       GROUP_CONCAT(failed_dims) as all_failed_dims
                FROM strategy_outcomes
                WHERE ts >= ?
                GROUP BY primary_goal, buyer_state
                ORDER BY count DESC
            """, (cutoff,)).fetchall()

            summary: list[dict] = []
            for r in rows:
                # Parse most common failed dimensions
                all_dims = [d for d in (r["all_failed_dims"] or "").split(",") if d]
                dim_freq: dict[str, int] = {}
                for d in all_dims:
                    dim_freq[d] = dim_freq.get(d, 0) + 1
                top_failures = sorted(dim_freq.items(), key=lambda x: x[1], reverse=True)[:3]

                summary.append({
                    "primary_goal":       r["primary_goal"],
                    "buyer_state":        r["buyer_state"],
                    "email_preset":       r["email_preset"],
                    "count":              r["count"],
                    "avg_adherence":      round(r["avg_score"] or 0, 1),
                    "regression_count":   r["regressions"],
                    "rep_violations":     r["total_rep_violations"],
                    "top_failed_dims":    [d for d, _ in top_failures],
                })

            return {
                "available":   True,
                "period_days": limit_days,
                "total_evals": sum(r["count"] for r in summary),
                "by_strategy": summary,
            }
        except Exception as e:
            print(f"[StrategyAnalytics] get_summary failed: {e}")
            return {"available": False, "error": str(e)}

    def get_worst_performing(self, limit: int = 5) -> list[dict]:
        """Return strategy combinations with lowest average adherence score."""
        if not self.available:
            return []
        try:
            rows = self._conn.execute("""
                SELECT primary_goal, buyer_state, email_preset,
                       AVG(adherence_score) as avg_score, COUNT(*) as count
                FROM strategy_outcomes
                WHERE count > 2
                GROUP BY primary_goal, buyer_state
                HAVING count >= 2
                ORDER BY avg_score ASC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[StrategyAnalytics] get_worst_performing failed: {e}")
            return []


# Module-level singleton
analytics = StrategyAnalytics()
