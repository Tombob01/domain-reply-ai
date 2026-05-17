"""
pipeline.py — Unified Input Analysis & Flow Coordinator (v2)
=============================================================
Sits between the raw user input and the prompt builders in main.py.

What it does (without replacing anything):
  1. Question detection      — finds questions (explicit + implicit)
  2. Question classification — labels each as factual / how_to /
                               clarification / comparison
  3. Multi-intent detection  — surfaces ALL matching intents, scored
  4. InputAnalysis dataclass — single object every prompt builder reads
  5. build_question_block()  — prompt injection: typed questions + guidance
  6. build_multi_intent_note()— prompt injection: secondary intent context
  7. build_flow_instruction() — prompt injection: reply order based on input
  8. analyse()               — single entry point for main.py

Question classification comes from intent_utils.QUESTION_TYPES.
Existing main.py, intent_registry.py, template_engine.py, quality_control.py
are NOT changed.

Usage (in main.py, before calling build_reply_prompt):
    from pipeline import analyse

    analysis = analyse(req.customer_message)
    base_prompt = build_reply_prompt(..., analysis=analysis)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from intent_utils import (
    INTENT_KEYWORDS,
    detect_intent,
    classify_question,
    classify_questions,
    get_question_guidance,
    QUESTION_TYPES,
)


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION DETECTION
# Finds questions in the raw message — explicit (?) and implicit patterns.
# Classification (factual / how_to / clarification / comparison) is done
# by classify_question() from intent_utils.
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that signal a question even without "?"
_IMPLICIT_QUESTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(does it|do you|is it|can i|will it|would it|has it|have you)\b", re.I),
    re.compile(r"\b(how much|how does|how do|how long|how many|how soon)\b", re.I),
    re.compile(r"\b(what('s| is)|when('s| is)|where('s| is)|who('s| is)|which)\b", re.I),
    re.compile(r"\b(tell me|let me know|wondering|curious|want to know)\b", re.I),
    re.compile(r"\b(any (traffic|visitors|searches|interest)|get (traffic|visitors|clicks))\b", re.I),
]


def detect_questions(message: str) -> list[str]:
    """
    Return a list of question strings found in the message.
    Captures both explicit (?) and implicit question patterns.
    """
    questions: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", message.strip())

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if "?" in s:
            questions.append(s)
            continue
        for pattern in _IMPLICIT_QUESTION_PATTERNS:
            if pattern.search(s):
                questions.append(s)
                break

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-INTENT DETECTION
# Returns ALL matching intents ranked by keyword hit count.
# ─────────────────────────────────────────────────────────────────────────────

def detect_all_intents(message: str) -> list[str]:
    """
    Return ALL intents whose keyword phrases match the message, ordered by
    number of matched phrases (strongest signal first).

    Differs from detect_intent() which returns only the top scorer.
    Preserved for backward compatibility — all existing callers still work.
    Confidence data is now computed separately in _score_intents().
    """
    return [intent for intent, _ in _score_intents(message)]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING HELPERS  (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def _score_intents(message: str) -> list[tuple[str, int]]:
    """
    Return a list of (intent, raw_score) sorted descending by raw_score.
    raw_score = number of keyword phrases from that intent's list that appear
    in the lowercased message.

    Returns [("general", 0)] if nothing matches so callers always get a result.
    """
    low    = message.lower()
    scored: list[tuple[int, str]] = []

    for intent, phrases in INTENT_KEYWORDS.items():
        count = sum(1 for p in phrases if p in low)
        if count > 0:
            scored.append((count, intent))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return [("general", 0)]

    return [(intent, count) for count, intent in scored]


def _compute_confidence(
    scores: list[tuple[str, int]],
    questions_found: bool,
) -> dict[str, float]:
    """
    Convert raw keyword-hit counts into normalised confidence values in [0.0, 1.0].

    Algorithm:
      1. Base score = raw_count / total_keyword_matches_across_all_intents
         (share of the total signal this intent accounts for)
      2. Phrase density bonus: if the primary intent accounts for > 60 % of
         all matches, add up to +0.15 to reflect a very clear signal.
      3. Competition penalty: for each additional intent that has a score
         ≥ 50 % of the top score, subtract 0.10 (max penalty 0.30).
         Competing intents reduce certainty.
      4. Question overlap nudge: if questions were detected AND this is the
         primary intent, add +0.05 (questions confirm engagement).
      5. Clamp result to [0.05, 0.98].

    The "general" fallback always receives 0.0 confidence.
    """
    if not scores or (len(scores) == 1 and scores[0][0] == "general"):
        return {"general": 0.0}

    total_hits = sum(s for _, s in scores)
    if total_hits == 0:
        return {intent: 0.0 for intent, _ in scores}

    top_score   = scores[0][1]
    result: dict[str, float] = {}

    for i, (intent, raw) in enumerate(scores):
        # 1. Base normalised share
        base = raw / total_hits

        # 2. Density bonus for primary intent only
        density_bonus = 0.0
        if i == 0 and top_score > 0 and (raw / total_hits) > 0.60:
            density_bonus = 0.15

        # 3. Competition penalty — count intents whose score ≥ 50 % of top
        if i == 0:
            competitors = sum(
                1 for _, s in scores[1:]
                if s >= top_score * 0.50
            )
            competition_penalty = min(competitors * 0.10, 0.30)
        else:
            competition_penalty = 0.0

        # 4. Question overlap nudge for primary intent
        question_nudge = 0.05 if (i == 0 and questions_found) else 0.0

        raw_conf = base + density_bonus - competition_penalty + question_nudge
        result[intent] = round(max(0.05, min(0.98, raw_conf)), 3)

    return result


def _ambiguity_level(
    confidence: float,
    scores: list[tuple[str, int]],
) -> str:
    """
    Classify how ambiguous the intent detection is.

    Rules:
      low    — primary confidence ≥ 0.70  OR  only one intent matched
      high   — primary confidence < 0.40  OR  3+ intents with score ≥ 50% of top
      medium — everything else

    Returns: "low" | "medium" | "high"
    """
    if len(scores) <= 1:
        return "low"

    top_score   = scores[0][1]
    competitors = sum(
        1 for _, s in scores[1:]
        if s >= top_score * 0.50
    )

    if confidence >= 0.70:
        return "low"
    if confidence < 0.40 or competitors >= 3:
        return "high"
    return "medium"


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING RECOMMENDATION ENGINE  (Phase 3)
#
# Determines which generation mode would produce the best result for this
# particular input, based entirely on signals already computed by the pipeline.
#
# RECOMMENDATION ONLY — this does not invoke generation and does not change
# any existing endpoint behaviour.  Callers read recommended_mode from
# InputAnalysis but are not required to act on it until Phase 4.
#
# Decision tree (evaluated top-to-bottom, first match wins):
#
#   1. No template coverage  → autonomous
#      The detected intent has no pre-built components.  Template and hybrid
#      modes would fall back to a generic reply.  Full AI generation is the
#      only option that can handle it well without pre-built assets.
#      Special case of no-coverage: "autonomous" is recommended, not "ai",
#      to signal that this intent should eventually get its own template.
#
#   2. High ambiguity  → ai
#      Three or more intents are competing at similar strength.  Template
#      components are built for single-intent clarity.  A Claude prompt that
#      sees the full analysis can navigate competing signals more gracefully.
#
#   3. Low confidence  → ai
#      The primary intent scored < 0.40.  The keyword evidence is too thin
#      to trust a template component choice.  AI mode handles uncertainty
#      better because the prompt includes the full scored intent list.
#
#   4. Questions detected AND medium-or-lower confidence  → ai
#      Questions need specific, factual answers.  When we are already unsure
#      which intent applies, templates cannot be trusted to answer them
#      correctly.  Only AI mode has the full context.
#
#   5. High confidence AND questions detected  → hybrid
#      We know what the prospect wants (high confidence) but they also asked
#      something specific.  Template mode cannot answer questions; AI polish
#      can.  Hybrid assembles the intent-matched structure via template and
#      lets AI handle the question answer within the polish pass.
#
#   6. Medium confidence AND no questions  → hybrid
#      The intent signal is reasonable but not definitive.  Template gives
#      a structurally sound reply; AI polish can soften the edges where the
#      template wording might feel off-target.
#
#   7. High confidence AND no questions AND template covered  → template
#      The clearest possible signal.  Template mode is fast, free, and
#      fully deterministic.  There is no quality reason to call an AI.
#
# Thresholds:
#   HIGH confidence  ≥ 0.70
#   MEDIUM confidence  0.40 – 0.69
#   LOW confidence   < 0.40
# ─────────────────────────────────────────────────────────────────────────────

# Intents that have pre-built COMPONENTS in template_engine.py.
# Kept as a frozenset for O(1) lookup.  Must be kept in sync with
# the COMPONENTS dict keys in template_engine.py.
# To add coverage: add the intent key to both COMPONENTS and this set.
_TEMPLATE_COVERED_INTENTS: frozenset[str] = frozenset({
    "agreed_no_pay",
    "angry",
    "cold_outreach",
    "competitor_comparison",
    "demo_offer",
    "development",
    "domain_metrics",
    "expired_owner",
    "extension",
    "feature_explanation",
    "follow_up",
    "follow_up_after_interest",
    "follow_up_after_pricing",
    "follow_up_no_response",
    "general",
    "general_response",
    "have_website",
    "how_it_works",
    "identity",
    "low_budget",
    "meeting_request",
    "negotiation",
    "no_thanks",
    "not_interested_ask_why",
    "not_now",
    "objection_handling",
    "partner",
    "payment_issue",
    "payment_method",
    "post_purchase",
    "price_inquiry",
    "price_negotiation",
    "price_too_high",
    "rank_well",
    "re_engagement",
    "refund",
    "related_domains",
    "renewal_fees",
    "request_info",
    "sales_pitch",
    "soft_pitch",
    "trust_building",
    "trust_issue",
    "value_reminder",
    "why_buy",
})

# Confidence thresholds
_CONF_HIGH   = 0.70
_CONF_MEDIUM = 0.40   # [0.40, 0.70) = medium;  < 0.40 = low


def _recommend_mode(
    primary_intent: str,
    confidence: float,
    ambiguity: str,
    has_questions: bool,
    has_multiple_intents: bool,
    num_competitors: int,
) -> tuple[str, str, str]:
    """
    Apply the routing decision tree and return:
        (recommended_mode, reason_code, plain_english_explanation)

    Args:
        primary_intent:      The highest-scoring intent label.
        confidence:          Normalised confidence for the primary intent [0, 1].
        ambiguity:           "low" | "medium" | "high"
        has_questions:       Whether the message contains detectable questions.
        has_multiple_intents:Whether more than one intent matched.
        num_competitors:     Count of intents scoring ≥ 50 % of the top score.

    Returns a 3-tuple — all three values are non-empty strings.
    """
    covered = primary_intent in _TEMPLATE_COVERED_INTENTS

    # ── Rule 1: No template coverage ─────────────────────────────────────────
    if not covered:
        return (
            "autonomous",
            "no_template_coverage",
            (
                f"The intent '{primary_intent}' has no pre-built template components. "
                "Full AI generation is recommended. Consider adding this intent to "
                "the template library to enable faster, cheaper responses in future."
            ),
        )

    # ── Rule 2: High ambiguity (3+ competing intents) ─────────────────────────
    if ambiguity == "high" or num_competitors >= 3:
        return (
            "ai",
            "high_ambiguity_competing_intents",
            (
                f"Multiple intents are competing at similar strength "
                f"({num_competitors} competitors). "
                "Template components are designed for single-intent clarity. "
                "AI mode can navigate the ambiguity and weigh all signals in context."
            ),
        )

    # ── Rule 3: Low confidence ────────────────────────────────────────────────
    if confidence < _CONF_MEDIUM:
        return (
            "ai",
            "low_confidence_insufficient_signal",
            (
                f"Primary intent confidence is low ({confidence:.0%}). "
                "The keyword signal is too thin to trust a template component selection. "
                "AI mode receives the full scored intent list and can adapt accordingly."
            ),
        )

    # ── Rule 4: Questions + medium-or-lower confidence ────────────────────────
    if has_questions and confidence < _CONF_HIGH:
        return (
            "ai",
            "questions_detected_medium_confidence",
            (
                f"Questions were detected and confidence is only {confidence:.0%}. "
                "Template mode cannot answer specific questions. "
                "When confidence is not high, AI mode is safer because it handles "
                "both the uncertain intent and the questions in a single pass."
            ),
        )

    # ── Rule 5: High confidence + questions ───────────────────────────────────
    if has_questions and confidence >= _CONF_HIGH:
        return (
            "hybrid",
            "high_confidence_questions_need_ai_polish",
            (
                f"Intent is clear ({primary_intent}, {confidence:.0%} confidence) "
                "but questions were detected. "
                "Template mode builds the structure; AI polish handles the questions. "
                "Hybrid gives the best of both: speed and specificity."
            ),
        )

    # ── Rule 6: Medium confidence, no questions ───────────────────────────────
    if _CONF_MEDIUM <= confidence < _CONF_HIGH:
        return (
            "hybrid",
            "medium_confidence_ai_polish_recommended",
            (
                f"Confidence is {confidence:.0%} — reasonable but not definitive. "
                "Template mode assembles a structurally sound reply; "
                "AI polish refines the wording to fit the specific context. "
                "Hybrid balances reliability with naturalness."
            ),
        )

    # ── Rule 7: High confidence, no questions, template covered ──────────────
    # (This is the only path that reaches here — all other cases returned above)
    return (
        "template",
        "high_confidence_single_intent_template_covered",
        (
            f"Strong, unambiguous intent signal ({primary_intent}, {confidence:.0%} confidence). "
            "No questions detected. Template components are available. "
            "Template mode is fast, free, and will produce a high-quality reply "
            "without any API calls."
        ),
    )


def _count_competitors(scores: list[tuple[str, int]]) -> int:
    """
    Count how many intents (beyond the top) score ≥ 50 % of the top score.
    Returns 0 for single-intent or general cases.
    """
    if len(scores) <= 1:
        return 0
    top = scores[0][1]
    if top == 0:
        return 0
    return sum(1 for _, s in scores[1:] if s >= top * 0.50)


def _log_router(a: "InputAnalysis") -> None:
    """
    Emit one structured [ROUTER] log line per request.

    Format:
        [ROUTER] intent=price_inquiry confidence=0.81 ambiguity=low
                 recommended_mode=template reason=high_confidence_single_intent_template_covered
    """
    print(
        f"[ROUTER] intent={a.primary_intent}"
        f" confidence={a.primary_intent_confidence:.2f}"
        f" ambiguity={a.ambiguity_level}"
        f" recommended_mode={a.recommended_mode}"
        f" reason={a.routing_reason}"
    )




@dataclass
class InputAnalysis:
    """Complete analysis of one prospect message."""
    raw_message: str

    # ── Intent ────────────────────────────────────────────────────────────────
    primary_intent: str           = "general"
    all_intents: list[str]        = field(default_factory=list)
    has_multiple_intents: bool    = False
    secondary_intents: list[str]  = field(default_factory=list)

    # ── Confidence (Phase 1) ──────────────────────────────────────────────────
    # Normalised confidence for the primary intent: 0.0 – 1.0
    primary_intent_confidence: float       = 0.0
    # Per-intent normalised confidence scores for all matched intents
    intent_scores: dict[str, float]        = field(default_factory=dict)
    # "low" | "medium" | "high" — reflects how many competing signals exist
    ambiguity_level: str                   = ""

    # ── Questions ─────────────────────────────────────────────────────────────
    questions: list[str]          = field(default_factory=list)
    has_questions: bool           = False

    # Question type classification
    # e.g. {"factual_question": ["How much?"], "how_to_question": ["How do I redirect?"]}
    question_types: dict[str, list[str]] = field(default_factory=dict)
    # Primary type: the type that appears most in this message
    primary_question_type: str    = ""

    # Per-type answer guidance strings
    answer_hints: list[str]       = field(default_factory=list)

    # ── Routing recommendation (Phase 3) ──────────────────────────────────────
    # Recommended generation mode based on confidence + coverage + complexity.
    # "template" | "hybrid" | "ai" | "autonomous"
    # This is a RECOMMENDATION ONLY — callers are not required to honour it.
    # Automatic routing is implemented in a future phase.
    recommended_mode: str         = ""
    # Short machine-readable reason code for the recommendation.
    # e.g. "high_confidence_template_covered_no_questions"
    routing_reason: str           = ""
    # Plain English explanation suitable for displaying to the user.
    routing_explanation: str      = ""

    # ── Prompt blocks (injected directly into prompts) ─────────────────────────
    question_block: str           = ""
    multi_intent_note: str        = ""
    debug_block: str              = ""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BLOCK BUILDERS
# Each returns a string ready to inject into a Claude prompt.
# ─────────────────────────────────────────────────────────────────────────────

def _build_question_block(
    questions: list[str],
    question_types: dict[str, list[str]],
    hints: list[str],
) -> str:
    """
    Build the QUESTIONS section injected into prompts.
    Now includes the classified type for each question so Claude knows
    exactly how to handle it (factual → state fact; how_to → give steps; etc.)
    """
    if not questions:
        return ""

    lines = ["DIRECT QUESTIONS DETECTED — ANSWER THESE FIRST:"]

    # List each question with its type label
    q_num = 1
    for qtype, qs in question_types.items():
        if not qs:
            continue
        type_label = qtype.replace("_", " ").title()
        for q in qs:
            lines.append(f"  Q{q_num} [{type_label}]: {q}")
            q_num += 1

    # Per-type guidance
    if hints:
        lines.append("\nHOW TO ANSWER BY TYPE:")
        for hint in hints:
            lines.append(f"  • {hint}")

    lines.append(
        "\nRULE: Answer every question above clearly and directly BEFORE moving "
        "to the sales strategy. Never replace a specific answer with a generic sales message."
    )
    return "\n".join(lines)


def _build_multi_intent_note(primary: str, secondary: list[str]) -> str:
    """Build the SECONDARY INTENTS section injected into prompts."""
    if not secondary:
        return ""

    sec_labels = ", ".join(s.replace("_", " ").title() for s in secondary)
    return (
        f"SECONDARY SIGNALS DETECTED: {sec_labels}\n"
        f"PRIMARY intent drives the strategy. "
        f"Acknowledge secondary signals naturally where relevant — "
        f"do not ignore them, but do not let them override the primary approach."
    )


def _build_debug_block(analysis: InputAnalysis) -> str:
    """
    Compact debug summary injected into prompt logs (not the final email).
    Shows detected question types, confidence, ambiguity, and routing
    recommendation so behaviour is fully traceable without cluttering
    the generated reply.
    """
    lines = [
        "── PIPELINE DEBUG ──────────────────────────────",
        f"  Primary intent       : {analysis.primary_intent}",
        f"  Confidence           : {analysis.primary_intent_confidence:.2f}",
        f"  Ambiguity            : {analysis.ambiguity_level}",
        f"  Recommended mode     : {analysis.recommended_mode}",
        f"  Routing reason       : {analysis.routing_reason}",
        f"  All intents          : {', '.join(analysis.all_intents)}",
        f"  Questions found      : {'yes' if analysis.has_questions else 'no'}",
    ]
    if analysis.questions:
        lines.append(f"  Primary question type: {analysis.primary_question_type or 'none'}")
        for qtype, qs in analysis.question_types.items():
            if qs:
                type_label = qtype.replace("_", " ").title()
                for q in qs:
                    lines.append(f"    [{type_label}] → {q}")
    lines.append("────────────────────────────────────────────────")
    return "\n".join(lines)


def _derive_answer_hints(question_types: dict[str, list[str]]) -> list[str]:
    """
    Build per-type answer hints from classified questions.
    One hint per question type present — no duplicates.
    """
    hints: list[str] = []
    for qtype in ["factual_question", "how_to_question", "clarification_question", "comparison_question"]:
        if question_types.get(qtype):
            hint = get_question_guidance(qtype)
            if hint not in hints:
                hints.append(hint)
    return hints


def _primary_question_type(question_types: dict[str, list[str]]) -> str:
    """Return the question type with the most questions, or empty string."""
    best = max(
        (qt for qt in question_types if question_types[qt]),
        key=lambda qt: len(question_types[qt]),
        default="",
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyse(message: str) -> InputAnalysis:
    """
    Run the full input analysis pipeline on one prospect message.

    Steps:
      1. Score all intents (keyword-hit counts)
      2. Compute normalised confidence per intent  [Phase 1]
      3. Determine ambiguity level                 [Phase 1]
      4. Detect questions (explicit + implicit)
      5. Classify each question by type
      6. Build per-type answer hints
      7. Assemble prompt injection blocks
      8. Emit structured [INTENT] debug log        [Phase 1]
      9. Compute routing recommendation            [Phase 3]
     10. Emit structured [ROUTER] debug log        [Phase 3]

    Returns an InputAnalysis dataclass with everything prompt builders need.
    All new fields default gracefully so existing callers need no changes.

    Example:
        a = analyse("Does it have traffic? Also, can you come down on price?")
        a.primary_intent              → "price_too_high"
        a.primary_intent_confidence   → 0.63
        a.ambiguity_level             → "medium"
        a.secondary_intents           → ["price_inquiry"]
        a.has_questions               → True
        a.recommended_mode            → "hybrid"
        a.routing_reason              → "high_confidence_questions_need_ai_polish"
    """
    a = InputAnalysis(raw_message=message)

    # ── Step 1: Score all intents ────────────────────────────────────────────
    scored_pairs = _score_intents(message)            # [(intent, raw_score), …]
    all_intents  = [intent for intent, _ in scored_pairs]

    a.primary_intent       = all_intents[0]
    a.all_intents          = all_intents
    a.secondary_intents    = all_intents[1:] if len(all_intents) > 1 else []
    a.has_multiple_intents = len(all_intents) > 1

    # ── Step 2: Question detection (needed before confidence) ────────────────
    a.questions    = detect_questions(message)
    a.has_questions = bool(a.questions)

    # ── Step 3: Confidence + ambiguity  [Phase 1] ────────────────────────────
    conf_map = _compute_confidence(scored_pairs, questions_found=a.has_questions)

    a.intent_scores               = conf_map
    a.primary_intent_confidence   = conf_map.get(a.primary_intent, 0.0)
    a.ambiguity_level             = _ambiguity_level(
        a.primary_intent_confidence, scored_pairs
    )

    # ── Step 4: Question classification ──────────────────────────────────────
    if a.has_questions:
        a.question_types        = classify_questions(a.questions)
        a.primary_question_type = _primary_question_type(a.question_types)
        a.answer_hints          = _derive_answer_hints(a.question_types)
    else:
        a.question_types        = {qt: [] for qt in QUESTION_TYPES}
        a.question_types["general_question"] = []
        a.primary_question_type = ""
        a.answer_hints          = []

    # ── Step 5: Build prompt blocks ───────────────────────────────────────────
    a.question_block    = _build_question_block(a.questions, a.question_types, a.answer_hints)
    a.multi_intent_note = _build_multi_intent_note(a.primary_intent, a.secondary_intents)

    # ── Step 6: Routing recommendation  [Phase 3] ────────────────────────────
    num_competitors          = _count_competitors(scored_pairs)
    mode, reason, explanation = _recommend_mode(
        primary_intent      = a.primary_intent,
        confidence          = a.primary_intent_confidence,
        ambiguity           = a.ambiguity_level,
        has_questions       = a.has_questions,
        has_multiple_intents= a.has_multiple_intents,
        num_competitors     = num_competitors,
    )
    a.recommended_mode    = mode
    a.routing_reason      = reason
    a.routing_explanation = explanation

    # ── Step 7: Debug block (includes routing fields) ─────────────────────────
    a.debug_block = _build_debug_block(a)

    # ── Step 8: Structured logs ───────────────────────────────────────────────
    _log_intent(a)
    _log_router(a)

    return a


def _log_intent(a: InputAnalysis) -> None:
    """
    Emit one structured log line per request.

    Format:
        [INTENT] primary=price_too_high confidence=0.82 ambiguity=low
                 secondary=price_inquiry,negotiation questions=1
    """
    secondary_str = ",".join(a.secondary_intents) if a.secondary_intents else "none"
    print(
        f"[INTENT] primary={a.primary_intent}"
        f" confidence={a.primary_intent_confidence:.2f}"
        f" ambiguity={a.ambiguity_level}"
        f" secondary={secondary_str}"
        f" questions={len(a.questions)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOW INSTRUCTION BUILDER
# Returns the REPLY FLOW ORDER section injected into every prompt.
# Adapts the order based on whether questions are present and their types.
# ─────────────────────────────────────────────────────────────────────────────

def build_flow_instruction(analysis: InputAnalysis) -> str:
    """
    Returns the REPLY FLOW ORDER section injected into every prompt.
    Tells Claude the exact order in which to structure the reply.

    Flow with questions:
        Answer questions (by type) → intent strategy → CTA

    Flow without questions:
        Intent strategy → value → CTA

    The flow adapts per question type:
      - factual      → state fact first, then strategy
      - how_to       → give steps first, then strategy
      - clarification → explain first, then strategy
      - comparison   → compare first, then strategy
    """
    if not analysis.has_questions:
        return (
            "REPLY FLOW — FOLLOW THIS ORDER:\n"
            "  1. Apply the intent-based strategy (goal, tone, approach)\n"
            "  2. Weave in relevant value proposition\n"
            "  3. Close with the appropriate CTA for this intent"
        )

    # Build type-specific step 1 instruction
    type_instructions: dict[str, str] = {
        "factual_question":      "State the requested fact directly (price / stat / availability)",
        "how_to_question":       "Give the process in simple numbered steps",
        "clarification_question":"Explain the concept clearly in plain language",
        "comparison_question":   "Acknowledge the comparison, then explain the key advantage",
        "general_question":      "Answer the question directly and specifically",
    }

    active_types = [
        qt for qt in ["factual_question", "how_to_question", "clarification_question",
                      "comparison_question", "general_question"]
        if analysis.question_types.get(qt)
    ]

    if len(active_types) == 1:
        step1 = type_instructions[active_types[0]]
    else:
        parts = [type_instructions[qt] for qt in active_types]
        step1 = " / ".join(parts)

    return (
        "REPLY FLOW — FOLLOW THIS ORDER EXACTLY:\n"
        f"  1. {step1}\n"
        "  2. Transition naturally into the intent-based strategy\n"
        "  3. Close with the appropriate CTA for this intent\n\n"
        "Do NOT open with a sales pitch before the questions are answered."
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG HELPER — print a full analysis to stdout (for testing)
# ─────────────────────────────────────────────────────────────────────────────

def print_analysis(analysis: InputAnalysis) -> None:
    """
    Print a readable analysis report to stdout.
    Used for testing and the /debug endpoint.
    Includes Phase 1 confidence + ambiguity and Phase 3 routing fields.
    """
    print(f"\n{'═'*60}")
    print(f"  PIPELINE ANALYSIS")
    print(f"{'═'*60}")
    print(f"  Input:  \"{analysis.raw_message[:80]}\"")
    print(f"\n  ── Intent ─────────────────────────────────────────")
    print(f"  Primary:     {analysis.primary_intent}")
    print(f"  Confidence:  {analysis.primary_intent_confidence:.2f}   Ambiguity: {analysis.ambiguity_level}")
    if analysis.secondary_intents:
        print(f"  Secondary:   {', '.join(analysis.secondary_intents)}")
    if analysis.intent_scores:
        top_scores = sorted(analysis.intent_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top scores:  {', '.join(f'{k}={v:.2f}' for k, v in top_scores)}")
    print(f"\n  ── Routing recommendation ──────────────────────────")
    print(f"  Mode:        {analysis.recommended_mode}")
    print(f"  Reason:      {analysis.routing_reason}")
    print(f"  Explanation: {analysis.routing_explanation}")
    print(f"\n  ── Questions ───────────────────────────────────────")
    if analysis.has_questions:
        print(f"  Found {len(analysis.questions)} question(s):")
        for qtype, qs in analysis.question_types.items():
            if qs:
                label = qtype.replace("_", " ").title()
                for q in qs:
                    print(f"    [{label}] {q}")
        print(f"\n  Primary question type: {analysis.primary_question_type}")
    else:
        print("  No questions detected.")
    print(f"\n  ── Flow instruction ────────────────────────────────")
    print(build_flow_instruction(analysis))
    print(f"{'═'*60}\n")
