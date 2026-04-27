"""
pipeline.py — Unified Input Analysis & Flow Coordinator
=========================================================
Sits between the raw user input and the prompt builders in main.py.

What it adds (without replacing anything):
  1. Question detection  — finds direct questions in the prospect's message
  2. Multi-intent detection — surfaces ALL matching intents, not just the first
  3. InputAnalysis dataclass — single object that every prompt builder can read
  4. build_question_block() — pre-answered question section injected into prompts
  5. build_multi_intent_note() — secondary-intent context injected into prompts
  6. analyse() — the one function main.py calls to get everything at once

Existing code in main.py, intent_utils.py, intent_registry.py, template_engine.py
and quality_control.py is NOT changed by this file.

Usage (in main.py, before calling build_reply_prompt):
    from pipeline import analyse

    analysis = analyse(req.customer_message)
    base_prompt = build_reply_prompt(..., analysis=analysis)

The prompt builders merge the analysis blocks in; see main.py for details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from intent_utils import INTENT_KEYWORDS, detect_intent


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate a direct question even without a "?"
_IMPLICIT_QUESTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(does it|do you|is it|can i|will it|would it|has it|have you)\b", re.I),
    re.compile(r"\b(how much|how does|how do|how long|how many|how soon)\b", re.I),
    re.compile(r"\b(what('s| is)|when('s| is)|where('s| is)|who('s| is)|which)\b", re.I),
    re.compile(r"\b(tell me|let me know|wondering|curious|want to know)\b", re.I),
    re.compile(r"\b(any (traffic|visitors|searches|interest)|get (traffic|visitors|clicks))\b", re.I),
]

# Domain-selling specific question types with their answer guidance
_QUESTION_ANSWER_HINTS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\b(traffic|visitor|click|search|monthly|view)\b", re.I),
        "Answer the traffic / search-volume question directly. "
        "If you have data, state it. If not, explain honestly what geo-targeted domains typically receive "
        "and offer to share any available stats."
    ),
    (
        re.compile(r"\b(how much|price|cost|what are you asking|asking price|what.{0,10}want)\b", re.I),
        "State the asking price clearly and immediately. "
        "Follow with one sentence of value context, then the CTA."
    ),
    (
        re.compile(r"\b(redirect|forward|how does it work|how do i use|technical|set.?up|point)\b", re.I),
        "Explain the redirect/setup process in plain English — no jargon. "
        "Emphasise it takes minutes and requires no changes to their existing site."
    ),
    (
        re.compile(r"\b(still available|available|still for sale|taken|sold)\b", re.I),
        "Confirm availability immediately and directly."
    ),
    (
        re.compile(r"\b(why (should|would|do)|what.{0,10}benefit|what.{0,10}point|help my|value)\b", re.I),
        "Answer the 'why buy' question with the single most relevant benefit for their business type, "
        "then support with one concrete second point."
    ),
    (
        re.compile(r"\b(legit|legitimate|scam|real|trust|verify|proof|escrow)\b", re.I),
        "Address the trust question head-on. Name a specific verifiable mechanism "
        "(DAN.com escrow, GoDaddy listing) and invite them to check independently."
    ),
    (
        re.compile(r"\b(negotiat|best price|lower|discount|offer|counter)\b", re.I),
        "Respond to the negotiation signal directly. "
        "Either counter with a specific number or invite their best offer — don't deflect."
    ),
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
        # Explicit question mark
        if "?" in s:
            questions.append(s)
            continue
        # Implicit question patterns
        for pattern in _IMPLICIT_QUESTION_PATTERNS:
            if pattern.search(s):
                questions.append(s)
                break

    return questions


def build_question_answer_hints(questions: list[str]) -> list[str]:
    """
    For each detected question, find the most relevant answer guidance.
    Returns a list of hint strings (one per matched question type, deduplicated).
    """
    hints_seen: set[str] = set()
    hints: list[str] = []
    combined = " ".join(questions)

    for pattern, hint in _QUESTION_ANSWER_HINTS:
        if pattern.search(combined) and hint not in hints_seen:
            hints_seen.add(hint)
            hints.append(hint)

    return hints


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_all_intents(message: str) -> list[str]:
    """
    Return ALL intents whose keyword phrases match the message, ordered by
    the number of matched phrases (strongest signal first).

    Differs from detect_intent() in intent_utils.py which returns only the
    first match in dict iteration order.
    """
    low = message.lower()
    scored: list[tuple[int, str]] = []

    for intent, phrases in INTENT_KEYWORDS.items():
        count = sum(1 for p in phrases if p in low)
        if count > 0:
            scored.append((count, intent))

    scored.sort(key=lambda x: x[0], reverse=True)
    intents = [intent for _, intent in scored]

    # Always fall back to "general" if nothing matched
    return intents if intents else ["general"]


# ─────────────────────────────────────────────────────────────────────────────
# INPUT ANALYSIS DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InputAnalysis:
    """
    Complete analysis of one prospect message.
    Produced by analyse() and consumed by prompt builders.
    """
    raw_message: str

    # Intent
    primary_intent: str          = "general"
    all_intents: list[str]       = field(default_factory=list)
    has_multiple_intents: bool   = False
    secondary_intents: list[str] = field(default_factory=list)

    # Questions
    questions: list[str]         = field(default_factory=list)
    has_questions: bool          = False
    answer_hints: list[str]      = field(default_factory=list)

    # Pre-built prompt blocks (injected directly into prompts)
    question_block: str          = ""
    multi_intent_note: str       = ""
    debug_block: str             = ""


def _build_question_block(questions: list[str], hints: list[str]) -> str:
    """Build the QUESTIONS section injected into prompts."""
    if not questions:
        return ""

    lines = ["DIRECT QUESTIONS DETECTED — ANSWER THESE FIRST:"]
    for i, q in enumerate(questions, 1):
        lines.append(f"  Q{i}: {q}")

    if hints:
        lines.append("\nHOW TO ANSWER EACH QUESTION:")
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


def _build_debug_block(analysis: "InputAnalysis") -> str:
    """Compact debug summary — included in prompt logs, not in final email."""
    lines = [
        "── PIPELINE DEBUG ──────────────────────────────",
        f"  Primary intent : {analysis.primary_intent}",
        f"  All intents    : {', '.join(analysis.all_intents)}",
        f"  Questions found: {'yes' if analysis.has_questions else 'no'}",
    ]
    if analysis.questions:
        for q in analysis.questions:
            lines.append(f"    → {q}")
    lines.append("────────────────────────────────────────────────")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyse(message: str) -> InputAnalysis:
    """
    Run the full input analysis pipeline on one prospect message.

    Steps:
      1. Detect all matching intents (multi-intent)
      2. Detect direct questions
      3. Build answer hints for each question type
      4. Assemble prompt blocks

    Returns an InputAnalysis with everything the prompt builders need.

    Example:
        analysis = analyse("Does it have traffic? Also, can you do a better price?")
        # analysis.primary_intent   → "price_too_high"  (or whichever scores highest)
        # analysis.secondary_intents→ ["negotiation"]
        # analysis.has_questions    → True
        # analysis.questions        → ["Does it have traffic?", "can you do a better price?"]
        # analysis.question_block   → multi-line prompt injection string
    """
    a = InputAnalysis(raw_message=message)

    # ── Step 1: Multi-intent ─────────────────────────────────────────────────
    all_intents            = detect_all_intents(message)
    a.primary_intent       = all_intents[0]
    a.all_intents          = all_intents
    a.secondary_intents    = all_intents[1:] if len(all_intents) > 1 else []
    a.has_multiple_intents = len(all_intents) > 1

    # ── Step 2: Question detection ───────────────────────────────────────────
    a.questions    = detect_questions(message)
    a.has_questions= bool(a.questions)
    a.answer_hints = build_question_answer_hints(a.questions) if a.has_questions else []

    # ── Step 3: Build prompt blocks ──────────────────────────────────────────
    a.question_block    = _build_question_block(a.questions, a.answer_hints)
    a.multi_intent_note = _build_multi_intent_note(a.primary_intent, a.secondary_intents)
    a.debug_block       = _build_debug_block(a)

    return a


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: flow-order instruction string
# ─────────────────────────────────────────────────────────────────────────────

def build_flow_instruction(analysis: InputAnalysis) -> str:
    """
    Returns the REPLY FLOW ORDER section that is injected into every prompt.
    Tells Claude the exact order in which to structure the reply.

    If questions present:   Answer questions → Intent strategy → CTA
    If no questions:        Intent strategy → Value → CTA
    """
    if analysis.has_questions:
        return (
            "REPLY FLOW — FOLLOW THIS ORDER EXACTLY:\n"
            "  1. Answer every detected question directly and specifically\n"
            "  2. Transition naturally into the intent-based strategy\n"
            "  3. Close with the appropriate CTA for this intent\n\n"
            "Do NOT open with a sales pitch before the questions are answered."
        )
    else:
        return (
            "REPLY FLOW — FOLLOW THIS ORDER:\n"
            "  1. Apply the intent-based strategy (goal, tone, approach)\n"
            "  2. Weave in relevant value proposition\n"
            "  3. Close with the appropriate CTA for this intent"
        )
