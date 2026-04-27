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
    """
    low = message.lower()
    scored: list[tuple[int, str]] = []

    for intent, phrases in INTENT_KEYWORDS.items():
        count = sum(1 for p in phrases if p in low)
        if count > 0:
            scored.append((count, intent))

    scored.sort(key=lambda x: x[0], reverse=True)
    intents = [intent for _, intent in scored]
    return intents if intents else ["general"]


# ─────────────────────────────────────────────────────────────────────────────
# INPUT ANALYSIS DATACLASS
# Single object produced by analyse() and consumed by prompt builders.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InputAnalysis:
    """Complete analysis of one prospect message."""
    raw_message: str

    # ── Intent ────────────────────────────────────────────────────────────────
    primary_intent: str           = "general"
    all_intents: list[str]        = field(default_factory=list)
    has_multiple_intents: bool    = False
    secondary_intents: list[str]  = field(default_factory=list)

    # ── Questions ─────────────────────────────────────────────────────────────
    questions: list[str]          = field(default_factory=list)
    has_questions: bool           = False

    # Question type classification (NEW)
    # e.g. {"factual_question": ["How much?"], "how_to_question": ["How do I redirect?"]}
    question_types: dict[str, list[str]] = field(default_factory=dict)
    # Primary type: the type that appears most in this message
    primary_question_type: str    = ""

    # Per-type answer guidance strings
    answer_hints: list[str]       = field(default_factory=list)

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
    Shows detected question types so behaviour is traceable.
    """
    lines = [
        "── PIPELINE DEBUG ──────────────────────────────",
        f"  Primary intent       : {analysis.primary_intent}",
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
      1. Detect all matching intents (multi-intent, scored)
      2. Detect questions (explicit + implicit)
      3. Classify each question by type (factual / how_to / clarification / comparison)
      4. Build per-type answer hints
      5. Assemble prompt injection blocks

    Returns an InputAnalysis dataclass with everything prompt builders need.

    Example:
        a = analyse("Does it have traffic? Also, can you come down on price?")
        a.primary_intent          → "price_too_high"
        a.secondary_intents       → ["price_inquiry"]
        a.has_questions           → True
        a.questions               → ["Does it have traffic?", "can you come down on price?"]
        a.question_types          → {"factual_question": ["Does it have traffic?"],
                                      "comparison_question": ["can you come down on price?"]}
        a.primary_question_type   → "factual_question"
        a.answer_hints            → [factual guidance, comparison guidance]
        a.question_block          → ready-to-inject prompt string
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
    a.has_questions = bool(a.questions)

    # ── Step 3: Question classification ─────────────────────────────────────
    if a.has_questions:
        a.question_types        = classify_questions(a.questions)
        a.primary_question_type = _primary_question_type(a.question_types)
        a.answer_hints          = _derive_answer_hints(a.question_types)
    else:
        a.question_types        = {qt: [] for qt in QUESTION_TYPES}
        a.question_types["general_question"] = []
        a.primary_question_type = ""
        a.answer_hints          = []

    # ── Step 4: Build prompt blocks ──────────────────────────────────────────
    a.question_block    = _build_question_block(a.questions, a.question_types, a.answer_hints)
    a.multi_intent_note = _build_multi_intent_note(a.primary_intent, a.secondary_intents)
    a.debug_block       = _build_debug_block(a)

    return a


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
    """
    print(f"\n{'═'*60}")
    print(f"  PIPELINE ANALYSIS")
    print(f"{'═'*60}")
    print(f"  Input:  \"{analysis.raw_message[:80]}\"")
    print(f"\n  ── Intent ─────────────────────────────────────────")
    print(f"  Primary:    {analysis.primary_intent}")
    if analysis.secondary_intents:
        print(f"  Secondary:  {', '.join(analysis.secondary_intents)}")
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
