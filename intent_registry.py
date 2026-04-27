"""
intent_registry.py — Intent Registry: Single Source of Truth
=============================================================
Connects the three existing modules without changing any of them.

The problem this solves:
  - INTENT_KEYWORDS  lives in main.py         (trigger phrases)
  - INTENT_STRATEGY  lives in quality_control.py  (strategy)
  - COMPONENTS       lives in template_engine.py  (template parts)

Each piece works on its own, but there is no single place where you can
look up one intent and see all three layers together.  The flow —

    input → intent detection → strategy selection → template generation

— is implied by the code, not explicit anywhere.

This file makes it explicit.  It does NOT duplicate or replace any
existing logic.  It:
  1. Imports the canonical data from each module
  2. Wraps it in a single per-intent view called INTENT_REGISTRY
  3. Exposes helper functions that wire the four stages together
  4. Validates on import that every registered intent is complete
     (triggers + strategy + template) and prints any gaps

Usage — anywhere in the project:
    from intent_registry import registry_for, full_pipeline, INTENT_REGISTRY

    # Look up one intent end-to-end
    rec = registry_for("trust_issue")
    print(rec["triggers"])    # list[str]  — from main.py INTENT_KEYWORDS
    print(rec["strategy"])    # dict       — from quality_control INTENT_STRATEGY
    print(rec["components"])  # dict       — from template_engine COMPONENTS

    # Run the full pipeline on a raw message
    result = full_pipeline("This looks like a scam", domain_name="LondonPlumber.com")
    # → { intent, strategy, reply, components_used, qc_report }

Architecture note:
  Nothing in this file changes the existing modules.
  If a module is updated, this file automatically reflects the change
  because it imports live references, not copies.
"""

from __future__ import annotations

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — live references to all three existing data sources
# ─────────────────────────────────────────────────────────────────────────────

# Layer 1 — Trigger phrases (intent detection)
# Source: main.py  →  INTENT_KEYWORDS + detect_intent()
from intent_utils import INTENT_KEYWORDS, detect_intent

# Layer 2 — Strategy (how to respond)
# Source: quality_control.py  →  INTENT_STRATEGY + get_strategy() + build_strategy_block()
from quality_control import INTENT_STRATEGY, get_strategy, build_strategy_block, run_full_qc

# Layer 3 — Template components (how the email is built)
# Source: template_engine.py  →  COMPONENTS + detect_template_intent() + build_template_reply()
from template_engine import (
    COMPONENTS,
    TEMPLATE_INTENT_KEYWORDS,
    detect_template_intent,
    build_template_reply,
    ai_polish_reply,
)


# ─────────────────────────────────────────────────────────────────────────────
# INTENT REGISTRY
# One dict entry per intent — all three layers in one place.
# Each entry is read-only metadata; no logic lives here.
# ─────────────────────────────────────────────────────────────────────────────

def _build_registry() -> dict[str, dict]:
    """
    Build the registry by joining all three data sources on intent key.
    Called once at import time.  Returns a plain dict — no classes needed.

    For each intent key, the registry entry contains:
      triggers    — keyword list from TEMPLATE_INTENT_KEYWORDS (primary)
                    falls back to INTENT_KEYWORDS (main.py) if not present
      strategy    — strategy dict from INTENT_STRATEGY
      components  — component dict from COMPONENTS
      gaps        — list of missing layers (empty = fully wired)
    """
    # Collect all known intent keys across all three sources
    all_keys: set[str] = set()
    all_keys.update(TEMPLATE_INTENT_KEYWORDS.keys())
    all_keys.update(INTENT_KEYWORDS.keys())
    all_keys.update(INTENT_STRATEGY.keys())
    all_keys.update(
        k for k in COMPONENTS.keys() if not k.startswith("_")
    )

    registry: dict[str, dict] = {}

    for key in sorted(all_keys):
        # Layer 1 — triggers
        triggers = (
            TEMPLATE_INTENT_KEYWORDS.get(key)
            or INTENT_KEYWORDS.get(key)
            or []
        )

        # Layer 2 — strategy
        strategy = INTENT_STRATEGY.get(key)

        # Layer 3 — template components
        components = COMPONENTS.get(key)

        # Detect gaps
        gaps: list[str] = []
        if not triggers:
            gaps.append("triggers")
        if strategy is None:
            gaps.append("strategy")
        if components is None:
            gaps.append("components")

        registry[key] = {
            "intent":     key,
            "triggers":   triggers,
            "strategy":   strategy or {},
            "components": components or {},
            "gaps":       gaps,
            # Convenience flags
            "complete":   len(gaps) == 0,
            "has_triggers":   bool(triggers),
            "has_strategy":   strategy is not None,
            "has_components": components is not None,
        }

    return registry


# Build once at import time
INTENT_REGISTRY: dict[str, dict] = _build_registry()


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — run at import, prints a gap report
# Does not raise — the system still works with partial intents
# ─────────────────────────────────────────────────────────────────────────────

def _validate_registry(registry: dict[str, dict]) -> None:
    """
    Print a structured gap report to stdout.
    Called once at import time so gaps are visible immediately.
    """
    total    = len(registry)
    complete = sum(1 for r in registry.values() if r["complete"])
    gapped   = total - complete

    print(f"\n[IntentRegistry] {total} intents registered — "
          f"{complete} complete, {gapped} with gaps")

    if gapped:
        print("[IntentRegistry] Gaps (intent → missing layers):")
        for key, rec in sorted(registry.items()):
            if rec["gaps"]:
                print(f"  {key:<32}  missing: {', '.join(rec['gaps'])}")
    else:
        print("[IntentRegistry] All intents fully wired ✓")

    print()


_validate_registry(INTENT_REGISTRY)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def registry_for(intent: str) -> dict:
    """
    Return the full registry entry for a single intent.

    Includes all three layers plus gap info.
    Falls back to 'general' if the intent is not registered.

    Example:
        rec = registry_for("trust_issue")
        print(rec["triggers"])    # keyword list
        print(rec["strategy"])    # goal, approach, tone_hint, avoid
        print(rec["components"])  # acknowledgment, body, closing
    """
    return INTENT_REGISTRY.get(intent, INTENT_REGISTRY.get("general", {}))


def detect(message: str) -> str:
    """
    Detect intent from a raw customer message.

    Uses detect_template_intent() from template_engine (primary — has
    the full expanded keyword set including new intents).  Falls back to
    detect_intent() from main.py for compatibility.

    Returns:
        str — intent key, or 'general' if nothing matches
    """
    return detect_template_intent(message)


def strategy_for(intent: str) -> dict:
    """
    Return just the strategy dict for an intent.

    Thin wrapper around quality_control.get_strategy() —
    exposed here so callers only need to import intent_registry.
    """
    return get_strategy(intent)


def strategy_prompt_for(intent: str) -> str:
    """
    Return the strategy as a formatted string block for prompt injection.

    Thin wrapper around quality_control.build_strategy_block().
    Ready to drop directly into a Claude prompt.
    """
    return build_strategy_block(intent)


def components_for(intent: str) -> dict:
    """
    Return just the template components dict for an intent.
    Falls back to 'general' if not found.
    """
    return COMPONENTS.get(intent, COMPONENTS.get("general", {}))


def all_triggers() -> dict[str, list[str]]:
    """
    Return a flat dict of every intent → its trigger phrase list.
    Useful for building UI dropdowns or debug inspection.
    """
    return {key: rec["triggers"] for key, rec in INTENT_REGISTRY.items()}


def all_intents() -> list[str]:
    """
    Return a sorted list of all registered intent keys.
    """
    return sorted(INTENT_REGISTRY.keys())


def complete_intents() -> list[str]:
    """
    Return only intents where all three layers are present.
    """
    return sorted(k for k, v in INTENT_REGISTRY.items() if v["complete"])


def gapped_intents() -> dict[str, list[str]]:
    """
    Return a dict of incomplete intents and their missing layers.
    Empty dict means everything is wired.
    """
    return {
        k: v["gaps"]
        for k, v in INTENT_REGISTRY.items()
        if v["gaps"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE RUNNER
# Wires all four stages in one call:
#   input → detect → strategy → build template → QC
#
# This is the canonical implementation of the intended flow.
# Nothing here is new logic — it calls existing functions in the right order.
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(
    message: str,
    domain_name:        Optional[str] = None,
    asking_price:       Optional[str] = None,
    force_intent:       Optional[str] = None,
    response_length:    str = "medium",
    tone:               str = "professional and persuasive",
    api_key:            Optional[str] = None,
    polish:             bool = False,
) -> dict:
    """
    Run the full pipeline on a single customer message.

    Stages
    ──────
    1. DETECT   — detect_template_intent(message)
    2. STRATEGY — get_strategy(intent)          → strategy dict
    3. GENERATE — build_template_reply(...)     → assembled email
    4. QC       — run_full_qc(reply, intent)    → validation report

    Optional stage (requires api_key + polish=True):
    5. POLISH   — ai_polish_reply(...)          → improved reply

    Args:
        message         Raw customer message text
        domain_name     Optional domain name to inject into the reply
        asking_price    Optional price string to inject
        force_intent    Skip detection and use this intent directly
        response_length 'short' | 'medium' | 'long'
        tone            Tone instruction string passed to AI polish
        api_key         Anthropic API key (only needed if polish=True)
        polish          Whether to run AI polish stage (requires api_key)

    Returns:
        {
          "stage_1_intent":    str   — detected (or forced) intent
          "stage_2_strategy":  dict  — full strategy dict for this intent
          "stage_3_reply":     str   — assembled template email
          "stage_4_qc":        dict  — QC report {passed, issues, fixes, ...}
          "stage_5_polished":  str   — AI-polished reply (or same as stage_3)
          "components_used":   dict  — which component variants were selected
          "registry_entry":    dict  — full registry record for this intent
          "pipeline_summary":  str   — one-line human-readable summary
        }
    """

    # ── Stage 1: Detect ───────────────────────────────────────────────────────
    intent = force_intent if force_intent else detect_template_intent(message)

    # ── Stage 2: Strategy ─────────────────────────────────────────────────────
    strategy = get_strategy(intent)

    # ── Stage 3: Generate ─────────────────────────────────────────────────────
    template_result = build_template_reply(
        customer_message=message,
        domain_name=domain_name,
        asking_price=asking_price,
        force_intent=intent,
        response_length=response_length,
    )
    raw_reply      = template_result["reply"]
    components_used = template_result["components_used"]

    # ── Stage 4: QC ───────────────────────────────────────────────────────────
    qc_report = run_full_qc(raw_reply, intent=intent)
    # Use the QC-fixed reply (may have greeting/closing auto-patched)
    reply_after_qc = qc_report["reply"]

    # ── Stage 5: Polish (optional) ────────────────────────────────────────────
    polished_reply = reply_after_qc  # default: unchanged
    polish_applied = False

    if polish and api_key:
        polish_result = ai_polish_reply(
            template_reply=reply_after_qc,
            customer_message=message,
            intent=intent,
            api_key=api_key,
            domain_name=domain_name,
            asking_price=asking_price,
            tone=tone,
        )
        polished_reply = polish_result.get("polished_reply", reply_after_qc)
        polish_applied = polish_result.get("ai_polish", False)

    # ── Summary ───────────────────────────────────────────────────────────────
    qc_status = "✓ QC passed" if qc_report["validation"]["passed"] else "⚠ QC issues"
    polish_status = ("+ AI polished" if polish_applied else "")
    summary = f"intent={intent} | {qc_status} | length={response_length} {polish_status}".strip()

    return {
        "stage_1_intent":   intent,
        "stage_2_strategy": strategy,
        "stage_3_reply":    raw_reply,
        "stage_4_qc":       qc_report,
        "stage_5_polished": polished_reply,
        "components_used":  components_used,
        "registry_entry":   registry_for(intent),
        "pipeline_summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIBE — debug helper
# Print a human-readable view of all three layers for a given intent.
# Useful for auditing or building documentation.
# ─────────────────────────────────────────────────────────────────────────────

def describe(intent: str) -> None:
    """
    Print a formatted view of all three layers for one intent.

    Shows exactly what triggers it, how the strategy directs the reply,
    and what template component variants are available.
    """
    rec = registry_for(intent)
    if not rec:
        print(f"[describe] Intent '{intent}' not found in registry.")
        return

    _hr = "─" * 68

    print(f"\n{'═' * 68}")
    print(f"  INTENT: {intent.upper()}")
    print(f"{'═' * 68}")

    # ── Layer 1: Triggers ─────────────────────────────────────────────────────
    print(f"\n  LAYER 1 — TRIGGERS  (intent detection)")
    print(f"  {_hr}")
    triggers = rec["triggers"]
    if triggers:
        for i, t in enumerate(triggers, 1):
            print(f"    {i:>2}. \"{t}\"")
    else:
        print("    (none registered)")

    # ── Layer 2: Strategy ─────────────────────────────────────────────────────
    print(f"\n  LAYER 2 — STRATEGY  (how to respond)")
    print(f"  {_hr}")
    s = rec["strategy"]
    if s:
        print(f"    Goal:   {s.get('goal', '—')}")
        print(f"    Tone:   {s.get('tone_hint', '—')}")
        print(f"    Avoid:  {s.get('avoid', '—')}")
        print(f"    Steps:")
        for step in s.get("approach", []):
            print(f"      • {step}")
    else:
        print("    (no strategy registered)")

    # ── Layer 3: Components ───────────────────────────────────────────────────
    print(f"\n  LAYER 3 — TEMPLATE COMPONENTS  (how the email is built)")
    print(f"  {_hr}")
    c = rec["components"]
    if c:
        for section in ("acknowledgment", "body", "closing"):
            variants = c.get(section, [])
            print(f"    {section.upper()} ({len(variants)} variant{'s' if len(variants) != 1 else ''}):")
            for v in variants:
                preview = (v[:80] + "…") if len(v) > 80 else v
                print(f"      · {preview}")
    else:
        print("    (no components registered)")

    # ── Gap status ────────────────────────────────────────────────────────────
    if rec["gaps"]:
        print(f"\n  ⚠  GAPS: {', '.join(rec['gaps'])}")
    else:
        print(f"\n  ✓  Fully wired — all three layers present")

    print(f"\n{'═' * 68}\n")


def describe_all() -> None:
    """Print a summary table of all intents and their layer status."""
    print(f"\n{'═' * 68}")
    print(f"  INTENT REGISTRY — FULL STATUS ({len(INTENT_REGISTRY)} intents)")
    print(f"{'═' * 68}")
    print(f"  {'INTENT':<32} {'TRIGGERS':>8} {'STRATEGY':>9} {'COMPONENTS':>11} {'STATUS':>8}")
    print(f"  {'─'*32} {'─'*8} {'─'*9} {'─'*11} {'─'*8}")

    for key in sorted(INTENT_REGISTRY.keys()):
        rec = INTENT_REGISTRY[key]
        t_icon = "✓" if rec["has_triggers"]   else "✗"
        s_icon = "✓" if rec["has_strategy"]   else "✗"
        c_icon = "✓" if rec["has_components"] else "✗"
        status = "complete" if rec["complete"] else f"gaps({len(rec['gaps'])})"
        print(f"  {key:<32} {t_icon:>8} {s_icon:>9} {c_icon:>11} {status:>8}")

    print(f"{'═' * 68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DIAGRAM — prints the flow for documentation or debugging
# ─────────────────────────────────────────────────────────────────────────────

def print_pipeline() -> None:
    """Print the four-stage pipeline as an annotated flow diagram."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              EMAIL REPLY PIPELINE — FLOW OVERVIEW                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  INPUT                                                               ║
║  ─────                                                               ║
║  Raw customer message text (+ optional domain_name, asking_price)   ║
║                                                                      ║
║      │                                                               ║
║      ▼                                                               ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 1 — INTENT DETECTION                                 │    ║
║  │  Function:   detect_template_intent(message)                │    ║
║  │  Source:     template_engine.py → TEMPLATE_INTENT_KEYWORDS  │    ║
║  │  Fallback:   main.py → detect_intent()                      │    ║
║  │  Output:     intent key  e.g. "trust_issue"                 │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║      │                                                               ║
║      ▼                                                               ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 2 — STRATEGY SELECTION                               │    ║
║  │  Function:   get_strategy(intent)                           │    ║
║  │  Source:     quality_control.py → INTENT_STRATEGY           │    ║
║  │  Output:     { goal, approach, tone_hint, avoid }           │    ║
║  │  Prompt use: build_strategy_block(intent) → injected into   │    ║
║  │              Claude prompt in main.py build_reply_prompt()  │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║      │                                                               ║
║      ▼                                                               ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 3 — TEMPLATE GENERATION                              │    ║
║  │  Function:   build_template_reply(message, intent, ...)     │    ║
║  │  Source:     template_engine.py → COMPONENTS                │    ║
║  │  Selects:    random.choice() per component section          │    ║
║  │  Output:     assembled email string + components_used dict  │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║      │                                                               ║
║      ▼                                                               ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 4 — QUALITY CONTROL                                  │    ║
║  │  Function:   run_full_qc(reply, intent)                     │    ║
║  │  Source:     quality_control.py → validate_email()          │    ║
║  │  Checks:     greeting, CTA, closing, word count, paragraphs │    ║
║  │  Output:     { passed, issues, fixes, fixed_reply }         │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║      │                                                               ║
║      ▼  (optional — requires api_key)                               ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 5 — AI POLISH                                        │    ║
║  │  Function:   ai_polish_reply(template_reply, ...)           │    ║
║  │  Source:     template_engine.py → calls Claude API          │    ║
║  │  Input:      QC-fixed reply + strategy context              │    ║
║  │  Output:     natural, polished reply (facts preserved)      │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║      │                                                               ║
║      ▼                                                               ║
║  OUTPUT                                                              ║
║  ──────                                                              ║
║  Final email reply — ready to send                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run directly to inspect, validate, or test the registry
#
#   python intent_registry.py               → print gap report + table
#   python intent_registry.py describe all  → full table
#   python intent_registry.py describe <intent>  → one intent detail
#   python intent_registry.py pipeline      → print pipeline diagram
#   python intent_registry.py test <msg>    → run full_pipeline on a message
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args or args[0] == "table":
        describe_all()

    elif args[0] == "pipeline":
        print_pipeline()

    elif args[0] == "describe":
        if len(args) < 2:
            print("Usage: python intent_registry.py describe <intent_key>")
            print("       python intent_registry.py describe all")
        elif args[1] == "all":
            for key in sorted(INTENT_REGISTRY.keys()):
                describe(key)
        else:
            describe(args[1])

    elif args[0] == "test":
        if len(args) < 2:
            print("Usage: python intent_registry.py test \"your message here\"")
        else:
            msg = " ".join(args[1:])
            print(f"\nRunning pipeline on: \"{msg}\"")
            result = full_pipeline(msg)
            print(f"\n── Stage 1 — Intent ────────────────────────────────────")
            print(f"  {result['stage_1_intent']}")
            print(f"\n── Stage 2 — Strategy ──────────────────────────────────")
            s = result["stage_2_strategy"]
            print(f"  Goal:  {s.get('goal', '—')}")
            print(f"  Tone:  {s.get('tone_hint', '—')}")
            print(f"\n── Stage 3 — Reply ─────────────────────────────────────")
            print(result["stage_3_reply"])
            print(f"\n── Stage 4 — QC ────────────────────────────────────────")
            print(f"  {result['stage_4_qc']['summary']}")
            print(f"\n── Summary ─────────────────────────────────────────────")
            print(f"  {result['pipeline_summary']}")

    elif args[0] == "gaps":
        gaps = gapped_intents()
        if gaps:
            print("\nIncomplete intents:")
            for k, v in sorted(gaps.items()):
                print(f"  {k:<32}  missing: {', '.join(v)}")
        else:
            print("\nAll intents fully wired ✓")

    else:
        print(f"Unknown command: {args[0]}")
        print("Commands: table | pipeline | describe <intent|all> | test <msg> | gaps")
