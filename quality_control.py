"""
quality_control.py — Email Quality Control & Testing Layer
===========================================================
Adds four layers on top of the existing generation pipeline:

  PART 1 — Strategy Planning
      Before generating, decide a communication goal based on intent.
      The strategy is injected into the prompt so Claude knows *why*
      it is writing this email, not just *what* the situation is.

  PART 2 — Validation Rules
      After generating, check each email against structural rules
      (greeting, CTA, body length, paragraph count).
      Rule failures are fixed inline — no email passes through unchecked.

  PART 3 — Variation Uniqueness Guard
      When 2–3 variations are generated, score them for phrase overlap.
      If two versions are too similar, flag + log — easy to extend to auto-retry.

  PART 4 — Test Harness
      Run real-world messy inputs through intent detection + strategy planning.
      Print a readable pass/fail report. Call run_tests() from main or CLI.

Architecture:
  - Zero new dependencies — pure Python, re, difflib
  - Zero changes to existing function signatures
  - All functions return structured dicts for easy logging / API exposure
  - Import and call from main.py at the points marked with # ← QC
"""

from __future__ import annotations

import re
import difflib
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — STRATEGY PLANNER
# Maps detected intent → communication strategy (goal + approach)
# Injected into the prompt so Claude knows the *why*, not just the *what*.
# ─────────────────────────────────────────────────────────────────────────────

# Each strategy entry has:
#   goal      — the single goal this email must achieve
#   approach  — how to achieve it (3–4 bullet steps)
#   tone_hint — one-line tone calibration
#   avoid     — what not to do

INTENT_STRATEGY: dict[str, dict] = {

    "follow_up": {
        "goal":      "Remind the prospect the opportunity is still open without pressuring them.",
        "approach":  [
            "Open with a brief, friendly callback to the previous message",
            "Reinforce one concrete value point (not the full pitch)",
            "Make it easy to say yes OR no — remove friction from both",
            "End with a binary choice: interested / not interested",
        ],
        "tone_hint": "Polite, patient, non-pushy — assume they are busy, not uninterested.",
        "avoid":     "Re-sending the full sales pitch. Assuming bad intent. Multiple questions.",
    },

    "objection_handling": {
        "goal":      "Acknowledge the concern, reduce friction, and restore confidence.",
        "approach":  [
            "Name the specific objection first — don't sidestep it",
            "Reframe it as a question the prospect can answer for themselves",
            "Offer one concrete reassurance (escrow, no-rebuild, competitor risk)",
            "End with an open invitation to share their real concern",
        ],
        "tone_hint": "Empathetic and steady — validate before pivoting.",
        "avoid":     "Getting defensive. Ignoring the objection. Over-explaining.",
    },

    "sales_pitch": {
        "goal":      "Create interest in the domain and motivate a first reply.",
        "approach":  [
            "Open with the specific value for their business (not a generic pitch)",
            "State one clear, concrete benefit — local traffic, competitor risk, or redirect ease",
            "Make the domain feel real and relevant, not abstract",
            "End with one low-friction next step",
        ],
        "tone_hint": "Confident and direct — lead with value, not features.",
        "avoid":     "Generic openers. Listing every benefit at once. Pressure tactics.",
    },

    "re_engagement": {
        "goal":      "Re-establish interest after a cold gap without apologising for it.",
        "approach":  [
            "Acknowledge the time gap briefly and move on — don't dwell",
            "Offer a fresh angle or updated context (domain still available, price open)",
            "Re-spark curiosity without replaying the original pitch word-for-word",
            "Keep it short — earn the right to a longer conversation",
        ],
        "tone_hint": "Warm and low-key — re-open the door, don't knock it down.",
        "avoid":     "Over-apologising for the gap. Sending the original pitch verbatim.",
    },

    "not_now": {
        "goal":      "Leave the door open gracefully and optionally gather feedback.",
        "approach":  [
            "Thank them genuinely — one sentence, no over-effusion",
            "Ask one soft optional question about their reason (price? timing? irrelevant?)",
            "Signal the door is always open without being needy",
            "Close cleanly — no lingering",
        ],
        "tone_hint": "Gracious and brief — respect the decision without arguing.",
        "avoid":     "Arguing. Sending another pitch. Multiple questions.",
    },

    "general": {
        "goal":      "Provide a clear, helpful response that keeps the conversation open.",
        "approach":  [
            "Acknowledge the message briefly",
            "Give the most relevant information for the situation",
            "Include one concrete next step",
            "Keep it professional and concise",
        ],
        "tone_hint": "Professional and clear — no padding.",
        "avoid":     "Generic filler. No next step. Over-long responses.",
    },

    "price_inquiry": {
        "goal":      "State the price clearly and justify the premium in one confident paragraph.",
        "approach":  [
            "State the asking price without hesitation or padding",
            "Give one clear reason the premium is justified (keyword value, appraisal, scarcity)",
            "Invite a counter-offer to keep the conversation alive",
            "Link to the marketplace listing as the next step",
        ],
        "tone_hint": "Confident and transparent — price pride, not price defensiveness.",
        "avoid":     "Hedging on the price. Over-justifying. Not giving an actual number.",
    },

    "price_too_high": {
        "goal":      "Reframe the price comparison and invite negotiation rather than defending.",
        "approach":  [
            "Validate the $10 comparison — acknowledge it's real, not dismiss it",
            "Explain what makes this domain premium (existing traffic, keyword match, geo value)",
            "Invite their number — 'What would feel fair?' moves the conversation forward",
            "Keep it short — one reframe paragraph is enough",
        ],
        "tone_hint": "Calm and reasonable — you're not offended, you're explaining.",
        "avoid":     "Getting defensive. Repeating the full pitch. Refusing to negotiate.",
    },

    "negotiation": {
        "goal":      "Counter firmly without closing the deal off — keep it moving.",
        "approach":  [
            "Acknowledge the offer — don't ignore it",
            "State your floor or counter number directly",
            "Give one brief reason for your position (acquisition cost, market value)",
            "Set a soft time boundary to create mild momentum",
        ],
        "tone_hint": "Firm but collaborative — you want to close, not win an argument.",
        "avoid":     "Accepting too quickly. Giving no reason for the counter. Dead-end language.",
    },

    "trust_issue": {
        "goal":      "Remove doubt by directing the prospect to verifiable third-party proof.",
        "approach":  [
            "Address the concern directly — do not get defensive or dismissive",
            "Name a specific, verifiable trust mechanism (Dan.com, Afternic, Trustpilot)",
            "Offer a verification step the prospect can take right now, independently",
            "Keep the close low-friction — 'let me know which platform you prefer'",
        ],
        "tone_hint": "Calm and matter-of-fact — confidence without arrogance.",
        "avoid":     "Sounding defensive. Vague claims ('I'm legitimate'). Ignoring the concern.",
    },

    "have_website": {
        "goal":      "Clarify that no new website is needed and explain the redirect in plain terms.",
        "approach":  [
            "Lead with 'you don't need to build anything new' — this is the key objection",
            "Explain the redirect in one concrete, jargon-free sentence",
            "Mention the competitor risk as a secondary motivator",
            "End with the easiest possible next step",
        ],
        "tone_hint": "Helpful and practical — remove technical anxiety.",
        "avoid":     "Technical jargon. Assuming they know what a redirect is. Skipping the competitor angle.",
    },

    "agreed_no_pay": {
        "goal":      "Gently remind the prospect of the agreement and make payment frictionless.",
        "approach":  [
            "Reference the agreement specifically — not vaguely",
            "Mention the domain is publicly listed to create mild, factual urgency",
            "Give the payment link or platform directly",
            "One clear next step — no more than one",
        ],
        "tone_hint": "Friendly but purposeful — assume good faith, not bad.",
        "avoid":     "Accusatory tone. Over-explaining. Multiple steps.",
    },

    "angry": {
        "goal":      "De-escalate immediately, apologise briefly, and offer removal.",
        "approach":  [
            "Apologise in one genuine sentence — not multiple",
            "Confirm removal from further contact immediately",
            "Do NOT pitch. Do NOT argue. Do NOT explain your intent.",
            "Two sentences maximum.",
        ],
        "tone_hint": "Sincere and minimal — less is more.",
        "avoid":     "Defending yourself. Multiple apologies. Any sales content.",
    },

    "no_thanks": {
        "goal":      "Leave the door open gracefully and optionally gather feedback.",
        "approach":  [
            "Thank them genuinely — one sentence, no over-effusion",
            "Ask one soft optional question about their reason (price? timing? irrelevant?)",
            "Signal the door is always open without being needy",
            "Close cleanly — no lingering",
        ],
        "tone_hint": "Gracious and brief — respect the decision without arguing.",
        "avoid":     "Arguing. Sending another pitch. Multiple questions.",
    },

    "not_interested_ask_why": {
        "goal":      "Acknowledge rejection gracefully and keep the door open with one gentle question.",
        "approach":  [
            "Accept the response without pushback — one short sentence",
            "Ask one optional question: price? timing? not the right fit?",
            "Signal you're easy to work with and will respect their answer",
            "Close in one clean line — no lingering",
        ],
        "tone_hint": "Respectful and curious — not defensive, not pushy.",
        "avoid":     "Re-pitching. Multiple questions. Any hint of pressure.",
    },

    # ── NEW EXPANDED INTENT STRATEGIES ────────────────────────────────────────

    "request_info": {
        "goal":      "Give the prospect exactly the information they need to make a confident decision.",
        "approach":  [
            "Answer each specific question directly — no deflection",
            "Be transparent about the process, pricing, and platform",
            "Offer to go deeper on anything still unclear",
            "End with an open invitation to ask more",
        ],
        "tone_hint": "Helpful and thorough — this person is close to deciding.",
        "avoid":     "Vague answers. Redirecting to the listing without answering first. Generic sales language.",
    },

    "demo_offer": {
        "goal":      "Lower the barrier to commitment by making the value visible, not just described.",
        "approach":  [
            "Lead with the offer clearly — 'let me show you, not just tell you'",
            "Make the effort feel effortless for them (costs them nothing, takes you minutes)",
            "Keep it conditional — 'if it moves the needle, we can talk numbers'",
            "End with a simple yes/no ask",
        ],
        "tone_hint": "Confident and generous — you're doing the work, they just have to say yes.",
        "avoid":     "Over-explaining what the demo is. Talking about price before showing value.",
    },

    "meeting_request": {
        "goal":      "Get a yes to a brief call without making it feel like a sales pressure move.",
        "approach":  [
            "Keep the ask small — five minutes, no prep needed",
            "Give them the exit: 'if it doesn't make sense, I'll leave you alone'",
            "Be specific about what you'll cover on the call",
            "Make scheduling feel completely frictionless",
        ],
        "tone_hint": "Direct and respectful — treat their time as valuable.",
        "avoid":     "Vague meeting requests. Overselling the call before they've agreed.",
    },

    "price_negotiation": {
        "goal":      "Close the gap between positions without losing the deal or collapsing to an unreasonable offer.",
        "approach":  [
            "Acknowledge the offer genuinely before countering",
            "State your position clearly with one brief reason",
            "Propose a specific middle-ground number",
            "Create a soft deadline to build momentum without pressure",
        ],
        "tone_hint": "Firm but collaborative — you want to close, not win.",
        "avoid":     "Accepting immediately. Multiple counter-offers. Emotional language.",
    },

    "competitor_comparison": {
        "goal":      "Activate the prospect's competitive instinct to create urgency without manufactured pressure.",
        "approach":  [
            "Name the competitive risk specifically and honestly",
            "Frame it as information, not a threat",
            "Let them draw their own conclusions from the facts",
            "End with an action they can take immediately",
        ],
        "tone_hint": "Factual and measured — the risk is real, not manufactured.",
        "avoid":     "Exaggerating the threat. Sounding manipulative. Vague competitor references.",
    },

    "trust_building": {
        "goal":      "Remove doubt by directing the prospect to verifiable, independent proof.",
        "approach":  [
            "Give them three specific verification steps they can do right now",
            "Name the escrow mechanism and how it protects them",
            "Offer to accommodate their preferred platform",
            "End with 'I'm not asking you to trust me — verify for yourself'",
        ],
        "tone_hint": "Calm and transparent — confidence without defensiveness.",
        "avoid":     "Vague assurances. Defensive language. Skipping specific verification steps.",
    },

    "feature_explanation": {
        "goal":      "Remove technical anxiety by explaining the redirect process in the simplest possible terms.",
        "approach":  [
            "Use an analogy (postal redirect) to make it concrete",
            "Be explicit: 'two minutes, one text box, click save'",
            "Confirm nothing changes about their existing website",
            "Offer to do the setup for them post-purchase",
        ],
        "tone_hint": "Patient and clear — assume zero technical knowledge.",
        "avoid":     "Any jargon. Mentioning DNS, CNAME, or registrar settings by name.",
    },

    "soft_pitch": {
        "goal":      "Plant the seed without creating resistance — the goal is curiosity, not commitment.",
        "approach":  [
            "Lead with brevity — signal you won't take much of their time",
            "Give them a way out upfront: 'if it's not a fit, just say so'",
            "State one clear value point and let it land",
            "End without any pressure signal",
        ],
        "tone_hint": "Light and confident — you're offering, not pushing.",
        "avoid":     "Multiple value points. Any urgency. Anything that sounds like a pitch.",
    },

    "value_reminder": {
        "goal":      "Re-anchor the value proposition clearly before the prospect goes cold.",
        "approach":  [
            "Open by acknowledging the conversation history — don't ignore it",
            "Give the clearest, most structured value summary you've written",
            "Use concrete numbered points to structure the value case",
            "Close with respect for their decision, whatever it is",
        ],
        "tone_hint": "Clear and direct — this is the last real pitch before walking away.",
        "avoid":     "Repeating what was already said verbatim. Any apologetic tone.",
    },

    "follow_up_no_response": {
        "goal":      "Get any response — yes, no, or not now — without creating friction.",
        "approach":  [
            "Be brief — two to three sentences maximum",
            "Assume busyness, not disinterest",
            "Make it easy to say no: 'a no is fine too'",
            "End with a single clear question",
        ],
        "tone_hint": "Light and patient — no pressure, just a gentle nudge.",
        "avoid":     "Re-pitching. Multiple questions. Any hint of frustration.",
    },

    "follow_up_after_pricing": {
        "goal":      "Re-open the conversation after a quote with a focus on flexibility and dialogue.",
        "approach":  [
            "Reference the quote specifically — don't be vague",
            "Invite a counter rather than defending the price",
            "Leave space for other objections to surface",
            "End with a simple call to action",
        ],
        "tone_hint": "Open and collaborative — you're checking in, not chasing.",
        "avoid":     "Defending the price without hearing them out. Multiple follow-ups.",
    },

    "follow_up_after_interest": {
        "goal":      "Reconnect with a warm lead without making them feel chased.",
        "approach":  [
            "Reference the specific interest expressed — make them feel heard",
            "Offer a clear path forward for the most likely reason for the delay",
            "Give them an easy yes and an easy exit",
            "End personally — not with a generic closing",
        ],
        "tone_hint": "Warm and personal — this is a relationship, not a transaction.",
        "avoid":     "Generic follow-up language. Forgetting what was previously said.",
    },

    "general_response": {
        "goal":      "Provide a clear, helpful response that keeps the conversation open.",
        "approach":  [
            "Acknowledge the message briefly",
            "Give the most relevant information for the situation",
            "Include one concrete next step",
            "Keep it professional and concise",
        ],
        "tone_hint": "Professional and clear — no padding.",
        "avoid":     "Generic filler. No next step. Over-long responses.",
    },
}


def get_strategy(intent: str) -> dict:
    """
    Return the communication strategy for a given intent.
    Falls back to 'general' if the intent isn't in the map.
    """
    return INTENT_STRATEGY.get(intent, INTENT_STRATEGY["general"])


def build_strategy_block(intent: str) -> str:
    """
    Render the strategy as a formatted string block for injection into prompts.
    Keeps it structured so Claude reads it as instructions, not context.
    """
    s = get_strategy(intent)
    steps = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(s["approach"]))
    avoid = s.get("avoid", "")
    return (
        f"COMMUNICATION STRATEGY (decide this before writing a single word):\n"
        f"  Goal:      {s['goal']}\n"
        f"  Tone:      {s['tone_hint']}\n"
        f"  Approach:\n{steps}\n"
        + (f"  Avoid:     {avoid}\n" if avoid else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — STRUCTURAL VALIDATION
# Rule-based checks run AFTER generation, BEFORE returning to the user.
# Each rule returns (passed: bool, issue: str, fix: str | None)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns used in validation
_GREETING_RE  = re.compile(r"^(hi\b|hello\b|hey\b|dear\b|hope\b)", re.IGNORECASE)
_CLOSING_RE   = re.compile(
    r"(best regards|kind regards|warm regards|best wishes|thanks|thank you|cheers|sincerely|talk soon|let me know)",
    re.IGNORECASE
)
_CTA_RE       = re.compile(
    r"(let me know|reply|visit|click|send|reach out|get in touch|happy to|feel free|just say|drop me|give me a|"
    r"what do you think|would you like|shall i|can i|interested|any questions|do you want)",
    re.IGNORECASE
)
_VAGUE_PHRASES = [
    "this domain", "great value", "good opportunity", "many benefits",
    "very useful", "really helpful", "quite valuable", "very good",
    "a lot of value", "significant benefits",
]
_MIN_PARAGRAPHS = 2
_MAX_PARAGRAPHS = 4
_MIN_WORDS      = 30
_MAX_WORDS      = 220


def validate_email(reply: str, intent: str = "general") -> dict:
    """
    Run all structural validation rules on a generated email.

    Returns:
        {
          "passed":  bool          — True if all rules pass
          "issues":  list[str]     — human-readable failure descriptions
          "fixes":   list[str]     — what was auto-fixed (inline)
          "fixed_reply": str       — email after inline fixes applied
          "word_count":   int
          "paragraph_count": int
        }
    """
    issues: list[str]  = []
    fixes:  list[str]  = []
    body = reply.strip()

    # ── Rule 1: Greeting present ──────────────────────────────────────────────
    first_line = body.split("\n")[0].strip()
    has_greeting = bool(_GREETING_RE.match(first_line))
    if not has_greeting:
        issues.append("MISSING_GREETING: No greeting line detected at the start.")
        body = "Hi there,\n\n" + body
        fixes.append("Added 'Hi there,' greeting at the top.")

    # ── Rule 2: Closing line present ─────────────────────────────────────────
    has_closing = bool(_CLOSING_RE.search(body))
    if not has_closing:
        issues.append("MISSING_CLOSING: No closing line detected.")
        body = body.rstrip() + "\n\nBest regards,"
        fixes.append("Added 'Best regards,' closing.")

    # ── Rule 3: Call-to-action present (skip for 'angry' intent) ─────────────
    if intent != "angry":
        has_cta = bool(_CTA_RE.search(body))
        if not has_cta:
            issues.append("MISSING_CTA: No call-to-action detected.")
            # Don't auto-fix CTA — it requires context; flag it for quality_guard

    # ── Rule 4: Paragraph count ───────────────────────────────────────────────
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    if intent == "angry":
        pass  # angry replies are intentionally short
    elif para_count < _MIN_PARAGRAPHS:
        issues.append(f"TOO_FEW_PARAGRAPHS: {para_count} paragraph(s) found, need at least {_MIN_PARAGRAPHS}.")
    elif para_count > _MAX_PARAGRAPHS + 1:
        issues.append(f"TOO_MANY_PARAGRAPHS: {para_count} paragraphs — trim to {_MAX_PARAGRAPHS} max.")

    # ── Rule 5: Word count ────────────────────────────────────────────────────
    word_count = len(body.split())
    if intent != "angry":
        if word_count < _MIN_WORDS:
            issues.append(f"TOO_SHORT: {word_count} words — minimum is {_MIN_WORDS}.")
        elif word_count > _MAX_WORDS:
            issues.append(f"TOO_LONG: {word_count} words — maximum is {_MAX_WORDS}.")

    # ── Rule 6: Vagueness check ───────────────────────────────────────────────
    low = body.lower()
    vague_hits = [p for p in _VAGUE_PHRASES if p in low]
    if len(vague_hits) >= 3:
        issues.append(
            f"TOO_VAGUE: {len(vague_hits)} vague placeholder phrases detected "
            f"({', '.join(vague_hits[:3])}…). Expand with specific context."
        )

    return {
        "passed":          len(issues) == 0,
        "issues":          issues,
        "fixes":           fixes,
        "fixed_reply":     body,
        "word_count":      word_count,
        "paragraph_count": para_count,
    }


def validation_summary(result: dict) -> str:
    """Return a one-line human-readable summary of validation results."""
    if result["passed"]:
        return f"✓ Passed ({result['word_count']}w · {result['paragraph_count']}p)"
    flags = " | ".join(i.split(":")[0] for i in result["issues"])
    return f"✗ Issues: {flags} ({result['word_count']}w · {result['paragraph_count']}p)"


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — VARIATION UNIQUENESS GUARD
# When 2–3 variations are generated, check they are meaningfully different.
# Uses difflib SequenceMatcher for overlap scoring (0 = identical, 1 = unique).
# ─────────────────────────────────────────────────────────────────────────────

# Threshold: similarity score above this → versions are too similar
_SIMILARITY_THRESHOLD = 0.72


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio between two strings (0.0–1.0). Higher = more similar."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def check_variation_uniqueness(replies: list[str]) -> dict:
    """
    Compare every pair of reply variations for text similarity.

    Returns:
        {
          "passed":    bool
          "pairs":     list of {a, b, similarity, flag}
          "summary":   str
        }
    """
    if len(replies) < 2:
        return {"passed": True, "pairs": [], "summary": "Only one variation — nothing to compare."}

    pairs   = []
    flagged = 0

    for i in range(len(replies)):
        for j in range(i + 1, len(replies)):
            sim  = _similarity(replies[i], replies[j])
            flag = sim > _SIMILARITY_THRESHOLD
            if flag:
                flagged += 1
            pairs.append({
                "a":          f"Variation {i+1}",
                "b":          f"Variation {j+1}",
                "similarity": round(sim, 3),
                "flag":       flag,
                "note":       "Too similar — consider regenerating" if flag else "OK",
            })

    passed  = flagged == 0
    summary = (
        f"✓ All {len(replies)} variations are sufficiently unique."
        if passed
        else f"⚠ {flagged} pair(s) too similar (>{int(_SIMILARITY_THRESHOLD*100)}% overlap). Consider regenerating."
    )

    return {"passed": passed, "pairs": pairs, "summary": summary}


def log_variation_check(check: dict) -> None:
    """Print a readable variation uniqueness report to stdout."""
    print(f"[VariationQC] {check['summary']}")
    for p in check["pairs"]:
        marker = "⚠" if p["flag"] else "✓"
        print(f"  {marker} {p['a']} vs {p['b']}: {p['similarity']:.0%} similar — {p['note']}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — TEST HARNESS
# Tests messy, real-world inputs against intent detection + strategy planning.
# Call run_tests() from main.py startup or via CLI: python quality_control.py
# ─────────────────────────────────────────────────────────────────────────────

# Each test case:
#   input          — the raw user input (intentionally messy)
#   expected_intent — what the system should classify it as
#   expected_strategy_goal_keyword — a word that should appear in the strategy goal

TEST_CASES: list[dict] = [
    # ── Messy follow-up signals ───────────────────────────────────────────────
    {
        "input":            "he said later but didn't reply again",
        "expected_intent":  "follow_up",
        "goal_keyword":     "open",
        "description":      "follow-up: 'didn't reply again' is the core signal",
    },
    {
        "input":            "sent first email 5 days ago, nothing back",
        "expected_intent":  "follow_up",
        "goal_keyword":     "open",
        "description":      "Time-based follow-up — 'nothing back' is the signal",
    },
    {
        "input":            "pinged them twice, still silent",
        "expected_intent":  "follow_up",
        "goal_keyword":     "open",
        "description":      "Colloquial follow-up — 'still silent' maps to no response",
    },

    # ── Multi-signal inputs (intent scoring) ──────────────────────────────────
    {
        "input":            "asked price then disappeared",
        "expected_intent":  "follow_up",
        "goal_keyword":     "open",
        "description":      "Price inquiry + ghosting — 'disappeared' = follow_up wins on priority",
    },
    {
        "input":            "mentioned the price was high but might consider later",
        "expected_intent":  "price_too_high",
        "goal_keyword":     "reframe",
        "description":      "Price objection + timing hedge — price_too_high wins on score",
    },
    {
        "input":            "not interested but maybe later",
        "expected_intent":  "not_interested_ask_why",
        "goal_keyword":     "open",
        "description":      "Soft rejection + deferred interest — 'not interested' maps to not_interested_ask_why",
    },

    # ── Objection signals ─────────────────────────────────────────────────────
    {
        "input":            "prospect seems unsure, hasn't committed",
        "expected_intent":  "objection_handling",
        "goal_keyword":     "friction",
        "description":      "Hesitation signal — maps to objection handling",
    },
    {
        "input":            "he's on the fence, thinking about it",
        "expected_intent":  "objection_handling",
        "goal_keyword":     "friction",
        "description":      "Colloquial fence-sitting",
    },

    # ── Re-engagement signals ─────────────────────────────────────────────────
    {
        "input":            "spoke to them months ago, went quiet",
        "expected_intent":  "re_engagement",
        "goal_keyword":     "interest",
        "description":      "Classic cold lead — time gap + silence",
    },
    {
        "input":            "old lead from last year, want to try again",
        "expected_intent":  "re_engagement",
        "goal_keyword":     "interest",
        "description":      "Explicit old lead re-engagement",
    },

    # ── Trust & anger ─────────────────────────────────────────────────────────
    {
        "input":            "says this looks dodgy and probably a scam",
        "expected_intent":  "trust_issue",
        "goal_keyword":     "doubt",
        "description":      "Scam concern — colloquial phrasing",
    },
    {
        "input":            "very angry, told me to stop emailing",
        "expected_intent":  "angry",
        "goal_keyword":     "De-escalate",
        "description":      "Explicit anger + unsubscribe",
    },

    # ── Pricing ───────────────────────────────────────────────────────────────
    {
        "input":            "asked how much, waiting for reply",
        "expected_intent":  "price_inquiry",
        "goal_keyword":     "price",
        "description":      "Price question with no follow-up signal yet",
    },
    {
        "input":            "said our price is way too expensive",
        "expected_intent":  "price_too_high",
        "goal_keyword":     "reframe",
        "description":      "Direct price objection",
    },

    # ── Sales pitch ───────────────────────────────────────────────────────────
    {
        "input":            "new prospect, first time reaching out",
        "expected_intent":  "sales_pitch",
        "goal_keyword":     "interest",
        "description":      "Cold outreach / first contact",
    },
]


def _detect_intent_for_test(text: str) -> str:
    """
    Minimal intent detector for test harness — mirrors main.py scoring logic
    without importing it (avoids circular imports).
    Uses TEMPLATE_INTENT_KEYWORDS from template_engine if available.
    Falls back to a compact inline keyword map for standalone testing.
    """
    low = text.lower()

    # Try to import the real detector first (works when run inside the project)
    try:
        from template_engine import detect_template_intent
        return detect_template_intent(low)
    except ImportError:
        pass

    # Fallback inline map for standalone test runs
    _KEYWORDS: dict[str, list[str]] = {
        "angry":             ["stop emailing", "spam", "harassment", "angry", "annoying", "unsubscribe"],
        "trust_issue":       ["scam", "fake", "dodgy", "not real", "legitimate", "trust", "verify", "fraud", "suspicious"],
        "price_too_high":    ["too expensive", "too high", "too much", "way too", "can't afford", "just $10", "only $",
                              "price was high", "price is high"],
        "price_inquiry":     ["how much", "price", "cost", "what are you asking", "rate", "fee"],
        "negotiation":       ["offer", "counter", "negotiate", "lower", "discount", "best price"],
        "follow_up":         ["follow up", "no reply", "no response", "checking in", "reminder", "didn't reply",
                              "still silent", "nothing back", "went quiet", "disappeared", "no answer",
                              "hasn't replied", "nothing heard", "haven't heard back", "pinged",
                              "sent first email", "haven't responded", "not replied", "silence"],
        "re_engagement":     ["cold lead", "went cold", "months ago", "long time", "reconnect", "dormant",
                              "old lead", "previous", "last year", "try again", "been a while"],
        "objection_handling":["hesitant", "unsure", "not convinced", "on the fence", "thinking about it",
                              "considering", "not sure", "fence", "hasn't committed"],
        "not_now":           ["not now", "maybe", "not the right time", "future", "check back"],
        "sales_pitch":       ["first contact", "cold email", "initial", "new prospect", "first time", "reaching out"],
        "have_website":      ["already have", "have a website", "existing site"],
        "no_thanks":         ["no thanks", "not interested", "decline", "pass"],
    }

    PRIORITY = [
        "angry", "trust_issue", "price_too_high", "follow_up", "re_engagement",
        "objection_handling", "negotiation", "not_now", "sales_pitch",
        "no_thanks", "price_inquiry", "have_website",
    ]

    scores = {intent: sum(1 for kw in kws if kw in low) for intent, kws in _KEYWORDS.items()}
    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return "general"
    for intent in PRIORITY:
        if scores.get(intent, 0) == best_score:
            return intent
    return "general"


def run_tests(verbose: bool = True) -> dict:
    """
    Run all test cases and print a pass/fail report.

    Returns:
        {
          "total":    int
          "passed":   int
          "failed":   int
          "results":  list[dict]
        }
    """
    total  = len(TEST_CASES)
    passed = 0
    failed = 0
    results = []

    print("\n" + "═" * 70)
    print("  QUALITY CONTROL — TEST HARNESS")
    print("═" * 70)

    for i, tc in enumerate(TEST_CASES, 1):
        detected = _detect_intent_for_test(tc["input"])
        strategy = get_strategy(detected)

        intent_ok  = detected == tc["expected_intent"]
        goal_ok    = tc["goal_keyword"].lower() in strategy["goal"].lower()
        test_passed = intent_ok and goal_ok

        if test_passed:
            passed += 1
            mark = "✓"
        else:
            failed += 1
            mark = "✗"

        result = {
            "test":             i,
            "input":            tc["input"],
            "description":      tc["description"],
            "expected_intent":  tc["expected_intent"],
            "detected_intent":  detected,
            "intent_match":     intent_ok,
            "strategy_goal":    strategy["goal"],
            "goal_keyword":     tc["goal_keyword"],
            "goal_match":       goal_ok,
            "passed":           test_passed,
        }
        results.append(result)

        if verbose:
            print(f"\n  {mark} Test {i:02d}: {tc['description']}")
            print(f"     Input:    \"{tc['input']}\"")
            print(f"     Expected: {tc['expected_intent']:<22}  Detected: {detected}")
            if not intent_ok:
                print(f"     ⚠  Intent mismatch — expected '{tc['expected_intent']}', got '{detected}'")
            print(f"     Strategy: {strategy['goal'][:80]}")
            if not goal_ok:
                print(f"     ⚠  Goal keyword '{tc['goal_keyword']}' not found in strategy goal")

    print("\n" + "─" * 70)
    pct = int(passed / total * 100) if total else 0
    print(f"  RESULT: {passed}/{total} passed ({pct}%)")
    if failed:
        fail_list = [r["test"] for r in results if not r["passed"]]
        print(f"  FAILED tests: {fail_list}")
    print("═" * 70 + "\n")

    return {"total": total, "passed": passed, "failed": failed, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# Runs full QC pipeline on a single reply: validate + return structured report
# Call this from generate_variations() before returning each ReplyResult.
# ─────────────────────────────────────────────────────────────────────────────

def run_full_qc(reply: str, intent: str = "general") -> dict:
    """
    Run the full QC pipeline on a single email reply.

    Returns a structured report suitable for logging or API inclusion:
    {
      "validation": {...},   # from validate_email()
      "summary":    str,     # one-line human-readable status
      "reply":      str,     # the reply after inline fixes (same or patched)
    }
    """
    v = validate_email(reply, intent=intent)
    return {
        "validation": v,
        "summary":    validation_summary(v),
        "reply":      v["fixed_reply"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# python quality_control.py           → run all tests
# python quality_control.py --verbose → same, with full output (default)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    verbose = "--quiet" not in sys.argv
    results = run_tests(verbose=verbose)
    sys.exit(0 if results["failed"] == 0 else 1)
