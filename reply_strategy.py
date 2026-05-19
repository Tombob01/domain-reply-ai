"""
reply_strategy.py — ReplyStrategy Reasoning Layer
===================================================
Phase 1 of the reasoning-first architecture.

PURPOSE
-------
Separates the "what should this reply DO?" decision from the "write the reply"
action. build_strategy() consumes structured signals and produces a typed
ReplyStrategy object. build_reply_prompt_ai() reads that object to produce
a concise, purposeful prompt brief instead of a wall of injected rules.

DESIGN PRINCIPLES
-----------------
- Zero model calls. All logic is deterministic Python.
- No string injection. All fields are typed values.
- Conflict resolution is explicit — when signals contradict, the most
  conservative decision wins. Desperation never wins.
- The old rule tables (INTENT_RULES, TONE_INSTRUCTIONS, etc.) remain in
  main.py. This module reads them as data; it does not duplicate them.
- Backward compatible — build_reply_prompt_ai() falls back to its original
  behaviour when no strategy is passed.

ROLLBACK
--------
Remove the build_strategy() call from the two generate_reply call sites.
The prompt builder reverts to its original behaviour with no other changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports — InputAnalysis lives in pipeline.py
    from pipeline import InputAnalysis


# ── Enumerations ──────────────────────────────────────────────────────────────
# Using Literal types rather than Enum so values are plain strings in logs/debug.

PrimaryGoal = Literal[
    "inform",           # prospect needs information, not a pitch
    "introduce",        # first contact — plant the idea, don't close
    "re_engage",        # bring a cold lead back without pressure
    "build_interest",   # warm lead — deepen engagement, move toward decision
    "counter_offer",    # active negotiation — respond to an offer
    "hold_position",    # negotiation — don't move on price yet
    "close",            # buying signals present — move to transaction
    "defuse",           # anger, trust issue, or hard rejection
    "follow_up",        # light touch re-contact after silence
    "final_contact",    # last outreach — graceful, no pressure
    "confirm_next_step", # deal agreed — focus on transaction only
]

BuyerState = Literal[
    "unknown",          # insufficient signal
    "curious",          # asking questions, exploring
    "interested",       # expressed positive intent
    "hesitating",       # interest + stall (budget, timing, approval)
    "anchoring",        # deliberately lowballing to test elasticity
    "negotiating",      # serious back-and-forth
    "buying",           # commitment language present
    "cooling",          # gone quiet after prior engagement
    "cold",             # no prior relationship
    "resistant",        # objecting but not hard-rejecting
    "rejecting",        # explicit refusal or opt-out
]

ConversationPosture = Literal[
    "open",             # introduce, inform, invite
    "engaging",         # build on interest, move forward
    "confident",        # hold position, counter clearly
    "gentle",           # low friction, low pressure
    "closing",          # move toward transaction
    "defusing",         # de-escalate, acknowledge, exit
    "neutral",          # informational, no sales pressure
]

CTAStyle = Literal[
    "none",             # no CTA — informational or defusing
    "soft_question",    # 'Would this be of interest?' — first contact
    "forward_question", # 'Would you like me to send the details?' — warm
    "specific_counter", # 'My counter is $X — does that work for you?'
    "decision_prompt",  # 'Ready to proceed whenever you are.'
    "transaction",      # 'You can purchase directly here: [link]'
    "exit_open_door",   # 'Happy to reconnect if timing changes.'
]

ReplyLength = Literal["short", "medium", "long"]

RepetitionPolicy = Literal[
    "no_repeat",        # do not re-pitch value already stated
    "brief_reminder",   # one sentence value reminder only
    "full_pitch",       # full value case appropriate (first contact)
]


# ── Strategy dataclass ────────────────────────────────────────────────────────

@dataclass
class ReplyStrategy:
    """
    Structured communication decision for a single reply.

    All fields are typed values, not prompt strings. The prompt builder
    translates this object into a concise brief — it does not inject
    these fields as raw text.
    """
    # What this reply must accomplish
    primary_goal:        PrimaryGoal

    # Psychological reading of the prospect at this moment
    buyer_state:         BuyerState

    # How assertive / gentle the reply should be
    conversation_posture: ConversationPosture

    # 0 = none, 1 = light value mention, 2 = moderate case, 3 = full pitch
    persuasion_level:    int

    # 0 = none, 1 = factual only ("domain is publicly listed"), 2 = active
    urgency_level:       int

    # What kind of call-to-action fits this situation
    cta_style:           CTAStyle

    # reply_length recommendation
    reply_length:        ReplyLength

    # Resolved tone description (short phrase, not a full instruction)
    tone_posture:        str

    # One sentence: the single job this reply must do
    reply_objective:     str

    # Things the model must NOT do in this reply
    prohibited_topics:   list[str] = field(default_factory=list)

    # Whether to re-state value propositions
    repetition_policy:   RepetitionPolicy = "brief_reminder"

    # Debug: which signals drove each decision
    reasoning_trace:     dict = field(default_factory=dict)


# ── Signal container ──────────────────────────────────────────────────────────

@dataclass
class StrategySignals:
    """
    All structured inputs that build_strategy() needs.
    Collected in one place so the call site is clean.
    """
    intent:           str
    message:          str
    stage:            str                     # from detect_conversation_stage()
    neg_state:        str                     # from _detect_negotiation_state()
    response_frame:   str                     # from _classify_response_frame()
    tone_requested:   str
    asking_price:     Optional[str]  = None
    outreach_count:   int            = 0
    has_questions:    bool           = False
    question_count:   int            = 0
    ambiguity_level:  str            = "low"  # low / medium / high
    has_multiple_intents: bool       = False
    secondary_intents: list[str]     = field(default_factory=list)
    intent_confidence: float         = 1.0
    email_preset:     Optional[str]  = None
    domain_name:      Optional[str]  = None
    lead_stage:       Optional[str]  = None
    offer_ratio:      Optional[float] = None  # prospect_offer / asking_price


# ── Decision tables ───────────────────────────────────────────────────────────
# These are pure data — not injected as text. build_strategy() reads them.

# intent → primary_goal
_INTENT_GOAL_MAP: dict[str, PrimaryGoal] = {
    "cold_outreach":            "introduce",
    "sales_pitch":              "introduce",
    "warm_outreach":            "build_interest",
    "follow_up":                "follow_up",
    "follow_up_no_response":    "follow_up",
    "follow_up_after_pricing":  "follow_up",
    "follow_up_after_interest": "build_interest",
    "re_engagement":            "re_engage",
    "negotiation":              "counter_offer",
    "price_negotiation":        "counter_offer",
    "price_too_high":           "hold_position",
    "objection_handling":       "build_interest",
    "trust_issue":              "defuse",
    "angry":                    "defuse",
    "no_thanks":                "defuse",
    "have_website":             "build_interest",
    "not_now":                  "follow_up",
    "agreed_no_pay":            "confirm_next_step",
    "price_inquiry":            "inform",
    "how_it_works":             "inform",
    "general":                  "inform",
    "general_response":         "inform",
    "request_info":             "inform",
    "feature_explanation":      "inform",
    "domain_metrics":           "inform",
    "renewal_fees":             "inform",
    "payment_method":           "inform",
    "why_buy":                  "build_interest",
    "rank_well":                "build_interest",
    "expired_owner":            "build_interest",
    "not_interested_ask_why":   "defuse",
    "soft_pitch":               "build_interest",
    "value_reminder":           "build_interest",
    "competitor_comparison":    "build_interest",
}

# stage → primary_goal (overrides intent map when stage has stronger signal)
_STAGE_GOAL_OVERRIDE: dict[str, PrimaryGoal] = {
    "first_outreach":  "introduce",
    "warm_lead":       "build_interest",
    "negotiation":     "counter_offer",
    "counteroffer":    "counter_offer",
    "stalled":         "re_engage",
    "final_follow_up": "final_contact",
    "accepted":        "confirm_next_step",
    "rejected":        "defuse",
}

# neg_state → buyer_state
_NEG_BUYER_MAP: dict[str, BuyerState] = {
    "low_anchor_offer":  "anchoring",
    "active_negotiation":"negotiating",
    "soft_interest":     "interested",
    "hesitation":        "hesitating",
    "hard_rejection":    "rejecting",
    "curiosity":         "curious",
    "urgency_signal":    "interested",
    "none":              "unknown",
}

# primary_goal → conversation_posture
_GOAL_POSTURE_MAP: dict[str, ConversationPosture] = {
    "inform":            "neutral",
    "introduce":         "open",
    "re_engage":         "gentle",
    "build_interest":    "engaging",
    "counter_offer":     "confident",
    "hold_position":     "confident",
    "close":             "closing",
    "defuse":            "defusing",
    "follow_up":         "gentle",
    "final_contact":     "gentle",
    "confirm_next_step": "closing",
}

# primary_goal → CTA style
_GOAL_CTA_MAP: dict[str, CTAStyle] = {
    "inform":            "none",
    "introduce":         "soft_question",
    "re_engage":         "soft_question",
    "build_interest":    "forward_question",
    "counter_offer":     "specific_counter",
    "hold_position":     "specific_counter",
    "close":             "transaction",
    "defuse":            "exit_open_door",
    "follow_up":         "soft_question",
    "final_contact":     "exit_open_door",
    "confirm_next_step": "transaction",
}

# primary_goal → persuasion_level
_GOAL_PERSUASION_MAP: dict[str, int] = {
    "inform":            0,
    "introduce":         2,
    "re_engage":         1,
    "build_interest":    2,
    "counter_offer":     1,
    "hold_position":     1,
    "close":             1,
    "defuse":            0,
    "follow_up":         1,
    "final_contact":     1,
    "confirm_next_step": 0,
}

# primary_goal → urgency_level
_GOAL_URGENCY_MAP: dict[str, int] = {
    "inform":            0,
    "introduce":         0,
    "re_engage":         0,
    "build_interest":    1,
    "counter_offer":     1,
    "hold_position":     1,
    "close":             1,
    "defuse":            0,
    "follow_up":         0,
    "final_contact":     1,
    "confirm_next_step": 1,
}

# primary_goal → reply length
_GOAL_LENGTH_MAP: dict[str, ReplyLength] = {
    "inform":            "medium",
    "introduce":         "short",
    "re_engage":         "short",
    "build_interest":    "medium",
    "counter_offer":     "medium",
    "hold_position":     "medium",
    "close":             "short",
    "defuse":            "short",
    "follow_up":         "short",
    "final_contact":     "short",
    "confirm_next_step": "short",
}

# primary_goal → repetition_policy
_GOAL_REPETITION_MAP: dict[str, RepetitionPolicy] = {
    "inform":            "no_repeat",
    "introduce":         "full_pitch",
    "re_engage":         "brief_reminder",
    "build_interest":    "brief_reminder",
    "counter_offer":     "no_repeat",
    "hold_position":     "no_repeat",
    "close":             "no_repeat",
    "defuse":            "no_repeat",
    "follow_up":         "brief_reminder",
    "final_contact":     "brief_reminder",
    "confirm_next_step": "no_repeat",
}


# ── Prohibition rules ─────────────────────────────────────────────────────────
# Each entry: (condition_fn, prohibited_items)
# Evaluated against the resolved strategy fields — not raw signals.

def _compute_prohibitions(
    goal:      PrimaryGoal,
    posture:   ConversationPosture,
    buyer:     BuyerState,
    urgency:   int,
    outreach:  int,
    has_questions: bool,
) -> list[str]:
    """
    Derive the list of things this reply must NOT contain.
    Each prohibition is a short instruction phrase for the prompt brief.
    """
    p: list[str] = []

    # Universal prohibitions — always apply
    p.append("no manufactured urgency or fake deadlines")
    p.append("no hype phrases (perfect domain, once in a lifetime, game-changing)")
    p.append("no AI-sounding openers (I hope this email finds you, I am writing to)")

    # Goal-specific
    if goal == "defuse":
        p.append("no pitch content of any kind")
        p.append("no defending your position or price")
        p.append("no re-engagement attempt")

    if goal == "confirm_next_step":
        p.append("do not re-sell — they have already decided")
        p.append("no value propositions")

    if goal in ("follow_up", "re_engage", "final_contact"):
        p.append("do not re-pitch the full value proposition")
        p.append("no guilt language or pressure")
        if outreach >= 2:
            p.append("do not repeat content from previous messages")

    if goal == "counter_offer":
        p.append("do not accept or approach their offer without countering")
        p.append("do not give a price range — use a specific figure")
        p.append("do not over-explain the price justification")

    if goal == "hold_position":
        p.append("do not lower the price or imply flexibility")
        p.append("do not apologise for the price")

    # Buyer-state specific
    if buyer == "buying":
        p.append("do not re-sell — buying signals are present, move to close")

    if buyer == "rejecting":
        p.append("no counter-pitch after rejection")
        p.append("two sentences maximum")

    if buyer == "anchoring":
        p.append("do not acknowledge the low offer as reasonable")
        p.append("do not meet them in the middle without justification")

    if buyer == "hesitating":
        p.append("no hard close — they need a low-friction next step")

    # Urgency gate — only factual urgency allowed unless close
    if urgency < 2:
        p.append("no active urgency language unless factual (domain is publicly listed)")

    # Question gate
    if has_questions:
        p.append("answer their question(s) before any pitch content")

    return p


# ── Tone resolution ───────────────────────────────────────────────────────────

def _resolve_tone_posture(
    tone_requested: str,
    goal:           PrimaryGoal,
    buyer:          BuyerState,
    posture:        ConversationPosture,
) -> str:
    """
    Resolve a short tone description from requested tone + strategic context.
    Returns a phrase, not a paragraph. The prompt brief uses this as a single
    line: 'Tone: {tone_posture}'.

    Conflict resolution:
    - defusing always overrides requested tone
    - rejecting/anchoring buyer → confident overrides warm
    - buying buyer → neutral/closing overrides persuasive
    """
    # Hard overrides based on situation
    if goal == "defuse" or buyer == "rejecting":
        return "brief and respectful"

    if posture == "closing" or buyer == "buying":
        return "direct and transaction-focused"

    if buyer == "anchoring":
        return "calm and confident — hold position without defensiveness"

    if posture == "confident":
        return "confident and specific — no hedging"

    if posture == "gentle":
        return "light and low-pressure — easy to reply to"

    if posture == "neutral":
        return "clear and informative — no sales pressure"

    # Fall through to requested tone (abbreviated)
    tone_map = {
        "professional and persuasive":  "professional and value-led",
        "warm and friendly":            "warm and conversational",
        "firm but respectful":          "firm and clear",
        "concise and direct":           "concise — every sentence earns its place",
        "empathetic and understanding": "empathetic — acknowledge before pitching",
        "highly persuasive and compelling": "persuasive with concrete specifics",
        "urgent and time-sensitive":    "factual urgency — information, not pressure",
        "premium and exclusive":        "confident and premium-positioned",
    }
    return tone_map.get(tone_requested, tone_requested)


# ── Objective builder ─────────────────────────────────────────────────────────

def _build_reply_objective(
    goal:        PrimaryGoal,
    buyer:       BuyerState,
    stage:       str,
    domain_name: Optional[str],
) -> str:
    """
    Build the single-sentence reply objective.
    This is the most important output of the strategy layer — it tells the
    model what success looks like for this specific reply.
    """
    domain = f" for {domain_name}" if domain_name else ""

    objectives: dict[str, str] = {
        "inform":            f"Answer their question directly and factually{domain}. No pitch unless it fits naturally.",
        "introduce":         f"Plant genuine interest{domain} without pressuring. One relevant value point. One easy question.",
        "re_engage":         f"Restart the conversation{domain} with minimal friction. Short, low-pressure, easy to reply to.",
        "build_interest":    f"Deepen their interest{domain} and move toward a decision. Reinforce one strong value point. Invite the next step.",
        "counter_offer":     f"Acknowledge their offer briefly, counter with a specific figure{domain}, one reason, one clear next step.",
        "hold_position":     f"Hold the price position{domain} with confidence. One brief justification. Keep the door open.",
        "close":             f"Move this toward a transaction{domain}. Buying signals are present — respond to them, don't re-pitch.",
        "defuse":            f"De-escalate or accept the decision gracefully{domain}. Brief, professional, no pitch.",
        "follow_up":         f"Re-engage gently{domain} after silence. Remind without repeating. Make replying feel easy.",
        "final_contact":     f"Close this thread professionally{domain}. One value reminder, graceful exit, door left open.",
        "confirm_next_step": f"Move the agreed deal{domain} to completion. Focus entirely on the next action.",
    }
    return objectives.get(goal, f"Write a focused, relevant reply{domain}.")


# ── Main entry point ──────────────────────────────────────────────────────────

def build_strategy(signals: StrategySignals) -> ReplyStrategy:
    """
    Consume structured signals and produce a typed ReplyStrategy.

    Decision priority (highest wins):
    1. Explicit lead stage from broker_memory
    2. Negotiation state (neg_state) — highest signal fidelity
    3. Conversation stage (stage) — from detect_conversation_stage()
    4. Intent — from analyse()
    5. Response frame — as tiebreaker for goal type

    Conflict resolution:
    - When goal and posture conflict, posture becomes more conservative
    - Urgency is capped by buyer_state (never urgent with rejecting buyer)
    - Persuasion is zeroed for defuse/inform goals regardless of intent
    """
    trace: dict = {}

    # ── 1. Resolve primary_goal ───────────────────────────────────────────────
    # Check stage override first (explicit broker_memory stage has highest trust)
    if signals.stage in _STAGE_GOAL_OVERRIDE and signals.stage != "unknown":
        goal: PrimaryGoal = _STAGE_GOAL_OVERRIDE[signals.stage]
        trace["goal_source"] = f"stage:{signals.stage}"
    elif signals.intent in _INTENT_GOAL_MAP:
        goal = _INTENT_GOAL_MAP[signals.intent]
        trace["goal_source"] = f"intent:{signals.intent}"
    else:
        # Response frame as fallback
        frame_goal_map: dict[str, PrimaryGoal] = {
            "strategic_advice":    "build_interest",
            "educational_answer":  "inform",
            "negotiation_analysis":"counter_offer",
            "brainstorming":       "build_interest",
            "direct_reply":        "build_interest",
            "inferred_reply":      "introduce",
            "mixed_request":       "inform",
        }
        goal = frame_goal_map.get(signals.response_frame, "build_interest")
        trace["goal_source"] = f"frame:{signals.response_frame}"

    # ── 2. Resolve buyer_state ────────────────────────────────────────────────
    buyer: BuyerState = _NEG_BUYER_MAP.get(signals.neg_state, "unknown")
    trace["buyer_source"] = f"neg_state:{signals.neg_state}"

    # Refine buyer from stage if neg_state gave no signal
    if buyer == "unknown":
        stage_buyer_map: dict[str, BuyerState] = {
            "first_outreach":  "cold",
            "warm_lead":       "interested",
            "negotiation":     "negotiating",
            "counteroffer":    "negotiating",
            "stalled":         "cooling",
            "final_follow_up": "cooling",
            "accepted":        "buying",
            "rejected":        "rejecting",
        }
        buyer = stage_buyer_map.get(signals.stage, "unknown")
        trace["buyer_source"] = f"stage:{signals.stage}"

    # ── 3. Conflict resolution: goal vs buyer ─────────────────────────────────
    # Hard rejection overrides everything — goal becomes defuse
    if buyer == "rejecting" and goal not in ("defuse", "inform"):
        goal = "defuse"
        trace["goal_override"] = "buyer=rejecting forced goal=defuse"

    # Buying signal overrides to close if not already defusing/informing
    if buyer == "buying" and goal in ("build_interest", "counter_offer", "follow_up"):
        goal = "close"
        trace["goal_override"] = "buyer=buying upgraded goal=close"

    # ── 4. Resolve conversation_posture ──────────────────────────────────────
    posture: ConversationPosture = _GOAL_POSTURE_MAP.get(goal, "engaging")

    # Soften posture for high ambiguity — we're less certain what they need
    if signals.ambiguity_level == "high" and posture == "confident":
        posture = "engaging"
        trace["posture_softened"] = "ambiguity=high"

    # ── 5. Resolve persuasion_level ───────────────────────────────────────────
    persuasion = _GOAL_PERSUASION_MAP.get(goal, 1)

    # Outreach count caps persuasion — more attempts = less pitch
    if signals.outreach_count >= 3:
        persuasion = min(persuasion, 1)
        trace["persuasion_capped"] = f"outreach_count={signals.outreach_count}"

    # Info intents and defuse always zero
    if goal in ("inform", "defuse", "confirm_next_step"):
        persuasion = 0

    # ── 6. Resolve urgency_level ──────────────────────────────────────────────
    urgency = _GOAL_URGENCY_MAP.get(goal, 0)

    # Never urgent with rejecting or cooling buyer
    if buyer in ("rejecting", "cooling"):
        urgency = 0
        trace["urgency_zeroed"] = f"buyer={buyer}"

    # Never urgent on first contact
    if goal == "introduce":
        urgency = 0

    # ── 7. Resolve CTA style ──────────────────────────────────────────────────
    cta: CTAStyle = _GOAL_CTA_MAP.get(goal, "soft_question")

    # Override: if they asked a question, no CTA until after the answer
    if signals.has_questions and goal not in ("counter_offer", "close", "confirm_next_step"):
        cta = "soft_question"
        trace["cta_softened"] = "has_questions=True"

    # Buying signal → transaction CTA
    if buyer == "buying":
        cta = "transaction"

    # ── 8. Resolve reply_length ───────────────────────────────────────────────
    length: ReplyLength = _GOAL_LENGTH_MAP.get(goal, "medium")

    # Multiple questions → at least medium (need space to answer)
    if signals.question_count >= 2:
        length = "medium" if length == "short" else length

    # High ambiguity → shorter (less risk of addressing wrong thing)
    if signals.ambiguity_level == "high" and length == "long":
        length = "medium"

    # ── 9. Resolve repetition_policy ─────────────────────────────────────────
    repetition: RepetitionPolicy = _GOAL_REPETITION_MAP.get(goal, "brief_reminder")

    # Already pitched multiple times → no repeat
    if signals.outreach_count >= 2 and repetition == "full_pitch":
        repetition = "brief_reminder"
        trace["repetition_reduced"] = f"outreach_count={signals.outreach_count}"

    # ── 10. Resolve tone_posture ──────────────────────────────────────────────
    tone_posture = _resolve_tone_posture(signals.tone_requested, goal, buyer, posture)

    # ── 11. Build prohibitions ────────────────────────────────────────────────
    prohibited = _compute_prohibitions(
        goal           = goal,
        posture        = posture,
        buyer          = buyer,
        urgency        = urgency,
        outreach       = signals.outreach_count,
        has_questions  = signals.has_questions,
    )

    # ── 12. Build reply objective ─────────────────────────────────────────────
    objective = _build_reply_objective(goal, buyer, signals.stage, signals.domain_name)

    trace.update({
        "final_goal":    goal,
        "final_buyer":   buyer,
        "final_posture": posture,
        "persuasion":    persuasion,
        "urgency":       urgency,
        "cta":           cta,
        "length":        length,
    })

    return ReplyStrategy(
        primary_goal          = goal,
        buyer_state           = buyer,
        conversation_posture  = posture,
        persuasion_level      = persuasion,
        urgency_level         = urgency,
        cta_style             = cta,
        reply_length          = length,
        tone_posture          = tone_posture,
        reply_objective       = objective,
        prohibited_topics     = prohibited,
        repetition_policy     = repetition,
        reasoning_trace       = trace,
    )


# ── Prompt brief builder ──────────────────────────────────────────────────────

def build_prompt_brief(strategy: ReplyStrategy, context_line: str = "") -> str:
    """
    Translate a ReplyStrategy into a compact prompt brief.
    This replaces the multi-section instruction wall currently assembled
    inside build_reply_prompt_ai().

    Target: under 300 tokens. Every line earns its place.
    """
    lines: list[str] = []

    if context_line:
        lines.append(context_line)

    lines.append(f"OBJECTIVE: {strategy.reply_objective}")
    lines.append(f"Tone: {strategy.tone_posture}")

    # Length
    length_map = {
        "short":  "Keep it short — 2 paragraphs maximum.",
        "medium": "Medium length — 2-3 focused paragraphs.",
        "long":   "Full response — answer completely, but do not pad.",
    }
    lines.append(length_map[strategy.reply_length])

    # CTA
    cta_map: dict[str, str] = {
        "none":             "No call to action.",
        "soft_question":    "End with one soft, easy question.",
        "forward_question": "End with a direct but low-pressure forward question.",
        "specific_counter": "End with your specific counter-price and one clear next-step question.",
        "decision_prompt":  "End with a direct decision prompt.",
        "transaction":      "End with the purchase link or payment step — no re-selling.",
        "exit_open_door":   "End by leaving the door open gracefully. No pressure.",
    }
    if strategy.cta_style in cta_map:
        lines.append(f"CTA: {cta_map[strategy.cta_style]}")

    # Persuasion
    if strategy.persuasion_level == 0:
        lines.append("No sales content — this reply is not a pitch.")
    elif strategy.persuasion_level == 1:
        lines.append("One value point only — no full pitch.")
    elif strategy.persuasion_level == 2:
        lines.append("Make a clear, relevant case — lead with the strongest benefit.")
    elif strategy.persuasion_level == 3:
        lines.append("Full value case — lead with the most relevant outcome for their business.")

    # Urgency
    if strategy.urgency_level == 0:
        lines.append("No urgency.")
    elif strategy.urgency_level == 1:
        lines.append("Urgency only if factual: domain is publicly listed.")

    # Repetition
    if strategy.repetition_policy == "no_repeat":
        lines.append("Do not repeat value points already made.")
    elif strategy.repetition_policy == "brief_reminder":
        lines.append("One brief value reminder — do not re-pitch from scratch.")

    # Prohibitions (top 4 most important only — avoid over-constraining)
    key_prohibitions = [
        p for p in strategy.prohibited_topics
        if not p.startswith("no manufactured")   # universal, model already knows
        and not p.startswith("no hype phrases")   # handled by humanizer
    ][:4]
    if key_prohibitions:
        lines.append("Do not: " + "; ".join(key_prohibitions) + ".")

    lines.append("\nWrite only the email body. No subject line. No commentary.")
    lines.append("Write the reply:")

    return "\n".join(lines)
