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
    # ── Required fields (no defaults) ────────────────────────────────────────
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

    # ── Optional / defaulted fields ───────────────────────────────────────────
    # Specific writing behaviours derived from tone + intent rules
    tone_guidance:       list[str] = field(default_factory=list)

    # What NEW ground this reply should cover vs prior outreach
    progression_goal:    str = ""

    # Value topics already covered in prior outreach — suppress re-pitching these
    suppressed_topics:   list[str] = field(default_factory=list)

    # Things the model must NOT do in this reply
    prohibited_topics:   list[str] = field(default_factory=list)

    # Whether to re-state value propositions
    repetition_policy:   RepetitionPolicy = "brief_reminder"

    # No grounding facts available — suppress any invented specifics
    no_domain_no_price:  bool            = False

    # Confidence in each major decision (0.0 – 1.0)
    stage_confidence:    float = 1.0
    buyer_confidence:    float = 1.0
    goal_confidence:     float = 1.0

    # Debug: which signals drove each decision
    reasoning_trace:     dict = field(default_factory=dict)

    # ── Phase 2 fields — Adaptive Memory + Angle Selection ────────────────────
    # All defaulted — existing call sites require zero changes.
    # When angle data is unavailable (no lead_id, DB offline, Phase 1 only),
    # these stay empty and build_prompt_brief() behaves identically to before.

    # The specific angle the reply should lead with (AngleId string or "")
    selected_angle:        str       = ""

    # Angles not yet used with this lead — informs progression note
    available_angles:      list[str] = field(default_factory=list)

    # Angles used >= exhaustion_threshold times — supersedes keyword suppression
    exhausted_angles:      list[str] = field(default_factory=list)

    # Objection type labels still unresolved for this lead
    unresolved_objections: list[str] = field(default_factory=list)


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
    no_domain_no_price: bool         = False  # True when neither domain_name nor asking_price known
    # Prior outreach email bodies — used for repetition suppression
    # Pass memory_db.get_outreach_history(lead_id) here when available
    prior_outreach_bodies: list[str] = field(default_factory=list)
    # How confident was detect_conversation_stage() in its stage label?
    # Pass the string explanation; strategy layer infers confidence from it
    stage_signal_strength: str       = "intent"  # "memory" | "offer" | "message" | "intent" | "count"

    # ── Phase 2 fields — Adaptive Memory + Angle Selection ────────────────────
    # All Optional/defaulted — existing call sites require zero changes.
    # When absent, build_strategy() falls back to keyword-scan suppression
    # exactly as before. No behaviour change when these are not supplied.

    # lead_id for the current lead — for trace/logging context
    lead_id:              Optional[int]    = None

    # Pre-built AngleInventory for this lead (from build_angle_inventory()).
    # When present, exhausted_angles supersedes prior_outreach_bodies keyword scan.
    # Type is Any to avoid hard import at module level — degrades gracefully.
    angle_inventory:      Optional[object] = None

    # Unresolved ObjectionRecord list for this lead.
    # Built by caller from memory_db.get_objection_history(lead_id, unresolved_only=True).
    unresolved_objection_records: list     = field(default_factory=list)


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


# ── Repetition suppression ─────────────────────────────────────────────────────

# Value topic labels and the phrases that indicate they've been mentioned
_VALUE_TOPIC_SIGNALS: dict[str, list[str]] = {
    "seo_benefit":       ["seo", "search ranking", "search engine", "rank", "google",
                          "organic", "local search", "exact match", "exact-match"],
    "traffic_benefit":   ["traffic", "visitors", "clicks", "searches", "targeted traffic",
                          "capture traffic", "drive traffic"],
    "brand_protection":  ["brand", "competitor", "protect", "snapping it up", "shield",
                          "before a competitor", "protect your brand", "brand asset"],
    "domain_forwarding": ["forward", "forwarding", "redirect", "point to your site",
                          "current site", "existing site"],
    "price_anchor":      ["asking price", "listed at", "priced at", "available for",
                          "only asking", "just asking", "our price"],
    "scarcity":          ["publicly listed", "available to anyone", "first to",
                          "won't be available", "act before"],
    "local_relevance":   ["local", "city", "toronto", "london", "chicago", "geographic",
                          "location", "near", "neighbourhood", "neighborhood"],
    "credibility_trust": ["escrow", "trusted", "secure transfer", "guarantee", "protection",
                          "safe", "legitimate", "verified"],
    "payment_options":   ["credit card", "escrow", "paypal", "payment plan", "instalment",
                          "installment", "payment method"],
}


def _extract_mentioned_topics(prior_bodies: list[str]) -> list[str]:
    """
    Scan prior outreach email bodies and return topic labels that have
    already been mentioned. Zero model calls — pure keyword matching.

    Returns a list of topic labels from _VALUE_TOPIC_SIGNALS.
    """
    if not prior_bodies:
        return []

    combined = " ".join(prior_bodies).lower()
    mentioned: list[str] = []
    for topic, signals in _VALUE_TOPIC_SIGNALS.items():
        if any(signal in combined for signal in signals):
            mentioned.append(topic)
    return mentioned


# ── Conversation progression logic ────────────────────────────────────────────

# What a reply should newly accomplish at each stage, given prior coverage
_PROGRESSION_LOGIC: dict[str, dict] = {
    "introduce": {
        "default":    "introduce the domain and plant genuine interest",
        "with_seo":   "introduce the domain — seo angle already covered, try a different hook",
    },
    "build_interest": {
        "default":    "deepen interest with a specific, relevant value point not yet mentioned",
        "many_covered": "move toward decision — value has been explained; reduce friction instead",
    },
    "follow_up":  {
        "default":    "re-engage with minimal friction — do not re-pitch what was already said",
        "first_fu":   "follow up on prior interest — acknowledge the previous message briefly",
    },
    "re_engage":  {
        "default":    "restart dialogue with a light touch — one new angle or a simple check-in",
    },
    "counter_offer": {
        "default":    "respond to their offer — acknowledge briefly, counter with a specific figure",
    },
    "hold_position": {
        "default":    "hold the price position clearly — one brief reason, keep the door open",
    },
    "close":      {
        "default":    "move toward transaction — buying signals present, respond not re-pitch",
    },
    "defuse":     {
        "default":    "acknowledge and close gracefully — no sales content",
    },
    "final_contact": {
        "default":    "last contact — brief value reminder, respectful exit, door left open",
    },
    "confirm_next_step": {
        "default":    "confirm next action — payment, transfer, or escrow — nothing else",
    },
    "inform":     {
        "default":    "answer their question directly and factually",
    },
}


def _build_progression_goal(
    goal:              PrimaryGoal,
    suppressed_topics: list[str],
    outreach_count:    int,
) -> str:
    """
    Determine what NEW ground this reply should cover.
    Returns a short instruction phrase used in the prompt brief.
    """
    logic = _PROGRESSION_LOGIC.get(goal, {})

    if goal == "introduce":
        if "seo_benefit" in suppressed_topics:
            return logic.get("with_seo", logic["default"])
        return logic["default"]

    if goal == "build_interest":
        # If most value points are exhausted, shift to friction reduction
        if len(suppressed_topics) >= 3:
            return logic.get("many_covered", logic["default"])
        return logic["default"]

    if goal == "follow_up":
        if outreach_count == 1:
            return logic.get("first_fu", logic["default"])
        return logic["default"]

    return logic.get("default", f"advance the conversation appropriately for this stage")


# ── Confidence scoring ─────────────────────────────────────────────────────────

def _compute_confidence(
    stage_signal_strength: str,
    neg_state:             str,
    intent:                str,
    intent_confidence:     float,
    ambiguity_level:       str,
    goal_source:           str,
) -> tuple[float, float, float]:
    """
    Compute (stage_confidence, buyer_confidence, goal_confidence).

    Signal strength ranking:
    - "memory" (from broker_memory stored stage) → 0.95
    - "offer"  (from offer log)                  → 0.90
    - "message" (from message keyword signals)   → 0.80
    - "intent"  (from intent classifier)         → 0.70
    - "count"   (from outreach count heuristic)  → 0.55
    - "unknown"                                  → 0.40
    """
    stage_conf_map = {
        "memory":  0.95,
        "offer":   0.90,
        "message": 0.80,
        "intent":  0.70,
        "count":   0.55,
        "unknown": 0.40,
    }
    stage_conf = stage_conf_map.get(stage_signal_strength, 0.65)

    # Buyer confidence: strong if neg_state has a clear signal
    clear_neg_states = {"low_anchor_offer", "active_negotiation", "hard_rejection",
                        "urgency_signal", "soft_interest"}
    if neg_state in clear_neg_states:
        buyer_conf = 0.90
    elif ambiguity_level == "high":
        buyer_conf = 0.50
    elif ambiguity_level == "medium":
        buyer_conf = 0.70
    else:
        buyer_conf = 0.80

    # Goal confidence: driven by goal_source and intent confidence
    if goal_source.startswith("email_preset:"):
        goal_conf = 0.95   # explicit broker choice
    elif goal_source.startswith("memory:") or goal_source.startswith("offer:"):
        goal_conf = 0.90
    elif goal_source.startswith("message:"):
        goal_conf = 0.80
    else:
        goal_conf = min(intent_confidence, 0.85)

    return round(stage_conf, 2), round(buyer_conf, 2), round(goal_conf, 2)


# ── Tone guidance extractor ───────────────────────────────────────────────────
# Pulls specific writing behaviours from TONE_INSTRUCTIONS and INTENT_RULES
# without injecting those tables wholesale into the prompt.

# Condensed behavioural extracts — what each tone actually requires the writer to DO
# Derived from TONE_INSTRUCTIONS but reduced to action phrases only
_TONE_BEHAVIOURS: dict[str, list[str]] = {
    "professional and persuasive":      ["state position clearly — no hedging",
                                          "open with the strongest benefit for their situation",
                                          "avoid weak qualifiers like 'might' or 'could possibly'"],
    "warm and friendly":                ["use short sentences",
                                          "ask one genuine question",
                                          "avoid anything that sounds like a sales script"],
    "firm but respectful":              ["state your number once, cleanly, without apology",
                                          "acknowledge their point in one sentence then restate position",
                                          "avoid caving or over-explaining"],
    "concise and direct":               ["2-3 sentences maximum",
                                          "state the point, the ask, the close — in that order"],
    "empathetic and understanding":     ["name their specific objection before addressing it",
                                          "do not pivot to pitch before they feel heard"],
    "highly persuasive and compelling": ["lead with the most compelling outcome for their business",
                                          "one specific outcome beats three vague benefits"],
    "urgent and time-sensitive":        ["state urgency as fact, not pressure",
                                          "one factual reason why timing matters"],
    "premium and exclusive":            ["project confidence in the price",
                                          "never apologise for or justify the asking price"],
}

# Intent-specific writing behaviours — what the INTENT_RULES carry that's actionable
# Reduced to 1-2 directive phrases per intent rather than full paragraphs
_INTENT_BEHAVIOURS: dict[str, list[str]] = {
    "follow_up":                ["acknowledge prior contact in one sentence",
                                  "do not repeat the full pitch from the first email"],
    "follow_up_no_response":    ["assume they missed the email — no guilt language",
                                  "one new hook or angle, not a re-send"],
    "price_too_high":           ["acknowledge their concern — one sentence",
                                  "hold the price or counter; do not apologise for it"],
    "objection_handling":       ["name their objection specifically before reframing it",
                                  "one clear reframe — do not list multiple counter-arguments"],
    "negotiation":              ["counter with a specific number, not a range",
                                  "one brief justification — no price defence essay"],
    "re_engagement":            ["light and low-friction — they've gone quiet",
                                  "short message, easy to reply to, no pressure"],
    "agreed_no_pay":            ["they agreed — focus only on payment or next step",
                                  "no re-selling of any kind"],
    "not_interested_ask_why":   ["accept the no gracefully — one sentence",
                                  "soft question only if it feels natural, otherwise close the thread"],
    "cold_outreach":            ["curiosity-first — one specific, relevant benefit",
                                  "end with a low-pressure question, not a call to buy"],
    "trust_issue":              ["address the concern directly — do not deflect",
                                  "one credibility signal (escrow, transfer process, etc.)"],
    "have_website":             ["acknowledge they have a site — they know",
                                  "reframe: forwarding / brand protection / SEO shield"],
}


def _extract_tone_guidance(
    tone_requested: str,
    intent:         str,
    goal:           PrimaryGoal,
    confidence:     float,
) -> list[str]:
    """
    Return 2-4 specific writing behaviour phrases derived from tone + intent rules.
    These replace wholesale TONE_INSTRUCTIONS injection — only the actionable
    directives are extracted, not the full paragraph.

    Low confidence → return only neutral guidance to avoid wrong assumptions.
    """
    guidance: list[str] = []

    if confidence < 0.55:
        return ["match the tone of their message", "keep it neutral and professional"]

    # Tone behaviours — max 2 to keep brief tight
    tone_behaviours = _TONE_BEHAVIOURS.get(tone_requested, [])
    guidance.extend(tone_behaviours[:2])

    # Intent behaviours — max 2
    intent_behaviours = _INTENT_BEHAVIOURS.get(intent, [])
    guidance.extend(intent_behaviours[:2])

    # Goal-level overrides for certain high-stakes situations
    if goal == "defuse":
        guidance = ["brief and respectful — two to three sentences maximum",
                    "no sales content of any kind"]
    elif goal == "confirm_next_step":
        guidance = ["focus entirely on the next action (payment / transfer)",
                    "do not re-sell"]

    return guidance[:4]  # hard cap — never more than 4 behaviour lines


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
    no_domain_no_price: bool = False,
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

    # No-invent gate — when there are no grounding facts available
    if no_domain_no_price:
        p.append("do not invent domain names, prices, traffic stats, or registration dates")

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
    1. email_preset — explicit broker intent, highest trust when set
    2. Explicit lead stage from broker_memory
    3. Negotiation state (neg_state) — highest signal fidelity for current message
    4. Conversation stage (stage) — from detect_conversation_stage()
    5. Intent — from analyse()
    6. Response frame — as tiebreaker for goal type

    Conflict resolution rules:
    - rejecting buyer always forces defuse goal
    - buying buyer upgrades to close (except defuse/inform)
    - hesitating buyer softens counter_offer → hold_position
    - email_preset overrides stage goal when preset is more specific
    - Urgency is capped by buyer_state (never urgent with rejecting/cooling buyer)
    - Persuasion is zeroed for defuse/inform goals regardless of intent
    """
    trace: dict = {}

    # ── 1. Resolve primary_goal ───────────────────────────────────────────────
    # email_preset has highest trust when explicitly set by the broker
    _PRESET_GOAL_MAP: dict[str, PrimaryGoal] = {
        "cold_outreach":    "introduce",
        "warm_outreach":    "build_interest",
        "follow_up":        "follow_up",
        "final_follow_up":  "final_contact",
        "negotiation":      "counter_offer",
        "counter_offer":    "counter_offer",
        "final_offer":      "hold_position",
        "closing":          "close",
        "payment_reminder": "confirm_next_step",
        "general":          "build_interest",
    }
    if signals.email_preset and signals.email_preset in _PRESET_GOAL_MAP:
        goal: PrimaryGoal = _PRESET_GOAL_MAP[signals.email_preset]
        trace["goal_source"] = f"email_preset:{signals.email_preset}"
    elif signals.stage in _STAGE_GOAL_OVERRIDE and signals.stage != "unknown":
        goal = _STAGE_GOAL_OVERRIDE[signals.stage]
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

    # Hesitating buyer: counter_offer is too aggressive — soften to hold_position
    # (they're not ready to be pushed into a specific number)
    if buyer == "hesitating" and goal == "counter_offer":
        goal = "hold_position"
        trace["goal_override"] = "buyer=hesitating softened counter_offer→hold_position"

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
        goal               = goal,
        posture            = posture,
        buyer              = buyer,
        urgency            = urgency,
        outreach           = signals.outreach_count,
        has_questions      = signals.has_questions,
        no_domain_no_price = signals.no_domain_no_price,
    )

    # ── 12. Build reply objective ─────────────────────────────────────────────
    objective = _build_reply_objective(goal, buyer, signals.stage, signals.domain_name)

    # ── 13. Confidence scoring ────────────────────────────────────────────────
    stage_conf, buyer_conf, goal_conf = _compute_confidence(
        stage_signal_strength = signals.stage_signal_strength,
        neg_state             = signals.neg_state,
        intent                = signals.intent,
        intent_confidence     = signals.intent_confidence,
        ambiguity_level       = signals.ambiguity_level,
        goal_source           = trace.get("goal_source", "intent"),
    )

    # Low overall confidence → cap aggressiveness
    overall_conf = min(stage_conf, buyer_conf, goal_conf)
    if overall_conf < 0.55:
        persuasion = min(persuasion, 1)
        urgency    = 0
        if posture == "confident":
            posture = "engaging"
        trace["confidence_dampened"] = f"overall={overall_conf:.2f}"

    # ── 14. Repetition suppression + Phase 2 angle selection ────────────────
    # Phase 2 path: AngleInventory supplied — use structured exhaustion data.
    # Phase 1 fallback: keyword scan of prior_outreach_bodies (unchanged).
    # The two paths produce identical suppressed_topics output format so all
    # downstream code (prohibitions, progression) works without modification.

    angle_sel      = None    # AngleSelection | None
    _available     : list[str] = []
    _exhausted_p2  : list[str] = []
    _unresolved_obj: list[str] = []

    if signals.angle_inventory is not None:
        # ── Phase 2 path ──────────────────────────────────────────────────────
        try:
            from angle_memory import _select_next_angle, ObjectionRecord
            inv = signals.angle_inventory
            angle_sel    = _select_next_angle(
                inventory             = inv,
                goal                  = goal,
                stage                 = signals.stage,
                unresolved_objections = signals.unresolved_objection_records,
            )
            _available    = list(inv.available_angles)
            _exhausted_p2 = list(inv.exhausted_angles)
            _unresolved_obj = [
                r.objection_type for r in signals.unresolved_objection_records
            ]
            # Use exhausted_angles as suppressed_topics — superset of keyword scan
            suppressed = list(inv.exhausted_angles)
            trace["angle_selection"] = {
                "selected":        angle_sel.selected_angle if angle_sel else None,
                "reason":          angle_sel.selection_reason if angle_sel else None,
                "confidence":      angle_sel.confidence if angle_sel else None,
                "available_count": len(_available),
                "exhausted_count": len(_exhausted_p2),
                "objections":      _unresolved_obj,
                "source":          "angle_inventory",
            }
        except Exception as _ae:
            # Degrade silently to Phase 1 path — never block reply generation
            print(f"[ReplyStrategy] angle selection error (non-blocking): {_ae}")
            suppressed = _extract_mentioned_topics(signals.prior_outreach_bodies)
            trace["angle_selection_error"] = str(_ae)
    else:
        # ── Phase 1 path (unchanged) ──────────────────────────────────────────
        suppressed = _extract_mentioned_topics(signals.prior_outreach_bodies)

    if suppressed:
        trace["suppressed_topics"] = suppressed
        # Add suppressed topics to prohibitions — identical to Phase 1 behaviour
        topic_labels = {
            "seo_benefit":       "SEO / search ranking (already covered)",
            "traffic_benefit":   "traffic benefits (already covered)",
            "brand_protection":  "brand protection angle (already covered)",
            "domain_forwarding": "domain forwarding explanation (already covered)",
            "scarcity":          "scarcity / availability urgency (already used)",
            "local_relevance":   "local relevance pitch (already covered)",
            "credibility_trust": "escrow / trust reassurance (already covered)",
        }
        for t in suppressed:
            label = topic_labels.get(t)
            if label:
                prohibited.append(f"do not re-explain {label}")

    # ── 15. Progression goal ──────────────────────────────────────────────────
    # Phase 2: when an AngleSelection was produced, its progression_note
    # replaces the generic _PROGRESSION_LOGIC string.
    # Phase 1: _build_progression_goal() runs unchanged.
    if angle_sel and angle_sel.progression_note:
        progression = angle_sel.progression_note
    else:
        progression = _build_progression_goal(goal, suppressed, signals.outreach_count)

    # ── 16. Tone guidance — extract behavioural directives from rule tables ───
    tone_guidance = _extract_tone_guidance(
        tone_requested = signals.tone_requested,
        intent         = signals.intent,
        goal           = goal,
        confidence     = overall_conf,
    )

    trace.update({
        "final_goal":    goal,
        "final_buyer":   buyer,
        "final_posture": posture,
        "persuasion":    persuasion,
        "urgency":       urgency,
        "cta":           cta,
        "length":        length,
        "stage_conf":    stage_conf,
        "buyer_conf":    buyer_conf,
        "goal_conf":     goal_conf,
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
        tone_guidance         = tone_guidance,
        reply_objective       = objective,
        progression_goal      = progression,
        suppressed_topics     = suppressed,
        prohibited_topics     = prohibited,
        repetition_policy     = repetition,
        no_domain_no_price    = signals.no_domain_no_price,
        stage_confidence      = stage_conf,
        buyer_confidence      = buyer_conf,
        goal_confidence       = goal_conf,
        reasoning_trace       = trace,
        # Phase 2 fields — empty when no inventory supplied
        selected_angle        = angle_sel.selected_angle if angle_sel else "",
        available_angles      = _available,
        exhausted_angles      = _exhausted_p2,
        unresolved_objections = _unresolved_obj,
    )


# ── Prompt brief builder ──────────────────────────────────────────────────────

def build_prompt_brief(
    strategy:     ReplyStrategy,
    context_line: str  = "",
    has_questions: bool = False,
    question_count: int = 0,
    no_domain_no_price: bool = False,
) -> str:
    """
    Translate a ReplyStrategy into a compact prompt brief.
    Target: under 250 tokens. Every line earns its place.

    Parameters
    ----------
    strategy          : resolved ReplyStrategy object
    context_line      : "Domain: X  Asking price: Y" header line (or empty)
    has_questions     : whether the prospect asked questions (from InputAnalysis)
    question_count    : number of distinct questions detected
    no_domain_no_price: True when neither domain_name nor asking_price is known
    """
    lines: list[str] = []

    if context_line:
        lines.append(context_line)

    # Objective — the single most important line
    lines.append(f"OBJECTIVE: {strategy.reply_objective}")

    # Phase 2: selected angle — positive instruction replaces generic hint.
    # When selected_angle is empty (Phase 1 / no inventory) this block is
    # skipped and behaviour is identical to before.
    if strategy.selected_angle:
        try:
            from angle_memory import _ANGLE_REGISTRY
            _entry = _ANGLE_REGISTRY.get(strategy.selected_angle, {})
            _label = _entry.get("label", strategy.selected_angle)
            lines.append(f"Lead with: {_label}.")
        except Exception:
            pass  # angle_memory not available — skip silently

    # Phase 2: unresolved objection instruction.
    # Surfaces as a positive directive before the progression note.
    # Empty list = no instruction emitted.
    if strategy.unresolved_objections:
        try:
            from angle_memory import _OBJECTION_REGISTRY
            for _obj_type in strategy.unresolved_objections[:1]:  # max 1 per reply
                _oi = _OBJECTION_REGISTRY.get(_obj_type, {})
                _hint = _oi.get("handling_hint", "")
                if _hint:
                    lines.append(f"Address objection: {_hint}")
        except Exception:
            pass  # angle_memory not available — skip silently

    # Progression — what NEW ground to cover (not present in first version)
    if strategy.progression_goal:
        lines.append(f"Focus: {strategy.progression_goal}")

    # Tone — posture first, then specific behavioural directives from rule tables
    lines.append(f"Tone: {strategy.tone_posture}")
    if strategy.tone_guidance:
        for g in strategy.tone_guidance:
            lines.append(f"  • {g}")

    # Length
    length_map = {
        "short":  "Keep it short — 2 paragraphs maximum.",
        "medium": "2-3 focused paragraphs.",
        "long":   "Full response — answer completely, do not pad.",
    }
    lines.append(length_map[strategy.reply_length])

    # Question priority — surfaces what the prohibitions already encode,
    # but as a positive instruction so the model sees it clearly
    if has_questions:
        q_note = (
            f"Answer their {question_count} questions first, then any sales content."
            if question_count > 1
            else "Answer their question first, then any sales content."
        )
        lines.append(q_note)

    # CTA
    cta_map: dict[str, str] = {
        "none":             "No call to action.",
        "soft_question":    "End with one easy, low-pressure question.",
        "forward_question": "End with a direct but low-pressure forward question.",
        "specific_counter": "End with your specific counter-price and one clear next-step question.",
        "decision_prompt":  "End with a direct decision prompt.",
        "transaction":      "End with the purchase link or payment step — no re-selling.",
        "exit_open_door":   "End by leaving the door open gracefully. No pressure.",
    }
    if strategy.cta_style in cta_map:
        lines.append(f"CTA: {cta_map[strategy.cta_style]}")

    # Persuasion + repetition combined — they always carry the same direction,
    # so one consolidated line avoids sending the same instruction twice
    if strategy.persuasion_level == 0:
        lines.append("No sales content.")
    elif strategy.persuasion_level == 1:
        repeat_note = " Do not repeat points already made." if strategy.repetition_policy == "no_repeat" else ""
        lines.append(f"One value point only — no full pitch.{repeat_note}")
    elif strategy.persuasion_level == 2:
        lines.append("Make a clear, relevant case — lead with the strongest benefit.")
    elif strategy.persuasion_level == 3:
        lines.append("Full value case — lead with the most relevant outcome for their business.")

    # Urgency — only emit when non-zero (zero is the default, no need to state it)
    if strategy.urgency_level == 1:
        lines.append("Urgency only if factual: the domain is publicly listed.")

    # No-invent guard — only when there are no domain/price details to ground from
    if no_domain_no_price:
        lines.append(
            "No domain details were provided. Answer from general domain industry knowledge. "
            "Do not invent registration dates, ages, traffic numbers, or specific facts."
        )

    # Prohibitions — filter universals already covered by system prompt and humanizer,
    # keep only the most situationally specific (max 3)
    _always_skip = {
        "no manufactured urgency or fake deadlines",
        "no hype phrases (perfect domain, once in a lifetime, game-changing)",
        "no AI-sounding openers (I hope this email finds you, I am writing to)",
    }
    key_prohibitions = [
        p for p in strategy.prohibited_topics
        if p not in _always_skip
    ][:3]
    if key_prohibitions:
        lines.append("Do not: " + "; ".join(key_prohibitions) + ".")

    lines.append("\nWrite only the email body. No subject line. No commentary.")
    lines.append("Write the reply:")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION & DEBUG TOOLING
# Zero external dependencies. Call from tests, API endpoints, or the REPL.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(signals: StrategySignals) -> dict:
    """
    Run build_strategy() and return a structured evaluation report.

    Useful for:
    - automated tests against known scenarios
    - the /qc/strategy-eval API endpoint
    - local debugging without starting the server

    Returns a dict safe to JSON-serialise.
    """
    strategy = build_strategy(signals)
    brief    = build_prompt_brief(
        strategy,
        has_questions      = signals.has_questions,
        question_count     = signals.question_count,
        no_domain_no_price = signals.no_domain_no_price,
    )

    return {
        "strategy": {
            "primary_goal":         strategy.primary_goal,
            "buyer_state":          strategy.buyer_state,
            "conversation_posture": strategy.conversation_posture,
            "persuasion_level":     strategy.persuasion_level,
            "urgency_level":        strategy.urgency_level,
            "cta_style":            strategy.cta_style,
            "reply_length":         strategy.reply_length,
            "tone_posture":         strategy.tone_posture,
            "tone_guidance":        strategy.tone_guidance,
            "reply_objective":      strategy.reply_objective,
            "progression_goal":     strategy.progression_goal,
            "suppressed_topics":    strategy.suppressed_topics,
            "repetition_policy":    strategy.repetition_policy,
            "no_domain_no_price":   strategy.no_domain_no_price,
            "prohibited_topics":    strategy.prohibited_topics,
            "confidence": {
                "stage": strategy.stage_confidence,
                "buyer": strategy.buyer_confidence,
                "goal":  strategy.goal_confidence,
            },
            # Phase 2 fields — empty strings/lists when no inventory supplied
            "selected_angle":        strategy.selected_angle,
            "available_angles":      strategy.available_angles,
            "exhausted_angles":      strategy.exhausted_angles,
            "unresolved_objections": strategy.unresolved_objections,
        },
        "reasoning_trace": strategy.reasoning_trace,
        "prompt_brief":    brief,
        "brief_tokens":    len(brief) // 4,
    }


def assert_strategy(
    signals:        StrategySignals,
    expected_goal:  Optional[str]  = None,
    expected_buyer: Optional[str]  = None,
    expected_cta:   Optional[str]  = None,
    expected_length: Optional[str] = None,
    label:          str            = "",
) -> dict:
    """
    Lightweight assertion helper for scenario-based unit tests.
    Returns a result dict; raises AssertionError with a clear message on failure.

    Usage:
        assert_strategy(
            StrategySignals(intent="cold_outreach", message="Hi", stage="first_outreach",
                            neg_state="none", response_frame="inferred_reply",
                            tone_requested="professional and persuasive"),
            expected_goal="introduce",
            expected_buyer="cold",
            expected_cta="soft_question",
            label="cold outreach — first contact"
        )
    """
    result   = evaluate_strategy(signals)
    strategy = result["strategy"]
    failures = []

    checks = [
        ("primary_goal",  expected_goal),
        ("buyer_state",   expected_buyer),
        ("cta_style",     expected_cta),
        ("reply_length",  expected_length),
    ]
    for field, expected in checks:
        if expected is not None and strategy[field] != expected:
            failures.append(
                f"{field}: expected '{expected}', got '{strategy[field]}' "
                f"(trace: {result['reasoning_trace'].get('goal_source','?')})"
            )

    if failures:
        tag = f" [{label}]" if label else ""
        raise AssertionError(f"Strategy assertion failed{tag}:\n" + "\n".join(failures))

    result["label"]  = label
    result["passed"] = True
    return result


def run_scenario_suite() -> list[dict]:
    """
    Canonical scenario test suite. Returns list of pass/fail results.
    Run with: python3 -c "from reply_strategy import run_scenario_suite; run_scenario_suite()"

    Each scenario represents a real broker situation and the strategy decisions
    we expect from it. Add new scenarios here as edge cases are discovered.
    """
    def _sig(**kwargs) -> StrategySignals:
        defaults = dict(
            neg_state="none", response_frame="inferred_reply",
            tone_requested="professional and persuasive",
            stage="unknown", outreach_count=0,
        )
        defaults.update(kwargs)
        return StrategySignals(**defaults)

    # Phase 2 helper — build minimal AngleInventory objects for scenario tests
    def _make_empty_inventory(lead_id: int):
        """An inventory with no angles used yet — all available."""
        try:
            from angle_memory import AngleInventory, get_angle_labels
            return AngleInventory(
                lead_id        = lead_id,
                all_angles     = get_angle_labels(),
                available_angles = get_angle_labels(),
            )
        except ImportError:
            return None

    def _make_exhausted_inventory(lead_id: int, exhausted: list):
        """An inventory where the given angles are already exhausted."""
        try:
            from angle_memory import AngleInventory, get_angle_labels
            all_a = get_angle_labels()
            return AngleInventory(
                lead_id          = lead_id,
                all_angles       = all_a,
                exhausted_angles = exhausted,
                available_angles = [a for a in all_a if a not in exhausted],
            )
        except ImportError:
            return None

    scenarios = [
        # ── Cold outreach ─────────────────────────────────────────────────────
        dict(
            label   = "cold outreach — zero prior contact",
            signals = _sig(intent="cold_outreach", message="Hi we own this domain",
                           stage="first_outreach"),
            expect  = dict(goal="introduce", buyer="cold", cta="soft_question", length="short"),
        ),
        # ── Warm lead ─────────────────────────────────────────────────────────
        dict(
            label   = "warm lead — expressed interest",
            signals = _sig(intent="warm_outreach", message="Sounds interesting tell me more",
                           stage="warm_lead", neg_state="soft_interest"),
            expect  = dict(goal="build_interest", buyer="interested", cta="forward_question"),
        ),
        # ── Negotiation — lowball ─────────────────────────────────────────────
        dict(
            label   = "negotiation — lowball offer",
            signals = _sig(intent="negotiation", message="I'll give you $50",
                           stage="negotiation", neg_state="low_anchor_offer"),
            expect  = dict(goal="counter_offer", buyer="anchoring", cta="specific_counter"),
        ),
        # ── Conflict: hesitating buyer + counter_offer goal ───────────────────
        dict(
            label   = "hesitating buyer softens counter_offer to hold_position",
            signals = _sig(intent="negotiation",
                           message="Let me think about it, maybe $200",
                           stage="negotiation", neg_state="hesitation"),
            expect  = dict(goal="hold_position", buyer="hesitating"),
        ),
        # ── Buying signal overrides build_interest ────────────────────────────
        # In production, detect_conversation_stage() returns "accepted" when
        # the message contains commitment language — that sets buyer=buying.
        dict(
            label   = "buying signal upgrades goal to close",
            signals = _sig(intent="follow_up_after_interest",
                           message="Ready to proceed, send the invoice",
                           stage="accepted", neg_state="none"),
            expect  = dict(goal="confirm_next_step", buyer="buying", cta="transaction"),
        ),
        # ── Hard rejection forces defuse ──────────────────────────────────────
        dict(
            label   = "hard rejection forces defuse",
            signals = _sig(intent="not_interested_ask_why",
                           message="Not interested, please stop emailing",
                           stage="rejected", neg_state="hard_rejection"),
            expect  = dict(goal="defuse", buyer="rejecting", length="short"),
        ),
        # ── email_preset overrides stage goal ─────────────────────────────────
        dict(
            label   = "final_offer preset overrides warm_lead stage",
            signals = _sig(intent="follow_up", message="Following up on my offer",
                           stage="warm_lead", email_preset="final_offer"),
            expect  = dict(goal="hold_position"),
        ),
        # ── Stalled lead ──────────────────────────────────────────────────────
        dict(
            label   = "stalled lead — multiple outreach attempts",
            signals = _sig(intent="follow_up_no_response", message="Just checking in",
                           stage="stalled", outreach_count=3),
            expect  = dict(goal="re_engage", length="short"),
        ),
        # ── Final follow-up ───────────────────────────────────────────────────
        dict(
            label   = "final follow-up — high outreach count",
            signals = _sig(intent="follow_up", message="One last note",
                           stage="final_follow_up", outreach_count=4),
            expect  = dict(goal="final_contact", cta="exit_open_door"),
        ),
        # ── Agreed, no payment ────────────────────────────────────────────────
        dict(
            label   = "agreed deal — next step only",
            signals = _sig(intent="agreed_no_pay", message="We agreed on $450",
                           stage="accepted"),
            expect  = dict(goal="confirm_next_step", cta="transaction",
                           buyer="buying", length="short"),
        ),
        # ── No domain/price — no-invent guard ────────────────────────────────
        dict(
            label   = "no domain/price — invent prohibition present",
            signals = _sig(intent="general", message="How does domain forwarding work?",
                           no_domain_no_price=True),
            expect  = dict(goal="inform"),
        ),

        # ── Phase 2 regression scenarios ─────────────────────────────────────
        # Verify that Phase 2 fields are empty (not erroring) when no inventory
        dict(
            label   = "Phase 2 — no inventory supplied, fields default to empty",
            signals = _sig(intent="cold_outreach", message="Hi",
                           stage="first_outreach"),
            expect  = dict(goal="introduce"),
        ),
        # Verify that when angle_inventory IS supplied, selected_angle is populated
        dict(
            label   = "Phase 2 — inventory supplied, selected_angle populated",
            signals = _sig(
                intent="warm_outreach", message="Interested, tell me more",
                stage="warm_lead", neg_state="soft_interest",
                angle_inventory=_make_empty_inventory(42),
            ),
            expect  = dict(goal="build_interest"),
        ),
        # Verify that exhausted angles from inventory feed suppressed_topics
        dict(
            label   = "Phase 2 — exhausted angles feed suppressed_topics",
            signals = _sig(
                intent="follow_up", message="Just following up",
                stage="stalled", outreach_count=3,
                angle_inventory=_make_exhausted_inventory(99, ["seo_benefit"]),
            ),
            expect  = dict(goal="re_engage"),
        ),
    ]

    results   = []
    passed    = 0
    failed    = 0

    for sc in scenarios:
        try:
            result = assert_strategy(
                signals        = sc["signals"],
                label          = sc["label"],
                expected_goal  = sc["expect"].get("goal"),
                expected_buyer = sc["expect"].get("buyer"),
                expected_cta   = sc["expect"].get("cta"),
                expected_length= sc["expect"].get("length"),
            )
            print(f"  ✓  {sc['label']}")
            results.append({"label": sc["label"], "passed": True})
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {sc['label']}\n     {e}")
            results.append({"label": sc["label"], "passed": False, "error": str(e)})
            failed += 1

    print(f"\n{passed}/{passed+failed} scenarios passed")
    return results
