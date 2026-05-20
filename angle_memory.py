"""
angle_memory.py — Angle & Objection Memory Structures
======================================================
Phase 1 of the Adaptive Memory + Angle Selection system.

THIS FILE IN PHASE 1
--------------------
Contains ONLY:
  - _ANGLE_REGISTRY     : typed dict of all defined persuasion angles
  - _OBJECTION_REGISTRY : typed dict of all tracked objection types
  - AngleRecord         : one logged use of an angle for a lead
  - AngleInventory      : full angle state for a lead at a point in time
  - AngleSelection      : result of the angle selection algorithm (Phase 2)
  - ObjectionRecord     : one logged objection for a lead

NO SELECTION LOGIC IN THIS FILE.
  build_angle_inventory() and _select_next_angle() are Phase 2.
  This file only defines the data structures they will consume.

ROLLBACK
--------
Delete this file.  Nothing in the existing codebase imports it yet.
It will be imported by reply_strategy.py in Phase 2.

DESIGN NOTES
------------
- All Literal types use plain strings (not Enum) so values survive JSON
  serialisation and print() without .value accessor noise.
- Every registry dict key is a stable identifier.  Do not rename keys
  between phases — they are stored as strings in angle_log.angle_id.
- _ANGLE_REGISTRY.detection_signals reuses the keyword lists already
  in reply_strategy._VALUE_TOPIC_SIGNALS.  In Phase 2 those lists will
  be derived from this registry, not the other way around.  For Phase 1
  both sets coexist — zero changes to reply_strategy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Literal types ─────────────────────────────────────────────────────────────

AngleId = Literal[
    "seo_benefit",       # local SEO / exact-match keyword advantage
    "traffic_benefit",   # direct / organic traffic the domain captures
    "brand_protection",  # owning the name before a competitor does
    "domain_forwarding", # instant redirect — no site rebuild needed
    "price_anchor",      # the asking price as a value anchor
    "scarcity",          # publicly listed; anyone can buy it
    "local_relevance",   # geo-specificity; city/service credibility
    "credibility_trust", # escrow, safe transfer, legitimate process
    "payment_options",   # instalment, escrow, credit card — low friction
    "competitor_risk",   # explicit: a competitor could snap it up
    "one_time_cost",     # domain = one-time purchase, not recurring ad spend
    "expired_reclaim",   # original owner: reclaim your traffic/brand
]

ObjectionType = Literal[
    "price_too_high",        # prospect says the price is too high
    "have_website",          # prospect says they already have a website
    "not_now",               # prospect says timing is wrong
    "no_budget",             # prospect says they have no budget
    "trust_concern",         # prospect is sceptical / thinks it's a scam
    "need_approval",         # prospect needs sign-off from someone else
    "competitor_preferred",  # prospect prefers a competing domain / TLD
    "not_relevant",          # prospect doesn't see why they'd want this domain
]

PitchDepth = Literal["primary", "secondary", "mentioned"]

ReplySentiment = Literal["positive", "neutral", "negative", "no_reply"]

AngleStageAffinity = Literal[
    "any",           # useful at any stage
    "early",         # best for first_outreach / warm_lead
    "mid",           # best for warm_lead / stalled re-engagement
    "late",          # best for negotiation / close
]


# ── Angle Registry ────────────────────────────────────────────────────────────
# Single source of truth for all defined persuasion angles.
#
# Fields per entry:
#   id                  — AngleId key (same as dict key)
#   label               — Human-readable name used in prompt briefs
#   detection_signals   — Keywords that indicate this angle was used in an
#                         outreach body.  Mirrors _VALUE_TOPIC_SIGNALS in
#                         reply_strategy.py.  Phase 2 replaces that dict
#                         with a derivation from here.
#   stage_affinity      — Which conversation stages this angle fits best.
#                         Phase 2 uses this to rank available angles.
#   goal_affinity       — Which primary_goals this angle pairs with.
#                         Phase 2 skips angles that don't match the current goal.
#   objection_response  — Which objection types this angle helps address.
#                         Phase 2 prioritises angles that resolve open objections.
#   exhaustion_threshold— How many times this angle can be pitched as 'primary'
#                         before Phase 2 treats it as exhausted.
#                         Default 2: pitched once is normal, twice is maximum.
#   position            — Priority order for first-contact use (0 = highest).
#                         Phase 2 uses this as a tiebreaker when multiple
#                         angles are available and equally affine.

_ANGLE_REGISTRY: dict[str, dict] = {
    "seo_benefit": {
        "id":                  "seo_benefit",
        "label":               "Local SEO / exact-match keyword advantage",
        "detection_signals":   [
            "seo", "search ranking", "search engine", "rank", "google",
            "organic", "local search", "exact match", "exact-match",
        ],
        "stage_affinity":      ["first_outreach", "warm_lead", "stalled"],
        "goal_affinity":       ["introduce", "build_interest", "re_engage"],
        "objection_response":  ["have_website", "not_relevant"],
        "exhaustion_threshold": 2,
        "position":            0,   # most commonly used first angle
    },

    "traffic_benefit": {
        "id":                  "traffic_benefit",
        "label":               "Organic / direct traffic the domain captures",
        "detection_signals":   [
            "traffic", "visitors", "clicks", "searches", "targeted traffic",
            "capture traffic", "drive traffic",
        ],
        "stage_affinity":      ["first_outreach", "warm_lead"],
        "goal_affinity":       ["introduce", "build_interest"],
        "objection_response":  ["not_relevant"],
        "exhaustion_threshold": 2,
        "position":            1,
    },

    "brand_protection": {
        "id":                  "brand_protection",
        "label":               "Brand protection — own it before a competitor does",
        "detection_signals":   [
            "brand", "competitor", "protect", "snapping it up", "shield",
            "before a competitor", "protect your brand", "brand asset",
        ],
        "stage_affinity":      ["warm_lead", "stalled", "final_follow_up"],
        "goal_affinity":       ["build_interest", "re_engage", "follow_up"],
        "objection_response":  ["not_relevant", "not_now"],
        "exhaustion_threshold": 2,
        "position":            2,
    },

    "competitor_risk": {
        "id":                  "competitor_risk",
        "label":               "Competitor risk — a rival could buy it and capture your traffic",
        "detection_signals":   [
            "competitor", "rival", "someone else", "another business",
            "first to", "get there first", "competitor buys",
        ],
        "stage_affinity":      ["warm_lead", "stalled", "final_follow_up"],
        "goal_affinity":       ["build_interest", "re_engage", "final_contact"],
        "objection_response":  ["not_now", "not_relevant"],
        "exhaustion_threshold": 1,   # strong fear angle — once is enough
        "position":            3,
    },

    "domain_forwarding": {
        "id":                  "domain_forwarding",
        "label":               "Easy setup — 5-minute redirect, no rebuild needed",
        "detection_signals":   [
            "forward", "forwarding", "redirect", "point to your site",
            "current site", "existing site",
        ],
        "stage_affinity":      ["warm_lead", "first_outreach"],
        "goal_affinity":       ["introduce", "build_interest"],
        "objection_response":  ["have_website"],
        "exhaustion_threshold": 1,
        "position":            4,
    },

    "one_time_cost": {
        "id":                  "one_time_cost",
        "label":               "One-time purchase — no recurring ad spend",
        "detection_signals":   [
            "one-time", "one time", "no monthly", "no ad spend", "permanent",
            "ongoing cost", "unlike ads", "instead of advertising",
        ],
        "stage_affinity":      ["warm_lead", "negotiation", "stalled"],
        "goal_affinity":       ["build_interest", "hold_position"],
        "objection_response":  ["price_too_high", "no_budget"],
        "exhaustion_threshold": 2,
        "position":            5,
    },

    "scarcity": {
        "id":                  "scarcity",
        "label":               "Publicly listed — available to any buyer",
        "detection_signals":   [
            "publicly listed", "available to anyone", "first to",
            "won't be available", "act before", "still available",
        ],
        "stage_affinity":      ["stalled", "final_follow_up", "negotiation"],
        "goal_affinity":       ["re_engage", "final_contact", "hold_position"],
        "objection_response":  ["not_now"],
        "exhaustion_threshold": 2,
        "position":            6,
    },

    "local_relevance": {
        "id":                  "local_relevance",
        "label":               "Geographic relevance — city/service authority signal",
        "detection_signals":   [
            "local", "city", "geographic", "location", "near",
            "neighbourhood", "neighborhood", "city-specific",
        ],
        "stage_affinity":      ["first_outreach", "warm_lead"],
        "goal_affinity":       ["introduce", "build_interest"],
        "objection_response":  ["not_relevant"],
        "exhaustion_threshold": 2,
        "position":            7,
    },

    "credibility_trust": {
        "id":                  "credibility_trust",
        "label":               "Secure transfer — escrow, trusted process",
        "detection_signals":   [
            "escrow", "trusted", "secure transfer", "guarantee", "protection",
            "safe", "legitimate", "verified",
        ],
        "stage_affinity":      ["warm_lead", "negotiation"],
        "goal_affinity":       ["build_interest", "counter_offer", "close"],
        "objection_response":  ["trust_concern"],
        "exhaustion_threshold": 1,
        "position":            8,
    },

    "payment_options": {
        "id":                  "payment_options",
        "label":               "Flexible payment — credit card, escrow, instalment",
        "detection_signals":   [
            "credit card", "escrow", "paypal", "payment plan", "instalment",
            "installment", "payment method",
        ],
        "stage_affinity":      ["negotiation", "close"],
        "goal_affinity":       ["counter_offer", "close", "confirm_next_step"],
        "objection_response":  ["price_too_high", "no_budget"],
        "exhaustion_threshold": 1,
        "position":            9,
    },

    "price_anchor": {
        "id":                  "price_anchor",
        "label":               "Price context — asking price reflects genuine value",
        "detection_signals":   [
            "asking price", "listed at", "priced at", "available for",
            "only asking", "just asking", "our price",
        ],
        "stage_affinity":      ["first_outreach", "warm_lead", "negotiation"],
        "goal_affinity":       ["introduce", "build_interest", "counter_offer"],
        "objection_response":  ["price_too_high"],
        "exhaustion_threshold": 2,
        "position":            10,
    },

    "expired_reclaim": {
        "id":                  "expired_reclaim",
        "label":               "Reclaim your domain — expired owner opportunity",
        "detection_signals":   [
            "expired", "used to own", "previously", "reclaim", "had it",
            "own it back", "direct traffic", "old customers",
        ],
        "stage_affinity":      ["first_outreach"],
        "goal_affinity":       ["introduce", "build_interest"],
        "objection_response":  [],
        "exhaustion_threshold": 1,   # highly specific — one mention is enough
        "position":            11,
    },
}


# ── Objection Registry ────────────────────────────────────────────────────────
# Maps each objection type to detection patterns and a brief handling hint.
#
# detection_signals — phrases in the prospect's message that trigger this objection.
# handling_hint     — a one-line instruction for Phase 2's prompt brief.
# angle_responses   — which angles most directly address this objection.

_OBJECTION_REGISTRY: dict[str, dict] = {
    "price_too_high": {
        "type":              "price_too_high",
        "label":             "Price objection — too expensive",
        "detection_signals": [
            "too expensive", "too much", "too high", "can't afford",
            "over my budget", "price is too", "that's a lot", "not worth",
        ],
        "handling_hint":     "Acknowledge the price concern in one sentence; pivot to value or payment options.",
        "angle_responses":   ["one_time_cost", "price_anchor", "payment_options"],
    },

    "have_website": {
        "type":              "have_website",
        "label":             "Already has a website",
        "detection_signals": [
            "already have a website", "have a site", "have our own",
            "already have one", "don't need another", "got a website",
        ],
        "handling_hint":     "Acknowledge they have a site; explain redirect — their site stays as-is.",
        "angle_responses":   ["domain_forwarding", "seo_benefit", "brand_protection"],
    },

    "not_now": {
        "type":              "not_now",
        "label":             "Not the right time",
        "detection_signals": [
            "not now", "not the right time", "maybe later", "not this year",
            "come back", "try again", "check in", "timing isn't",
        ],
        "handling_hint":     "Respect the timing; plant mild urgency (publicly listed) and offer a check-back.",
        "angle_responses":   ["scarcity", "competitor_risk"],
    },

    "no_budget": {
        "type":              "no_budget",
        "label":             "No budget available",
        "detection_signals": [
            "no budget", "out of budget", "tight on cash", "can't justify",
            "not in the budget", "financially", "cash flow",
        ],
        "handling_hint":     "Acknowledge budget constraint; frame as one-time cost vs ongoing ad spend.",
        "angle_responses":   ["one_time_cost", "payment_options"],
    },

    "trust_concern": {
        "type":              "trust_concern",
        "label":             "Trust / legitimacy concern",
        "detection_signals": [
            "scam", "not legitimate", "don't trust", "suspicious", "fraud",
            "how do i know", "verify", "seems sketchy", "is this real",
        ],
        "handling_hint":     "Address the concern directly; offer one concrete verification step (listing link, escrow).",
        "angle_responses":   ["credibility_trust"],
    },

    "need_approval": {
        "type":              "need_approval",
        "label":             "Needs approval from someone else",
        "detection_signals": [
            "need to ask", "check with", "my partner", "my boss", "board",
            "management", "approval", "sign off", "run it by",
        ],
        "handling_hint":     "Acknowledge the process; offer to provide a one-pager they can share.",
        "angle_responses":   ["seo_benefit", "one_time_cost"],
    },

    "competitor_preferred": {
        "type":              "competitor_preferred",
        "label":             "Prefers a competing domain or TLD",
        "detection_signals": [
            ".co.uk", ".net", "other domain", "different domain", "prefer the .com",
            "already got", "we have the .co", "another option",
        ],
        "handling_hint":     "Acknowledge their existing domain; position this as a complementary asset.",
        "angle_responses":   ["brand_protection", "seo_benefit"],
    },

    "not_relevant": {
        "type":              "not_relevant",
        "label":             "Doesn't see relevance to their business",
        "detection_signals": [
            "not relevant", "not for us", "don't see how", "what would we",
            "not sure why", "doesn't apply", "not in that space", "wrong business",
        ],
        "handling_hint":     "Connect the domain specifically to their business type with a concrete use case.",
        "angle_responses":   ["seo_benefit", "local_relevance", "traffic_benefit"],
    },
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AngleRecord:
    """
    One logged use of a persuasion angle for a specific lead.
    Populated from angle_log rows returned by MemoryDB.get_angle_history().

    Fields match angle_log columns exactly — no transformation needed.
    """
    angle_id:         str                     # AngleId key
    outreach_seq:     int                     # 1-based email number for this lead
    pitched_as:       PitchDepth              # primary | secondary | mentioned
    logged_at:        float                   # unix timestamp

    # Optional — populated when prospect reply data is available
    prospect_replied: Optional[bool]  = None  # True if prospect sent a reply
    reply_sentiment:  Optional[str]   = None  # ReplySentiment or None

    # Optional DB keys — not needed for Phase 2 logic
    id:               Optional[int]   = None
    lead_id:          Optional[int]   = None
    outreach_id:      Optional[int]   = None

    @classmethod
    def from_db_row(cls, row: dict) -> "AngleRecord":
        """
        Construct from a broker_memory.get_angle_history() dict row.
        Handles None / integer-boolean conversions transparently.
        """
        prospect_replied = row.get("prospect_replied")
        if prospect_replied is not None:
            prospect_replied = bool(prospect_replied)
        return cls(
            id               = row.get("id"),
            lead_id          = row.get("lead_id"),
            outreach_id      = row.get("outreach_id"),
            angle_id         = row["angle_id"],
            outreach_seq     = row.get("outreach_seq", 1),
            pitched_as       = row.get("pitched_as", "primary"),
            prospect_replied = prospect_replied,
            reply_sentiment  = row.get("reply_sentiment"),
            logged_at        = row.get("logged_at", 0.0),
        )


@dataclass
class AngleInventory:
    """
    Full angle state for a single lead at the current point in time.

    Built by Phase 2's build_angle_inventory() from angle_log rows.
    In Phase 1 this dataclass is defined but never instantiated.

    Purpose: answer every question the angle selection algorithm needs
    without additional DB queries after construction.
    """
    lead_id:             int

    # All AngleId keys defined in _ANGLE_REGISTRY
    all_angles:          list[str]        = field(default_factory=list)

    # AngleRecord objects from angle_log, sorted oldest-first
    used_angles:         list[AngleRecord] = field(default_factory=list)

    # Angle IDs pitched as 'primary' >= exhaustion_threshold times
    exhausted_angles:    list[str]        = field(default_factory=list)

    # all_angles - any angle that has ever been used (primary/secondary/mentioned)
    available_angles:    list[str]        = field(default_factory=list)

    # Angles where prospect_replied=True and sentiment in (positive, neutral)
    effective_angles:    list[str]        = field(default_factory=list)

    # Angles where prospect_replied=True and sentiment = negative, or no_reply
    ineffective_angles:  list[str]        = field(default_factory=list)

    # Count of how many times each angle was pitched as 'primary'
    primary_use_count:   dict[str, int]   = field(default_factory=dict)

    # Timestamp of last primary use per angle
    last_used_at:        dict[str, float] = field(default_factory=dict)

    def angle_info(self, angle_id: str) -> Optional[dict]:
        """Return the registry entry for an angle_id, or None if not found."""
        return _ANGLE_REGISTRY.get(angle_id)

    def is_exhausted(self, angle_id: str) -> bool:
        """True if this angle has been used >= its exhaustion_threshold."""
        return angle_id in self.exhausted_angles

    def times_used_as_primary(self, angle_id: str) -> int:
        """How many times this angle was pitched as 'primary' for this lead."""
        return self.primary_use_count.get(angle_id, 0)


@dataclass
class AngleSelection:
    """
    Result of the angle selection algorithm (Phase 2).

    In Phase 1 this dataclass is defined but never instantiated.
    The fields are defined here so reply_strategy.py can add them to
    ReplyStrategy without any further changes to this file.

    selected_angle    — the AngleId that build_prompt_brief() should use.
    selection_reason  — why this angle was chosen (used in reasoning_trace).
    confidence        — 0.0–1.0; how certain the algorithm is in its choice.
    fallback_angle    — if selected_angle can't be used, use this one instead.
    progression_note  — what the reply should accomplish with this angle;
                        replaces the generic _PROGRESSION_LOGIC string.
    objection_to_address — if the angle was chosen to resolve an objection,
                         the objection_type it is addressing.
    """
    selected_angle:         str
    selection_reason:       str
    confidence:             float       = 1.0
    fallback_angle:         Optional[str] = None
    progression_note:       str         = ""
    objection_to_address:   Optional[str] = None


@dataclass
class ObjectionRecord:
    """
    One logged objection for a specific lead.
    Populated from objection_log rows returned by MemoryDB.get_objection_history().

    Fields match objection_log columns exactly.
    """
    objection_type:        str           # ObjectionType key
    detected_at:           float         # unix timestamp
    addressed:             bool          = False
    source_snippet:        Optional[str] = None
    addressed_at:          Optional[float] = None
    addressed_outreach_id: Optional[int]   = None

    # Optional DB keys
    id:                    Optional[int] = None
    lead_id:               Optional[int] = None

    @classmethod
    def from_db_row(cls, row: dict) -> "ObjectionRecord":
        """
        Construct from a broker_memory.get_objection_history() dict row.
        """
        return cls(
            id                    = row.get("id"),
            lead_id               = row.get("lead_id"),
            objection_type        = row["objection_type"],
            source_snippet        = row.get("source_snippet"),
            addressed             = bool(row.get("addressed", 0)),
            addressed_at          = row.get("addressed_at"),
            addressed_outreach_id = row.get("addressed_outreach_id"),
            detected_at           = row.get("detected_at", 0.0),
        )

    @property
    def registry_info(self) -> Optional[dict]:
        """Return the _OBJECTION_REGISTRY entry for this objection type."""
        return _OBJECTION_REGISTRY.get(self.objection_type)

    @property
    def handling_hint(self) -> str:
        """Quick access to the handling instruction for prompt briefs."""
        info = self.registry_info
        return info["handling_hint"] if info else ""


# ── Convenience helpers ───────────────────────────────────────────────────────
# Pure functions with no DB dependency — safe to call anywhere.

def get_angle_labels() -> list[str]:
    """Return all defined AngleId strings in position order."""
    return [
        entry["id"]
        for entry in sorted(_ANGLE_REGISTRY.values(), key=lambda e: e["position"])
    ]


def angle_detection_signals(angle_id: str) -> list[str]:
    """Return the keyword list for a given angle_id.  Empty list if unknown."""
    entry = _ANGLE_REGISTRY.get(angle_id, {})
    return entry.get("detection_signals", [])


def objection_angle_responses(objection_type: str) -> list[str]:
    """Return AngleId strings that help address a given objection type."""
    entry = _OBJECTION_REGISTRY.get(objection_type, {})
    return entry.get("angle_responses", [])


def angles_for_goal(goal: str) -> list[str]:
    """
    Return AngleId strings whose goal_affinity includes the given primary_goal.
    Ordered by position (most commonly useful first).
    """
    return [
        entry["id"]
        for entry in sorted(_ANGLE_REGISTRY.values(), key=lambda e: e["position"])
        if goal in entry.get("goal_affinity", [])
    ]


def angles_for_stage(stage: str) -> list[str]:
    """
    Return AngleId strings whose stage_affinity includes the given stage.
    Ordered by position.
    """
    return [
        entry["id"]
        for entry in sorted(_ANGLE_REGISTRY.values(), key=lambda e: e["position"])
        if stage in entry.get("stage_affinity", [])
    ]


# ── Phase 2: Inventory builder ────────────────────────────────────────────────

def build_angle_inventory(lead_id: int, db: object) -> AngleInventory:
    """
    Construct a complete AngleInventory for a lead from broker_memory rows.

    db must be a MemoryDB instance (or any object implementing
    get_angle_history(lead_id) and get_prospect_replies(lead_id)).

    This function is the single source of truth for the question:
    "what is the current angle state for this lead?"

    Returns a fully populated AngleInventory.
    Never raises — returns an empty inventory on any failure.
    """
    all_angle_ids = get_angle_labels()

    try:
        angle_rows   = db.get_angle_history(lead_id)
        reply_rows   = db.get_prospect_replies(lead_id)
    except Exception as e:
        print(f"[AngleMemory] build_angle_inventory db error: {e}")
        return AngleInventory(lead_id=lead_id, all_angles=all_angle_ids)

    # Build a quick lookup: outreach_id → reply sentiment + replied flag
    # Key: in_reply_to_outreach_id (can be None → skip)
    outreach_reply_map: dict[int, dict] = {}
    for r in reply_rows:
        oid = r.get("in_reply_to_outreach_id")
        if oid is not None:
            outreach_reply_map[oid] = {
                "sentiment":     r.get("sentiment"),
                "has_questions": bool(r.get("has_questions", 0)),
                "word_count":    r.get("word_count", 0),
            }

    # Materialise AngleRecord objects, enriching with reply data where available
    used_records: list[AngleRecord] = []
    primary_use_count: dict[str, int]   = {}
    last_used_at:      dict[str, float] = {}

    for row in angle_rows:
        rec = AngleRecord.from_db_row(row)

        # Enrich with prospect reply data if the angle has a linked outreach_id
        if rec.outreach_id and rec.outreach_id in outreach_reply_map:
            reply_info = outreach_reply_map[rec.outreach_id]
            rec.prospect_replied = True
            rec.reply_sentiment  = reply_info["sentiment"]
        elif rec.prospect_replied is None:
            # No reply data linked — leave as unknown (None)
            pass

        used_records.append(rec)

        if rec.pitched_as == "primary":
            primary_use_count[rec.angle_id] = primary_use_count.get(rec.angle_id, 0) + 1
        if rec.logged_at > last_used_at.get(rec.angle_id, 0.0):
            last_used_at[rec.angle_id] = rec.logged_at

    # Classify angles
    exhausted_angles:   list[str] = []
    available_angles:   list[str] = []
    effective_angles:   list[str] = []
    ineffective_angles: list[str] = []

    used_ids = {r.angle_id for r in used_records}

    for angle_id in all_angle_ids:
        entry     = _ANGLE_REGISTRY[angle_id]
        threshold = entry.get("exhaustion_threshold", 2)
        times     = primary_use_count.get(angle_id, 0)

        if times >= threshold:
            exhausted_angles.append(angle_id)
        elif angle_id not in used_ids:
            available_angles.append(angle_id)
        # else: used but not exhausted — still available, counted in available too
        # (secondary/mentioned uses don't count toward exhaustion)

    # Angles that were used AND got a positive/neutral reply
    angles_with_reply = {
        r.angle_id for r in used_records
        if r.prospect_replied and r.reply_sentiment in ("positive", "neutral")
    }
    effective_angles = [a for a in used_ids if a in angles_with_reply]

    # Angles used but got negative or no reply
    angles_without_positive = {
        r.angle_id for r in used_records
        if r.reply_sentiment in ("negative", "no_reply") or
           (r.prospect_replied is False)
    }
    ineffective_angles = [a for a in used_ids if a in angles_without_positive
                          and a not in effective_angles]

    return AngleInventory(
        lead_id            = lead_id,
        all_angles         = all_angle_ids,
        used_angles        = used_records,
        exhausted_angles   = exhausted_angles,
        available_angles   = available_angles,
        effective_angles   = effective_angles,
        ineffective_angles = ineffective_angles,
        primary_use_count  = primary_use_count,
        last_used_at       = last_used_at,
    )


# ── Phase 2: Angle selector ───────────────────────────────────────────────────

def _select_next_angle(
    inventory:            AngleInventory,
    goal:                 str,
    stage:                str,
    unresolved_objections: list["ObjectionRecord"],
) -> Optional["AngleSelection"]:
    """
    Select the best next persuasion angle for a lead's reply.

    Selection priority (highest first):
    1. Objection-responsive angle   — unresolved objection present and an
                                      available angle directly addresses it
    2. Effective + stage-affine     — was used before, got a positive/neutral
                                      reply, fits current stage
    3. Stage + goal affine unused   — has never been used, fits stage and goal
    4. Goal-affine unused           — has never been used, fits goal only
    5. Stage-affine unused          — has never been used, fits stage only
    6. Any remaining available      — never used, any stage/goal
    7. Brief reminder of best prior — all matching angles exhausted; re-use
                                      the most effective one at low intensity

    Returns None only if inventory.all_angles is empty (impossible in practice).
    Never raises.
    """
    if not inventory.all_angles:
        return None

    exhausted  = set(inventory.exhausted_angles)
    used_ids   = {r.angle_id for r in inventory.used_angles}
    stage_set  = set(angles_for_stage(stage))
    goal_set   = set(angles_for_goal(goal))

    # ── Priority 1: objection-responsive available angle ──────────────────────
    for obj_rec in unresolved_objections:
        responding_angles = objection_angle_responses(obj_rec.objection_type)
        for angle_id in responding_angles:
            if angle_id in exhausted:
                continue
            return AngleSelection(
                selected_angle       = angle_id,
                selection_reason     = f"unresolved objection: {obj_rec.objection_type}",
                confidence           = 0.95,
                progression_note     = (
                    f"{_ANGLE_REGISTRY[angle_id]['label']} — "
                    f"{_OBJECTION_REGISTRY[obj_rec.objection_type]['handling_hint']}"
                ),
                objection_to_address = obj_rec.objection_type,
            )

    # ── Priority 2: effective + stage-affine (re-use what worked, stage-matched) ──
    effective_stage = [
        a for a in inventory.effective_angles
        if a in stage_set and a not in exhausted
    ]
    if effective_stage:
        chosen = min(effective_stage, key=lambda a: _ANGLE_REGISTRY[a]["position"])
        return AngleSelection(
            selected_angle   = chosen,
            selection_reason = "previously effective + stage-affine",
            confidence       = 0.80,
            progression_note = (
                f"Re-engage with {_ANGLE_REGISTRY[chosen]['label']} — "
                "prospect responded positively to this angle before."
            ),
        )

    # ── Priority 3: stage + goal affine, never used ───────────────────────────
    stage_goal_unused = [
        a for a in (stage_set & goal_set)
        if a not in used_ids and a not in exhausted
    ]
    # Sort by registry position for determinism
    stage_goal_unused.sort(key=lambda a: _ANGLE_REGISTRY[a]["position"])
    if stage_goal_unused:
        chosen = stage_goal_unused[0]
        fallback = stage_goal_unused[1] if len(stage_goal_unused) > 1 else None
        return AngleSelection(
            selected_angle   = chosen,
            selection_reason = f"unused + stage={stage} + goal={goal}",
            confidence       = 0.90,
            fallback_angle   = fallback,
            progression_note = (
                f"Introduce {_ANGLE_REGISTRY[chosen]['label']} — "
                "not yet covered with this prospect."
            ),
        )

    # ── Priority 4: goal-affine, never used ───────────────────────────────────
    goal_unused = [
        a for a in goal_set
        if a not in used_ids and a not in exhausted
    ]
    goal_unused.sort(key=lambda a: _ANGLE_REGISTRY[a]["position"])
    if goal_unused:
        chosen = goal_unused[0]
        return AngleSelection(
            selected_angle   = chosen,
            selection_reason = f"unused + goal={goal}",
            confidence       = 0.82,
            fallback_angle   = goal_unused[1] if len(goal_unused) > 1 else None,
            progression_note = (
                f"Lead with {_ANGLE_REGISTRY[chosen]['label']} — "
                "relevant to current goal, not yet pitched."
            ),
        )

    # ── Priority 5: stage-affine, never used ─────────────────────────────────
    stage_unused = [
        a for a in stage_set
        if a not in used_ids and a not in exhausted
    ]
    stage_unused.sort(key=lambda a: _ANGLE_REGISTRY[a]["position"])
    if stage_unused:
        chosen = stage_unused[0]
        return AngleSelection(
            selected_angle   = chosen,
            selection_reason = f"unused + stage={stage}",
            confidence       = 0.75,
            progression_note = (
                f"Try {_ANGLE_REGISTRY[chosen]['label']} — "
                "appropriate for this conversation stage."
            ),
        )

    # ── Priority 6: any available (never used, any stage/goal) ───────────────
    fully_available = [
        a for a in inventory.all_angles
        if a not in used_ids and a not in exhausted
    ]
    fully_available.sort(key=lambda a: _ANGLE_REGISTRY[a]["position"])
    if fully_available:
        chosen = fully_available[0]
        return AngleSelection(
            selected_angle   = chosen,
            selection_reason = "only remaining unused angle",
            confidence       = 0.65,
            progression_note = (
                f"Try {_ANGLE_REGISTRY[chosen]['label']} — "
                "all stage/goal-affine angles already covered."
            ),
        )

    # ── Priority 7: all angles exhausted — brief reminder of most effective ───
    # Find the angle with the best reply history, not-recently-used preferred
    if inventory.effective_angles:
        # Among effective, pick least recently used
        chosen = min(
            inventory.effective_angles,
            key=lambda a: inventory.last_used_at.get(a, 0.0),
        )
    elif inventory.used_angles:
        # No positive signal at all — least recently used primary angle
        primary_angles = [r.angle_id for r in inventory.used_angles
                          if r.pitched_as == "primary"]
        if primary_angles:
            chosen = min(primary_angles,
                         key=lambda a: inventory.last_used_at.get(a, 0.0))
        else:
            chosen = inventory.used_angles[-1].angle_id
    else:
        # Absolute fallback — first angle in registry
        chosen = inventory.all_angles[0]

    return AngleSelection(
        selected_angle   = chosen,
        selection_reason = "all angles exhausted — brief reminder only",
        confidence       = 0.50,
        progression_note = (
            f"Brief reminder of {_ANGLE_REGISTRY.get(chosen, {}).get('label', chosen)} — "
            "all major value angles have been covered. Keep it very short."
        ),
    )
