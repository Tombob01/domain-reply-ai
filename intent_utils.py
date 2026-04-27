"""
intent_utils.py — Shared intent detection data & helpers
=========================================================
Extracted from main.py to break the circular import between
main.py and intent_registry.py.

Both main.py and intent_registry.py import from here.
Nothing in this file imports from main.py or intent_registry.py.

Also contains:
  - QUESTION_TYPES  : classifier for detected questions
  - classify_question() : returns one of four question type labels
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# INTENT KEYWORDS
# Single canonical source — used by intent_utils, main.py, pipeline.py,
# and intent_registry.py.  main.py's local INTENT_KEYWORDS now imports
# from here instead of defining its own copy.
# ─────────────────────────────────────────────────────────────────────────────

INTENT_KEYWORDS: dict[str, list[str]] = {
    # ── Core response intents ─────────────────────────────────────────────────
    "no_thanks":             ["no thanks", "not interested", "pass", "decline", "don't need",
                              "not for us", "no thank you", "remove me"],
    "price_inquiry":         ["how much", "price", "cost", "what are you asking", "rate", "fee",
                              "what's the price", "pricing"],
    "price_too_high":        ["too expensive", "too high", "too much", "can't afford",
                              "register for", "just $10", "only $8", "only $10", "only $12",
                              "costs nothing", "regular domain", "price was high",
                              "price is high", "bit expensive", "quite expensive",
                              "very expensive", "a lot of money", "that's a lot"],
    "negotiation":           ["offer", "counter", "negotiate", "lower", "discount",
                              "best price", "bottom", "lowest", "deal", "make an offer"],
    "follow_up":             ["follow up", "following up", "no reply", "no response",
                              "checking in", "reminder", "still available", "any update",
                              "didn't reply", "didn't respond", "nothing back", "still silent",
                              "disappeared", "no answer", "haven't replied", "not replied",
                              "haven't responded", "not responded", "pinged", "silence",
                              "went quiet", "ghosted", "nothing heard", "haven't heard back"],
    "trust_issue":           ["scam", "fake", "not real", "legitimate", "trust",
                              "verify", "proof", "fraud", "suspicious", "doubt",
                              "worried", "concern", "is this real", "dodgy"],
    "have_website":          ["already have", "have a website", "have a domain",
                              "don't need another", "existing site", "current website"],
    "rank_well":             ["already rank", "rank fine", "seo is fine",
                              "first page already", "rank well", "good ranking"],
    "how_it_works":          ["how does", "redirect", "forward", "how do i", "technical",
                              "it guy", "developer", "walk me through", "step by step",
                              "buy it", "how to buy", "process"],
    "why_buy":               ["why", "benefits", "what's the point", "explain",
                              "value", "help my business", "what will it do", "how will it help"],
    "not_now":               ["not now", "not the right time", "check back",
                              "few months", "come back later", "try later"],
    "partner":               ["partner", "team", "discuss", "boss", "colleague",
                              "approval", "need to talk", "business partner"],
    "agreed_no_pay":         ["agreed", "deal", "haven't paid", "no payment",
                              "still waiting", "we agreed", "send the link"],
    "payment_issue":         ["link not working", "payment failed", "can't checkout",
                              "portal", "error", "not working", "checkout issue"],
    "angry":                 ["stop emailing", "spam", "harassment", "angry",
                              "annoying", "unsubscribe", "remove", "leave me alone"],
    "expired_owner":         ["used to be", "our domain", "how did you get",
                              "previously owned", "that was ours", "we owned"],
    "extension":             [".net", ".org", ".io", "other extension",
                              "already own the", "have the .net", "have the .org"],
    "not_interested_ask_why": ["not interested", "no interest", "thanks but no"],
    "cold_outreach":         ["first contact", "cold email", "new prospect",
                              "reaching out", "initial email"],
    # ── Post-sale / admin ─────────────────────────────────────────────────────
    "post_purchase":         ["already paid", "sent payment", "made payment",
                              "where is my domain", "when will i receive", "how long does transfer",
                              "transfer take", "waiting for domain", "haven't received",
                              "payment confirmed", "after i pay", "next step"],
    "refund":                ["refund", "money back", "cancel my order",
                              "changed my mind", "want my money", "return"],
    "payment_method":        ["paypal", "crypto", "bitcoin", "bank transfer",
                              "wire transfer", "how do i pay", "payment method",
                              "pay by", "payment options", "can i pay"],
    "renewal_fees":          ["annual fee", "renewal fee", "ongoing fee", "yearly fee",
                              "recurring cost", "how much per year", "after i buy",
                              "maintenance fee", "keep it running"],
    "domain_metrics":        ["domain authority", "da score", "dr score", "backlinks",
                              "traffic data", "seo score", "moz", "ahrefs",
                              "how many visitors", "monthly traffic", "analytics",
                              "blacklisted", "penalised", "penalty", "spam score"],
    "identity":              ["who are you", "what company", "your company",
                              "who is this", "where are you from", "your name",
                              "why are you contacting", "how did you find me",
                              "who sent this", "are you a broker"],
    "low_budget":            ["low budget", "tight budget", "small business",
                              "can't afford much", "limited budget", "not much to spend",
                              "small budget", "what's the minimum"],
    "related_domains":       ["other domains", "similar domains", "other cities",
                              "portfolio", "do you have more", "what else",
                              "different domain", "other options", "multiple domains"],
    "development":           ["build a website", "build a new site", "develop it",
                              "create a website", "make a site", "new website on",
                              "host a site", "build on this", "develop the domain"],
    # ── Expanded intents ─────────────────────────────────────────────────────
    "request_info":          ["more information", "more info", "can you tell me",
                              "before i decide", "questions about", "details please",
                              "need to know", "curious about", "what exactly",
                              "tell me more", "few questions", "quick question"],
    "demo_offer":            ["show me", "can i see", "mock up", "mock-up", "example",
                              "visual", "what would it look like", "proof of concept",
                              "demonstrate", "show what", "see how it looks"],
    "meeting_request":       ["can we talk", "quick call", "phone call", "schedule a call",
                              "five minutes", "speak with you", "jump on a call",
                              "available to talk", "book a call", "prefer to chat"],
    "price_negotiation":     ["meet in the middle", "split the difference",
                              "room to negotiate", "come down", "what's your bottom",
                              "lowest you'll go", "any flexibility", "wiggle room"],
    "competitor_comparison": ["competitor", "rival", "others in my space", "who else",
                              "what about my competition", "what if someone else buys",
                              "niche competitors", "beat competition"],
    "trust_building":        ["how do i know", "prove it", "can you verify",
                              "how can i trust", "show proof", "confirm ownership",
                              "how do i verify", "is this legitimate"],
    "feature_explanation":   ["what does redirect mean", "plain english",
                              "explain simply", "layman terms", "what does it mean to",
                              "i don't understand", "break it down", "in simple terms",
                              "what exactly happens"],
    "soft_pitch":            ["just wanted to mention", "thought this might",
                              "no pressure", "take it or leave it",
                              "in case it's useful", "just letting you know", "fyi"],
    "value_reminder":        ["remind me why", "value recap", "full picture",
                              "benefits again", "what's the value again",
                              "summarize the value", "not convinced yet",
                              "still not sure"],
    "follow_up_no_response": ["no reply", "no response", "haven't heard back",
                              "sent last week", "sent an email", "nothing back",
                              "no answer", "still no response", "checking back in",
                              "haven't responded", "silence after"],
    "follow_up_after_pricing":["sent pricing", "price i quoted", "following up on price",
                              "after quote", "sent the cost", "pricing information",
                              "shared the rate", "sent the fee"],
    "follow_up_after_interest":["you were interested", "you mentioned interest",
                              "seemed keen", "expressed interest", "you said maybe",
                              "after showing interest", "you seemed interested",
                              "you were considering"],
    "general_response":      ["general inquiry", "misc", "other question",
                              "not sure which", "various questions"],
    # ── Situation-mode intents ────────────────────────────────────────────────
    "sales_pitch":           ["first contact", "cold email", "initial outreach",
                              "new prospect", "reaching out", "never contacted",
                              "introduce", "presenting", "first time emailing",
                              "first pitch", "haven't spoken before"],
    "re_engagement":         ["cold lead", "went cold", "lost contact", "stopped replying",
                              "months ago", "long time", "reconnect", "revive",
                              "dormant", "inactive", "old lead", "previous conversation",
                              "been a while", "haven't heard", "time has passed"],
    "objection_handling":    ["hesitant", "unsure", "not convinced", "on the fence",
                              "needs convincing", "doubtful", "skeptical",
                              "thinking about it", "considering", "not sure if",
                              "hard to decide", "difficult to commit"],
}


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Priority order for intent resolution when scores tie
_PRIORITY_ORDER: list[str] = [
    "angry", "refund", "post_purchase", "payment_issue", "payment_method",
    "agreed_no_pay", "trust_issue", "trust_building", "expired_owner",
    "price_too_high", "price_negotiation", "extension", "rank_well",
    "have_website", "partner", "not_now", "negotiation", "low_budget",
    "how_it_works", "feature_explanation", "development", "why_buy",
    "value_reminder", "renewal_fees", "domain_metrics", "related_domains",
    "identity", "competitor_comparison", "objection_handling", "re_engagement",
    "follow_up_after_interest", "follow_up_after_pricing", "follow_up_no_response",
    "follow_up", "not_interested_ask_why", "no_thanks", "price_inquiry",
    "meeting_request", "demo_offer", "request_info", "sales_pitch",
    "soft_pitch", "cold_outreach", "general_response",
]


def detect_intent(msg: str) -> str:
    """
    Return the highest-scoring intent label, or 'general' if none match.
    Uses keyword scoring + priority order to break ties.
    Single canonical detector — used by main.py, pipeline.py, registry.
    """
    low = msg.lower()
    scores: dict[str, int] = {}

    for intent, phrases in INTENT_KEYWORDS.items():
        count = sum(1 for p in phrases if p in low)
        if count > 0:
            scores[intent] = count

    if not scores:
        return "general"

    best_score = max(scores.values())
    # Among tied top scorers, pick by priority order
    for intent in _PRIORITY_ORDER:
        if scores.get(intent, 0) == best_score:
            return intent

    # Fallback: first key with best score
    return max(scores, key=lambda k: scores[k])


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION TYPE CLASSIFIER
# Classifies a detected question into one of four types.
# Used by pipeline.py to decide how to handle the question in generation.
#
# Types:
#   factual_question     — price, availability, traffic, ownership facts
#   how_to_question      — setup, redirect, process, steps
#   clarification_question — explanation of terms or concepts
#   comparison_question  — domain vs alternatives, .com vs .net etc.
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_TYPES: dict[str, dict] = {
    "factual_question": {
        "description": "Asks for a specific fact: price, traffic, availability, ownership.",
        "patterns": [
            re.compile(r"\b(how much|price|cost|rate|fee|asking)\b", re.I),
            re.compile(r"\b(still available|available|for sale|taken|sold)\b", re.I),
            re.compile(r"\b(traffic|visitor|click|search|monthly|view|stat)\b", re.I),
            re.compile(r"\b(who owns|ownership|registered|whois|when (was|did))\b", re.I),
            re.compile(r"\b(how (long|many|often|soon))\b", re.I),
        ],
        "answer_guidance": (
            "Answer directly and specifically. "
            "State the fact first — price, stat, or availability — then one sentence of context. "
            "Do not pad with sales content before giving the answer."
        ),
    },
    "how_to_question": {
        "description": "Asks how to do something: setup, redirect, purchase, transfer.",
        "patterns": [
            re.compile(r"\b(how do i|how does|how to|how can i|how would)\b", re.I),
            re.compile(r"\b(redirect|forward|point|set.?up|configure|install)\b", re.I),
            re.compile(r"\b(step|process|procedure|walk me through|guide)\b", re.I),
            re.compile(r"\b(buy it|purchase|transfer|move|migrate)\b", re.I),
            re.compile(r"\b(technical|developer|it (guy|person|team))\b", re.I),
        ],
        "answer_guidance": (
            "Explain in plain English with simple steps. "
            "Use no jargon. Lead with 'here is how it works' then numbered steps. "
            "Confirm no new website or technical knowledge is needed."
        ),
    },
    "clarification_question": {
        "description": "Asks what something means or how a concept works.",
        "patterns": [
            re.compile(r"\b(what (is|does|exactly|do you mean)|what.{0,8}mean)\b", re.I),
            re.compile(r"\b(explain|clarify|elaborate|tell me more|what.{0,8}difference)\b", re.I),
            re.compile(r"\b(don.t understand|confused|not sure what|plain english|layman)\b", re.I),
            re.compile(r"\b(what is (a |an )?(domain|redirect|escrow|geo|keyword))\b", re.I),
        ],
        "answer_guidance": (
            "Explain clearly using plain language and a simple analogy. "
            "Keep it to 2–3 sentences. Avoid technical terms unless you define them inline."
        ),
    },
    "comparison_question": {
        "description": "Asks about differences, alternatives, or comparisons.",
        "patterns": [
            re.compile(r"\b(vs\.?|versus|compared? (to|with)|difference between|better than)\b", re.I),
            re.compile(r"\b(why (not|not just)|why (this|yours) (over|instead))\b", re.I),
            re.compile(r"\b(alternative|option|other (domain|choice|way)|instead of)\b", re.I),
            re.compile(r"\b(\.net|\.org|\.io|other extension|already own the \.)\b", re.I),
            re.compile(r"\b(what.{0,10}(better|best|difference|advantage))\b", re.I),
        ],
        "answer_guidance": (
            "Acknowledge the alternative directly, then explain the specific advantage. "
            "Be honest — do not dismiss the comparison. "
            "Focus on the one most relevant differentiator."
        ),
    },
}


def classify_question(question_text: str) -> str:
    """
    Classify a single question string into one of four types.

    Returns one of:
        'factual_question'
        'how_to_question'
        'clarification_question'
        'comparison_question'
        'general_question'  (fallback if nothing matches)

    Scoring: counts pattern matches per type, returns highest scorer.
    On tie: priority order is factual → how_to → clarification → comparison.
    """
    scores: dict[str, int] = {}
    for qtype, data in QUESTION_TYPES.items():
        count = sum(1 for pat in data["patterns"] if pat.search(question_text))
        if count > 0:
            scores[qtype] = count

    if not scores:
        return "general_question"

    best = max(scores.values())
    # Priority order on tie
    for qtype in ["factual_question", "how_to_question", "clarification_question", "comparison_question"]:
        if scores.get(qtype, 0) == best:
            return qtype

    return "general_question"


def classify_questions(questions: list[str]) -> dict[str, list[str]]:
    """
    Classify a list of question strings.
    Returns a dict mapping question_type → list of questions of that type.

    Example:
        {
          "factual_question":      ["How much is it?"],
          "how_to_question":       ["How do I redirect it?"],
          "clarification_question": [],
          "comparison_question":   [],
        }
    """
    result: dict[str, list[str]] = {qt: [] for qt in QUESTION_TYPES}
    result["general_question"] = []

    for q in questions:
        qtype = classify_question(q)
        result.setdefault(qtype, []).append(q)

    return result


def get_question_guidance(question_type: str) -> str:
    """Return the answer_guidance string for a given question type."""
    if question_type in QUESTION_TYPES:
        return QUESTION_TYPES[question_type]["answer_guidance"]
    return "Answer the question directly and clearly before continuing."
