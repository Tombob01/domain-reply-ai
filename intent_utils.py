"""
intent_utils.py — Shared intent detection data & helpers
=========================================================
Extracted from main.py to break the circular import between
main.py and intent_registry.py.

Both main.py and intent_registry.py import from here.
Nothing in this file imports from main.py or intent_registry.py.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# INTENT KEYWORDS
# Maps intent labels to trigger phrases found in prospect messages.
# ─────────────────────────────────────────────────────────────────────────────

INTENT_KEYWORDS: dict[str, list[str]] = {
    "no_thanks":      ["no thanks", "not interested", "pass", "decline", "don't need"],
    "price_inquiry":  ["how much", "price", "cost", "what are you asking", "rate", "fee"],
    "price_too_high": ["too expensive", "too high", "too much", "can't afford", "$10", "register for"],
    "negotiation":    ["offer", "counter", "negotiate", "lower", "discount", "best price", "bottom"],
    "follow_up":      ["follow up", "following up", "no reply", "no response", "checking in", "reminder"],
    "trust_issue":    ["scam", "fake", "not real", "legitimate", "trust", "verify", "proof",
                       "fraud", "suspicious", "doubt", "worried", "concern"],
    "have_website":   ["already have", "have a website", "have a domain", "don't need another"],
    "rank_well":      ["already rank", "rank fine", "seo is fine", "first page already"],
    "how_it_works":   ["how does", "redirect", "forward", "how do i", "technical", "it guy", "developer"],
    "why_buy":        ["why", "benefits", "what's the point", "explain", "value", "help my business"],
    "not_now":        ["later", "not now", "maybe", "not the right time", "future"],
    "partner":        ["partner", "team", "discuss", "boss", "colleague", "approval"],
    "agreed_no_pay":  ["agreed", "deal", "haven't paid", "no payment", "still waiting"],
    "payment_issue":  ["link not working", "payment failed", "can't checkout", "portal", "error"],
    "angry":          ["stop emailing", "spam", "harassment", "angry", "annoying", "unsubscribe"],
    "expired_owner":  ["used to be", "our domain", "how did you get", "previously owned"],
    "extension":      [".net", ".org", ".io", "other extension", "already own the"],
}


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_intent(msg: str) -> str:
    """Return the first matching intent label, or 'general' if none match."""
    low = msg.lower()
    for intent, phrases in INTENT_KEYWORDS.items():
        if any(p in low for p in phrases):
            return intent
    return "general"
