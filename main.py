"""
Domain Email Reply Generator — FastAPI + Ollama (qwen2.5:7b)
=============================================================
Ollama-first architecture — no external API keys required.
All generation routes through the local Ollama server at http://localhost:11434.

What's new in v7 — Ollama-first, zero cloud dependency:
  ✓ All generation uses local Ollama (qwen2.5:7b) — no Anthropic key needed
  ✓ template mode  → fully offline, zero AI calls
  ✓ hybrid mode    → template + Ollama polish
  ✓ ai mode        → Ollama direct generation
  ✓ Graceful 503 if Ollama is unreachable — no silent fallback
  ✓ Confidence scoring + routing recommendation (Phase 1/2/3 preserved)
  ✓ All QC, intent detection, pipeline analysis unchanged
  ✓ Claude support preserved as optional — set ANTHROPIC_API_KEY to re-enable
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, field_validator
from intent_utils import INTENT_KEYWORDS, detect_intent, classify_question, QUESTION_TYPES
from intent_registry import INTENT_REGISTRY, full_pipeline, registry_for
from pipeline import analyse, build_flow_instruction, InputAnalysis, print_analysis
from template_engine import build_template_reply, ai_polish_reply, detect_template_intent, TEMPLATE_INTENT_KEYWORDS
from quality_control import build_strategy_block, run_full_qc, check_variation_uniqueness, log_variation_check
from reply_strategy import (
    ReplyStrategy, StrategySignals, build_strategy, build_prompt_brief,
)
from strategy_eval import (
    evaluate_strategy_adherence, StrategyAnalytics, analytics as strategy_analytics,
)

# Broker memory — graceful fallback if module unavailable
try:
    from broker_memory import memory_db
    _MEMORY_AVAILABLE = memory_db.available
except Exception as _mem_err:
    print(f"[MemoryDB] Not available: {_mem_err} — lead history disabled")
    memory_db = None
    _MEMORY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL        = "qwen2.5:3b"           # Ollama model — local, no API key needed
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "180"))   # seconds per request

# ── Groq (cloud, free tier) ───────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_TIMEOUT     = int(os.getenv("GROQ_TIMEOUT", "30"))
# Groq model strings use "groq:" prefix so the router knows to use Groq, not Ollama
GROQ_MODELS: dict[str, str] = {
    "groq:llama3.1-70b":  "llama-3.3-70b-versatile",   # best quality (3.1-70b decommissioned)
    "groq:mixtral-8x7b":  "mixtral-8x7b-32768",         # fast + smart
}
GROQ_DEFAULT = "groq:llama3.1-70b"
EMBED_MODEL  = "voyage-3"
MAX_TOKENS        = 400
MAX_TOKENS_MULTI  = 1200   # higher limit when generating 3 variations
MIN_REPLY_WORDS   = 30     # quality guard lower bound
MAX_REPLY_WORDS   = 200    # quality guard upper bound
DEFAULT_SENDER    = "Alex" # default signature name
TOP_K        = 4
DATA_FILE    = Path(__file__).parent / "past_replies.json"
INDEX_FILE   = Path(__file__).parent / "embeddings_index.json"

FILLER_PHRASES = [
    "i hope this email finds you well", "trust you are doing great",
    "hope you are having a great day", "i hope you are doing well",
    "i hope all is well", "i wanted to reach out",
    "please do not hesitate", "feel free to reach out",
    "as per my last email", "going forward",
    "at the end of the day", "it is what it is",
]

# Phrases banned from model OUTPUT — injected into system prompts as AVOID list.
# These are marketing clichés that make replies sound like AI-generated ad copy.
_BANNED_OUTPUT_PHRASES = (
    "unlock the power, take your business to the next level, cutting-edge, "
    "revolutionary, maximize your online presence, supercharge, game-changing, "
    "leverage the full potential, skyrocket, don't miss out, act now, "
    "limited time, exclusive opportunity, transform your business, "
    "elevate your brand, boost your visibility, dominate your market, "
    "once in a lifetime, perfect domain, ideal fit, this won't last long, "
    "don't wait, incredible opportunity, powerful opportunity, before it's too late, "
    "we're excited, act quickly, amazing opportunity, unique opportunity"
)

# Shared writing rules injected into all four system prompts.
# Covers structure, tone, formatting, CTA, buy-now link, and banned language.
_WRITING_RULES = (
    "WRITING RULES — follow on every output:\n"
    "Structure: Short readable paragraphs of 1-3 sentences each. "
    "Space between paragraphs. Never write a wall of text. "
    "No bullet points or headers unless explicitly requested.\n"
    "Length: 150-220 words maximum unless user specifies otherwise. "
    "2-5 paragraphs. Clean readable sentences.\n"
    "Tone: Professional, confident, conversational, persuasive without being pushy. "
    "Sound like a real experienced broker — never robotic, never overhyped.\n"
    "CTA: End with one clear low-friction action. "
    "Examples: 'Let me know if you'd like pricing details.' / "
    "'Happy to send over the purchase link.' / "
    "'Let me know if you'd like to discuss further.'\n"
    "Buy-now link: If a listing or purchase link is provided, mention it naturally near the CTA. "
    "Low-pressure. Example: 'You can view the domain details here: [LINK]'\n"
    f"Never use these phrases: {_BANNED_OUTPUT_PHRASES}.\n"
    "Never invent company operations, employees, services, warranties, or business history "
    "not stated in the prompt. Treat any domain as a digital asset and SEO opportunity only.\n"
    "Write ONLY the requested output. No preamble, no self-commentary, no metadata."
)

# Keep _AVOID_FILLER_LINE as alias so any external references still work
_AVOID_FILLER_LINE = _WRITING_RULES

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIG FLAGS
# All optional AI work respects these flags.
# Set in .env or environment — no code changes needed to toggle.
# ─────────────────────────────────────────────────────────────────────────────

def _flag(name: str, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val in ("1", "true", "yes"):  return True
    if val in ("0", "false", "no"):  return False
    return default

ENABLE_VARIATIONS  = _flag("ENABLE_VARIATIONS",  False)  # False = 1 reply by default (faster)
ENABLE_AI_SCORING  = _flag("ENABLE_AI_SCORING",  False)  # False = skip Ollama scoring pass
ENABLE_AI_SUBJECT  = _flag("ENABLE_AI_SUBJECT",  False)  # False = use template subject (fast)
ENABLE_QC_REWRITE  = _flag("ENABLE_QC_REWRITE",  True)   # True  = allow Ollama QC fix pass

# Fast-path: how many variations to generate when ENABLE_VARIATIONS=false
_FAST_PATH_VARIATIONS = 1

# Intents where MISSING_CTA and TOO_MANY_PARAGRAPHS QC rules are suppressed
# because the reply is structural/informational, not persuasive
_QC_RELAXED_INTENTS = frozenset({
    "angry", "no_thanks", "follow_up", "follow_up_no_response",
    "general", "how_it_works", "request_info", "feature_explanation",
    "domain_metrics", "renewal_fees", "payment_method", "refund",
})

# ─────────────────────────────────────────────────────────────────────────────
# TIMING INSTRUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class _Timer:
    """Lightweight context manager for structured timing logs."""
    def __init__(self, label: str):
        self.label = label
        self._t0   = 0.0

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_):
        ms = int((time.monotonic() - self._t0) * 1000)
        print(f"[TIMING] {self.label}_ms={ms}")
        return False

    @staticmethod
    def log(label: str, ms: int) -> None:
        print(f"[TIMING] {label}_ms={ms}")

# ─────────────────────────────────────────────────────────────────────────────
# SITUATION INTENT DETECTION
# Maps natural-language situation descriptions to intent labels
# ─────────────────────────────────────────────────────────────────────────────

SITUATION_KEYWORDS: dict[str, list[str]] = {
    "follow_up":          ["no response", "no reply", "didn't reply", "hasn't responded",
                           "no answer", "silence", "ghost", "days ago", "week ago",
                           "checking in", "follow up", "following up", "chasing"],
    "sales_pitch":        ["first contact", "initial outreach", "cold email", "introduce",
                           "presenting", "pitch", "new prospect", "first time", "reaching out"],
    "re_engagement":      ["cold lead", "went cold", "lost contact", "stopped replying",
                           "months ago", "long time", "reconnect", "revive", "dormant",
                           "inactive", "old lead", "previous conversation"],
    "objection_handling": ["hesitant", "unsure", "not convinced", "on the fence",
                           "needs convincing", "doubtful", "skeptical", "thinking about it",
                           "considering", "not sure if"],
    "no_thanks":          ["said not interested", "said no", "declined", "rejected", "turned down"],
    "price_inquiry":      ["asked about price", "wants to know cost", "asked how much",
                           "price question", "cost question"],
    "price_too_high":     ["said too expensive", "price too high", "over budget", "too much money"],
    "negotiation":        ["wants to negotiate", "made an offer", "counter offer", "haggling"],
    "trust_issue":        ["thinks it's a scam", "doesn't trust", "suspicious", "worried about fraud"],
    "have_website":       ["has a website", "already has site", "existing website"],
    "agreed_no_pay":      ["agreed but not paid", "accepted but no payment", "deal agreed"],
    "angry":              ["angry", "upset", "spam complaint", "wants to unsubscribe"],
}

# ─────────────────────────────────────────────────────────────────────────────
# PITCH INTENSITY
# ─────────────────────────────────────────────────────────────────────────────

INTENT_TO_INTENSITY: dict[str, str] = {
    "follow_up":          "low",
    "sales_pitch":        "high",
    "re_engagement":      "medium",
    "objection_handling": "medium",
    "no_thanks":          "low",
    "price_inquiry":      "medium",
    "price_too_high":     "medium",
    "negotiation":        "medium",
    "trust_issue":        "low",
    "have_website":       "medium",
    "rank_well":          "medium",
    "why_buy":            "high",
    "not_now":            "low",
    "agreed_no_pay":      "low",
    "angry":              "low",
    "general":            "medium",
}

VALUE_PROPOSITIONS: dict[str, str] = {
    "low": (
        "If relevant, briefly mention one concrete benefit: "
        "owning a geo-targeted domain means every local search for that service "
        "could land on their business — it's a permanent digital asset, not an ad spend."
    ),
    "medium": (
        "Weave in 2-3 of these value points naturally:\n"
        "- Geo-targeted domains rank faster in local search (city + service = exact-match keyword)\n"
        "- Redirects to their existing site in minutes — no rebuilding needed\n"
        "- One-time purchase: no ongoing ad spend, no monthly fees\n"
        "- If a competitor buys it first, that traffic goes to them permanently\n"
        "Lead with the one most relevant to their situation."
    ),
    "high": (
        "Build a compelling case using the most relevant angles for this prospect:\n"
        "- Local SEO: Exact-match domains carry strong geo-signals for 'near me' searches\n"
        "- Competitor risk: Once it's gone, a rival owns it — and gets that traffic forever\n"
        "- Easy setup: Redirect takes minutes. Their existing site stays exactly as it is\n"
        "- Permanent asset: Unlike ad spend, a domain is owned outright after one purchase\n"
        "- Local credibility: A city-specific .com signals relevance to local customers\n"
        "Pick 2-3 of these — the ones that actually fit this prospect's situation."
    ),
}

SOFT_CTA: dict[str, str] = {
    "low":    "End with a low-friction close: 'Happy to answer any questions — just let me know either way.'",
    "medium": "End with a clear but gentle CTA: 'Would it help if I sent over more details?' or 'Shall I hold this for you while you decide?'",
    "high":   "End with a direct CTA: give them the next concrete step — a link, an offer to call, or a specific question that moves things forward.",
}

URGENCY_LAYER: dict[str, str] = {
    "low":    "",
    "medium": "Optional: add one sentence of light urgency only if it fits naturally — e.g. 'This domain does get enquiries occasionally.' Do not manufacture pressure.",
    "high":   "Add genuine urgency where true: domains are publicly listed and can sell at any time. Frame it as information, not pressure: 'Just so you know, this one does get interest.'",
}

# ─────────────────────────────────────────────────────────────────────────────
# INTENT + TONE + PERSUASION MAPS
# ─────────────────────────────────────────────────────────────────────────────

# INTENT_KEYWORDS and detect_intent are imported from intent_utils above.
# The canonical full keyword set lives in intent_utils.py.

TONE_INSTRUCTIONS: dict[str, str] = {
    "professional and persuasive": (
        "Confident, professional, value-led. "
        "Open with the strongest benefit for their situation. "
        "State your position clearly — do not hedge. "
        "AVOID: corporate filler, over-apologising, weak qualifiers like 'might' or 'could possibly'."
    ),
    "warm and friendly": (
        "Conversational and genuine — as if talking to a neighbour you want to help. "
        "Use short sentences. Ask one genuine question. "
        "AVOID: overselling, exclamation marks, anything that sounds like a sales script."
    ),
    "firm but respectful": (
        "Hold your position clearly. State your number or stance once, cleanly, without apology. "
        "Acknowledge their point in one sentence, then restate your position. "
        "AVOID: caving, over-explaining, repeating the same point twice."
    ),
    "concise and direct": (
        "2 to 3 sentences maximum. Every word earns its place. "
        "State the point, the ask, the close — in that order. "
        "AVOID: any sentence that could be deleted without losing meaning."
    ),
    "empathetic and understanding": (
        "Lead with genuine acknowledgment of their concern — not dismissal of it. "
        "Name their objection specifically before addressing it. "
        "AVOID: pivoting to your pitch before they feel heard."
    ),
    "highly persuasive and compelling": (
        "Make the strongest honest case for why they should act now. "
        "Lead with the most compelling outcome for their specific business. "
        "Use concrete language: numbers, local search, competitor risk. "
        "AVOID: vague claims, generic benefits, anything a competitor could also say."
    ),
    "urgent and time-sensitive": (
        "Create genuine urgency — domains are publicly listed and can sell any time. "
        "Frame urgency as information, not pressure: 'Just so you know, this does get enquiries.' "
        "AVOID: manufactured deadlines, pressure tactics, anything that feels dishonest."
    ),
    "premium and exclusive": (
        "Position the domain as a premium asset with real scarcity value. "
        "Use language that reflects investment, not just cost. "
        "Reference the long-term value: permanent traffic, brand authority, competitor barrier. "
        "AVOID: discounting, apologising for the price, comparing to cheap alternatives."
    ),
}

INTENT_RULES: dict[str, str] = {
    "follow_up": (
        "They've gone quiet — but silence usually means busy, not gone. Keep it very short. "
        "Don't re-pitch. Give them an easy out ('just let me know either way') — "
        "paradoxically, this gets more replies than pushing. Two or three sentences max."
    ),
    "sales_pitch": (
        "This is first contact. Make it feel relevant to them specifically, not like a blast. "
        "Lead with a concrete outcome for their type of business, mention that they don't need "
        "to build a new site (this kills the main objection before it's raised), "
        "and end with one clear next step."
    ),
    "re_engagement": (
        "You're coming back after a gap — acknowledge it briefly, then move on. "
        "Don't over-explain the re-contact. Give them a fresh angle or new reason "
        "to reconsider. Keep it short: earn the right to a longer conversation."
    ),
    "objection_handling": (
        "Name their specific hesitation before addressing it — this is what builds trust. "
        "Then reframe: most objections are really requests for more information. "
        "Turn their concern into a question they can answer for themselves. "
        "End by inviting them to share what's really holding them back."
    ),
    "price_too_high": (
        "Don't dismiss the $10 GoDaddy comparison — they're not wrong, they just don't "
        "understand premium domain pricing yet. Validate it briefly, then explain the actual "
        "difference: existing search signal, exact-match keyword, competitor risk. "
        "Reframe it as investment vs expense. Invite their number rather than defending yours."
    ),
    "negotiation": (
        "Their first offer is almost never their limit — it's an anchoring attempt. "
        "Acknowledge it in one sentence without accepting or rejecting it. "
        "Counter with a specific number, not a range (ranges invite them to pick the low end). "
        "Give one brief reason. Create mild movement pressure without a fake deadline. "
        "Always leave the next step clear."
    ),
    "trust_issue": (
        "Address the concern head-on in the first sentence — don't warm up to it. "
        "Name a specific, verifiable trust mechanism immediately: "
        "DAN.com listing, GoDaddy marketplace, Escrow.com payment. "
        "Give them something they can check independently right now. "
        "Confidence is the cure here — don't get defensive."
    ),
    "have_website": (
        "Lead with the fact that they don't need to change anything about their current site. "
        "Explain the redirect in plain terms: five minutes, their existing site stays the same. "
        "Then pivot to competitor risk: this domain exists regardless — the question is who owns it."
    ),
    "no_thanks": (
        "Accept the decision graciously without arguing. "
        "Ask one soft optional question about their reasoning — sometimes this reverses the decision. "
        "Leave the door genuinely open."
    ),
    "price_inquiry": (
        "State the price clearly in the first sentence — don't bury it. "
        "Follow with one concrete reason that price reflects real value. "
        "End with how to proceed."
    ),
    "agreed_no_pay": (
        "Don't re-sell — they already agreed. Reference the agreement briefly, "
        "create mild urgency (the domain is publicly listed), "
        "and make the payment step as frictionless as possible."
    ),
    "angry": (
        "Two sentences only. Apologise sincerely in one, offer immediate removal in the other. "
        "No pitch. No explanation. No defending yourself."
    ),
    "expired_owner": (
        "Explain calmly how expired domains enter the open market — they may not know this is legal. "
        "Mention the direct traffic if known. "
        "Frame the offer as an opportunity to reclaim what they had."
    ),
    "rank_well": (
        "Acknowledge their ranking genuinely — it's a real achievement. "
        "Then pivot: even strong rankings can be threatened by a competitor "
        "owning the exact-match domain. Frame ownership as insurance, not a replacement."
    ),
    "why_buy": (
        "Lead with the single strongest benefit for their specific business type. "
        "Use concrete local search language. End with a direct question."
    ),
    "not_now": (
        "Respect the timing without pushing back. "
        "Offer a specific check-back date. "
        "Plant mild urgency as information: the domain is publicly listed."
    ),
    # ── General / informational intents ─────────────────────────────────────
    "general": (
        "Answer the question directly and factually. "
        "Keep any sales angle minimal — earn trust with useful information first."
    ),
    "how_it_works": (
        "Explain the process in plain English. Cover purchase → transfer → redirect → done. "
        "No jargon. Imagine explaining to a busy non-technical business owner."
    ),
    "request_info": (
        "Give the specific information requested clearly. "
        "Don't pad with sales content. Offer to answer follow-up questions."
    ),
    "feature_explanation": (
        "Explain the feature in simple, practical terms with a concrete example "
        "relevant to their business or industry."
    ),
    "domain_metrics": (
        "Answer factually on age, traffic, authority, backlinks, search volume. "
        "Translate technical metrics into business value. "
        "If metrics aren't available, say so and point them to where they can verify."
    ),
    "renewal_fees": (
        "State the annual renewal cost clearly (typically $10-15/year at major registrars). "
        "Distinguish the one-time purchase price from the annual renewal fee. "
        "Recommend reputable registrars."
    ),
    "payment_method": (
        "List accepted payment methods clearly. "
        "Mention escrow options: Dan.com, Escrow.com. "
        "Explain the transfer briefly: payment → push/auth code → 5-7 days."
    ),
}

CLOSING_TECHNIQUES: dict[str, str] = {
    "no_thanks":          "End with a genuinely open door — make it feel easy and non-awkward to come back later. No pitch.",
    "price_inquiry":      "End with a direct CTA: give them the next concrete step (buy link, how to proceed, or offer to hold it).",
    "price_too_high":     "End by inviting their number: 'What would feel fair to you?' — this removes the standoff and keeps the negotiation alive.",
    "negotiation":        "End with a specific counter number and a soft time frame: 'I can hold it at [counter] for the next 48 hours.' Never leave the close open-ended.",
    "trust_issue":        "End by naming one verifiable action they can take right now — a link to the listing, an escrow service, or a reference they can check independently.",
    "have_website":       "End with the competitor fear close: 'The only risk is someone else getting there first.' Make it information, not pressure.",
    "follow_up":          "End with a binary choice: interested or not — make it genuinely easy to say either. 'Just a yes or no is fine.'",
    "sales_pitch":        "End with the most compelling outcome for their specific business, then one clear action step — not two options.",
    "re_engagement":      "End with a fresh hook — something new or different from the first conversation to justify the re-contact.",
    "objection_handling": "End with an open question that invites them to share the real concern beneath the stated objection.",
    "agreed_no_pay":      "End with the purchase link and a friendly soft deadline. Make the action one click away.",
    "angry":              "End with a genuine release — no pitch, no future contact offer unless they ask. Just acknowledge and let go.",
    "why_buy":            "End with a vivid picture of the outcome: 'Every local search could end with your site.' Then one clear next step.",
    "not_now":            "End with a specific check-back date so it doesn't get forgotten: 'Mind if I follow up in [timeframe]?'",
    "rank_well":          "End with the ownership angle: 'Owning it costs less than defending against someone who does.'",
}

ALTERNATIVE_ANGLES: list[dict] = [
    {"label": "Lead with competitor risk",
     "instruction": "Open by highlighting the risk of a competitor buying this domain first. Make that the central argument."},
    {"label": "Lead with SEO value",
     "instruction": "Open by explaining the keyword and search traffic value. Focus on how this domain could drive free organic traffic."},
    {"label": "Lead with social proof",
     "instruction": "Open by referencing how other businesses in similar cities/industries have used geo-targeted domains successfully."},
]

# ─────────────────────────────────────────────────────────────────────────────
# EMAIL TYPE PRESETS
# Each preset shapes tone, structure, CTA style, urgency, and persuasion
# strategy. Injected into all three prompt builders when email_preset is set.
# Defaults to None — existing behaviour fully preserved when not set.
# ─────────────────────────────────────────────────────────────────────────────

EMAIL_PRESET_INSTRUCTIONS: dict[str, dict] = {
    "cold_outreach": {
        "label":       "Cold Outreach",
        "structure":   "Short personalised opener → introduce the domain naturally → explain business value (branding, SEO, local authority) → simple low-friction CTA.",
        "tone":        "Confident, professional, conversational. Sound like a real person reaching out — not a mass campaign.",
        "cta":         "End with one soft question: 'Would this be of interest?' or 'Happy to share more details if useful.'",
        "urgency":     "No urgency. This is a first introduction — leave room for a conversation.",
        "persuasion":  "Lead with relevance, not pressure. Explain why this specific domain fits their specific business.",
        "subject_hint":"Short and specific — mention the domain or a clear benefit. Examples: 'Re: [Domain]' / 'A domain that fits [City][Industry]'",
    },
    "warm_outreach": {
        "label":       "Warm Outreach",
        "structure":   "Reference prior contact or expressed interest → reintroduce the domain briefly → reinforce the strongest value point → direct CTA.",
        "tone":        "Warmer and more direct than cold outreach. They already know you — skip the full introduction.",
        "cta":         "More direct: 'Would you like me to send the purchase link?' or 'Ready to move forward when you are.'",
        "urgency":     "Light natural urgency: 'Still available if you'd like to revisit.'",
        "persuasion":  "Build on what they already expressed interest in. Don't repeat the full pitch.",
        "subject_hint":"Reference previous contact: 'Following up — [Domain]' or 'Re: [Domain] — still available'",
    },
    "follow_up": {
        "label":       "Follow-Up",
        "structure":   "Brief acknowledgment of previous outreach → one sentence on why you're following up → single value reminder → simple CTA.",
        "tone":        "Shorter than the original. Friendly, not pushy. Assume they're busy.",
        "cta":         "'Just wanted to check if you had a chance to look at this.' or 'Happy to answer any questions.'",
        "urgency":     "Gentle: 'Still available and worth a look.'",
        "persuasion":  "Minimal. Remind, don't re-pitch. One value point maximum.",
        "subject_hint":"'Following up — [Domain]' or 'Quick check-in on [Domain]'",
    },
    "final_follow_up": {
        "label":       "Final Follow-Up",
        "structure":   "Acknowledge this is the last message → brief closing value statement → genuine release with an open door.",
        "tone":        "Calm, confident, respectful. Not desperate. This is a graceful exit.",
        "cta":         "'If timing changes, feel free to reach out.' or 'Happy to reconnect whenever makes sense.'",
        "urgency":     "Natural scarcity only: 'Planning to open this to other buyers soon.' Never manufactured pressure.",
        "persuasion":  "None. One clear statement of value, then let go.",
        "subject_hint":"'Last note on [Domain]' or 'Closing out — [Domain]'",
    },
    "negotiation": {
        "label":       "Negotiation",
        "structure":   "Acknowledge their position → justify your price with one clear reason → counter or hold → keep the door open.",
        "tone":        "Calm, confident, experienced. Never defensive or desperate.",
        "cta":         "'Let me know what works for you.' or 'Happy to discuss — what's your best number?'",
        "urgency":     "Real scarcity only: 'I do have other interest in this domain.' Never invented.",
        "persuasion":  "Value-based. Justify the price logically — don't just repeat it. One strong reason.",
        "subject_hint":"'Re: [Domain] — pricing discussion' or 'Re: your offer on [Domain]'",
    },
    "counter_offer": {
        "label":       "Counter Offer",
        "structure":   "Acknowledge their offer briefly → give your counter with one clear reason → name your number specifically → invite their response.",
        "tone":        "Direct, businesslike, fair. Not aggressive — just clear.",
        "cta":         "'Let me know if that works.' or 'Open to discussing if you have a different number in mind.'",
        "urgency":     "Optional soft deadline: 'I can hold this for 48 hours at that price.'",
        "persuasion":  "State your counter and the reason. One sentence. Don't over-justify.",
        "subject_hint":"'Re: [Domain] — counteroffer' or 'Counter: [Domain] at [Price]'",
    },
    "seo_pitch": {
        "label":       "SEO Pitch",
        "structure":   "Open with the SEO angle → explain how the exact-match domain affects local search ranking → connect to their business specifically → CTA.",
        "tone":        "Knowledgeable but accessible. Explain SEO value clearly without being technical.",
        "cta":         "'Happy to walk you through how this works if useful.' or 'Let me know if you'd like the details.'",
        "urgency":     "Competitor risk: 'Once a competitor owns this, the SEO advantage goes to them permanently.'",
        "persuasion":  "Lead with search visibility and 'near me' traffic. Make it concrete: this domain = those searches.",
        "subject_hint":"'[Domain] — local SEO advantage' or 'How [Domain] could rank for your city'",
    },
    "brandability_pitch": {
        "label":       "Brandability Pitch",
        "structure":   "Open with the brand angle → explain memorability, credibility, and authority → connect to their specific market → CTA.",
        "tone":        "Creative but professional. Focus on long-term brand value.",
        "cta":         "'Let me know if you'd like to discuss.' or 'Happy to share the listing.'",
        "urgency":     "Light: 'A name like this is rare in your market.'",
        "persuasion":  "Lead with what customers will think when they see the domain. Make the brand case, not the SEO case.",
        "subject_hint":"'A domain that fits [Brand/City]' or '[Domain] — brand opportunity'",
    },
    "buy_it_now": {
        "label":       "Buy-It-Now Outreach",
        "structure":   "Direct opener → domain and price upfront → one clear value statement → purchase link → simple CTA.",
        "tone":        "Efficient and direct. Respect their time — give them everything they need to act immediately.",
        "cta":         "Include a direct purchase link: 'You can purchase directly here: [LINK]' or 'Direct link: [LINK]'",
        "urgency":     "Clear availability: 'Available now at [Price].'",
        "persuasion":  "Lead with the offer. Don't over-explain. Price + domain + one reason + link.",
        "subject_hint":"'[Domain] — available at [Price]' or 'Buy [Domain] — [Price]'",
    },
}


def _build_preset_block(preset_key: Optional[str]) -> str:
    """
    Return a formatted prompt instruction block for the given email preset.
    Returns empty string when preset is None — preserves existing behaviour.
    Used by all three prompt builders: build_reply_prompt, build_situation_prompt,
    build_reply_prompt_ai.
    """
    if not preset_key:
        return ""
    preset = EMAIL_PRESET_INSTRUCTIONS.get(preset_key)
    if not preset:
        return ""
    return (
        f"EMAIL TYPE: {preset['label']}\n"
        f"Structure: {preset['structure']}\n"
        f"Tone guidance: {preset['tone']}\n"
        f"CTA: {preset['cta']}\n"
        f"Urgency: {preset['urgency']}\n"
        f"Persuasion: {preset['persuasion']}\n"
        f"Subject hint: {preset['subject_hint']}\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING INDEX
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingIndex:
    def __init__(self):
        self.entries: list[dict] = []
        self.built_at: float     = 0.0
        self.kb_size: int        = 0
        self.ready: bool         = False

    def is_stale(self, n: int) -> bool:
        return n != self.kb_size

    def load_cache(self) -> bool:
        if not INDEX_FILE.exists():
            return False
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.entries  = d["entries"]
            self.built_at = d.get("built_at", 0.0)
            self.kb_size  = d.get("kb_size", 0)
            self.ready    = True
            print(f"[Embed] Cache loaded — {self.kb_size} entries")
            return True
        except Exception as e:
            print(f"[Embed] Cache load failed: {e}")
            return False

    def save_cache(self) -> None:
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"built_at": self.built_at, "kb_size": self.kb_size, "entries": self.entries}, f, indent=2)

    def build(self, replies: list[dict], api_key: str) -> None:
        import voyageai
        print(f"[Embed] Building for {len(replies)} replies…")
        vo    = voyageai.Client(api_key=api_key)
        texts = [f"{r.get('category','')} | {r['customer_message']} | {r['reply']}" for r in replies]
        embs: list[list[float]] = []
        for i in range(0, len(texts), 50):
            embs.extend(vo.embed(texts[i:i+50], model=EMBED_MODEL, input_type="document").embeddings)
        self.entries  = [{**r, "embedding": e} for r, e in zip(replies, embs)]
        self.built_at = time.time()
        self.kb_size  = len(replies)
        self.ready    = True
        self.save_cache()
        print(f"[Embed] Done — {len(self.entries)} entries saved")

    def query(self, text: str, api_key: str, top_k: int) -> list[dict]:
        import voyageai
        vo  = voyageai.Client(api_key=api_key)
        vec = vo.embed([text], model=EMBED_MODEL, input_type="query").embeddings[0]
        return sorted(self.entries, key=lambda e: _cosine(vec, e["embedding"]), reverse=True)[:top_k]


_index = EmbeddingIndex()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma  = math.sqrt(sum(x * x for x in a))
    mb  = math.sqrt(sum(y * y for y in b))
    return dot / (ma * mb) if ma and mb else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Domain Email Reply Generator",
    description="FastAPI + Ollama (qwen2.5:7b) — local AI, no external API keys required",
    version="7.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def on_startup():
    loaded  = _index.load_cache()
    replies = load_replies()
    if loaded and not _index.is_stale(len(replies)):
        print("[Startup] Semantic index ready.")
    else:
        print("[Startup] Index stale/missing — will build on first request.")
        _index.ready = False

    # ── PART 4: Run QC test harness on startup ────────────────────────────────
    # Prints a pass/fail report to logs so you can spot intent detection
    # regressions every time the server restarts. Non-blocking.
    try:
        from quality_control import run_tests
        run_tests(verbose=True)
    except Exception as e:
        print(f"[QC Tests] Could not run test harness: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    customer_message: str
    tone: Optional[str]         = "professional and persuasive"
    api_key: Optional[str]      = None
    domain_name: Optional[str]  = None
    asking_price: Optional[str] = None
    sender_name: Optional[str]  = None
    prospect_name: Optional[str]= None
    num_variations: int         = 3
    mode: Optional[str]         = "ai"
    model: Optional[str]        = None
    email_preset: Optional[str] = None   # cold_outreach | warm_outreach | follow_up | etc.
    lead_id: Optional[int]      = None   # broker memory — fetch lead history for context

    @field_validator("customer_message")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_message cannot be empty.")
        return v.strip()


class SituationRequest(BaseModel):
    """Situation-based generation — describe what's happening, not what was said."""
    situation: str
    tone: Optional[str]            = "professional and persuasive"
    api_key: Optional[str]         = None
    domain_name: Optional[str]     = None
    asking_price: Optional[str]    = None
    force_intent: Optional[str]    = None
    force_intensity: Optional[str] = None
    include_urgency: bool          = False
    sender_name: Optional[str]     = None
    prospect_name: Optional[str]   = None
    num_variations: int            = 3
    model: Optional[str]           = None
    email_preset: Optional[str]    = None   # cold_outreach | warm_outreach | follow_up | etc.
    lead_id: Optional[int]         = None   # broker memory

    @field_validator("situation")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("situation cannot be empty.")
        return v.strip()


class ReplyResult(BaseModel):
    reply: str
    subject: Optional[str] = None          # generated subject line
    label: Optional[str] = None            # e.g. "Safe", "Persuasive", "Short"
    confidence_score: int = 75
    confidence_reason: str = ""
    angle: Optional[str] = None
    quality_report: Optional[dict] = None  # heuristic_score_reply() output — None if not run


class GenerateResponse(BaseModel):
    subject: str
    replies: list[ReplyResult]
    detected_intent: str
    retrieval_method: str
    similar_examples_used: list[dict]
    model_used: str
    model_requested: Optional[str] = None   # what the frontend sent; None if omitted
    tone_applied: str
    pipeline_debug: Optional[dict] = None


class SituationResponse(BaseModel):
    subject: str
    replies: list[ReplyResult]
    detected_intent: str
    pitch_intensity: str
    situation_interpreted: str
    model_used: str
    model_requested: Optional[str] = None
    tone_applied: str
    pipeline_debug: Optional[dict] = None


class AlternativesResponse(BaseModel):
    alternatives: list[ReplyResult]
    detected_intent: str
    model_used: str


class AddReplyRequest(BaseModel):
    category: str
    customer_message: str
    reply: str

    @field_validator("category", "customer_message", "reply")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip()


class SaveReplyRequest(BaseModel):
    """Save a generated reply into the KB as a reusable template."""
    category: str
    customer_message: str        # original situation / prospect message
    reply: str                   # the generated reply to save
    subject: Optional[str] = None
    make_template: bool = False  # if True, also adds to template_engine KB

    @field_validator("category", "customer_message", "reply")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip()


class TemplateRequest(BaseModel):
    customer_message: str
    domain_name: Optional[str]  = None
    asking_price: Optional[str] = None
    force_intent: Optional[str] = None
    ai_polish: bool              = False
    api_key: Optional[str]       = None
    tone: str                    = "professional and persuasive"
    response_length: Optional[str]   = "medium"
    length_instructions: Optional[str] = None
    include_urgency: bool            = False
    force_intensity: Optional[str]   = None
    model: Optional[str]             = None
    email_preset: Optional[str]      = None   # cold_outreach | warm_outreach | follow_up | etc.
    lead_id: Optional[int]           = None   # broker memory

    @field_validator("customer_message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_message cannot be empty.")
        return v.strip()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_lead_context(lead_id: Optional[int]) -> Optional[str]:
    """Fetch lead history summary for prompt injection. Returns None if unavailable."""
    if not lead_id or not _MEMORY_AVAILABLE or memory_db is None:
        return None
    return memory_db.lead_summary(lead_id)


def load_replies() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_replies(data: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# detect_intent() is imported from intent_utils — no local definition needed.

def detect_situation_intent(situation: str) -> str:
    """Checks SITUATION_KEYWORDS first (proactive), then falls back to standard detection."""
    low = situation.lower()
    for intent, phrases in SITUATION_KEYWORDS.items():
        if any(p in low for p in phrases):
            return intent
    return detect_intent(situation)


def strip_filler(text: str) -> str:
    low = text.lower()
    for phrase in FILLER_PHRASES:
        if phrase in low:
            text = re.compile(re.escape(phrase), re.IGNORECASE).sub("", text)
    return text.strip()


_STOP = {
    "i","me","my","we","you","your","the","a","an","is","it","in","on","at",
    "to","for","of","and","or","but","have","has","had","do","did","was",
    "were","be","been","am","are","this","that","with","from","by","as","so",
    "if","not","just","will","would","could","should","can","may","might",
    "please","hi","hello","dear","regards","thanks","thank","yes","no",
}


def _tok(t: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", t.lower())) - _STOP


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _keyword_retrieve(msg: str, replies: list[dict], intent: str, k: int) -> list[dict]:
    q = _tok(msg)
    scored = []
    for r in replies:
        s = 0.5 * _jaccard(q, _tok(r.get("category","")) | _tok(r.get("customer_message",""))) \
          + 0.5 * _jaccard(q, _tok(r.get("reply","")))
        if intent != "general" and intent.replace("_","") in r.get("category","").replace("_",""):
            s += 0.25
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for s, r in scored[:k] if s > 0.01]


def retrieve(msg: str, replies: list[dict], intent: str, api_key: str, k: int = TOP_K) -> tuple[list[dict], str]:
    if _index.is_stale(len(replies)):
        if not api_key:
            pass  # No Voyage key — skip build, use keyword retrieval
        else:
            try:
                _index.build(replies, api_key)
            except Exception as e:
                print(f"[Embed] Build failed: {e}")
                _index.ready = False

    if _index.ready and _index.entries:
        try:
            results = _index.query(msg, api_key, top_k=k * 2)
            if intent != "general":
                matched   = [e for e in results if intent.replace("_","") in e.get("category","").replace("_","")]
                unmatched = [e for e in results if e not in matched]
                return (matched + unmatched)[:k], "semantic"
            return results[:k], "semantic"
        except Exception as e:
            print(f"[Embed] Query failed: {e}")

    return _keyword_retrieve(msg, replies, intent, k), "keyword"


def get_ollama_client() -> "OllamaClient":
    """
    Return the shared OllamaClient singleton.
    Imported here (not at top level) to keep the import lazy and testable.
    """
    from ollama_client import get_default_client
    return get_default_client(
        model    = MODEL,
        base_url = OLLAMA_BASE_URL,
        timeout  = OLLAMA_TIMEOUT,
    )


def _require_ollama(result: Optional[str], label: str = "") -> str:
    """
    Assert that an Ollama generate() call returned a non-empty result.
    Raises HTTP 503 if Ollama is unreachable or returned nothing.
    Never falls back to Claude — the error is explicit and actionable.
    """
    if result:
        return result
    detail = (
        f"Ollama is unavailable or returned an empty response"
        + (f" ({label})" if label else "")
        + ". Make sure Ollama is running: `ollama serve` and model is pulled: `ollama pull qwen2.5:7b`"
    )
    raise HTTPException(status_code=503, detail=detail)


def call_ollama(system: str, user: str, label: str = "", model: str = MODEL) -> str:
    """
    Call Ollama for a single-turn generation.
    model defaults to global MODEL for backward compatibility.
    Raises HTTP 503 if Ollama is unavailable.
    """
    client = _get_client_for_model(model)
    print(f"[AI_BACKEND] backend={('groq' if model.startswith('groq:') else 'ollama')} model={model} label={label or 'generate'}")
    result = client.generate(prompt=user, system=system, temperature=0.7, max_tokens=MAX_TOKENS)
    return _require_ollama(result, label)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS — model-aware behavioural briefs
# 3b: tighter, more explicit constraints (small model needs precise instruction)
# 7b: richer reasoning scaffold (larger model can use nuance)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_3B = (
    "You are a professional domain broker with years of experience selling premium .com domains "
    "to local businesses. You write outbound sales emails, follow-ups, and negotiation replies "
    "that sound natural, credible, and human — never templated, never spammy.\n\n"
    "When you write outreach: start with a short personalised opener, introduce the domain naturally, "
    "explain why it matters for their business (branding, SEO, local authority, lead generation), "
    "and end with one simple low-friction CTA.\n\n"
    "When you write follow-ups: keep it shorter than the first email, acknowledge the prior message, "
    "add light scarcity if it fits naturally, end with a direct simple CTA.\n\n"
    "When you handle negotiation: stay calm and confident, justify value logically, "
    "never sound desperate, never drop price immediately.\n\n"
    + _WRITING_RULES
)

SYSTEM_PROMPT_7B = (
    "You are an experienced domain broker who has sold hundreds of premium .com domains to local "
    "businesses across every industry. You know how geo-targeted domains affect local SEO, how "
    "buyers think about price, and how to move a stalled deal forward without being pushy.\n\n"
    "OUTREACH: Short personalised opener. Introduce the domain naturally. Explain the value — "
    "branding, SEO, local authority, memorability, lead generation. End with one simple CTA. "
    "Mention a buy-now link or marketplace listing naturally near the CTA if available.\n\n"
    "FOLLOW-UP: Shorter than the first email. Acknowledge prior contact. "
    "Light urgency if it fits genuinely. Direct simple CTA.\n\n"
    "NEGOTIATION: Calm and confident. Justify value logically. Never desperate. "
    "Never drop price immediately. Leave room for discussion.\n\n"
    "You read exactly what the prospect said and respond to that specifically — "
    "not to a generic template. You never invent numbers, stats, or business details "
    "not given to you.\n\n"
    + _WRITING_RULES
)

# Legacy alias — endpoints select the right one via effective_model
SYSTEM_PROMPT = SYSTEM_PROMPT_3B


def _select_system_prompt(model: str) -> str:
    """Return the appropriate system prompt for the given model.
    Used by Hybrid mode — always returns the broker/sales identity."""
    return SYSTEM_PROMPT_7B if "7b" in model else SYSTEM_PROMPT_3B


# ─────────────────────────────────────────────────────────────────────────────
# AI MODE SYSTEM PROMPTS
# Separate identity from Hybrid mode.
# No broker role. No CTA mandate. No persuasion scaffolds.
# Context-aware: adapts to educational, advisory, strategic, and conversational
# requests without assuming every interaction is a sales opportunity.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_AI_3B = (
    "You are a professional domain broker and industry advisor. "
    "You handle real situations — outreach, follow-ups, objections, negotiations, "
    "informational questions, and strategy requests.\n\n"
    "When you draft a domain email: follow professional domain sales writing — "
    "short paragraphs, natural tone, one clear CTA, no hype, no bullet points.\n\n"
    "When you answer a question: give a direct, specific, accurate answer. "
    "No generic padding. No unnecessary sales content.\n\n"
    "When you give strategy or advice: be practical and specific to the situation given.\n\n"
    "Read exactly what the person asked and respond to that — not to what a template expects.\n\n"
    + _WRITING_RULES
)

SYSTEM_PROMPT_AI_7B = (
    "You are a professional domain broker and advisor with deep experience in domain sales, "
    "negotiation, local SEO, pricing, and business outreach. "
    "You help brokers handle real situations — prospect replies, pricing standoffs, "
    "objections, follow-ups, re-engagement, and domain industry questions.\n\n"
    "WHEN DRAFTING A DOMAIN EMAIL:\n"
    "Follow professional domain sales structure. "
    "Outreach: personalised opener → domain introduction → business value → simple CTA. "
    "Follow-up: shorter, acknowledge prior contact, light genuine urgency, direct CTA. "
    "Negotiation: calm, confident, value-justified, never desperate, never immediately discounting.\n\n"
    "WHEN ANSWERING A QUESTION:\n"
    "Answer directly and accurately first. "
    "No sales padding before the answer. No invented facts. "
    "Educational questions get educational answers.\n\n"
    "WHEN GIVING STRATEGY OR ADVICE:\n"
    "Be specific to the exact situation described. "
    "Think through it properly before responding. "
    "Give a practical recommendation, not a generic framework.\n\n"
    "You read exactly what was said and respond to that. "
    "You don't pad. You don't inject sales content where it doesn't belong. "
    "You give people what they actually need, in the format they actually need it.\n\n"
    + _WRITING_RULES
)


def _select_system_prompt_for_mode(model: str, mode: str = "hybrid") -> str:
    """
    Return the appropriate system prompt based on both model and mode.

    mode='hybrid'  → broker/sales identity (existing behaviour, unchanged)
    mode='ai'      → context-aware advisor identity (no sales assumption)
    mode='template'→ not called for template mode (no Ollama)
    """
    if mode == "ai":
        return SYSTEM_PROMPT_AI_7B if "7b" in model else SYSTEM_PROMPT_AI_3B
    # hybrid / template / unknown → existing broker prompts
    return SYSTEM_PROMPT_7B if "7b" in model else SYSTEM_PROMPT_3B

SCORE_SYSTEM = (
    "You are a sales email quality reviewer. "
    "You assess domain-selling email replies and return a JSON object with two fields: "
    "'score' (integer 0-100) and 'reason' (one plain English sentence explaining the score). "
    "Criteria: Does it directly address the specific situation? Is the tone appropriate? "
    "Is there a clear next step? Is it free of filler and placeholders? "
    "Does it avoid being generic? Return ONLY the JSON object, nothing else."
)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE FRAME CLASSIFIER
# Pure heuristic — classifies what kind of response the input needs.
# Called once per request inside _build_context_frame. Zero latency.
#
# Frames:
#   direct_reply        — prospect message needing a ready-to-send reply
#   strategic_advice    — user wants to know how to handle a situation
#   educational_answer  — factual or explanatory question
#   negotiation_analysis— analyse an offer, leverage, or counteroffer
#   brainstorming       — user wants multiple ideas or options
#   mixed_request       — contains 2+ frame signals
# ─────────────────────────────────────────────────────────────────────────────

_FRAME_SIGNALS: dict[str, list[str]] = {
    "strategic_advice": [
        "how should i", "what should i", "what's the best way",
        "how do i handle", "how do i respond", "best approach",
        "what would you do", "advice on", "tips for", "strategy for",
        "should i accept", "should i counter", "should i follow up",
        "how to deal with", "how to approach",
    ],
    "educational_answer": [
        "what is", "what does", "how does", "explain", "tell me about",
        "what are", "can you explain", "what exactly", "plain english",
        "how do domains", "what is a redirect", "what is escrow",
        "what does geo", "why do domains", "how does seo",
    ],
    "negotiation_analysis": [
        "analyse", "analyze", "what do you think of this offer",
        "is this a good offer", "is this fair", "should i take",
        "they offered", "he offered", "she offered", "their offer",
        "counter with", "what should i counter", "leverage",
        "negotiation position", "are they serious", "low ball",
    ],
    "brainstorming": [
        "give me ideas", "brainstorm", "options for", "what are my options",
        "different ways", "multiple approaches", "several ideas",
        "what could i say", "alternatives to", "other ways to",
        "come up with", "list of", "ideas for",
    ],
    "direct_reply": [
        "write a reply", "draft a reply", "write an email", "reply to this",
        "respond to this", "draft an email", "write back", "send a reply",
        "write something", "help me reply", "what should i write",
    ],
}


def _classify_response_frame(message: str) -> str:
    """
    Classify the message into a response frame using keyword scoring.
    Returns one of: direct_reply | strategic_advice | educational_answer |
                    negotiation_analysis | brainstorming | mixed_request | inferred_reply

    'inferred_reply' = no explicit frame signals but message looks like a
    direct prospect quote — generate a reply directly.
    """
    low = message.lower()
    scores: dict[str, int] = {}

    for frame, signals in _FRAME_SIGNALS.items():
        count = sum(1 for s in signals if s in low)
        if count > 0:
            scores[frame] = count

    if not scores:
        # No explicit frame signal — infer from context
        # If it reads like a prospect talking (short, contains keywords like
        # price/domain/not interested) treat as direct_reply
        return "inferred_reply"

    if len(scores) >= 2:
        return "mixed_request"

    return max(scores, key=lambda k: scores[k])


# ─────────────────────────────────────────────────────────────────────────────
# NEGOTIATION STATE DETECTOR
# Pure heuristic — detects the negotiation posture of the prospect.
# Runs inside _build_context_frame. Zero latency.
#
# States:
#   low_anchor_offer    — suspect offer well below market value
#   soft_interest       — interest expressed but no commitment
#   hard_rejection      — firm no with emotional language
#   curiosity           — information-seeking without intent signals
#   hesitation          — interested but stalling
#   urgency_signal      — prospect has a deadline or time pressure
#   active_negotiation  — explicit counter / back-and-forth in progress
#   none                — no negotiation signals detected
# ─────────────────────────────────────────────────────────────────────────────

_NEG_SIGNALS: dict[str, list[str]] = {
    "low_anchor_offer": [
        "i'll give you", "i'll offer", "how about", "i can do",
        "would you take", "would you accept", "$50", "$100", "$150",
        "$200", "fifty", "hundred dollars", "two hundred",
        "that's all i can", "maximum i can", "most i can pay",
    ],
    "soft_interest": [
        "maybe", "might", "possibly", "could be interesting",
        "let me think", "i'll consider", "sounds interesting",
        "could work", "not sure yet", "thinking about it",
        "i'll get back", "will think about", "keep me posted",
    ],
    "hard_rejection": [
        "absolutely not", "no way", "never", "stop contacting",
        "not a chance", "waste of time", "don't contact me again",
        "remove me", "unsubscribe", "this is spam",
    ],
    "curiosity": [
        "just curious", "wondering", "just asking", "out of interest",
        "purely hypothetical", "just want to know", "no obligation",
        "not committing", "just exploring",
    ],
    "hesitation": [
        "not sure", "i don't know", "hard to say", "difficult to decide",
        "need to think", "have to ask", "need approval",
        "need to check", "not the right time", "maybe later",
        "let me speak to", "need to discuss",
    ],
    "urgency_signal": [
        "need it by", "asap", "as soon as possible", "urgent",
        "deadline", "by end of", "this week", "immediately",
        "right away", "quickly", "time sensitive",
    ],
    "active_negotiation": [
        "counter", "counteroffer", "counter offer", "my offer is",
        "i offered", "we can meet", "split the difference",
        "meet in the middle", "final offer", "last offer",
        "best i can do", "take it or leave it",
    ],
}


def _detect_negotiation_state(message: str, asking_price: Optional[str]) -> str:
    """
    Detect the negotiation posture from the message.
    Returns one of the state labels above or 'none'.
    Uses keyword scoring — highest scorer wins.
    """
    low = message.lower()
    scores: dict[str, int] = {}

    for state, signals in _NEG_SIGNALS.items():
        count = sum(1 for s in signals if s in low)
        if count > 0:
            scores[state] = count

    # Hard rejection always wins on any match
    if scores.get("hard_rejection", 0) > 0:
        return "hard_rejection"

    if not scores:
        return "none"

    return max(scores, key=lambda k: scores[k])


def _negotiation_guidance(state: str, asking_price: Optional[str]) -> str:
    """
    Return a plain-English instruction for the model based on the
    detected negotiation state. No AI call — pure lookup.
    """
    price_str = f" Your asking price is {asking_price}." if asking_price else ""

    guidance = {
        "low_anchor_offer": (
            f"Prospect has made a low-anchor offer.{price_str} "
            "Do NOT accept or come close to it. "
            "Acknowledge the offer in one sentence, then counter with a specific figure. "
            "Give one brief reason for your counter — not a lecture. "
            "Keep the door open; do not close the negotiation."
        ),
        "soft_interest": (
            "Prospect is interested but uncommitted. "
            "Do not push harder — that triggers retreat. "
            "Reduce friction: answer any implied questions, make the next step feel small and easy. "
            "One gentle forward-moving question at the end."
        ),
        "hard_rejection": (
            "Prospect has firmly rejected contact. "
            "Do NOT pitch. Do NOT defend. "
            "Apologise briefly (one sentence), offer immediate removal, and stop. "
            "Two sentences maximum."
        ),
        "curiosity": (
            "Prospect is exploring, not buying. "
            "Treat this as an information request — answer factually and helpfully. "
            "A light CTA at the end is fine; anything heavier will feel premature."
        ),
        "hesitation": (
            "Prospect is interested but stalling. "
            "Identify the most likely stall reason from the message (approval needed, budget, timing). "
            "Address that specific friction point — not a generic reassurance. "
            "End with a low-commitment next step."
        ),
        "urgency_signal": (
            "Prospect has a time constraint. "
            "Match their urgency: be concise, lead with the most relevant information, "
            "give them exactly what they need to act. "
            "Do not waste their time with build-up."
        ),
        "active_negotiation": (
            f"Active back-and-forth negotiation in progress.{price_str} "
            "Hold your position — do not fold pre-emptively. "
            "Counter with a specific number (not a range). "
            "One brief reason, one clear next step. "
            "Keep momentum: never leave the close open-ended."
        ),
    }
    return guidance.get(state, "")


def _build_context_frame(message: str, analysis: "InputAnalysis",
                         asking_price: Optional[str]) -> str:
    """
    Infer prospect psychology, response frame, and negotiation state from
    the raw message. Surfaces everything as plain English for the model.
    No AI call — pure heuristics. Zero added latency.

    Replaces the raw debug_block injection which caused metadata bleed.
    Now also classifies:
      - response_frame      (direct_reply / strategic_advice / etc.)
      - negotiation_state   (low_anchor / soft_interest / hesitation / etc.)
    And emits a [REASONING] debug log for every request.
    """
    lines: list[str] = []
    msg_low = message.lower()

    # ── 1. Response frame ────────────────────────────────────────────────────
    response_frame = _classify_response_frame(message)

    frame_instructions = {
        "strategic_advice": (
            "RESPONSE TYPE: Strategic Advice — "
            "explain your recommended approach first, then provide a suggested reply. "
            "Do not open with the reply draft."
        ),
        "educational_answer": (
            "RESPONSE TYPE: Educational — "
            "answer the question factually and completely first. "
            "Keep sales content minimal. Earn trust with useful information."
        ),
        "negotiation_analysis": (
            "RESPONSE TYPE: Negotiation Analysis — "
            "analyse the offer or situation first: assess the leverage, "
            "buyer intent, and realistic outcome range. "
            "Then provide a recommended reply or counter."
        ),
        "brainstorming": (
            "RESPONSE TYPE: Brainstorming — "
            "provide 3 distinct concise ideas or approaches. "
            "Label each clearly. Keep each idea to 2-3 sentences."
        ),
        "mixed_request": (
            "RESPONSE TYPE: Mixed — "
            "this message contains multiple request types. "
            "Address each part in order: information first, strategy second, reply draft last."
        ),
        "direct_reply": (
            "RESPONSE TYPE: Direct Reply — "
            "output a ready-to-send reply. No preamble, no strategy explanation."
        ),
        "inferred_reply": (
            "RESPONSE TYPE: Inferred Reply — "
            "this appears to be a direct prospect message. "
            "Write a ready-to-send reply that addresses it specifically."
        ),
    }

    if response_frame in frame_instructions:
        lines.append(frame_instructions[response_frame])

    # ── 2. Negotiation state ─────────────────────────────────────────────────
    neg_state = _detect_negotiation_state(message, asking_price)
    neg_guidance = _negotiation_guidance(neg_state, asking_price)
    if neg_guidance:
        lines.append(neg_guidance)

    # ── 3. Emotional tone signals ────────────────────────────────────────────
    if any(w in msg_low for w in ["unfortunately", "can't afford", "tight budget",
                                   "struggling", "small business", "just starting",
                                   "don't have much", "limited budget", "very small"]):
        lines.append(
            "Prospect appears genuinely budget-constrained. "
            "Lead with value before price. Do not open with the price or defend it."
        )
    elif any(w in msg_low for w in ["maybe", "might", "i guess", "i suppose",
                                     "possibly", "not sure", "kind of", "i think so"]):
        if neg_state not in ("low_anchor_offer", "active_negotiation"):
            lines.append(
                "Prospect is uncertain, not opposed. "
                "They need reassurance and a smaller next step — not a harder sell."
            )
    elif any(w in msg_low for w in ["love it", "great", "sounds good", "interested",
                                     "tell me more", "how do i", "let's do it",
                                     "ready to", "want to proceed"]):
        lines.append(
            "Prospect is warm and engaged. "
            "Move toward closing — do not re-sell what they already accept. "
            "Make the next step frictionless."
        )
    elif any(w in msg_low for w in ["confused", "don't understand", "not sure what you mean",
                                     "what do you mean", "can you explain"]):
        lines.append(
            "Prospect is confused or needs clarification. "
            "Explain clearly in plain English first. No pitch until they understand."
        )

    # ── 4. Question count signal ─────────────────────────────────────────────
    if analysis.has_questions:
        q_count = len(analysis.questions)
        if q_count > 1:
            lines.append(
                f"Prospect asked {q_count} questions — answer every one of them "
                "before any pitch content. Skipping a question ends the deal."
            )
        else:
            lines.append(
                "Prospect asked a direct question — answer it first, specifically. "
                "Do not replace the answer with generic sales content."
            )

    # ── 5. Multi-intent / ambiguity signal ───────────────────────────────────
    if analysis.has_multiple_intents and analysis.ambiguity_level in ("medium", "high"):
        secondaries = ", ".join(s.replace("_", " ") for s in analysis.secondary_intents[:2])
        lines.append(
            f"Message has mixed signals ({secondaries}). "
            "Address the primary intent but do not ignore the secondary concern — "
            "it is often the real reason they haven't committed."
        )

    # ── 6. Low confidence fallback ───────────────────────────────────────────
    fallback_mode = False
    if hasattr(analysis, "confidence") and analysis.confidence < 0.3:
        fallback_mode = True
        lines.append(
            "FALLBACK MODE: Intent confidence is low. "
            "Do not force intent-specific rules aggressively. "
            "Prioritise contextual relevance — read the message carefully and respond to what is actually being said."
        )

    # ── 7. Generic reply prevention ──────────────────────────────────────────
    lines.append(
        "RELEVANCE CHECK: Your reply must directly address the specific situation above. "
        "If any sentence could appear in a reply to a completely different message, delete it."
    )

    # ── 8. Debug log ─────────────────────────────────────────────────────────
    print(
        f"[REASONING] frame={response_frame} "
        f"neg_state={neg_state} "
        f"fallback={fallback_mode} "
        f"questions={len(analysis.questions) if analysis.has_questions else 0} "
        f"ambiguity={getattr(analysis, 'ambiguity_level', 'unknown')}"
    )

    if not lines:
        return ""

    return (
        "PROSPECT PSYCHOLOGY & RESPONSE FRAMING "
        "(inferred — use to guide tone, structure, and content):\n"
        + "\n".join(f"• {l}" for l in lines)
        + "\n"
    )


def build_reply_prompt(message, intent, examples, tone, domain_name, asking_price,
                       retrieval_method, angle_instruction=None,
                       analysis: Optional[InputAnalysis] = None,
                       email_preset: Optional[str] = None,
                       lead_context: Optional[str] = None):
    tone_inst   = TONE_INSTRUCTIONS.get(tone, f"Tone: {tone}.")
    method_note = "by semantic meaning" if retrieval_method == "semantic" else "by keyword"

    domain_block = ""
    if domain_name or asking_price:
        parts = []
        if domain_name:  parts.append(f"Domain being sold: {domain_name}")
        if asking_price: parts.append(f"Asking price: {asking_price}")
        domain_block = "\n".join(parts) + "\n"

    ex_block = ""
    if examples:
        ex_block = f"REFERENCE EXAMPLES (retrieved {method_note} — style only, do NOT copy):\n\n"
        for i, ex in enumerate(examples, 1):
            ex_block += (f"  [{i}] {ex.get('category','general')}\n"
                         f"       Situation: {ex['customer_message']}\n"
                         f"       Reply: {ex['reply']}\n\n")

    rules       = INTENT_RULES.get(intent, "Respond naturally and professionally.")
    closing     = CLOSING_TECHNIQUES.get(intent, "End with a clear next step.")
    angle_block = f"\nAngle to use: {angle_instruction}\n" if angle_instruction else ""
    preset_block  = _build_preset_block(email_preset)
    lead_block    = f"LEAD HISTORY:\n{lead_context}\n" if lead_context else ""

    if analysis is None:
        analysis = analyse(message)

    context_frame    = _build_context_frame(message, analysis, asking_price)
    question_section = f"\n{analysis.question_block}\n" if analysis.question_block else ""

    return (
        f"{domain_block}"
        f"{ex_block}"
        f"{angle_block}"
        f"{preset_block}"
        f"{lead_block}"
        f"Message from prospect:\n\"{strip_filler(message)}\"\n\n"
        f"{context_frame}"
        f"{question_section}"
        f"How to handle this: {rules}\n\n"
        f"How to close: {closing}\n\n"
        f"Tone: {tone_inst}\n\n"
        f"Write a ready-to-send email body. No filler openers, no placeholders. "
        f"Sound like a real person.\n\nWrite the reply:"
    )

def build_situation_prompt(situation, intent, intensity, examples, tone,
                           domain_name, asking_price, retrieval_method, include_urgency=False,
                           analysis: Optional[InputAnalysis] = None,
                           email_preset: Optional[str] = None,
                           lead_context: Optional[str] = None):
    """Prompt for situation-mode — injects pitch intensity, value props, soft CTAs, urgency."""
    tone_inst   = TONE_INSTRUCTIONS.get(tone, f"Tone: {tone}.")
    method_note = "by meaning" if retrieval_method == "semantic" else "by keyword"

    # ── Context ───────────────────────────────────────────────────────────────
    context_parts = []
    if domain_name:  context_parts.append(f"Domain: {domain_name}")
    if asking_price: context_parts.append(f"Asking price: {asking_price}")
    context_line = "  ".join(context_parts) + "\n" if context_parts else ""

    # ── Examples ──────────────────────────────────────────────────────────────
    ex_block = ""
    if examples:
        ex_block = f"A few past replies for style reference (retrieved {method_note} — do not copy):\n"
        for i, ex in enumerate(examples, 1):
            ex_block += f"  {i}. [{ex.get('category','general')}] {ex['reply'][:120]}…\n"
        ex_block += "\n"

    # ── Intent guidance ───────────────────────────────────────────────────────
    rule = INTENT_RULES.get(intent, "Respond naturally and professionally.")
    intent_guidance = f"For this kind of situation: {rule}\n\n"

    # ── Value propositions (intensity-calibrated) ─────────────────────────────
    value_p = VALUE_PROPOSITIONS.get(intensity, VALUE_PROPOSITIONS["medium"])
    cta     = SOFT_CTA.get(intensity, SOFT_CTA["medium"])
    urgency = URGENCY_LAYER.get(intensity, "") if include_urgency else ""

    intensity_note = {
        "low":    "Keep value mentions brief and non-pushy.",
        "medium": "Weave in 2-3 value points naturally — be helpful, not salesy.",
        "high":   "Make the strongest relevant case, but keep it human.",
    }.get(intensity, "Be helpful and clear.")

    if analysis is None:
        analysis = analyse(situation)

    # ── Debug ─────────────────────────────────────────────────────────────────
    preset_block = _build_preset_block(email_preset)
    lead_block   = f"LEAD HISTORY:\n{lead_context}\n" if lead_context else ""

    print(
        f"[REASONING] mode=situation intent={intent} intensity={intensity} "
        f"urgency={include_urgency} preset={email_preset or 'none'}"
    )

    return (
        f"{context_line}"
        f"{ex_block}"
        f"{preset_block}"
        f"{lead_block}"
        f"Situation (described by the broker):\n\"{strip_filler(situation)}\"\n\n"
        f"{intent_guidance}"
        f"Pitch level: {intensity_note}\n"
        f"Value points to weave in naturally:\n{value_p}\n\n"
        f"How to close: {cta}\n"
        + (f"Urgency note: {urgency}\n\n" if urgency else "\n")
        + f"Tone: {tone_inst}\n\n"
        f"Write a ready-to-send email body. No filler openers, no placeholders, "
        f"no subject line. Sound like a real person who knows their business.\n\n"
        f"Write the email:"
    )



# ─────────────────────────────────────────────────────────────────────────────
# AI MODE PROMPT BUILDER
# Parallel to build_reply_prompt() — used only when effective_mode == "ai".
#
# Key differences from build_reply_prompt():
#   - No CLOSING_TECHNIQUES injection (CTA only when context supports it)
#   - No strategy_block (replaces with context_frame which is already richer)
#   - No "One clear next step" quality gate
#   - INTENT_RULES injected only for sales/negotiation intents
#   - quality_gate is context-aware: checks relevance, not CTA presence
#   - Response frame drives the output shape, not a fixed email structure
#
# Existing build_reply_prompt() is NOT changed — used by hybrid mode as-is.
# ─────────────────────────────────────────────────────────────────────────────

# Intents that warrant full sales scaffolding even in AI mode
_AI_SALES_INTENTS = {
    "sales_pitch", "cold_outreach", "follow_up", "follow_up_no_response",
    "follow_up_after_pricing", "follow_up_after_interest", "re_engagement",
    "negotiation", "price_negotiation", "objection_handling",
    "agreed_no_pay", "not_interested_ask_why", "soft_pitch",
    "value_reminder", "competitor_comparison",
}

# Intents that are purely informational — suppress sales scaffolding entirely
_AI_INFO_INTENTS = {
    "general", "general_response", "how_it_works", "request_info",
    "feature_explanation", "domain_metrics", "identity", "why_buy",
    "renewal_fees", "payment_method", "trust_building", "development",
}


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION-STAGE INTELLIGENCE
# Pure heuristics — zero model calls.
# Detects stage from outreach count, offer history, intent, and message text.
# ─────────────────────────────────────────────────────────────────────────────

# All valid conversation stages
_CONV_STAGES = {
    "first_outreach",   # no prior contact
    "warm_lead",        # replied with interest, no offer yet
    "negotiation",      # offer/counter-offer in progress
    "stalled",          # no reply for a while / gone cold
    "final_follow_up",  # last-chance outreach
    "accepted",         # deal agreed
    "rejected",         # definitively declined
    "counteroffer",     # specific counter-offer situation
    "unknown",          # insufficient data
}

# Stage → compact prompt instruction block
_STAGE_INSTRUCTIONS: dict[str, dict] = {
    "first_outreach": {
        "tone":       "Curious and professional. No pressure.",
        "length":     "Concise — 2 short paragraphs maximum.",
        "cta":        "One low-friction question. Not 'buy now'.",
        "urgency":    "None. Do not manufacture urgency.",
        "posture":    "Introduction-focused. Plant the seed; don't close.",
        "avoid":      "Avoid price anchoring too early. Avoid pushy language.",
    },
    "warm_lead": {
        "tone":       "Engaged and direct. They've shown interest — match their energy.",
        "length":     "Medium — 2-3 paragraphs.",
        "cta":        "Stronger CTA. Move toward a next step: call, offer, or decision.",
        "urgency":    "Light natural urgency is fine — domain availability is real.",
        "posture":    "Value positioning. Make the case clearly without over-explaining.",
        "avoid":      "Avoid re-introducing the domain as if they've never heard of it.",
    },
    "negotiation": {
        "tone":       "Calm, confident, logical. No desperation.",
        "length":     "Medium — tight. Every sentence earns its place.",
        "cta":        "Specific next step: a counter figure or a clear decision prompt.",
        "urgency":    "Factual only — 'domain is listed publicly' if true. Never fake.",
        "posture":    "Hold the pricing position. Counter with a specific number, not a range. "
                      "One brief reason. Keep momentum — never leave it open-ended.",
        "avoid":      "Avoid folding pre-emptively. Avoid over-explaining your price. "
                      "Avoid phrases like 'I understand your budget constraints'.",
    },
    "counteroffer": {
        "tone":       "Business-like and precise.",
        "length":     "Short to medium. Get to the counter fast.",
        "cta":        "Name your counter-price and ask if they'd like to proceed.",
        "urgency":    "Factual only.",
        "posture":    "Acknowledge their offer in one sentence. Counter with a specific figure. "
                      "Brief rationale. Close with a direct question.",
        "avoid":      "Avoid lengthy justification. Avoid hedging on price.",
    },
    "stalled": {
        "tone":       "Light, low-friction, no pressure.",
        "length":     "Short — 2 paragraphs maximum.",
        "cta":        "Easy yes/no question. Remove all friction from replying.",
        "urgency":    "Very light or none. Don't lecture them.",
        "posture":    "Re-engage gently. Reference prior interest briefly. "
                      "Make replying feel easy and natural.",
        "avoid":      "Avoid 'just following up'. Avoid guilt language. "
                      "Avoid re-pitching the whole value proposition.",
    },
    "final_follow_up": {
        "tone":       "Respectful, non-pushy, slightly closing.",
        "length":     "Short — 2 paragraphs.",
        "cta":        "Closing question or a graceful exit offer.",
        "urgency":    "Optional, very soft — 'happy to close this thread if timing isn't right'.",
        "posture":    "This is the last contact. Be professional. Leave the door open.",
        "avoid":      "Avoid manipulation. Avoid fake FOMO. Avoid desperation.",
    },
    "accepted": {
        "tone":       "Professional, warm, action-oriented.",
        "length":     "Short — next steps only.",
        "cta":        "Payment link / escrow setup / document signing.",
        "urgency":    "Natural — deal is agreed, move it forward.",
        "posture":    "Focus entirely on completing the transaction smoothly.",
        "avoid":      "Avoid re-selling. They've decided.",
    },
    "rejected": {
        "tone":       "Gracious, brief.",
        "length":     "Very short — 2-3 sentences.",
        "cta":        "Leave the door open lightly, or close professionally.",
        "urgency":    "None.",
        "posture":    "Accept the decision. No counter-pitch.",
        "avoid":      "Avoid pleading. Avoid repeating value points.",
    },
}


def detect_conversation_stage(
    intent:         str,
    message:        str,
    outreach_count: int           = 0,
    offers:         list[dict]    = None,   # list of offer_log dicts from broker_memory
    lead_stage:     Optional[str] = None,   # stored stage from leads table
    asking_price:   Optional[str] = None,
) -> tuple[str, str]:
    """
    Infer conversation stage from available signals. Zero model calls.

    Returns (stage_label, explanation) where stage_label is one of _CONV_STAGES.

    Priority:
    1. Explicit stored stage from broker_memory (highest trust)
    2. Offer history signals
    3. Intent + message signals
    4. Outreach count signals
    """
    offers = offers or []
    msg_low = message.lower()

    # ── 1. Hard signals from stored lead stage ────────────────────────────────
    if lead_stage == "agreed":
        return "accepted", "Lead stage is 'agreed' in broker memory."
    if lead_stage == "closed":
        return "accepted", "Lead stage is 'closed' in broker memory."

    # ── 2. Offer history analysis ─────────────────────────────────────────────
    sent_offers     = [o for o in offers if o.get("direction") == "sent"]
    received_offers = [o for o in offers if o.get("direction") == "received"]

    if received_offers and sent_offers:
        return "counteroffer", (
            f"Active counter-offer exchange: {len(received_offers)} received, "
            f"{len(sent_offers)} sent."
        )
    if received_offers and not sent_offers:
        # They made an offer, we haven't countered yet
        return "negotiation", (
            f"Prospect has made {len(received_offers)} offer(s) — awaiting counter."
        )

    # ── 3. Message-level signals ──────────────────────────────────────────────
    # Hard rejection
    rejection_signals = [
        "not interested", "no thanks", "remove me", "unsubscribe",
        "stop emailing", "do not contact", "please stop",
    ]
    if any(s in msg_low for s in rejection_signals):
        return "rejected", "Prospect has explicitly declined or requested removal."

    # Acceptance signals
    acceptance_signals = [
        "let's do it", "i'll take it", "we agree", "send the invoice",
        "send payment", "how do i pay", "ready to proceed", "we have a deal",
        "accepted", "agreed",
    ]
    if any(s in msg_low for s in acceptance_signals):
        return "accepted", "Prospect has signalled acceptance."

    # Negotiation signals (current message)
    offer_signals = [
        "offer", "willing to pay", "could do", "how about", "would you accept",
        "can you do", "best price", "counter", "meet in the middle",
    ]
    if any(s in msg_low for s in offer_signals):
        return "negotiation", "Current message contains an offer or counter-offer signal."

    # Warm interest signals
    warm_signals = [
        "interested", "tell me more", "sounds good", "like this", "makes sense",
        "could work", "let me think", "good option", "we're considering",
    ]
    if any(s in msg_low for s in warm_signals):
        return "warm_lead", "Prospect has expressed genuine interest."

    # ── 4. Intent-based signals ───────────────────────────────────────────────
    intent_stage_map = {
        "negotiation":           "negotiation",
        "price_negotiation":     "negotiation",
        "price_too_high":        "negotiation",
        "follow_up":             "stalled",
        "follow_up_no_response": "stalled",
        "re_engagement":         "stalled",
        "cold_outreach":         "first_outreach",
        "sales_pitch":           "first_outreach",
        "agreed_no_pay":         "accepted",
        "angry":                 "rejected",
        "no_thanks":             "rejected",
    }
    if intent in intent_stage_map:
        mapped = intent_stage_map[intent]
        return mapped, f"Intent '{intent}' maps to stage '{mapped}'."

    # ── 5. Outreach count heuristics ─────────────────────────────────────────
    if outreach_count == 0:
        return "first_outreach", "No prior outreach logged."
    if outreach_count == 1:
        return "warm_lead", "One prior outreach — prospect is in early dialogue."
    if outreach_count >= 4:
        return "final_follow_up", f"{outreach_count} outreach attempts — treat as final follow-up."
    if outreach_count >= 2:
        return "stalled", f"{outreach_count} prior outreach attempts with no clear progression."

    return "unknown", "Insufficient signals to determine stage."


def _stage_prompt_block(stage: str, explanation: str) -> str:
    """
    Build a compact, targeted prompt block for the detected conversation stage.
    Injected into build_reply_prompt_ai() — kept deliberately short to avoid prompt bloat.
    """
    if stage not in _STAGE_INSTRUCTIONS or stage == "unknown":
        return ""

    inst = _STAGE_INSTRUCTIONS[stage]
    stage_label = stage.replace("_", " ").title()

    lines = [
        f"CONVERSATION STAGE: {stage_label}",
        f"  Tone: {inst['tone']}",
        f"  Length: {inst['length']}",
        f"  CTA: {inst['cta']}",
        f"  Urgency: {inst['urgency']}",
        f"  Posture: {inst['posture']}",
        f"  Avoid: {inst['avoid']}",
    ]
    return "\n".join(lines) + "\n"


def _offer_intelligence(
    offers:       list[dict],
    asking_price: Optional[str],
    message:      str,
) -> str:
    """
    Analyse offer history and current message to produce a compact negotiation
    intelligence note for the prompt. Zero model calls.

    Returns empty string if no useful signals.
    """
    if not offers and not asking_price:
        return ""

    lines: list[str] = []
    msg_low = message.lower()

    # ── Parse asking price ────────────────────────────────────────────────────
    ask_num: Optional[float] = None
    if asking_price:
        digits = re.sub(r"[^\d.]", "", asking_price)
        try:
            ask_num = float(digits)
        except ValueError:
            pass

    # ── Extract latest received offer from history ────────────────────────────
    received = sorted(
        [o for o in offers if o.get("direction") == "received"],
        key=lambda o: o.get("offered_at", 0),
    )
    sent = sorted(
        [o for o in offers if o.get("direction") == "sent"],
        key=lambda o: o.get("offered_at", 0),
    )

    # ── Offer ratio analysis ──────────────────────────────────────────────────
    if received and ask_num and ask_num > 0:
        latest_offer = received[-1].get("amount", 0)
        ratio = latest_offer / ask_num

        if ratio < 0.25:
            lines.append(
                f"OFFER INTELLIGENCE: Prospect's latest offer (${latest_offer:,.0f}) is "
                f"{int(ratio*100)}% of your asking price — a low anchor. "
                "Do NOT move close to it. Counter firmly with a specific figure near asking. "
                "Acknowledge briefly, counter confidently."
            )
        elif ratio < 0.60:
            lines.append(
                f"OFFER INTELLIGENCE: Prospect offered ${latest_offer:,.0f} — "
                f"{int(ratio*100)}% of asking. Serious intent but a significant gap. "
                "Counter at no lower than 80% of asking. Offer one reason, one next step."
            )
        elif ratio < 0.85:
            lines.append(
                f"OFFER INTELLIGENCE: Prospect offered ${latest_offer:,.0f} — "
                f"{int(ratio*100)}% of asking. Close gap. "
                "You can counter modestly or hold — don't fold to asking without a small counter."
            )
        elif ratio >= 0.85:
            lines.append(
                f"OFFER INTELLIGENCE: Prospect offered ${latest_offer:,.0f} — "
                f"{int(ratio*100)}% of asking. Near full ask. "
                "This is a serious buyer. Close efficiently — counter minimally or accept. "
                "Don't risk losing the deal over a small gap."
            )

    # ── Offer trend ───────────────────────────────────────────────────────────
    if len(received) >= 2:
        trend = received[-1].get("amount", 0) - received[-2].get("amount", 0)
        if trend > 0:
            lines.append(
                f"OFFER TREND: Prospect is moving up (last increase: +${trend:,.0f}). "
                "Keep negotiating — they have room to move."
            )
        elif trend == 0:
            lines.append(
                "OFFER TREND: Prospect's offer has not changed. "
                "They may be anchoring — give them a reason to move or call the stall."
            )

    # ── Buying signals in current message ────────────────────────────────────
    buying_signals = [
        "ready", "proceed", "move forward", "let's do it", "sounds good",
        "works for me", "agreed", "deal", "transfer", "payment",
    ]
    has_buying = any(s in msg_low for s in buying_signals)
    if has_buying:
        lines.append(
            "BUYING SIGNAL: Current message contains commitment language. "
            "Do not re-pitch. Move toward closing: payment, escrow, or next action."
        )

    # ── Hesitation signals ────────────────────────────────────────────────────
    hesitation_signals = [
        "need to think", "let me check", "ask my partner", "speak to my",
        "not sure yet", "maybe", "possibly", "considering",
    ]
    has_hesitation = any(s in msg_low for s in hesitation_signals)
    if has_hesitation:
        lines.append(
            "HESITATION SIGNAL: Prospect is stalling on a decision. "
            "Identify the friction point from their message and address it specifically. "
            "Offer a low-commitment next step — not a hard close."
        )

    return "\n".join(lines) + "\n" if lines else ""


def _get_lead_context_rich(lead_id: Optional[int]) -> dict:
    """
    Fetch full lead data from broker_memory for stage detection and prompt injection.
    Returns a dict with: summary_text, outreach_count, outreach, offers, lead_stage.
    Always safe to call — returns empty defaults if memory unavailable.
    """
    empty = {"summary_text": None, "outreach_count": 0, "outreach": [], "offers": [], "lead_stage": None}
    if not lead_id or not _MEMORY_AVAILABLE or memory_db is None:
        return empty
    try:
        history        = memory_db.full_history(lead_id)
        outreach_list  = history.get("outreach", [])
        offers         = history.get("offers", [])
        lead           = history.get("lead", {})
        lead_stage     = lead.get("stage") if lead else None
        summary_text   = memory_db.lead_summary(lead_id)
        return {
            "summary_text":   summary_text,
            "outreach_count": len(outreach_list),
            "outreach":       outreach_list,
            "offers":         offers,
            "lead_stage":     lead_stage,
        }
    except Exception as e:
        print(f"[STAGE_INTEL] _get_lead_context_rich failed: {e}")
        return empty


def retrieve_stage_aware(
    message: str,
    replies: list[dict],
    intent:  str,
    api_key: str,
    top_k:   int,
    stage:   Optional[str] = None,
) -> tuple[list[dict], str]:
    """
    Stage-aware retrieval wrapper around the existing retrieve() function.
    Biases results toward same-stage/same-intent examples without replacing
    the existing retrieval system.

    Strategy:
    1. Run normal retrieve() to get top_k results
    2. If stage is known, up-rank examples whose category or preset matches the stage
    3. Return re-ranked results with method label

    This is lightweight — no vector DB, no extra model calls.
    """
    # Stage → KB category/preset keywords that signal the same stage
    _STAGE_KB_KEYWORDS: dict[str, list[str]] = {
        "first_outreach":  ["cold_outreach", "sales_pitch", "warm_outreach"],
        "warm_lead":       ["warm_outreach", "follow_up_after_interest", "soft_pitch"],
        "negotiation":     ["negotiation", "price_negotiation", "counter_offer", "price_too_high"],
        "counteroffer":    ["counter_offer", "negotiation", "price_negotiation"],
        "stalled":         ["follow_up", "follow_up_no_response", "re_engagement"],
        "final_follow_up": ["follow_up", "re_engagement", "final_offer"],
        "accepted":        ["agreed_no_pay", "closing", "payment_reminder"],
        "rejected":        ["no_thanks", "angry"],
    }

    base_results, method = retrieve(message, replies, intent, api_key, top_k)

    if not stage or stage == "unknown" or not base_results:
        return base_results, method

    stage_keywords = _STAGE_KB_KEYWORDS.get(stage, [])
    if not stage_keywords:
        return base_results, method

    # Up-rank examples that match the stage keywords
    def _stage_score(ex: dict) -> int:
        cat = (ex.get("category") or "").lower()
        preset = (ex.get("email_preset") or "").lower()
        return sum(1 for kw in stage_keywords if kw in cat or kw in preset)

    re_ranked = sorted(base_results, key=_stage_score, reverse=True)
    boosted = sum(1 for ex in re_ranked if _stage_score(ex) > 0)
    if boosted > 0:
        print(f"[STAGE_RETRIEVAL] stage={stage} boosted {boosted}/{len(re_ranked)} examples")

    return re_ranked, method


def build_reply_prompt_ai(
    message: str,
    intent: str,
    examples: list,
    tone: str,
    domain_name: Optional[str],
    asking_price: Optional[str],
    retrieval_method: str,
    analysis: Optional["InputAnalysis"] = None,
    email_preset: Optional[str] = None,
    lead_context: Optional[str] = None,
    stage_block: Optional[str] = None,
    offer_intel: Optional[str] = None,
    strategy: Optional["ReplyStrategy"] = None,
) -> str:
    """
    AI-mode prompt builder.

    When strategy is provided (Phase 1+): uses build_prompt_brief() to produce
    a concise, purposeful brief driven by the ReplyStrategy object.
    The examples, lead context, offer intel, and message are still injected —
    only the instruction section changes.

    When strategy is None (fallback / backward compat): original behaviour,
    all rule tables and inline logic as before.
    """
    if analysis is None:
        analysis = analyse(message)

    # ── Context: domain + price (shared by both paths) ────────────────────────
    context_parts = []
    if domain_name:  context_parts.append(f"Domain: {domain_name}")
    if asking_price: context_parts.append(f"Asking price: {asking_price}")
    context_line = "  ".join(context_parts) + "\n" if context_parts else ""

    # ── Reference examples (shared by both paths) ─────────────────────────────
    method_note = "by meaning" if retrieval_method == "semantic" else "by keyword"
    ex_block = ""
    if examples:
        ex_block = f"Past replies for style reference (retrieved {method_note} — do not copy):\n"
        for i, ex in enumerate(examples, 1):
            ex_block += f"  {i}. [{ex.get('category','general')}] {ex['reply'][:120]}…\n"
        ex_block += "\n"

    lead_block    = f"LEAD HISTORY:\n{lead_context}\n\n" if lead_context else ""
    intel_section = f"{offer_intel}\n" if offer_intel else ""

    # ── STRATEGY-DRIVEN PATH ──────────────────────────────────────────────────
    if strategy is not None:
        print(
            f"[STRATEGY] goal={strategy.primary_goal} buyer={strategy.buyer_state} "
            f"posture={strategy.conversation_posture} persuasion={strategy.persuasion_level} "
            f"urgency={strategy.urgency_level} cta={strategy.cta_style} "
            f"length={strategy.reply_length}"
        )
        brief = build_prompt_brief(
            strategy,
            context_line       = context_line.strip(),
            has_questions      = analysis.has_questions if analysis else False,
            question_count     = len(analysis.questions) if analysis and analysis.has_questions else 0,
            no_domain_no_price = strategy.no_domain_no_price,
        )
        return (
            f"{ex_block}"
            f"{lead_block}"
            f"{intel_section}"
            f"Message or situation:\n\"{strip_filler(message)}\"\n\n"
            f"{brief}"
        )

    # ── ORIGINAL PATH (fallback — strategy=None) ──────────────────────────────
    response_frame = _classify_response_frame(message)
    neg_state      = _detect_negotiation_state(message, asking_price)
    neg_guidance   = _negotiation_guidance(neg_state, asking_price)

    intent_guidance = ""
    if intent in _AI_SALES_INTENTS:
        rule = INTENT_RULES.get(intent, "")
        if rule:
            intent_guidance = f"For this kind of message: {rule}\n\n"

    neg_block = f"Negotiation note: {neg_guidance}\n\n" if neg_guidance else ""
    tone_inst = TONE_INSTRUCTIONS.get(tone, f"Tone: {tone}.")

    frame_inst = {
        "strategic_advice": (
            "The broker is asking for advice, not a draft. "
            "Explain the recommended approach first, then include a suggested reply at the end."
        ),
        "educational_answer": (
            "This is a question that needs a direct, informative answer. "
            "Answer it completely. Sales content only if it fits naturally."
        ),
        "negotiation_analysis": (
            "Analyse the offer or situation: what does it signal, what's the realistic counter, "
            "what should happen next. Then write the recommended reply."
        ),
        "brainstorming": (
            "Give three distinct approaches, each 2-3 sentences. Label them simply."
        ),
        "direct_reply": (
            "Write a complete, ready-to-send email reply. "
            "Start with a natural greeting. Write 2-3 focused paragraphs. "
            "End with one clear call to action and a sign-off. "
            "Do NOT ask clarifying questions — just write the email."
        ),
        "inferred_reply": (
            "The broker has described a situation. Write a complete, ready-to-send email for it. "
            "Start with a natural greeting. Write 2-3 focused paragraphs. "
            "End with one clear call to action and a sign-off. "
            "Do NOT ask clarifying questions — just write the email."
        ),
        "mixed_request": (
            "This message has multiple parts. Address information first, then write the email reply."
        ),
    }.get(response_frame, (
        "Write a complete, ready-to-send email reply. "
        "Start with a natural greeting. Write 2-3 focused paragraphs. "
        "End with one clear call to action and a sign-off."
    ))

    question_note = ""
    if analysis.has_questions:
        q_count = len(analysis.questions)
        if q_count > 1:
            question_note = (
                f"They asked {q_count} questions. Answer all of them before anything else.\n\n"
            )
        else:
            question_note = "They asked a direct question — answer it first.\n\n"

    msg_low = message.lower()
    emotion_note = ""
    if any(w in msg_low for w in ["can't afford", "tight budget", "small business",
                                    "limited budget", "struggling"]):
        emotion_note = "They seem genuinely budget-constrained. Lead with value before price.\n\n"
    elif any(w in msg_low for w in ["love it", "sounds good", "interested", "let's do it",
                                      "want to proceed", "tell me more"]):
        emotion_note = "They're warm and engaged — move toward closing, don't re-sell.\n\n"
    elif any(w in msg_low for w in ["confused", "don't understand", "what do you mean"]):
        emotion_note = "They need clarity. Explain simply before any pitch.\n\n"

    print(
        f"[REASONING] frame={response_frame} neg={neg_state} "
        f"intent={intent} questions={len(analysis.questions) if analysis.has_questions else 0}"
    )

    no_invent_note = ""
    if not domain_name and not asking_price and intent in (
        "general", "general_response", "how_it_works", "domain_metrics",
        "renewal_fees", "payment_method", "feature_explanation", "request_info"
    ):
        no_invent_note = (
            "No domain details have been provided. "
            "Answer from general domain industry knowledge only. "
            "Do NOT invent registration dates, ages, traffic numbers, or any specific facts.\n\n"
        )

    preset_block  = _build_preset_block(email_preset)
    stage_section = f"{stage_block}\n" if stage_block else ""

    return (
        f"{context_line}"
        f"{ex_block}"
        f"{preset_block}"
        f"{stage_section}"
        f"{intel_section}"
        f"{lead_block}"
        f"Message or situation:\n\"{strip_filler(message)}\"\n\n"
        f"{question_note}"
        f"{emotion_note}"
        f"{no_invent_note}"
        f"{intent_guidance}"
        f"{neg_block}"
        f"What to do: {frame_inst}\n\n"
        f"Tone: {tone_inst}\n\n"
        f"Write only the email body — no subject line, no metadata, no commentary. "
        f"No filler openers like 'I hope this finds you well'. "
        f"Do NOT ask clarifying questions — write the email.\n\n"
        f"Write the reply:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI MODE VARIATION STYLES
# Parallel to VARIATION_STYLES — used only when effective_mode == 'ai'.
# Removes urgency language and broker framing from non-sales variations.
# VARIATION_STYLES (hybrid) is NOT changed.
# ─────────────────────────────────────────────────────────────────────────────

AI_VARIATION_STYLES = [
    {
        "label":       "Direct",
        "instruction": (
            "Write a clear, grounded reply that speaks directly to the situation. "
            "Match the register of what was asked — answer questions, respond to negotiations, "
            "or pitch where appropriate. No padding, no build-up."
        ),
    },
    {
        "label":       "Detailed",
        "instruction": (
            "Give a more complete response. If there was a question, answer it fully. "
            "If it's a sales context, make the most relevant case with real specifics. "
            "Add context only where it genuinely helps — not to fill space. "
            "No marketing language, no exaggeration, no phrases that sound like ad copy."
        ),
    },
    {
        "label":       "Concise",
        "instruction": (
            "Write the shortest version that still fully addresses the situation. "
            "Three sentences is the ceiling. Cut anything that doesn't earn its place."
        ),
    },
]


def generate_variations_ai(
    base_prompt: str,
    num: int,
    situation: str,
    prospect_name: Optional[str],
    sender_name: Optional[str],
    intent: str,
    domain_name: Optional[str],
    model: str = MODEL,
    strategy: Optional["ReplyStrategy"] = None,
) -> list["ReplyResult"]:
    """
    AI-mode variation generator. Uses AI_VARIATION_STYLES instead of
    VARIATION_STYLES — no urgency language, no broker framing in non-sales
    variations. Uses AI-mode system prompt.

    Otherwise identical structure to generate_variations() — same QC,
    same scoring, same format_email_body calls. Existing generate_variations()
    is NOT changed.
    """
    effective_num = max(1, min(3, num)) if ENABLE_VARIATIONS else _FAST_PATH_VARIATIONS
    if not ENABLE_VARIATIONS and num > 1:
        print(f"[TIMING] variations_requested={num} effective=1 (ENABLE_VARIATIONS=false)")

    styles     = AI_VARIATION_STYLES[:effective_num]
    results    = []
    ollama     = _get_client_for_model(model)
    qc_relaxed = intent in _QC_RELAXED_INTENTS
    sys_prompt = _select_system_prompt_for_mode(model, mode="ai")

    for style in styles:
        _t_var = time.monotonic()
        variation_prompt = base_prompt + (
            f"\n\nWrite this version: {style['instruction']}\nWrite the reply:"
        )
        print(f"[AI_BACKEND] backend={('groq' if model.startswith('groq:') else 'ollama')} model={model} mode=ai label=variation_{style['label'].lower()} intent={intent}")
        raw = ollama.generate(
            prompt      = variation_prompt,
            system      = sys_prompt,
            temperature = 0.72,
            max_tokens  = MAX_TOKENS,
        )
        _require_ollama(raw, f"variation_{style['label']}")

        formatted = format_email_body(raw.strip(), prospect_name=prospect_name,
                                      sender_name=sender_name)

        with _Timer("qc"):
            qc = run_full_qc(formatted, intent=intent)
        if not qc["validation"]["passed"] and not qc_relaxed:
            print(f"[ValidationQC] AI {style['label']}: ✗ Issues: "
                  f"{', '.join(qc['validation'].get('issues', []))} "
                  f"({qc['validation'].get('word_count', 0)}w · "
                  f"{qc['validation'].get('paragraph_count', 0)}p)")
        final = qc["reply"]

        # ── Humanization layer ────────────────────────────────────────
        with _Timer("humanize"):
            final, quality_report = humanize_reply(
                reply        = final,
                situation    = situation,
                intent       = intent,
                model        = model,
                domain_name  = domain_name,
                asking_price = None,
            )

        score, reason = (
            score_reply_ollama(situation, final, model=model)
            if ENABLE_AI_SCORING
            else (75, "AI scoring disabled")
        )

        elapsed = int((time.monotonic() - _t_var) * 1000)
        print(f"[TIMING] ai_variation_{style['label'].lower()}_ms={elapsed}")

        # ── Strategy adherence evaluation (non-blocking) ──────────────────
        if strategy is not None and quality_report is not None:
            try:
                eval_result = evaluate_strategy_adherence(final, strategy)
                if eval_result:
                    quality_report["strategy_adherence"]   = eval_result.get("strategy_adherence")
                    quality_report["repetition_violations"]= eval_result.get("repetition_violations", [])
                    quality_report["progression_result"]   = eval_result.get("progression_result")
                    quality_report["confidence_alignment"] = eval_result.get("confidence_alignment")
                    strategy_analytics.record(strategy, eval_result)
                    adh_score = eval_result.get("strategy_adherence", {}).get("adherence_score", "?")
                    prog      = eval_result.get("progression_result", {}).get("verdict", "?")
                    print(f"[STRATEGY_EVAL] adherence={adh_score}/100 progression={prog}")
            except Exception as _eval_err:
                print(f"[STRATEGY_EVAL] non-blocking error: {_eval_err}")

        results.append(ReplyResult(
            reply=final,
            label=style["label"],
            confidence_score=score,
            confidence_reason=reason,
            quality_report=quality_report,
        ))

    check_variation_uniqueness(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT LINE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_TEMPLATES: dict[str, list[str]] = {
    "follow_up":          ["Quick follow-up", "Still available — {domain}", "Checking in"],
    "sales_pitch":        ["A domain that could bring more customers to {business}", "{domain} — is this a fit?", "Opportunity for your business"],
    "re_engagement":      ["Coming back to this — {domain}", "Still available if you're interested", "Revisiting our conversation"],
    "objection_handling": ["Happy to answer your questions on {domain}", "Let me address your concerns", "More info on {domain}"],
    "no_thanks":          ["Understood — keeping the door open", "No problem at all"],
    "price_inquiry":      ["Pricing for {domain}", "Your inquiry about {domain}"],
    "price_too_high":     ["Let's find a number that works", "Re: pricing on {domain}"],
    "negotiation":        ["Re: your offer on {domain}", "Counteroffer — {domain}"],
    "trust_issue":        ["Verifying {domain} — here's how", "Proof of ownership + escrow options"],
    "have_website":       ["You don't need to change a thing — re: {domain}", "{domain} would work alongside your site"],
    "agreed_no_pay":      ["Your domain is ready — payment link inside", "Action needed: {domain}"],
    "angry":              ["Removing you now — apologies for the interruption"],
    "why_buy":            ["Why {domain} could be your best marketing move", "The case for {domain}"],
    "not_now":            ["No rush — {domain} is still here", "Whenever you're ready"],
    "rank_well":          ["Even top rankers benefit from owning {domain}", "{domain} — a different angle"],
    "general":            ["Following up on {domain}", "Quick note about {domain}"],
}

def generate_subject(intent: str, domain_name: Optional[str] = None) -> str:
    """Pick the most relevant subject template for this intent and fill it in."""
    templates = SUBJECT_TEMPLATES.get(intent, SUBJECT_TEMPLATES["general"])
    template  = templates[0]  # use the first (best) option
    domain    = domain_name or "the domain"
    business  = domain_name.replace(".com","").replace(".co.uk","") if domain_name else "your business"
    return template.replace("{domain}", domain).replace("{business}", business)


def generate_subject_ai(intent: str, reply_body: str,
                        domain_name: Optional[str] = None,
                        model: str = MODEL) -> str:
    """
    Generate subject line. Respects ENABLE_AI_SUBJECT flag.
    When disabled, uses the fast template subject instantly.
    When enabled, asks Ollama for a custom subject via the specified model.
    model defaults to the global MODEL constant for backward compatibility.
    """
    if not ENABLE_AI_SUBJECT:
        return generate_subject(intent, domain_name)

    domain_hint = f" The domain being sold is {domain_name}." if domain_name else ""
    prompt = (
        f"Write a short email subject line (under 8 words) for this domain sales email.{domain_hint}\n"
        f"Intent: {intent.replace('_', ' ')}\n"
        f"Email preview: {reply_body[:200]}\n\n"
        f"Rules:\n"
        f"- Do NOT use spammy words like 'Amazing', 'Urgent', 'Act Now'\n"
        f"- Sound natural, like a real person wrote it\n"
        f"- No punctuation at the end\n"
        f"Return ONLY the subject line, nothing else."
    )
    try:
        client = _get_client_for_model(model)
        print(f"[AI_BACKEND] backend={('groq' if model.startswith('groq:') else 'ollama')} model={model} label=subject_line")
        result = client.generate(prompt=prompt, temperature=0.5, max_tokens=40)
        if result:
            return result.strip().strip('"').strip("'")
    except Exception as e:
        print(f"[AI_BACKEND] subject_line failed: {e}")
    return generate_subject(intent, domain_name)


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_email_body(raw_body: str, prospect_name: Optional[str] = None,
                      sender_name: Optional[str] = None) -> str:
    """
    Wrap a raw reply body in proper email structure:
      Greeting → Body paragraphs → Closing → Signature
    Handles cases where Claude already included a greeting or closing.
    """
    sender  = sender_name or DEFAULT_SENDER
    body    = raw_body.strip()

    # Detect if Claude already added a greeting line (starts with Hi/Hello/Dear)
    has_greeting = bool(re.match(r"^(hi|hello|dear|hey)\b", body, re.IGNORECASE))
    # Detect if Claude already added a closing line
    has_closing  = bool(re.search(
        r"(best regards|kind regards|warm regards|best wishes|thanks|thank you|cheers|sincerely)",
        body, re.IGNORECASE
    ))

    # Build greeting
    if not has_greeting:
        if prospect_name:
            greeting = f"Hi {prospect_name.strip()},"
        else:
            greeting = "Hi,"
        body = greeting + "\n\n" + body

    # Ensure paragraphs are separated by double newlines (not single)
    body = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", body)

    # Add closing + signature if missing
    if not has_closing:
        body = body.rstrip() + f"\n\nBest regards,\n{sender}"

    return body


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY GUARD
# ─────────────────────────────────────────────────────────────────────────────

def quality_guard(reply: str, situation: str, intent: str = "", model: str = MODEL) -> str:
    """
    Check the reply for critical quality issues and fix via Ollama if needed.
    Respects ENABLE_QC_REWRITE flag.
    model defaults to global MODEL for backward compatibility.
    """
    if not ENABLE_QC_REWRITE:
        return reply

    words      = len(reply.split())
    is_relaxed = intent in _QC_RELAXED_INTENTS

    issues = []
    if words < MIN_REPLY_WORDS:
        issues.append(f"TOO SHORT ({words} words) — expand to at least {MIN_REPLY_WORDS} words")
    if words > MAX_REPLY_WORDS and not is_relaxed:
        issues.append(f"TOO LONG ({words} words) — trim to under {MAX_REPLY_WORDS} words")

    if not issues:
        return reply

    fix_prompt = (
        f"Fix the following email reply. Issues found:\n"
        + "\n".join(f"- {i}" for i in issues)
        + f"\n\nOriginal reply:\n{reply}\n\n"
        f"Situation: {situation}\n\n"
        f"Return ONLY the corrected reply, nothing else."
    )
    try:
        client = _get_client_for_model(model)
        print(f"[AI_BACKEND] backend={('groq' if model.startswith('groq:') else 'ollama')} model={model} label=quality_guard intent={intent} issues={len(issues)}")
        result = client.generate(prompt=fix_prompt, temperature=0.4, max_tokens=MAX_TOKENS)
        if result:
            print(f"[ValidationQC] quality_guard fixed: {', '.join(issues)}")
            return result.strip()
    except Exception as e:
        print(f"[ValidationQC] quality_guard Ollama fix failed: {e}")

    print(f"[ValidationQC] quality_guard skipped fix (Ollama unavailable): {', '.join(issues)}")
    return reply


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID MODE (Template → AI polish)
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_mode(customer_message: str, intent: str,
                    domain_name: Optional[str], asking_price: Optional[str],
                    tone: str, model: str = MODEL) -> str:
    """
    Hybrid flow: template → polish.
    Routes through _get_client_for_model so Groq and Ollama both work.
    """
    template_result = build_template_reply(
        customer_message=customer_message,
        domain_name=domain_name,
        asking_price=asking_price,
        force_intent=intent,
    )
    template_reply = template_result.get("reply", "")

    print(f"[AI_BACKEND] backend={('groq' if model.startswith('groq:') else 'ollama')} model={model} mode=hybrid intent={intent}")

    if model.startswith("groq:"):
        # Groq path — ai_polish_reply doesn't support Groq, polish directly
        from quality_control import build_strategy_block
        polish_prompt = (
            f"You are polishing a domain sales email reply.\n\n"
            f"Original prospect message:\n\"{customer_message}\"\n\n"
            f"Template reply to polish:\n\"{template_reply}\"\n\n"
            f"Improve the reply so it sounds natural and human. "
            f"Keep the same intent and structure. "
            f"Do not add facts not in the template. "
            f"Tone: {tone}.\n\n"
            f"Write only the polished email body:"
        )
        try:
            client = _get_client_for_model(model)
            sys = _select_system_prompt(model)
            polished = client.generate(prompt=polish_prompt, system=sys,
                                       temperature=0.7, max_tokens=MAX_TOKENS)
            return polished.strip() if polished and polished.strip() else template_reply
        except Exception as e:
            print(f"[AI_BACKEND] Groq polish failed: {e} — returning template")
            return template_reply
    else:
        result = ai_polish_reply(
            template_reply   = template_reply,
            customer_message = customer_message,
            intent           = intent,
            api_key          = "",
            domain_name      = domain_name,
            asking_price     = asking_price,
            tone             = tone,
            backend          = "ollama",
            ollama_model     = model,
            ollama_base_url  = OLLAMA_BASE_URL,
            ollama_timeout   = OLLAMA_TIMEOUT,
        )
        return result.get("polished_reply", template_reply)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-VARIATION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

VARIATION_STYLES = [
    {
        "label":       "Safe",
        "instruction": (
            "Write a balanced, professional reply — friendly but not pushy. "
            "This should be the one most brokers would feel comfortable sending."
        ),
    },
    {
        "label":       "Persuasive",
        "instruction": (
            "Make the strongest relevant case. Lead with the most compelling value point, "
            "be confident and direct, and add light urgency only if it fits naturally and is true. "
            "No exaggerated language, no marketing clichés, no phrases that sound like an ad."
        ),
    },
    {
        "label":       "Short",
        "instruction": (
            "Three sentences maximum. Get to the point immediately. "
            "Nothing that doesn't earn its place."
        ),
    },
]

def generate_variations(
    base_prompt: str,
    num: int,
    situation: str,
    prospect_name: Optional[str],
    sender_name: Optional[str],
    intent: str,
    domain_name: Optional[str],
    model: str = MODEL,
) -> list[ReplyResult]:
    """
    Generate reply variations via Ollama.
    Respects ENABLE_VARIATIONS, ENABLE_AI_SCORING, ENABLE_QC_REWRITE flags.
    Selects model-appropriate system prompt automatically.
    """
    effective_num = max(1, min(3, num)) if ENABLE_VARIATIONS else _FAST_PATH_VARIATIONS
    if not ENABLE_VARIATIONS and num > 1:
        print(f"[TIMING] variations_requested={num} effective=1 (ENABLE_VARIATIONS=false)")

    styles     = VARIATION_STYLES[:effective_num]
    results    = []
    ollama     = _get_client_for_model(model)
    qc_relaxed = intent in _QC_RELAXED_INTENTS
    sys_prompt = _select_system_prompt(model)   # model-aware system prompt

    for style in styles:
        _t_var = time.monotonic()
        variation_prompt = base_prompt + (
            f"\n\nWrite this version: {style['instruction']}\nWrite the reply:"
        )
        print(f"[AI_BACKEND] backend={('groq' if model.startswith('groq:') else 'ollama')} model={model} label=variation_{style['label'].lower()} intent={intent}")
        raw = ollama.generate(
            prompt      = variation_prompt,
            system      = sys_prompt,
            temperature = 0.75,
            max_tokens  = MAX_TOKENS,
        )
        _require_ollama(raw, f"variation_{style['label']}")

        formatted = format_email_body(raw.strip(), prospect_name=prospect_name, sender_name=sender_name)

        with _Timer("qc"):
            qc = run_full_qc(formatted, intent=intent)
        if not qc["validation"]["passed"] and not qc_relaxed:
            print(f"[ValidationQC] {style['label']}: {qc['summary']}")
            formatted = qc["reply"]

        if ENABLE_QC_REWRITE:
            fixed = quality_guard(formatted, situation, intent=intent, model=model)
        else:
            fixed = formatted

        if ENABLE_AI_SCORING:
            score, reason = score_reply_ollama(situation, fixed, model=model)
        else:
            score, reason = 75, "AI scoring disabled (ENABLE_AI_SCORING=false)"

        _Timer.log(f"variation_{style['label'].lower()}", int((time.monotonic() - _t_var) * 1000))

        results.append(ReplyResult(
            reply=fixed,
            label=style["label"],
            confidence_score=score,
            confidence_reason=reason,
        ))

    if len(results) > 1:
        uniqueness = check_variation_uniqueness([r.reply for r in results])
        log_variation_check(uniqueness)

    return results


def score_reply_ollama(situation: str, reply_text: str, model: str = MODEL) -> tuple[int, str]:
    """
    Score a reply via Ollama. Respects ENABLE_AI_SCORING flag.
    model defaults to global MODEL for backward compatibility.
    """
    if not ENABLE_AI_SCORING:
        return 75, "AI scoring disabled (set ENABLE_AI_SCORING=true to enable)"

    prompt = (
        f"SITUATION: {situation}\n\n"
        f"REPLY TO ASSESS:\n{reply_text}\n\n"
        "Return a JSON object with exactly two fields:\n"
        '{"score": <integer 0-100>, "reason": "<one plain English sentence>"}\n'
        "Return ONLY the JSON. No explanation, no markdown, no backticks."
    )
    try:
        client = _get_client_for_model(model)
        print(f"[AI_BACKEND] backend={('groq' if model.startswith('groq:') else 'ollama')} model={model} label=score_reply")
        with _Timer("scoring"):
            result = client.generate(prompt=prompt, temperature=0.2, max_tokens=120)
        if result:
            clean = re.sub(r"```(?:json)?", "", result.strip()).strip().rstrip("```").strip()
            data  = json.loads(clean)
            score  = max(0, min(100, int(data.get("score", 75))))
            reason = str(data.get("reason", "")).strip() or "Reply looks good."
            return score, reason
    except Exception as e:
        print(f"[AI_BACKEND] score_reply failed: {e}")
    return 75, "Could not auto-score this reply."


# ─────────────────────────────────────────────────────────────────────────────
# HUMANIZATION & QUALITY SCORING LAYER
# ─────────────────────────────────────────────────────────────────────────────
# Flow:  generation → run_full_qc() → humanize_reply() → score_reply_ollama()
#
# This layer is heuristics-first (zero latency), model-rewrite second (optional).
# Gated by ENABLE_HUMANIZER env flag — set to "false" to disable entirely.
# ─────────────────────────────────────────────────────────────────────────────

ENABLE_HUMANIZER = _flag("ENABLE_HUMANIZER", True)

# ── Hype / spam / AI-phrase detection patterns ────────────────────────────────
_HYPE_PHRASES: list[tuple[str, str]] = [
    # (regex_pattern,  human-readable flag label)
    (r"\bperfect\s+domain\b",            "hype:perfect_domain"),
    (r"\bdream\s+domain\b",              "hype:dream_domain"),
    (r"\bonce[\s-]in[\s-]a[\s-]lifetime\b", "hype:once_in_lifetime"),
    (r"\bgame[\s-]chang(?:ing|er)\b",    "hype:game_changer"),
    (r"\bunique\s+opportunit(?:y|ies)\b","hype:unique_opportunity"),
    (r"\bdon't\s+miss\s+out\b",          "hype:dont_miss_out"),
    (r"\bact\s+(?:fast|now|quickly)\b",  "hype:act_fast"),
    (r"\blimited\s+time\b",              "hype:limited_time"),
    (r"\bexclusive\s+offer\b",           "hype:exclusive_offer"),
    (r"\bincredible\s+(?:value|deal|opportunity)\b", "hype:incredible"),
    (r"\bamazing\s+(?:domain|opportunity|deal)\b",   "hype:amazing"),
    (r"\bworld[\s-]class\b",             "hype:world_class"),
    (r"\bpremium\s+domain\b",            "hype:premium_domain"),
]

_AI_PHRASES: list[tuple[str, str]] = [
    (r"\bI\s+hope\s+this\s+(?:email|message)\s+finds\s+you\b", "ai:hope_finds_you"),
    (r"\bI\s+hope\s+you(?:'re|\s+are)\s+(?:doing\s+)?well\b",  "ai:hope_doing_well"),
    (r"\bplease\s+don't\s+hesitate\s+to\b",                     "ai:dont_hesitate"),
    (r"\bfeel\s+free\s+to\s+reach\s+out\b",                     "ai:feel_free_reach_out"),
    (r"\bI\s+wanted\s+to\s+(?:reach\s+out|touch\s+base)\b",    "ai:wanted_to_reach_out"),
    (r"\bI\s+am\s+writing\s+to\b",                              "ai:i_am_writing_to"),
    (r"\bas\s+per\s+(?:my|our|your)\s+(?:previous|last|prior)\b", "ai:as_per_previous"),
    (r"\bI\s+trust\s+this\s+(?:finds|email|message)\b",         "ai:i_trust_this"),
    (r"\bthank\s+you\s+for\s+your\s+(?:time\s+and\s+)?consideration\b", "ai:thanks_consideration"),
    (r"\bI\s+look\s+forward\s+to\s+hearing\s+from\s+you\b",    "ai:look_forward"),
    (r"\bkind(?:est)?\s+regards\b",                              "ai:kind_regards"),
    (r"\bwarm\s+regards\b",                                      "ai:warm_regards"),
    (r"\bI\s+sincerely\s+hope\b",                                "ai:sincerely_hope"),
]

_WEAK_CTA: list[tuple[str, str]] = [
    (r"\blet\s+me\s+know\s+if\s+you\s+have\s+any\s+questions?\b", "weak_cta:let_me_know_questions"),
    (r"\bfeel\s+free\s+to\s+contact\s+me\b",                       "weak_cta:feel_free_contact"),
    (r"\bdo\s+not\s+hesitate\s+to\s+contact\b",                    "weak_cta:do_not_hesitate"),
    (r"\breach\s+out\s+(?:if|any)\b",                              "weak_cta:generic_reach_out"),
]

_STRUCTURE_RULES: list[tuple[str, str]] = [
    (r"(?m)^(?:.{200,}(?:\n|$)){3,}",   "structure:wall_of_text"),     # 3+ lines >200 chars
    (r"(?i)^(?:dear\s+sir|to\s+whom\s+it\s+may\s+concern|hello\s+there)[,\s]", "structure:generic_opener"),
    (r"\b(\w+)\b(?:\s+\w+){0,5}\s+\1\b","structure:word_repetition"),  # same word within 6 words
]

# ── Per-dimension heuristic scorer ─────────────────────────────────────────────

def heuristic_score_reply(reply: str, intent: str = "") -> dict:
    """
    Fast, regex/heuristic multi-dimensional scoring. Zero model calls.

    Returns a dict:
    {
        "dimensions": {
            "naturalness":           int 0-20,
            "readability":           int 0-20,
            "spamminess":            int 0-20,   (inverted: 20=not spammy)
            "cta_quality":           int 0-10,
            "human_tone":            int 0-15,
            "personalization":       int 0-5,
            "structure":             int 0-10,
        },
        "total":      int 0-100,
        "flags":      list[str],   # all triggered pattern labels
        "needs_humanization": bool,
        "summary":    str,
    }
    """
    text     = reply.strip()
    words    = text.split()
    word_ct  = len(words)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sent_ct  = len([s for s in sentences if s.strip()])
    paras    = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    para_ct  = len(paras)

    flags: list[str] = []

    # ── Detect hype ─────────────────────────────────────────────────
    hype_hits = 0
    for pattern, label in _HYPE_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(label)
            hype_hits += 1

    # ── Detect AI phrasing ───────────────────────────────────────────
    ai_hits = 0
    for pattern, label in _AI_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(label)
            ai_hits += 1

    # ── Detect weak CTA ─────────────────────────────────────────────
    cta_hits = 0
    for pattern, label in _WEAK_CTA:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(label)
            cta_hits += 1

    # ── Detect no CTA at all (for negotiation/sales intents) ────────
    has_question      = "?" in text
    has_strong_cta    = bool(re.search(
        r"\b(?:reply|respond|let(?:'s|\s+us)\s+(?:schedule|set up|talk|discuss)|"
        r"would\s+you\s+(?:be\s+open|consider|like)|"
        r"open\s+to\s+a|what(?:'s|\s+is)\s+your\s+(?:budget|timeline)|"
        r"can\s+we|shall\s+we|give\s+(?:me|us)\s+a\s+call)\b",
        text, re.IGNORECASE
    ))
    sales_intent = intent not in _QC_RELAXED_INTENTS
    missing_cta  = sales_intent and not has_question and not has_strong_cta
    if missing_cta:
        flags.append("cta:missing")

    # ── Detect structure issues ──────────────────────────────────────
    if para_ct == 1 and word_ct > 100:
        flags.append("structure:no_paragraph_breaks")
    if para_ct > 5:
        flags.append("structure:too_many_paragraphs")
    long_paras = sum(1 for p in paras if len(p.split()) > 80)
    if long_paras >= 2:
        flags.append("structure:long_paragraphs")

    # generic opener detection
    if re.search(r"(?i)^(?:dear\s+sir|to\s+whom\s+it\s+may\s+concern|hello\s+there)[,\s]", text):
        flags.append("structure:generic_opener")

    # ── Detect word repetition (same non-trivial word ≥3×) ──────────
    word_freq: dict[str, int] = {}
    stop = {"the","a","an","and","or","but","to","of","in","for","it","is","this","that","we","you","i","my","your","our"}
    for w in words:
        w_clean = re.sub(r"[^a-z]", "", w.lower())
        if len(w_clean) > 4 and w_clean not in stop:
            word_freq[w_clean] = word_freq.get(w_clean, 0) + 1
    repeated = [w for w, c in word_freq.items() if c >= 3]
    if repeated:
        flags.append(f"style:word_repetition({'|'.join(repeated[:3])})")

    # ── Avg sentence length ──────────────────────────────────────────
    avg_sent_len = word_ct / max(sent_ct, 1)
    if avg_sent_len > 30:
        flags.append("readability:long_sentences")
    if avg_sent_len < 5 and sent_ct > 3:
        flags.append("readability:fragment_sentences")

    # ── Weak opener ──────────────────────────────────────────────────
    first_sent = sentences[0] if sentences else ""
    weak_opener = bool(re.search(
        r"\b(?:I\s+hope|Thank\s+you\s+for|I\s+am\s+writing|I\s+wanted\s+to)\b",
        first_sent, re.IGNORECASE
    ))
    if weak_opener:
        flags.append("opener:weak")

    # ── SCORING ──────────────────────────────────────────────────────
    # Naturalness (0-20): penalise hype + AI phrases
    naturalness = max(0, 20 - (hype_hits * 4) - (ai_hits * 2))

    # Readability (0-20): penalise long sentences, no breaks, wall of text
    readability = 20
    if "readability:long_sentences"        in flags: readability -= 6
    if "structure:no_paragraph_breaks"     in flags: readability -= 5
    if "structure:long_paragraphs"         in flags: readability -= 4
    if "readability:fragment_sentences"    in flags: readability -= 3
    readability = max(0, readability)

    # Spamminess (0-20, higher=better i.e. less spammy)
    spamminess = max(0, 20 - (hype_hits * 5) - (cta_hits * 1))

    # CTA quality (0-10)
    if has_strong_cta:       cta_quality = 10
    elif has_question:       cta_quality = 7
    elif not missing_cta:    cta_quality = 6   # relaxed intent, no CTA needed
    elif cta_hits:           cta_quality = 3   # has a weak CTA
    else:                    cta_quality = 0   # no CTA at all

    # Human tone (0-15): penalise AI phrases + weak opener
    human_tone = max(0, 15 - (ai_hits * 3) - (3 if weak_opener else 0))

    # Personalization (0-5): simple heuristic — presence of a name or domain
    has_name_or_domain = bool(re.search(r'\b[A-Z][a-z]+\b', text))
    personalization = 5 if has_name_or_domain else 2

    # Structure (0-10)
    structure = 10
    if "structure:generic_opener"      in flags: structure -= 3
    if "structure:too_many_paragraphs" in flags: structure -= 3
    if "structure:no_paragraph_breaks" in flags: structure -= 4
    if "style:word_repetition"         in flags: structure -= 2
    structure = max(0, structure)

    total = naturalness + readability + spamminess + cta_quality + human_tone + personalization + structure

    needs_humanization = (
        total < 65
        or hype_hits >= 2
        or ai_hits  >= 3
        or "cta:missing"          in flags
        or "opener:weak"          in flags
        or "structure:no_paragraph_breaks" in flags
    )

    # Build summary line
    issue_groups: list[str] = []
    if hype_hits:   issue_groups.append(f"{hype_hits} hype phrase{'s' if hype_hits>1 else ''}")
    if ai_hits:     issue_groups.append(f"{ai_hits} AI-sounding phrase{'s' if ai_hits>1 else ''}")
    if cta_hits:    issue_groups.append("weak CTA")
    if missing_cta: issue_groups.append("no CTA")
    if "opener:weak" in flags: issue_groups.append("weak opener")
    if "structure:no_paragraph_breaks" in flags: issue_groups.append("wall of text")
    if "style:word_repetition" in flags: issue_groups.append("word repetition")

    summary = f"Score {total}/100" + (f" — issues: {', '.join(issue_groups)}" if issue_groups else " — looks good")

    return {
        "dimensions": {
            "naturalness":    naturalness,
            "readability":    readability,
            "spamminess":     spamminess,
            "cta_quality":    cta_quality,
            "human_tone":     human_tone,
            "personalization": personalization,
            "structure":      structure,
        },
        "total":               total,
        "flags":               flags,
        "needs_humanization":  needs_humanization,
        "summary":             summary,
    }


# ── Humanization rewrite system ────────────────────────────────────────────────

_HUMANIZE_SYSTEM = (
    "You are a domain broker email editor making targeted, conservative improvements.\n\n"
    "YOUR ONLY JOB: Fix specific flagged issues while leaving everything else untouched.\n\n"
    "WHAT YOU MUST PRESERVE — do not change these under any circumstances:\n"
    "- Domain name(s), asking price, offer amounts, any links\n"
    "- Negotiation position and stance (firm, open, counter, etc.)\n"
    "- The broker's persuasion angle and emotional tone\n"
    "- The CTA meaning — only improve its wording, not its intent\n"
    "- Overall structure and paragraph order\n"
    "- Conversation stage context (follow-up, counter-offer, closing, etc.)\n"
    "- Reply length — do not significantly expand or shrink\n\n"
    "WHAT YOU SHOULD FIX (sentence-level edits only):\n"
    "- Robotic AI openers — replace with a direct, natural opening sentence\n"
    "- Hype phrases (perfect domain, once in a lifetime, game-changing) — soften naturally, "
    "keeping the underlying point\n"
    "- Weak or generic CTAs — reword to feel more human and specific\n"
    "- Overly long sentences — split into two shorter ones\n"
    "- Walls of text — add a paragraph break where there is a natural topic shift\n"
    "- Repeated words — vary phrasing once\n\n"
    "WHAT YOU MUST NOT DO:\n"
    "- Do not rewrite sections that were not flagged\n"
    "- Do not invent new claims, benefits, or urgency\n"
    "- Do not weaken or soften the negotiation stance\n"
    "- Do not change the pricing or domain name\n"
    "- Do not add new paragraphs or significantly restructure the email\n"
    "- Do not replace the broker's voice with a generic sales voice\n\n"
    "STRATEGY: Prefer the minimum edit that fixes the issue. "
    "The result must feel like a polished version of the same email, not a new one.\n\n"
    "Write ONLY the corrected reply body. No explanation, no preamble, no subject line."
)

def _build_humanize_prompt(reply: str, flags: list[str], situation: str) -> str:
    """
    Build a surgical, section-targeted humanization prompt.

    Each flag group becomes a specific, scoped instruction so the model
    knows exactly which sentences to touch and which to leave alone.
    """
    fix_lines: list[str] = []

    hype_flags   = [f for f in flags if f.startswith("hype:")]
    ai_flags     = [f for f in flags if f.startswith("ai:")]
    cta_flags    = [f for f in flags if f.startswith(("cta:", "weak_cta:"))]
    opener_flags = [f for f in flags if f.startswith("opener:")]
    struct_flags = [f for f in flags if f.startswith(("structure:", "readability:"))]
    style_flags  = [f for f in flags if f.startswith("style:")]

    if opener_flags:
        fix_lines.append(
            "OPENER: The opening sentence sounds like a template. "
            "Replace it with a direct, natural sentence that gets straight to the point. "
            "Do not use 'I hope', 'I am writing', or 'I wanted to'. "
            "Keep everything after the opener exactly as-is."
        )

    if ai_flags:
        phrases = ", ".join(f.split(":",1)[1].replace("_"," ") for f in ai_flags)
        fix_lines.append(
            f"AI PHRASING: The following phrases sound robotic — fix only these sentences: {phrases}. "
            "Rewrite each affected sentence naturally. Leave all other sentences unchanged."
        )

    if hype_flags:
        phrases = ", ".join(f.split(":",1)[1].replace("_"," ") for f in hype_flags)
        fix_lines.append(
            f"HYPE LANGUAGE: Soften these phrases without removing the underlying point: {phrases}. "
            "Replace each with plain, credible language. "
            "Do not remove the value claim — just express it without exaggeration."
        )

    if cta_flags:
        if "cta:missing" in cta_flags:
            fix_lines.append(
                "CTA: There is no clear call-to-action. Add one natural closing question or "
                "next-step invitation at the end. Keep it brief and specific to the situation. "
                "Do not change anything else."
            )
        else:
            fix_lines.append(
                "CTA: The closing call-to-action is generic. Reword only that sentence to feel "
                "more direct and human. Preserve its intent — only improve the phrasing."
            )

    if struct_flags:
        struct_issues = [f.split(":",1)[1].replace("_"," ") for f in struct_flags]
        if "no paragraph breaks" in struct_issues or "long paragraphs" in struct_issues:
            fix_lines.append(
                "STRUCTURE: Add one paragraph break where there is a natural topic shift. "
                "Do not reorder sentences or change any wording."
            )
        if "long sentences" in struct_issues:
            fix_lines.append(
                "READABILITY: Split any sentence over 30 words into two shorter sentences. "
                "Do not change the meaning."
            )

    if style_flags:
        fix_lines.append(
            "REPETITION: A word or phrase is repeated too often. "
            "Replace one or two of the repeated instances with a synonym. "
            "Do not change anything else."
        )

    if not fix_lines:
        # Fallback — generic light polish, still conservative
        fix_lines.append(
            "Make light improvements to naturalness and readability. "
            "Do not change the meaning, structure, pricing, or CTA intent."
        )

    issues_block = "\n\n".join(fix_lines)

    return (
        f"BROKER SITUATION:\n{situation}\n\n"
        f"ORIGINAL REPLY (change as little as possible):\n{reply}\n\n"
        f"TARGETED FIXES REQUIRED:\n{issues_block}\n\n"
        "Important: Only edit the specific sentences described above. "
        "Leave all other sentences word-for-word. "
        "Write the corrected reply now:"
    )


def _model_tier(model: str) -> str:
    """Classify model capability tier for humanization decisions."""
    if model.startswith("groq:"):   return "groq"
    if "7b" in model.lower():       return "7b"
    return "3b"


def humanize_reply(
    reply:     str,
    situation: str,
    intent:    str = "",
    model:     str = MODEL,
    domain_name:  Optional[str] = None,
    asking_price: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Heuristic-first humanization layer.

    1. Run heuristic_score_reply() — zero latency
    2. If score is good enough: return as-is with the quality report
    3. If needs_humanization: rewrite via model (respects model tier)
    4. Re-score the rewrite to confirm improvement

    Returns (final_reply, quality_report_dict).
    The quality_report is always populated — callers can attach it to ReplyResult.
    Respects ENABLE_HUMANIZER flag.
    """
    # Always score first — report is always returned even if rewrite is disabled
    report = heuristic_score_reply(reply, intent=intent)

    if not ENABLE_HUMANIZER:
        return reply, report

    if not report["needs_humanization"]:
        print(f"[HUMANIZER] score={report['total']}/100 — no rewrite needed")
        return reply, report

    tier = _model_tier(model)

    # 3B: rewrite only for severe issues (score < 50 or critical flags)
    critical_flags = {"hype:perfect_domain", "hype:once_in_lifetime", "hype:game_changer",
                      "hype:amazing", "cta:missing", "opener:weak"}
    has_critical   = bool(critical_flags & set(report["flags"]))

    if tier == "3b" and report["total"] >= 50 and not has_critical:
        print(f"[HUMANIZER] 3B tier — score={report['total']}/100, no critical flags — skipping rewrite")
        return reply, report

    print(
        f"[HUMANIZER] tier={tier} score={report['total']}/100 "
        f"flags={len(report['flags'])} — running rewrite"
    )

    humanize_prompt = _build_humanize_prompt(reply, report["flags"], situation)

    try:
        client   = _get_client_for_model(model)
        # Temperature slightly higher for humanization — we want natural variety
        temp     = 0.65 if tier == "groq" else 0.55
        rewritten = client.generate(
            prompt      = humanize_prompt,
            system      = _HUMANIZE_SYSTEM,
            temperature = temp,
            max_tokens  = MAX_TOKENS,
        )
        if not rewritten or not rewritten.strip():
            print("[HUMANIZER] model returned empty — keeping original")
            return reply, report

        rewritten = rewritten.strip()

        # Safety check: don't accept a rewrite that drops pricing/domain info
        if domain_name and domain_name.lower() not in rewritten.lower():
            print(f"[HUMANIZER] rewrite dropped domain '{domain_name}' — keeping original")
            return reply, report
        if asking_price:
            # Extract numeric part for loose check (e.g. "$350" → "350")
            price_num = re.sub(r"[^\d]", "", asking_price)
            if price_num and price_num not in rewritten.replace(",", ""):
                print(f"[HUMANIZER] rewrite dropped price '{asking_price}' — keeping original")
                return reply, report

        # Re-score the rewrite to confirm it's an improvement
        new_report = heuristic_score_reply(rewritten, intent=intent)
        if new_report["total"] >= report["total"] - 5:   # accept if not worse by more than 5pts
            print(
                f"[HUMANIZER] rewrite accepted: {report['total']} → {new_report['total']}/100 "
                f"({len(new_report['flags'])} flags remaining)"
            )
            new_report["original_score"]  = report["total"]
            new_report["rewrite_applied"] = True
            return rewritten, new_report
        else:
            print(
                f"[HUMANIZER] rewrite rejected (score dropped {report['total']} → {new_report['total']}) "
                f"— keeping original"
            )
            report["rewrite_applied"] = False
            return reply, report

    except Exception as e:
        print(f"[HUMANIZER] rewrite failed: {e} — keeping original")
        report["rewrite_applied"] = False
        return reply, report


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — ACTIVE ROUTING
# ─────────────────────────────────────────────────────────────────────────────

_VALID_MODES    = {"template", "hybrid", "ai"}
_AUTO_SENTINELS = {None, "", "auto", "Auto", "AUTO"}


def select_effective_mode(
    requested_mode: Optional[str],
    analysis: "InputAnalysis",
) -> tuple[str, bool]:
    """
    Single source of truth for mode selection.
    Returns (effective_mode, router_acted).
    """
    if requested_mode not in _AUTO_SENTINELS:
        cleaned = (requested_mode or "").strip().lower()
        if cleaned in _VALID_MODES:
            print(f"[ROUTER_EXECUTION] requested={requested_mode} effective={cleaned} decided_by=caller reason=caller_override")
            return cleaned, False
        print(f"[ROUTER_EXECUTION] unknown mode '{requested_mode}' — treating as auto")

    rec = (getattr(analysis, "recommended_mode", "") or "").strip().lower()
    if rec in _VALID_MODES:
        print(f"[ROUTER_EXECUTION] requested=auto effective={rec} decided_by=router reason={analysis.routing_reason}")
        return rec, True
    if rec == "autonomous":
        print(f"[ROUTER_EXECUTION] requested=auto effective=ai decided_by=router reason=autonomous_mapped_to_ai")
        return "ai", True

    print(f"[ROUTER_EXECUTION] requested=auto effective=hybrid decided_by=router reason=fallback_no_valid_recommendation")
    return "hybrid", True


# ─────────────────────────────────────────────────────────────────────────────
# SMART EXTRACTION — conversational input → structured GenerateRequest fields
# Lightweight extraction layer. Does NOT touch the generation pipeline.
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_INTENTS = [
    "cold_outreach", "warm_outreach", "follow_up", "negotiation",
    "objection_handling", "price_discussion", "closing", "general",
]

_EXTRACT_TONES = [
    "professional", "friendly", "firm", "empathetic",
    "urgent", "concise", "persuasive",
]

_EXTRACT_PRESETS = [
    "cold_outreach", "warm_outreach", "follow_up", "counter_offer",
    "final_offer", "closing", "payment_reminder", "general",
]

_SMART_EXTRACT_SYSTEM = """You are a domain broker workflow assistant.
Your ONLY job is to extract structured fields from a broker's natural language description.

Output ONLY a valid JSON object — no markdown, no explanation, no preamble.

Fields to extract:
{
  "customer_message": "<the core prospect message or situation, cleaned — required>",
  "domain_name": "<domain being sold, e.g. ChicagoPlumber.com — null if not mentioned>",
  "asking_price": "<broker's asking price as string, e.g. '$350' — null if not mentioned>",
  "prospect_offer": "<prospect's offer as string, e.g. '$50' — null if not mentioned>",
  "prospect_name": "<prospect's first name — null if unknown>",
  "tone": "<one of: professional | friendly | firm | empathetic | urgent | concise | persuasive>",
  "intent": "<one of: cold_outreach | warm_outreach | follow_up | negotiation | objection_handling | price_discussion | closing | general>",
  "email_preset": "<one of: cold_outreach | warm_outreach | follow_up | counter_offer | final_offer | closing | payment_reminder | general — or null>",
  "goal": "<short phrase describing what the broker wants to achieve — null if unclear>",
  "stage": "<one of: new | contacted | negotiating | agreed | closed — best guess>",
  "urgency": <true if broker signals time pressure, else false>
}

Rules:
- customer_message must be a clean restatement useful for the AI reply generator (not just copied verbatim)
- If asking_price and prospect_offer are both present, infer intent = "negotiation" and email_preset = "counter_offer"
- Never hallucinate details not present in the input
- If a field truly cannot be determined, use null
- Output ONLY the JSON object"""


class SmartExtractRequest(BaseModel):
    raw_input: str
    lead_id: Optional[int] = None    # inject lead history from broker_memory if provided
    model: Optional[str]  = None     # override extraction model; None = auto (Groq > Ollama)

    @field_validator("raw_input")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw_input cannot be empty.")
        return v.strip()


class SmartExtractResponse(BaseModel):
    # Core GenerateRequest-compatible fields
    customer_message: str
    domain_name:      Optional[str] = None
    asking_price:     Optional[str] = None
    prospect_offer:   Optional[str] = None
    prospect_name:    Optional[str] = None
    tone:             Optional[str] = "professional"
    intent:           Optional[str] = "general"
    email_preset:     Optional[str] = None
    goal:             Optional[str] = None
    stage:            Optional[str] = None
    urgency:          bool          = False
    # Metadata
    backend_used:     str           = "ollama"
    model_used:       str           = ""
    timing_ms:        int           = 0
    lead_history_used: bool         = False


@app.post("/smart-extract", response_model=SmartExtractResponse)
async def smart_extract(req: SmartExtractRequest):
    """
    Convert a natural language broker description into structured GenerateRequest fields.

    - Uses Groq (fast) if GROQ_API_KEY is set, Ollama as fallback.
    - If lead_id is provided and broker_memory is available, injects lead history
      into the extraction prompt so the model can detect conversation stage.
    - Returns fields ready to populate the frontend form and feed into /generate-reply.
    - Does NOT call the generation pipeline — extraction only.
    """
    _t0 = time.monotonic()

    # ── Resolve extraction model ──────────────────────────────────────────────
    if req.model:
        extract_model = req.model
    elif GROQ_API_KEY:
        extract_model = GROQ_DEFAULT
    else:
        extract_model = MODEL

    backend_label = "groq" if extract_model.startswith("groq:") else "ollama"

    # ── Optionally inject lead history ────────────────────────────────────────
    lead_history_used = False
    lead_context = ""
    if req.lead_id and _MEMORY_AVAILABLE and memory_db:
        summary = memory_db.lead_summary(req.lead_id)
        if summary:
            lead_context = f"\n\nLEAD HISTORY (use to improve extraction accuracy):\n{summary}\n"
            lead_history_used = True

    # ── Build extraction prompt ───────────────────────────────────────────────
    extract_prompt = (
        f"Extract structured fields from this broker's description:{lead_context}\n\n"
        f"BROKER INPUT:\n{req.raw_input}\n\n"
        f"Return the JSON object now:"
    )

    # ── Call extraction model ─────────────────────────────────────────────────
    raw_json = ""
    fallback_used = False
    try:
        client   = _get_client_for_model(extract_model)
        raw_json = client.generate(
            prompt      = extract_prompt,
            system      = _SMART_EXTRACT_SYSTEM,
            temperature = 0.1,   # deterministic — we want consistent structured output
            max_tokens  = 400,
        )
    except Exception as primary_err:
        print(f"[SMART_EXTRACT] primary model {extract_model!r} failed: {primary_err}")
        raw_json = ""

    # ── Groq → Ollama fallback ────────────────────────────────────────────────
    if not raw_json and backend_label == "groq":
        print(f"[SMART_EXTRACT] falling back to ollama model={MODEL}")
        fallback_used  = True
        extract_model  = MODEL
        backend_label  = "ollama"
        try:
            client   = _get_client_for_model(MODEL)
            raw_json = client.generate(
                prompt      = extract_prompt,
                system      = _SMART_EXTRACT_SYSTEM,
                temperature = 0.1,
                max_tokens  = 400,
            )
        except Exception as fallback_err:
            print(f"[SMART_EXTRACT] ollama fallback also failed: {fallback_err}")

    if not raw_json:
        raise HTTPException(
            status_code=503,
            detail="Smart extraction failed — both Groq and Ollama unavailable. Use manual form input."
        )

    # ── Parse JSON safely ─────────────────────────────────────────────────────
    parsed: dict = {}
    try:
        # Strip markdown fences if the model wrapped it anyway
        clean = raw_json.strip()
        if clean.startswith("```"):
            clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
        # Find the JSON object
        brace_start = clean.find("{")
        brace_end   = clean.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            clean = clean[brace_start:brace_end]
        parsed = json.loads(clean)
    except Exception as parse_err:
        print(f"[SMART_EXTRACT] JSON parse error: {parse_err}\nRaw: {raw_json[:300]}")
        # Return a graceful minimum: use raw_input as the customer_message
        parsed = {"customer_message": req.raw_input}

    # ── Validate and sanitise fields ──────────────────────────────────────────
    customer_message = str(parsed.get("customer_message") or req.raw_input).strip()
    domain_name      = parsed.get("domain_name")  or None
    asking_price     = parsed.get("asking_price") or None
    prospect_offer   = parsed.get("prospect_offer") or None
    prospect_name    = parsed.get("prospect_name") or None
    goal             = parsed.get("goal") or None
    stage            = parsed.get("stage") or None
    urgency          = bool(parsed.get("urgency", False))

    tone = parsed.get("tone", "professional")
    if tone not in _EXTRACT_TONES:
        tone = "professional"

    intent = parsed.get("intent", "general")
    if intent not in _EXTRACT_INTENTS:
        intent = "general"

    email_preset = parsed.get("email_preset") or None
    if email_preset and email_preset not in _EXTRACT_PRESETS:
        email_preset = None

    timing_ms = int((time.monotonic() - _t0) * 1000)
    print(
        f"[SMART_EXTRACT] backend={backend_label} model={extract_model} "
        f"intent={intent} preset={email_preset} timing_ms={timing_ms}"
    )

    return SmartExtractResponse(
        customer_message  = customer_message,
        domain_name       = domain_name,
        asking_price      = asking_price,
        prospect_offer    = prospect_offer,
        prospect_name     = prospect_name,
        tone              = tone,
        intent            = intent,
        email_preset      = email_preset,
        goal              = goal,
        stage             = stage,
        urgency           = urgency,
        backend_used      = backend_label,
        model_used        = extract_model,
        timing_ms         = timing_ms,
        lead_history_used = lead_history_used,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/generate-reply", response_model=GenerateResponse)
async def generate_reply(req: GenerateRequest):
    """
    Standard reply — for direct prospect messages.
    mode=None/auto → router selects. Explicit mode always honoured.
    model=None → falls back to global MODEL constant.
    """
    _t_total = time.monotonic()
    tone     = req.tone or "professional and persuasive"

    # ── Model resolution ──────────────────────────────────────────────────────
    effective_model = req.model or MODEL
    print(f"[MODEL_ROUTER] requested={req.model!r} effective={effective_model} endpoint=/generate-reply")

    # ── Analysis ─────────────────────────────────────────────────────────────
    with _Timer("analysis"):
        analysis = analyse(req.customer_message)
    intent = analysis.primary_intent

    # ── Retrieval ─────────────────────────────────────────────────────────────
    with _Timer("retrieval"):
        replies = load_replies()
        api_key_for_embed = os.getenv("VOYAGE_API_KEY", "")
        examples, method = retrieve(req.customer_message, replies, intent, api_key_for_embed, TOP_K)

    # ── Routing ───────────────────────────────────────────────────────────────
    effective_mode, router_acted = select_effective_mode(req.mode, analysis)

    # ── Generation ────────────────────────────────────────────────────────────
    with _Timer("generation"):
        if effective_mode == "template":
            template_result = build_template_reply(
                customer_message=req.customer_message,
                domain_name=req.domain_name,
                asking_price=req.asking_price,
                force_intent=intent,
            )
            raw       = template_result.get("reply", "")
            formatted = format_email_body(raw, req.prospect_name, req.sender_name)
            with _Timer("qc"):
                qc = run_full_qc(formatted, intent=intent)
            final = qc["reply"]
            variations = [ReplyResult(reply=final, label="Template", confidence_score=75,
                                      confidence_reason="Template mode — no AI scoring")]
            subject = generate_subject(intent, req.domain_name)

        elif effective_mode == "hybrid":
            base_body  = run_hybrid_mode(req.customer_message, intent,
                                         req.domain_name, req.asking_price, tone,
                                         model=effective_model)
            formatted  = format_email_body(base_body, req.prospect_name, req.sender_name)
            fixed      = quality_guard(formatted, req.customer_message, intent=intent,
                                       model=effective_model)
            # ── Humanization ──────────────────────────────────────────
            with _Timer("humanize"):
                fixed, quality_report = humanize_reply(
                    reply        = fixed,
                    situation    = req.customer_message,
                    intent       = intent,
                    model        = effective_model,
                    domain_name  = req.domain_name,
                    asking_price = req.asking_price,
                )
            if ENABLE_AI_SCORING:
                score, reason = score_reply_ollama(req.customer_message, fixed,
                                                   model=effective_model)
            else:
                score, reason = 75, "AI scoring disabled"
            variations = [ReplyResult(reply=fixed, label="Hybrid",
                                      confidence_score=score, confidence_reason=reason,
                                      quality_report=quality_report)]
            subject = generate_subject_ai(intent, variations[0].reply, req.domain_name,
                                          model=effective_model)

        else:  # effective_mode == "ai"
            # ── Conversation-stage intelligence ───────────────────────────────
            lead_id  = getattr(req, "lead_id", None)
            lead_ctx = _get_lead_context_rich(lead_id)
            stage, stage_reason = detect_conversation_stage(
                intent         = intent,
                message        = req.customer_message,
                outreach_count = lead_ctx["outreach_count"],
                offers         = lead_ctx["offers"],
                lead_stage     = lead_ctx["lead_stage"],
                asking_price   = req.asking_price,
            )
            print(f"[STAGE_INTEL] stage={stage} reason={stage_reason}")
            stage_block  = _stage_prompt_block(stage, stage_reason)
            offer_intel  = _offer_intelligence(
                lead_ctx["offers"], req.asking_price, req.customer_message
            )

            # ── Stage-aware retrieval ─────────────────────────────────────────
            examples, method = retrieve_stage_aware(
                req.customer_message, load_replies(), intent,
                os.getenv("VOYAGE_API_KEY", ""), TOP_K, stage=stage,
            )

            # ── Phase 2 — prefetch angle inventory + objections ──────────────────
            # Non-blocking: failures leave _p2_inventory=None and the strategy
            # layer falls back to Phase 1 keyword-scan path automatically.
            _p2_inventory  = None
            _p2_objections = []
            if lead_id and _MEMORY_AVAILABLE and memory_db:
                try:
                    from angle_memory import build_angle_inventory, ObjectionRecord
                    _p2_inventory = build_angle_inventory(lead_id, memory_db)
                    _p2_obj_rows  = memory_db.get_objection_history(
                        lead_id, unresolved_only=True
                    )
                    _p2_objections = [
                        ObjectionRecord.from_db_row(r) for r in _p2_obj_rows
                    ]
                except Exception as _p2_err:
                    print(f"[P2] angle inventory fetch failed (non-blocking): {_p2_err}")

            # ── ReplyStrategy reasoning layer ─────────────────────────────────
            strategy = build_strategy(StrategySignals(
                intent            = intent,
                message           = req.customer_message,
                stage             = stage,
                neg_state         = _detect_negotiation_state(req.customer_message, req.asking_price),
                response_frame    = _classify_response_frame(req.customer_message),
                tone_requested    = tone,
                asking_price      = req.asking_price,
                outreach_count    = lead_ctx["outreach_count"],
                has_questions     = analysis.has_questions,
                question_count    = len(analysis.questions) if analysis.has_questions else 0,
                ambiguity_level   = getattr(analysis, "ambiguity_level", "low"),
                has_multiple_intents = getattr(analysis, "has_multiple_intents", False),
                secondary_intents = getattr(analysis, "secondary_intents", []),
                intent_confidence = getattr(analysis, "primary_intent_confidence", 1.0),
                email_preset      = getattr(req, "email_preset", None),
                domain_name       = req.domain_name,
                lead_stage        = lead_ctx["lead_stage"],
                no_domain_no_price = not req.domain_name and not req.asking_price,
                prior_outreach_bodies = [
                    o.get("body", "") for o in lead_ctx.get("outreach", [])
                    if o.get("body")
                ],
                stage_signal_strength = (
                    "memory"  if lead_ctx["lead_stage"] else
                    "offer"   if lead_ctx["offers"] else
                    "intent"
                ),
                # Phase 2 — angle inventory + objection context
                lead_id              = lead_id if "lead_id" in dir() else None,
                angle_inventory      = _p2_inventory,
                unresolved_objection_records = _p2_objections,
            ))

            base_prompt = build_reply_prompt_ai(
                req.customer_message, intent, examples, tone,
                req.domain_name, req.asking_price, method,
                analysis     = analysis,
                email_preset = getattr(req, "email_preset", None),
                lead_context = lead_ctx["summary_text"],
                stage_block  = stage_block,
                offer_intel  = offer_intel,
                strategy     = strategy,
            )
            variations = generate_variations_ai(
                base_prompt,
                num=req.num_variations,
                situation=req.customer_message,
                prospect_name=req.prospect_name,
                sender_name=req.sender_name,
                intent=intent,
                domain_name=req.domain_name,
                model=effective_model,
                strategy=strategy if "strategy" in dir() else None,
            )
            subject = generate_subject_ai(intent, variations[0].reply, req.domain_name,
                                          model=effective_model)

            # ── Phase 1 silent memory logging ─────────────────────────────────
            # Records which value angles were suppressed (= already used) and
            # which objections the prospect's message contains.
            # Non-blocking: failures are logged but never affect reply delivery.
            # Rollback: remove this block.  Zero other changes required.
            if lead_id and _MEMORY_AVAILABLE and memory_db:
                try:
                    _outreach_seq = lead_ctx["outreach_count"] + 1
                    _sel = getattr(strategy, "selected_angle", "")
                    if _sel:
                        memory_db.log_angle(
                            lead_id      = lead_id,
                            angle_id     = _sel,
                            outreach_seq = _outreach_seq,
                            pitched_as   = "primary",
                        )
                    for _topic in getattr(strategy, "exhausted_angles", []):
                        if _topic != _sel:
                            memory_db.log_angle(
                                lead_id      = lead_id,
                                angle_id     = _topic,
                                outreach_seq = _outreach_seq,
                                pitched_as   = "mentioned",
                            )
                    print(f"[ANGLE_LOG] lead={lead_id} seq={_outreach_seq} "
                          f"selected={_sel!r} "
                          f"exhausted={getattr(strategy, 'exhausted_angles', [])}")
                except Exception as _log_err:
                    print(f"[ANGLE_LOG] silent logging failed (non-blocking): {_log_err}")

                try:
                    from angle_memory import _OBJECTION_REGISTRY
                    _msg_low = req.customer_message.lower()
                    for _obj_type, _obj_info in _OBJECTION_REGISTRY.items():
                        _signals = _obj_info.get("detection_signals", [])
                        if any(_sig in _msg_low for _sig in _signals):
                            _snippet = req.customer_message[:200]
                            memory_db.log_objection(
                                lead_id        = lead_id,
                                objection_type = _obj_type,
                                source_snippet = _snippet,
                            )
                            print(f"[OBJECTION_LOG] lead={lead_id} type={_obj_type}")
                except Exception as _obj_err:
                    print(f"[OBJECTION_LOG] silent logging failed (non-blocking): {_obj_err}")

    _Timer.log("total", int((time.monotonic() - _t_total) * 1000))

    return GenerateResponse(
        subject=subject,
        replies=variations,
        detected_intent=intent,
        retrieval_method=method,
        similar_examples_used=[{"category": ex.get("category",""), "snippet": ex["customer_message"][:80]} for ex in examples],
        model_used=effective_model,
        model_requested=req.model,
        tone_applied=tone,
        pipeline_debug={
            "primary_intent":         analysis.primary_intent,
            "secondary_intents":      analysis.secondary_intents,
            "has_questions":          analysis.has_questions,
            "questions":              analysis.questions,
            "primary_question_type":  analysis.primary_question_type,
            "question_types":         {k: v for k, v in analysis.question_types.items() if v},
            "answer_hints":           analysis.answer_hints,
            "intent_confidence":      round(analysis.primary_intent_confidence, 3),
            "ambiguity_level":        analysis.ambiguity_level,
            "intent_scores":          {k: round(v, 3) for k, v in sorted(
                                        analysis.intent_scores.items(),
                                        key=lambda x: x[1], reverse=True)},
            "recommended_mode":       getattr(analysis, "recommended_mode", ""),
            "routing_reason":         analysis.routing_reason,
            "requested_mode":         req.mode if req.mode not in {None,"","auto"} else "auto",
            "effective_mode":         effective_mode,
            "router_acted":           router_acted,
            "reply_strategy": (
                {
                    "goal":             strategy.primary_goal,
                    "buyer":            strategy.buyer_state,
                    "posture":          strategy.conversation_posture,
                    "cta":              strategy.cta_style,
                    "length":           strategy.reply_length,
                    "persuasion":       strategy.persuasion_level,
                    "urgency":          strategy.urgency_level,
                    "objective":        strategy.reply_objective,
                    "progression_goal": strategy.progression_goal,
                    "suppressed_topics": strategy.suppressed_topics,
                    "tone_guidance":    strategy.tone_guidance,
                    "confidence": {
                        "stage": strategy.stage_confidence,
                        "buyer": strategy.buyer_confidence,
                        "goal":  strategy.goal_confidence,
                    },
                    "trace":    strategy.reasoning_trace,
                }
                if "strategy" in dir() and strategy is not None else None
            ),
            "config": {
                "variations":  ENABLE_VARIATIONS,
                "ai_scoring":  ENABLE_AI_SCORING,
                "ai_subject":  ENABLE_AI_SUBJECT,
                "qc_rewrite":  ENABLE_QC_REWRITE,
            },
        },
    )
@app.post("/generate-reply/situation", response_model=SituationResponse)
async def generate_reply_situation(req: SituationRequest):
    """
    Situation-based generation.
    model=None → falls back to global MODEL constant.
    """
    _t_total  = time.monotonic()
    tone      = req.tone or "professional and persuasive"

    # ── Model resolution ──────────────────────────────────────────────────────
    effective_model = req.model or MODEL
    print(f"[MODEL_ROUTER] requested={req.model!r} effective={effective_model} endpoint=/generate-reply/situation")

    with _Timer("analysis"):
        analysis  = analyse(req.situation)
    intent    = req.force_intent or detect_situation_intent(req.situation)
    intensity = req.force_intensity or INTENT_TO_INTENSITY.get(intent, "medium")

    with _Timer("retrieval"):
        replies = load_replies()
        api_key_for_embed = os.getenv("VOYAGE_API_KEY", "")
        examples, method = retrieve(req.situation, replies, intent, api_key_for_embed, TOP_K)

    effective_mode, router_acted = select_effective_mode(
        getattr(req, "mode", None), analysis
    )

    with _Timer("generation"):
        if effective_mode == "template":
            template_result = build_template_reply(
                customer_message=req.situation,
                domain_name=req.domain_name,
                asking_price=req.asking_price,
                force_intent=intent,
            )
            raw       = template_result.get("reply", "")
            formatted = format_email_body(raw, req.prospect_name, req.sender_name)
            with _Timer("qc"):
                qc = run_full_qc(formatted, intent=intent)
            final = qc["reply"]
            variations = [ReplyResult(reply=final, label="Template", confidence_score=75,
                                      confidence_reason="Template mode — no AI scoring")]
            subject = generate_subject(intent, req.domain_name)

        elif effective_mode == "hybrid":
            base_body = run_hybrid_mode(req.situation, intent,
                                        req.domain_name, req.asking_price, tone,
                                        model=effective_model)
            formatted = format_email_body(base_body, req.prospect_name, req.sender_name)
            fixed     = quality_guard(formatted, req.situation, intent=intent,
                                      model=effective_model)
            if ENABLE_AI_SCORING:
                score, reason = score_reply_ollama(req.situation, fixed, model=effective_model)
            else:
                score, reason = 75, "AI scoring disabled"
            variations = [ReplyResult(reply=fixed, label="Hybrid",
                                      confidence_score=score, confidence_reason=reason)]
            subject = generate_subject_ai(intent, variations[0].reply, req.domain_name,
                                          model=effective_model)

        else:  # effective_mode == "ai"
            # ── Conversation-stage intelligence ───────────────────────────────
            lead_id  = getattr(req, "lead_id", None)
            lead_ctx = _get_lead_context_rich(lead_id)
            stage, stage_reason = detect_conversation_stage(
                intent         = intent,
                message        = req.situation,
                outreach_count = lead_ctx["outreach_count"],
                offers         = lead_ctx["offers"],
                lead_stage     = lead_ctx["lead_stage"],
                asking_price   = req.asking_price,
            )
            print(f"[STAGE_INTEL] stage={stage} reason={stage_reason}")
            stage_block = _stage_prompt_block(stage, stage_reason)
            offer_intel = _offer_intelligence(
                lead_ctx["offers"], req.asking_price, req.situation
            )
            examples, method = retrieve_stage_aware(
                req.situation, load_replies(), intent,
                os.getenv("VOYAGE_API_KEY", ""), TOP_K, stage=stage,
            )

            # ── Phase 2 — prefetch angle inventory + objections ──────────────────
            # Non-blocking: failures leave _p2_inventory=None and the strategy
            # layer falls back to Phase 1 keyword-scan path automatically.
            _p2_inventory  = None
            _p2_objections = []
            if lead_id and _MEMORY_AVAILABLE and memory_db:
                try:
                    from angle_memory import build_angle_inventory, ObjectionRecord
                    _p2_inventory = build_angle_inventory(lead_id, memory_db)
                    _p2_obj_rows  = memory_db.get_objection_history(
                        lead_id, unresolved_only=True
                    )
                    _p2_objections = [
                        ObjectionRecord.from_db_row(r) for r in _p2_obj_rows
                    ]
                except Exception as _p2_err:
                    print(f"[P2] angle inventory fetch failed (non-blocking): {_p2_err}")

            # ── ReplyStrategy reasoning layer ─────────────────────────────────
            strategy = build_strategy(StrategySignals(
                intent            = intent,
                message           = req.situation,
                stage             = stage,
                neg_state         = _detect_negotiation_state(req.situation, req.asking_price),
                response_frame    = _classify_response_frame(req.situation),
                tone_requested    = tone,
                asking_price      = req.asking_price,
                outreach_count    = lead_ctx["outreach_count"],
                has_questions     = analysis.has_questions,
                question_count    = len(analysis.questions) if analysis.has_questions else 0,
                ambiguity_level   = getattr(analysis, "ambiguity_level", "low"),
                has_multiple_intents = getattr(analysis, "has_multiple_intents", False),
                secondary_intents = getattr(analysis, "secondary_intents", []),
                intent_confidence = getattr(analysis, "primary_intent_confidence", 1.0),
                email_preset      = getattr(req, "email_preset", None),
                domain_name       = req.domain_name,
                lead_stage        = lead_ctx["lead_stage"],
                no_domain_no_price = not req.domain_name and not req.asking_price,
                prior_outreach_bodies = [
                    o.get("body", "") for o in lead_ctx.get("outreach", [])
                    if o.get("body")
                ],
                stage_signal_strength = (
                    "memory"  if lead_ctx["lead_stage"] else
                    "offer"   if lead_ctx["offers"] else
                    "intent"
                ),
                # Phase 2 — angle inventory + objection context
                lead_id              = lead_id if "lead_id" in dir() else None,
                angle_inventory      = _p2_inventory,
                unresolved_objection_records = _p2_objections,
            ))

            base_prompt = build_reply_prompt_ai(
                req.situation, intent, examples, tone,
                req.domain_name, req.asking_price, method,
                analysis     = analysis,
                email_preset = getattr(req, "email_preset", None),
                lead_context = lead_ctx["summary_text"],
                stage_block  = stage_block,
                offer_intel  = offer_intel,
                strategy     = strategy,
            )
            variations = generate_variations_ai(
                base_prompt,
                num=req.num_variations,
                situation=req.situation,
                prospect_name=req.prospect_name,
                sender_name=req.sender_name,
                intent=intent,
                domain_name=req.domain_name,
                model=effective_model,
                strategy=strategy if "strategy" in dir() else None,
            )
            subject = generate_subject_ai(intent, variations[0].reply, req.domain_name,
                                          model=effective_model)

            # ── Phase 1 silent memory logging ─────────────────────────────────
            # Mirrors the same block in /generate-reply.
            # Non-blocking: failures never affect reply delivery.
            # Rollback: remove this block.  Zero other changes required.
            if lead_id and _MEMORY_AVAILABLE and memory_db:
                try:
                    _outreach_seq = lead_ctx["outreach_count"] + 1
                    _sel = getattr(strategy, "selected_angle", "")
                    if _sel:
                        memory_db.log_angle(
                            lead_id      = lead_id,
                            angle_id     = _sel,
                            outreach_seq = _outreach_seq,
                            pitched_as   = "primary",
                        )
                    for _topic in getattr(strategy, "exhausted_angles", []):
                        if _topic != _sel:
                            memory_db.log_angle(
                                lead_id      = lead_id,
                                angle_id     = _topic,
                                outreach_seq = _outreach_seq,
                                pitched_as   = "mentioned",
                            )
                    print(f"[ANGLE_LOG] lead={lead_id} seq={_outreach_seq} "
                          f"selected={_sel!r} "
                          f"exhausted={getattr(strategy, 'exhausted_angles', [])}")
                except Exception as _log_err:
                    print(f"[ANGLE_LOG] silent logging failed (non-blocking): {_log_err}")

                try:
                    from angle_memory import _OBJECTION_REGISTRY
                    _msg_low = req.situation.lower()
                    for _obj_type, _obj_info in _OBJECTION_REGISTRY.items():
                        _signals = _obj_info.get("detection_signals", [])
                        if any(_sig in _msg_low for _sig in _signals):
                            _snippet = req.situation[:200]
                            memory_db.log_objection(
                                lead_id        = lead_id,
                                objection_type = _obj_type,
                                source_snippet = _snippet,
                            )
                            print(f"[OBJECTION_LOG] lead={lead_id} type={_obj_type}")
                except Exception as _obj_err:
                    print(f"[OBJECTION_LOG] silent logging failed (non-blocking): {_obj_err}")

    _Timer.log("total", int((time.monotonic() - _t_total) * 1000))

    signals = []
    if analysis.has_questions:
        signals.append(f"Questions: {len(analysis.questions)} detected")
    if analysis.has_multiple_intents:
        signals.append(f"Secondary: {', '.join(s.replace('_',' ').title() for s in analysis.secondary_intents)}")
    signals_note = " · ".join(signals)

    return SituationResponse(
        subject=subject,
        replies=variations,
        detected_intent=intent,
        pitch_intensity=intensity,
        situation_interpreted=(
            f"Intent: {intent.replace('_', ' ').title()} · "
            f"Intensity: {intensity.title()} · "
            f"Mode: {effective_mode}"
            + (f" · {signals_note}" if signals_note else "")
        ),
        model_used=effective_model,
        model_requested=req.model,
        tone_applied=tone,
        pipeline_debug={
            "primary_intent":         analysis.primary_intent,
            "secondary_intents":      analysis.secondary_intents,
            "has_questions":          analysis.has_questions,
            "questions":              analysis.questions,
            "primary_question_type":  analysis.primary_question_type,
            "question_types":         {k: v for k, v in analysis.question_types.items() if v},
            "answer_hints":           analysis.answer_hints,
            "intent_confidence":      round(analysis.primary_intent_confidence, 3),
            "ambiguity_level":        analysis.ambiguity_level,
            "intent_scores":          {k: round(v, 3) for k, v in sorted(
                                        analysis.intent_scores.items(),
                                        key=lambda x: x[1], reverse=True)},
            "recommended_mode":       getattr(analysis, "recommended_mode", ""),
            "routing_reason":         analysis.routing_reason,
            "requested_mode":         getattr(req, "mode", None) or "auto",
            "effective_mode":         effective_mode,
            "router_acted":           router_acted,
            "config": {
                "variations": ENABLE_VARIATIONS,
                "ai_scoring": ENABLE_AI_SCORING,
                "ai_subject": ENABLE_AI_SUBJECT,
                "qc_rewrite": ENABLE_QC_REWRITE,
            },
        },
    )




@app.post("/generate-reply/template")
async def generate_reply_template(req: TemplateRequest):
    """
    Template mode: keyword matching + component assembly.

    Intelligent fallback logic:
    - If the template produces a relevant, intent-matched reply → use it
      (and polish it if ai_polish=True)
    - If the message is a question or informational request that the template
      system cannot answer meaningfully (intent=general, no strong match) →
      bypass the template entirely and answer directly via Ollama regardless
      of whether ai_polish is checked
    This prevents the AI from polishing a generic placeholder in response to
    a specific question it has no template for.
    """
    result = build_template_reply(
        customer_message=req.customer_message,
        domain_name=req.domain_name,
        asking_price=req.asking_price,
        force_intent=req.force_intent,
        response_length=getattr(req, "response_length", "medium") or "medium",
        length_instructions=getattr(req, "length_instructions", None),
    )

    detected_intent = result.get("detected_intent", "general")
    effective_model = req.model or MODEL

    # ── Relevance check ──────────────────────────────────────────────────────
    # Determine whether the template result actually answers the message.
    # A template reply is NOT relevant when:
    #   - intent resolved to general/general_response (no specific template matched)
    #   - AND the message looks like an informational/educational question
    # In that case, bypass the template and answer directly via Ollama.

    _GENERIC_INTENTS = {"general", "general_response"}
    _QUESTION_SIGNALS = [
        "?", "what is", "what are", "what does", "how does", "how do",
        "how long", "how much", "explain", "tell me", "can you explain",
        "what happens", "what's the", "whats the", "why does", "why is",
        "when does", "when is", "who is", "which is", "is it", "does it",
        "can i", "will it", "do i need", "what should",
    ]

    msg_low = req.customer_message.lower()
    is_generic_intent  = detected_intent in _GENERIC_INTENTS
    is_question        = any(sig in msg_low for sig in _QUESTION_SIGNALS)
    template_irrelevant = is_generic_intent and is_question

    if template_irrelevant:
        # ── Direct answer mode — template bypassed ───────────────────────────
        # Use req.model if sent, otherwise default to 7b for factual accuracy.
        bypass_model = req.model or "qwen2.5:7b"
        print(f"[TEMPLATE] intent={detected_intent} question_detected=True → bypassing template, answering directly (model={bypass_model})")

        context_parts = []
        if req.domain_name:   context_parts.append(f"Domain: {req.domain_name}")
        if req.asking_price:  context_parts.append(f"Asking price: {req.asking_price}")
        context_line = "\n".join(context_parts) + "\n" if context_parts else ""

        tone_with_length = req.tone or "professional and helpful"
        resp_len = getattr(req, "response_length", "medium") or "medium"
        if resp_len == "short":
            length_note = "1-2 sentences maximum. Be direct and clear. No padding."
        elif resp_len == "long":
            length_note = "Answer in 150-250 words. Be thorough but natural."
        else:
            length_note = "Answer in 60-100 words. 2-3 sentences. No padding."

        direct_prompt = (
            f"{context_line}"
            f"Message or question:\n\"{req.customer_message}\"\n\n"
            f"Write a short email reply that answers this directly and accurately. "
            f"Start with 'Hi,' — do NOT use placeholders like [Name] or [Recipient]. "
            f"Answer the question clearly and factually in the body. "
            f"End with a natural sign-off like 'Best regards' — do NOT use [Your Name]. "
            f"Do not treat this as a sales pitch. "
            f"Do not add a call to action unless the situation clearly calls for one. "
            f"IMPORTANT: If no domain details are provided above, do NOT invent any — "
            f"no registration dates, no ages, no traffic numbers, no specific facts. "
            f"Answer only from general domain industry knowledge.\n\n"
            f"Tone: {tone_with_length}.\n"
            f"{length_note}\n\n"
            f"Write only the email body — no subject line, no metadata:"
        )

        sys_prompt = _select_system_prompt_for_mode(bypass_model, mode="ai")

        try:
            ollama  = _get_client_for_model(bypass_model)
            print(f"[AI_BACKEND] backend={('groq' if bypass_model.startswith('groq:') else 'ollama')} model={bypass_model} label=template_direct_answer intent={detected_intent}")
            raw = ollama.generate(
                prompt      = direct_prompt,
                system      = sys_prompt,
                temperature = 0.4,
                max_tokens  = MAX_TOKENS,
            )
            if raw and raw.strip():
                answer = raw.strip()
                return {
                    "reply":                   answer,
                    "polished_reply":          answer,
                    "original_template_reply": result.get("reply", ""),
                    "detected_intent":         detected_intent,
                    "mode":                    "direct_answer",
                    "ai_polish":               True,
                    "components_used":         result.get("components_used", {}),
                    "note":                    "Template bypassed — question answered directly by AI",
                }
        except Exception as e:
            print(f"[TEMPLATE] direct_answer failed: {e} — falling back to template result")

    # ── Normal flow — template is relevant (or Ollama unavailable) ───────────
    if not req.ai_polish:
        return result

    # ai_polish=True → polish the template reply via Ollama
    tone_with_length = req.tone or "professional and persuasive"
    resp_len = getattr(req, "response_length", "medium") or "medium"
    if resp_len == "long":
        tone_with_length += ". Write a full response of 160–320 words across 2–3 natural paragraphs. No bullet points, no hype language."
    elif resp_len == "short":
        tone_with_length += ". Keep the reply under 90 words — 2 to 3 sentences, direct and conversational."

    print(f"[MODEL_ROUTER] requested={req.model!r} effective={effective_model} endpoint=/generate-reply/template")

    if effective_model.startswith("groq:"):
        # Groq path — polish directly
        template_reply = result["reply"]
        polish_prompt = (
            f"You are polishing a domain sales email reply.\n\n"
            f"Original prospect message:\n\"{req.customer_message}\"\n\n"
            f"Template reply to polish:\n\"{template_reply}\"\n\n"
            f"Improve the reply so it sounds natural and human. "
            f"Keep the same intent and structure. "
            f"Do not add facts not in the template. "
            f"Tone: {tone_with_length}.\n\n"
            f"Write only the polished email body:"
        )
        try:
            client = _get_client_for_model(effective_model)
            sys = _select_system_prompt(effective_model)
            polished = client.generate(prompt=polish_prompt, system=sys,
                                       temperature=0.7, max_tokens=MAX_TOKENS)
            if polished and polished.strip():
                return {**result, "polished_reply": polished.strip(), "ai_polish": True}
        except Exception as e:
            print(f"[AI_BACKEND] Groq template polish failed: {e} — returning template")
        return result
    else:
        return ai_polish_reply(
            template_reply=result["reply"], customer_message=req.customer_message,
            intent=result["detected_intent"], api_key="",
            domain_name=req.domain_name, asking_price=req.asking_price, tone=tone_with_length,
            backend="ollama", ollama_model=effective_model,
            ollama_base_url=OLLAMA_BASE_URL, ollama_timeout=OLLAMA_TIMEOUT,
        )


@app.post("/generate-reply/template/detect-intent")
async def detect_intent_template(req: TemplateRequest):
    intent = req.force_intent or detect_template_intent(req.customer_message)
    return {"customer_message": req.customer_message, "detected_intent": intent,
            "available_intents": list(TEMPLATE_INTENT_KEYWORDS.keys())}


@app.post("/debug/analyse")
async def debug_analyse(body: dict):
    """
    Debug endpoint — runs the full pipeline analysis on any message and
    returns every layer of the decision process.

    Useful for testing question detection, intent scoring, and flow logic.

    Request body:
        { "message": "Does it have traffic? Also can you come down on price?" }

    Response includes:
        - primary_intent + secondary_intents
        - all matched intents with scores
        - detected questions
        - question type for each question (factual / how_to / clarification / comparison)
        - answer hints per type
        - the exact prompt blocks that will be injected into Claude
        - the flow instruction that tells Claude the reply order
    """
    message = body.get("message", "")
    if not message:
        return {"error": "Provide a 'message' field in the request body."}

    analysis = analyse(message)

    return {
        "input": message,
        # ── Step 1: Intent ─────────────────────────────────────────────────
        "step_1_intent": {
            "primary_intent":     analysis.primary_intent,
            "secondary_intents":  analysis.secondary_intents,
            "all_intents_ranked": analysis.all_intents,
            "has_multiple":       analysis.has_multiple_intents,
        },
        # ── Phase 1: Confidence ────────────────────────────────────────────
        "step_1b_confidence": {
            "primary_intent_confidence": round(analysis.primary_intent_confidence, 3),
            "ambiguity_level":           analysis.ambiguity_level,
            "intent_scores":             {
                k: round(v, 3)
                for k, v in sorted(
                    analysis.intent_scores.items(),
                    key=lambda x: x[1], reverse=True
                )
            },
            "interpretation": (
                "High confidence — proceed with primary intent"
                if analysis.primary_intent_confidence >= 0.70
                else "Medium confidence — secondary intents noted"
                if analysis.primary_intent_confidence >= 0.40
                else "Low confidence — reply may need extra care"
            ),
        },
        # ── Step 2: Questions ─────────────────────────────────────────────
        "step_2_questions": {
            "has_questions":          analysis.has_questions,
            "questions_found":        analysis.questions,
            "primary_question_type":  analysis.primary_question_type,
            "by_type":                {k: v for k, v in analysis.question_types.items() if v},
        },
        # ── Step 3: Strategy ──────────────────────────────────────────────
        "step_3_strategy": {
            "answer_hints":  analysis.answer_hints,
            "flow_order":    build_flow_instruction(analysis),
        },
        # ── Prompt blocks (what gets injected into Claude) ─────────────────
        "prompt_blocks": {
            "question_block":    analysis.question_block or "(none — no questions detected)",
            "multi_intent_note": analysis.multi_intent_note or "(none — single intent)",
            "debug_block":       analysis.debug_block,
        },
        # ── Phase 3: Routing recommendation ───────────────────────────────
        "step_4_routing": {
            "recommended_mode":  getattr(analysis, "recommended_mode", ""),
            "routing_reason":    analysis.routing_reason,
            "explanation":       analysis.routing_explanation,
            "template_covered":  analysis.primary_intent in (
                # Mirror of _TEMPLATE_COVERED_INTENTS — checked at response time
                # so this field is always accurate even if the set changes.
                # Importing the set directly would be cleaner; this avoids
                # adding a new import to main.py for a display-only field.
                "agreed_no_pay","angry","cold_outreach","competitor_comparison",
                "demo_offer","development","domain_metrics","expired_owner",
                "extension","feature_explanation","follow_up","follow_up_after_interest",
                "follow_up_after_pricing","follow_up_no_response","general",
                "general_response","have_website","how_it_works","identity",
                "low_budget","meeting_request","negotiation","no_thanks",
                "not_interested_ask_why","not_now","objection_handling","partner",
                "payment_issue","payment_method","post_purchase","price_inquiry",
                "price_negotiation","price_too_high","rank_well","re_engagement",
                "refund","related_domains","renewal_fees","request_info","sales_pitch",
                "soft_pitch","trust_building","trust_issue","value_reminder","why_buy",
            ),
            "acting_on_recommendation": False,   # Phase 4 will flip this to True
            "note": (
                "This is a recommendation only. Current endpoints do not automatically "
                "route based on this value. Automatic routing will be enabled in Phase 4."
            ),
        },
    }


@app.post("/generate-reply/stream")
async def generate_reply_stream(req: GenerateRequest):
    """Streaming reply via Ollama, chunked as SSE. Respects mode routing and model selection."""
    effective_model = req.model or MODEL
    print(f"[MODEL_ROUTER] requested={req.model!r} effective={effective_model} endpoint=/generate-reply/stream")

    with _Timer("analysis"):
        analysis = analyse(req.customer_message)
    intent   = analysis.primary_intent
    replies  = load_replies()
    api_key_for_embed = os.getenv("VOYAGE_API_KEY", "")
    examples, method = retrieve(req.customer_message, replies, intent, api_key_for_embed, TOP_K)
    effective_mode, router_acted = select_effective_mode(req.mode, analysis)

    async def stream() -> AsyncGenerator[str, None]:
        try:
            if effective_mode == "template":
                template_result = build_template_reply(
                    customer_message=req.customer_message,
                    domain_name=req.domain_name,
                    asking_price=req.asking_price,
                    force_intent=intent,
                )
                full_text = format_email_body(
                    template_result.get("reply", ""),
                    req.prospect_name, req.sender_name
                )
            else:
                prompt_builder = (
                    build_reply_prompt_ai if effective_mode == "ai"
                    else build_reply_prompt
                )
                user_prompt = prompt_builder(
                    req.customer_message, intent, examples,
                    req.tone or "professional and persuasive",
                    req.domain_name, req.asking_price, method,
                    analysis=analysis,
                )
                print(f"[AI_BACKEND] backend={('groq' if effective_model.startswith('groq:') else 'ollama')} model={effective_model} label=stream effective_mode={effective_mode}")
                client    = _get_client_for_model(effective_model)
                full_text = client.generate(
                    prompt=user_prompt,
                    system=_select_system_prompt_for_mode(effective_model, effective_mode),
                    temperature=0.7, max_tokens=MAX_TOKENS,
                )
                if not full_text:
                    yield "data: [ERROR] Ollama unavailable or returned empty response\n\n"
                    return

            words = full_text.split(" ")
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i+5])
                if i + 5 < len(words): chunk += " "
                yield f"data: {chunk.replace(chr(10), chr(92)+'n')}\n\n"

            score, reason = score_reply_ollama(req.customer_message, full_text,
                                               model=effective_model)
            meta = json.dumps({
                "intent": intent, "retrieval_method": method,
                "score": score, "reason": reason,
                "effective_mode": effective_mode, "router_acted": router_acted,
                "model_used": effective_model, "model_requested": req.model,
                "examples": [{"category": ex.get("category",""), "snippet": ex["customer_message"][:60]}
                              for ex in examples],
            })
            yield f"data: [META] {meta}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/generate-reply/alternatives", response_model=AlternativesResponse)
async def generate_alternatives(req: GenerateRequest):
    """
    3 alternative replies using different persuasion angles.
    Always uses ai mode (angles require free-form generation).
    """
    _t_total = time.monotonic()
    effective_model = req.model or MODEL
    print(f"[MODEL_ROUTER] requested={req.model!r} effective={effective_model} endpoint=/generate-reply/alternatives")

    with _Timer("analysis"):
        analysis = analyse(req.customer_message)
    intent   = analysis.primary_intent
    effective_mode, _ = select_effective_mode(req.mode, analysis)
    if effective_mode != "ai":
        print(f"[ROUTER_EXECUTION] alternatives override effective_mode={effective_mode}→ai (by design)")

    replies = load_replies()
    api_key_for_embed = os.getenv("VOYAGE_API_KEY", "")
    examples, method = retrieve(req.customer_message, replies, intent, api_key_for_embed, TOP_K)
    ollama   = _get_client_for_model(effective_model)
    results: list[ReplyResult] = []

    prompt_builder = (
        build_reply_prompt_ai if effective_mode == "ai"
        else build_reply_prompt
    )

    for angle in ALTERNATIVE_ANGLES:
        user_prompt = prompt_builder(
            req.customer_message, intent, examples,
            req.tone or "professional and persuasive",
            req.domain_name, req.asking_price, method,
            analysis=analysis,
        )
        lbl = angle["label"].replace(" ","_").lower()
        print(f"[AI_BACKEND] backend={('groq' if effective_model.startswith('groq:') else 'ollama')} model={effective_model} mode={effective_mode} label=alternative_{lbl}")
        raw = ollama.generate(
            prompt=user_prompt,
            system=_select_system_prompt_for_mode(effective_model, effective_mode),
            temperature=0.75, max_tokens=MAX_TOKENS,
        )
        _require_ollama(raw, f"alternative_{angle['label']}")
        reply_text = raw.strip()
        score, reason = score_reply_ollama(req.customer_message, reply_text, model=effective_model)
        results.append(ReplyResult(reply=reply_text, confidence_score=score,
                                   confidence_reason=reason, angle=angle["label"]))

    _Timer.log("total", int((time.monotonic() - _t_total) * 1000))
    return AlternativesResponse(alternatives=results, detected_intent=intent,
                                model_used=effective_model)



@app.post("/generate-reply/regenerate")
async def regenerate_reply(req: GenerateRequest):
    """
    Regenerate endpoint — same as /generate-reply but forces a fresh result.
    Randomises the variation style order so you get different output each time.
    Frontend "Regenerate" button should call this.
    """
    import random
    # Shuffle variation styles for a different result
    random.shuffle(VARIATION_STYLES)
    return await generate_reply(req)


# ─────────────────────────────────────────────────────────────────────────────
# GROQ VARIATION REWRITE SYSTEM
# Takes an existing reply and rewrites it in a different style.
# Completely separate from the generation path — variations only.
# Supports 10 named styles; Ollama fallback if Groq unavailable.
# ─────────────────────────────────────────────────────────────────────────────

GROQ_REWRITE_STYLES: dict[str, dict] = {
    "safe": {
        "label": "Safe",
        "instruction": (
            "Rewrite this reply in a balanced, professional tone. "
            "Friendly but not pushy. Suitable for any prospect. "
            "Preserve all factual details, pricing, and links exactly."
        ),
        "temperature": 0.65,
    },
    "persuasive": {
        "label": "Persuasive",
        "instruction": (
            "Rewrite this reply to make a stronger case. Lead with the most compelling "
            "value point. Be confident and direct. Add light urgency only if it fits naturally. "
            "No marketing clichés. Preserve all pricing and links exactly."
        ),
        "temperature": 0.72,
    },
    "short": {
        "label": "Short",
        "instruction": (
            "Rewrite this reply in the shortest possible form. "
            "Three sentences maximum. Keep only what earns its place. "
            "Preserve pricing and links. Cut everything else."
        ),
        "temperature": 0.60,
    },
    "professional": {
        "label": "Professional",
        "instruction": (
            "Rewrite this reply in a polished, formal business tone. "
            "Clear structure. Confident. No contractions, no casual phrasing. "
            "Preserve all facts, pricing, and links exactly."
        ),
        "temperature": 0.60,
    },
    "softer": {
        "label": "Softer",
        "instruction": (
            "Rewrite this reply with a warmer, more empathetic tone. "
            "Acknowledge the prospect's perspective before making any case. "
            "Sound like a helpful human, not a sales script. "
            "Preserve all facts, pricing, and links exactly."
        ),
        "temperature": 0.72,
    },
    "urgent": {
        "label": "More Urgent",
        "instruction": (
            "Rewrite this reply to convey genuine urgency — domains are publicly listed "
            "and can sell at any time. Frame urgency as information, not pressure. "
            "Be specific, not vague. Preserve all facts, pricing, and links exactly."
        ),
        "temperature": 0.68,
    },
    "executive": {
        "label": "Executive Tone",
        "instruction": (
            "Rewrite this reply for a senior decision-maker or executive audience. "
            "Lead with business impact. Remove any informal language. "
            "Be precise and strategic. Preserve all facts, pricing, and links exactly."
        ),
        "temperature": 0.58,
    },
    "friendly": {
        "label": "Friendly",
        "instruction": (
            "Rewrite this reply in a conversational, approachable tone. "
            "Sound like a trusted colleague, not a salesperson. "
            "Short sentences. One genuine question. "
            "Preserve all facts, pricing, and links exactly."
        ),
        "temperature": 0.75,
    },
    "seo": {
        "label": "SEO Focused",
        "instruction": (
            "Rewrite this reply to emphasise local SEO value. "
            "Explain how exact-match geo-targeted domains affect local search ranking. "
            "Be specific: this domain = those local searches. "
            "Accessible language — no jargon. Preserve pricing and links exactly."
        ),
        "temperature": 0.65,
    },
    "brand": {
        "label": "Brandability Focused",
        "instruction": (
            "Rewrite this reply to lead with brand value: memorability, authority, credibility. "
            "Focus on what customers will think when they see the domain. "
            "Long-term brand asset framing. Creative but professional. "
            "Preserve all facts, pricing, and links exactly."
        ),
        "temperature": 0.72,
    },
}

_GROQ_REWRITE_SYSTEM = (
    "You are an experienced domain broker rewriting email replies. "
    "Your job is to rephrase the reply in a specific style while preserving ALL factual content. "
    "Rules:\n"
    "- NEVER change pricing information, domain names, or purchase links\n"
    "- NEVER invent new facts, services, or business details not in the original\n"
    "- NEVER add hallucinated details about traffic, rankings, or domain history\n"
    "- Sound natural and human — not robotic, not like ad copy\n"
    "- Preserve the greeting and sign-off structure\n"
    "- Write ONLY the rewritten email body. No preamble, no commentary."
)


class VariationRewriteRequest(BaseModel):
    original_reply:  str
    rewrite_style:   str            = "safe"     # key from GROQ_REWRITE_STYLES
    variation_backend: str          = "auto"     # "auto" | "groq" | "ollama"
    model:           Optional[str]  = None       # override model; None = auto-select
    domain_name:     Optional[str]  = None
    asking_price:    Optional[str]  = None
    intent:          Optional[str]  = "general"

    @field_validator("original_reply")
    @classmethod
    def reply_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("original_reply cannot be empty.")
        return v.strip()

    @field_validator("rewrite_style")
    @classmethod
    def style_valid(cls, v: str) -> str:
        if v not in GROQ_REWRITE_STYLES:
            available = ", ".join(GROQ_REWRITE_STYLES.keys())
            raise ValueError(f"rewrite_style '{v}' not recognised. Available: {available}")
        return v


class VariationRewriteResponse(BaseModel):
    rewritten_reply: str
    style_label:     str
    backend_used:    str       # "groq" | "ollama"
    model_used:      str
    timing_ms:       int
    fallback:        bool = False   # True if Groq was requested but fell back to Ollama


def _select_variation_backend(requested: str, model_override: Optional[str]) -> str:
    """
    Resolve which model to use for variation rewriting.

    Logic:
    - model_override takes priority (caller knows exactly what they want)
    - requested == 'groq' → use GROQ_DEFAULT if GROQ_API_KEY is set, else fallback to Ollama
    - requested == 'ollama' → use global MODEL
    - requested == 'auto' → use Groq if available, else Ollama
    """
    if model_override:
        return model_override
    if requested == "ollama":
        return MODEL
    if requested == "groq":
        if GROQ_API_KEY:
            return GROQ_DEFAULT
        print("[VARIATION_BACKEND] groq requested but GROQ_API_KEY not set — falling back to ollama")
        return MODEL
    # auto
    if GROQ_API_KEY:
        return GROQ_DEFAULT
    return MODEL


@app.post("/generate-reply/variations/rewrite", response_model=VariationRewriteResponse)
async def rewrite_variation(req: VariationRewriteRequest):
    """
    Rewrite an existing reply in a different style using Groq (preferred) or Ollama.

    This is a dedicated rewrite-only endpoint — it does NOT run the full generation
    pipeline. It takes an existing reply and reshapes it stylistically while
    preserving all factual content, pricing, and links.

    Supports 10 named styles:
      safe, persuasive, short, professional, softer, urgent,
      executive, friendly, seo, brand

    Backends:
      auto   → Groq if GROQ_API_KEY is set, Ollama otherwise
      groq   → Groq (falls back to Ollama if key missing or call fails)
      ollama → local Ollama only

    Returns HTTP 422 if rewrite_style is unknown.
    Returns HTTP 503 if both Groq and Ollama fail.
    """
    _t0 = time.monotonic()
    style = GROQ_REWRITE_STYLES[req.rewrite_style]

    # ── Resolve backend model ───────────────────────────────────────────────
    resolved_model  = _select_variation_backend(req.variation_backend, req.model)
    is_groq         = resolved_model.startswith("groq:")
    fallback_used   = False

    # Build context hint for the rewrite
    context_parts = []
    if req.domain_name:   context_parts.append(f"Domain: {req.domain_name}")
    if req.asking_price:  context_parts.append(f"Asking price: {req.asking_price}")
    context_line = "\n".join(context_parts) + "\n\n" if context_parts else ""

    rewrite_prompt = (
        f"{context_line}"
        f"ORIGINAL REPLY:\n{req.original_reply}\n\n"
        f"REWRITE INSTRUCTION:\n{style['instruction']}\n\n"
        f"Write the rewritten reply now:"
    )

    backend_label = "groq" if is_groq else "ollama"
    print(
        f"[VARIATION_REWRITE] backend={backend_label} model={resolved_model} "
        f"style={req.rewrite_style} intent={req.intent}"
    )

    # ── Primary attempt ─────────────────────────────────────────────────────
    result_text = ""
    try:
        client = _get_client_for_model(resolved_model)
        result_text = client.generate(
            prompt      = rewrite_prompt,
            system      = _GROQ_REWRITE_SYSTEM,
            temperature = style["temperature"],
            max_tokens  = MAX_TOKENS,
        )
    except Exception as primary_err:
        print(f"[VARIATION_REWRITE] primary attempt failed: {primary_err}")
        result_text = ""

    # ── Groq fallback → Ollama ──────────────────────────────────────────────
    if not result_text and is_groq:
        print(f"[VARIATION_REWRITE] groq failed — falling back to ollama model={MODEL}")
        fallback_used  = True
        resolved_model = MODEL
        backend_label  = "ollama"
        try:
            client = _get_client_for_model(MODEL)
            result_text = client.generate(
                prompt      = rewrite_prompt,
                system      = _GROQ_REWRITE_SYSTEM,
                temperature = style["temperature"],
                max_tokens  = MAX_TOKENS,
            )
        except Exception as fallback_err:
            print(f"[VARIATION_REWRITE] ollama fallback also failed: {fallback_err}")

    if not result_text:
        raise HTTPException(
            status_code=503,
            detail=(
                "Both Groq and Ollama backends failed to generate a rewrite. "
                "Check that Ollama is running or that GROQ_API_KEY is valid."
            ),
        )

    timing_ms = int((time.monotonic() - _t0) * 1000)
    print(f"[TIMING] variation_rewrite_{req.rewrite_style}_ms={timing_ms} backend={backend_label}")

    return VariationRewriteResponse(
        rewritten_reply = result_text.strip(),
        style_label     = style["label"],
        backend_used    = backend_label,
        model_used      = resolved_model,
        timing_ms       = timing_ms,
        fallback        = fallback_used,
    )


@app.get("/generate-reply/variations/styles")
async def list_variation_styles():
    """Return all available rewrite styles and their metadata."""
    return {
        "styles": [
            {
                "key":   key,
                "label": s["label"],
                "description": s["instruction"][:120] + "…",
            }
            for key, s in GROQ_REWRITE_STYLES.items()
        ],
        "backends": ["auto", "groq", "ollama"],
        "groq_available": bool(GROQ_API_KEY),
        "default_backend": "groq" if GROQ_API_KEY else "ollama",
    }


@app.post("/replies/save-generated")
async def save_generated_reply(req: SaveReplyRequest):
    """
    Save a generated reply into the knowledge base for future retrieval.
    If make_template=True, also logs it as a template for reuse.
    """
    all_r  = load_replies()

    # Duplicate check — skip if identical reply already exists
    reply_text_clean = req.reply.strip().lower()
    for existing in all_r:
        if existing.get("reply", "").strip().lower() == reply_text_clean:
            return {
                "message": "Reply already exists in knowledge base (duplicate skipped).",
                "id": existing.get("id"),
                "entry": existing,
                "duplicate": True,
            }

    new_id = max((r.get("id", 0) for r in all_r), default=0) + 1
    entry  = {
        "id":               new_id,
        "category":         req.category,
        "customer_message": req.customer_message,
        "reply":            req.reply,
        "subject":          req.subject or "",
        "source":           "generated",   # distinguish from manually added
    }
    all_r.append(entry)
    save_replies(all_r)
    _index.kb_size = -1  # mark index stale so it rebuilds next request

    msg = f"Reply #{new_id} saved to knowledge base."
    if req.make_template:
        msg += " (Template flag noted — add to template_engine.py manually to enable template matching.)"

    return {"message": msg, "id": new_id, "entry": entry}

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE CRUD
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/replies")
async def list_replies():
    return load_replies()

@app.get("/replies/search")
async def search_replies(q: str):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    q_low   = q.lower()
    results = [r for r in load_replies()
               if q_low in r.get("category","").lower()
               or q_low in r.get("customer_message","").lower()
               or q_low in r.get("reply","").lower()]
    return {"query": q, "count": len(results), "results": results}

@app.get("/categories")
async def list_categories():
    from collections import Counter
    cats = Counter(r.get("category","unknown") for r in load_replies())
    return {"total_categories": len(cats), "categories": dict(sorted(cats.items()))}

@app.post("/replies")
async def add_reply(req: AddReplyRequest):
    all_r  = load_replies()
    new_id = max((r.get("id",0) for r in all_r), default=0) + 1
    all_r.append({"id": new_id, "category": req.category,
                  "customer_message": req.customer_message, "reply": req.reply})
    save_replies(all_r)
    _index.kb_size = -1
    return {"message": "Reply added. Index will rebuild on next generate call.", "id": new_id}

@app.delete("/replies/{reply_id}")
async def delete_reply(reply_id: int):
    all_r   = load_replies()
    updated = [r for r in all_r if r.get("id") != reply_id]
    if len(updated) == len(all_r):
        raise HTTPException(status_code=404, detail=f"Reply {reply_id} not found.")
    save_replies(updated)
    _index.kb_size = -1
    return {"message": f"Reply {reply_id} deleted."}


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/embed/status")
async def embed_status():
    replies = load_replies()
    stale   = _index.is_stale(len(replies))
    return {"semantic_ready": _index.ready and not stale, "index_size": len(_index.entries),
            "kb_size": len(replies), "is_stale": stale,
            "retrieval_method": "semantic" if (_index.ready and not stale) else "keyword"}

@app.post("/embed/rebuild")
async def embed_rebuild(body: dict = None):
    """Rebuild the semantic embedding index. Requires a Voyage AI API key."""
    key = (body or {}).get("api_key") or os.getenv("VOYAGE_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400,
            detail="Voyage AI API key required for semantic embeddings. "
                   "Pass 'api_key' in the request body or set VOYAGE_API_KEY in .env. "
                   "The app works without this — keyword retrieval is used as fallback.")
    try:
        _index.build(load_replies(), key)
        return {"message": "Index rebuilt.", "total": len(_index.entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY CONTROL ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/qc/test")
async def qc_run_tests():
    """
    Run the full QC test harness and return structured pass/fail results.
    Useful for checking intent detection health without restarting the server.
    """
    from quality_control import run_tests
    results = run_tests(verbose=True)
    return {
        "total":  results["total"],
        "passed": results["passed"],
        "failed": results["failed"],
        "pass_rate": f"{int(results['passed']/results['total']*100)}%" if results["total"] else "0%",
        "results": [
            {
                "test":            r["test"],
                "input":           r["input"],
                "expected_intent": r["expected_intent"],
                "detected_intent": r["detected_intent"],
                "intent_match":    r["intent_match"],
                "strategy_goal":   r["strategy_goal"],
                "passed":          r["passed"],
            }
            for r in results["results"]
        ],
    }


@app.post("/qc/validate")
async def qc_validate_reply(body: dict):
    """
    Validate a single email reply against structural rules.
    Pass: {"reply": "...", "intent": "follow_up"}
    Returns validation result with issues, fixes applied, and summary.
    """
    from quality_control import run_full_qc
    reply  = (body.get("reply") or "").strip()
    intent = body.get("intent", "general")
    if not reply:
        raise HTTPException(status_code=400, detail="'reply' field is required.")
    return run_full_qc(reply, intent=intent)


@app.post("/qc/score")
async def qc_score_reply(body: dict):
    """
    Run the heuristic multi-dimensional quality scorer on an existing reply.
    Fast — zero model calls.

    Body: { "reply": "...", "intent": "negotiation" }

    Returns:
    {
        "dimensions": { naturalness, readability, spamminess, cta_quality,
                        human_tone, personalization, structure },
        "total": int,           // 0-100
        "flags": [...],         // triggered pattern labels
        "needs_humanization": bool,
        "summary": str
    }
    """
    reply  = (body.get("reply") or "").strip()
    intent = body.get("intent", "general")
    if not reply:
        raise HTTPException(status_code=400, detail="'reply' field is required.")
    return heuristic_score_reply(reply, intent=intent)


@app.post("/qc/strategy-eval")
async def strategy_eval(body: dict):
    """
    Evaluate strategy selection for a given situation without generating a reply.
    Useful for debugging, testing, and understanding how the system reasons.

    Body:
    {
        "message":       str,   // required — the prospect message or situation
        "intent":        str,   // optional — override detected intent
        "stage":         str,   // optional — override detected stage
        "tone":          str,   // optional — default "professional and persuasive"
        "domain_name":   str,   // optional
        "asking_price":  str,   // optional
        "email_preset":  str,   // optional
        "outreach_count": int,  // optional
        "neg_state":     str,   // optional — override negotiation state
    }

    Returns:
    {
        "strategy":        { all strategy fields },
        "reasoning_trace": { which signal drove each decision },
        "prompt_brief":    str,   // the brief that would be sent to the model
        "brief_tokens":    int,   // estimated token count
    }
    """
    from reply_strategy import evaluate_strategy, StrategySignals

    message      = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="'message' field is required.")

    domain_name  = body.get("domain_name")  or None
    asking_price = body.get("asking_price") or None

    # Detect or accept overrides
    intent         = body.get("intent")       or detect_intent(message)
    stage_override = body.get("stage")        or "unknown"
    neg_override   = body.get("neg_state")    or _detect_negotiation_state(message, asking_price)
    frame          = _classify_response_frame(message)

    sig = StrategySignals(
        intent             = intent,
        message            = message,
        stage              = stage_override,
        neg_state          = neg_override,
        response_frame     = frame,
        tone_requested     = body.get("tone", "professional and persuasive"),
        asking_price       = asking_price,
        domain_name        = domain_name,
        email_preset       = body.get("email_preset") or None,
        outreach_count     = int(body.get("outreach_count", 0)),
        no_domain_no_price = not domain_name and not asking_price,
    )
    return evaluate_strategy(sig)


@app.get("/qc/strategy-scenarios")
async def strategy_scenarios():
    """
    Run the full scenario test suite and return pass/fail results.
    Useful for CI checks and regression testing after strategy changes.
    """
    from reply_strategy import run_scenario_suite
    results = run_scenario_suite()
    passed  = sum(1 for r in results if r.get("passed"))
    return {
        "total":   len(results),
        "passed":  passed,
        "failed":  len(results) - passed,
        "results": results,
    }


@app.post("/qc/replay-strategy")
async def replay_strategy(body: dict):
    """
    Evaluate an existing reply against a given strategy without regenerating.

    Useful for:
    - Debugging why a reply scored poorly
    - Testing strategy decisions against real examples
    - Manual calibration of the strategy layer

    Body:
    {
        "reply":    str,       // required — the generated reply to evaluate
        "message":  str,       // required — the original prospect message
        "intent":   str,       // optional — detected intent
        "stage":    str,       // optional — conversation stage
        "goal":     str,       // optional — primary_goal override
        "domain_name":   str,  // optional
        "asking_price":  str,  // optional
        "email_preset":  str,  // optional
        "outreach_count": int, // optional
        "suppressed_topics": list[str],  // optional — for repetition check
        "tone":     str,       // optional
    }

    Returns:
    {
        "strategy":            { resolved strategy fields },
        "evaluation":          { adherence_score, failed_dimensions, ... },
        "repetition_violations": [...],
        "progression_result":  { verdict, note },
        "confidence_alignment": { verdict, note },
        "suggested_adjustments": [...],
    }
    """
    from reply_strategy import evaluate_strategy, StrategySignals
    from strategy_eval import evaluate_strategy_adherence

    reply   = (body.get("reply")   or "").strip()
    message = (body.get("message") or "").strip()
    if not reply:
        raise HTTPException(400, "'reply' is required.")
    if not message:
        raise HTTPException(400, "'message' is required.")

    domain_name  = body.get("domain_name")  or None
    asking_price = body.get("asking_price") or None
    intent       = body.get("intent")       or detect_intent(message)
    stage        = body.get("stage")        or "unknown"
    neg_state    = _detect_negotiation_state(message, asking_price)
    frame        = _classify_response_frame(message)

    sig = StrategySignals(
        intent             = intent,
        message            = message,
        stage              = stage,
        neg_state          = neg_state,
        response_frame     = frame,
        tone_requested     = body.get("tone", "professional and persuasive"),
        asking_price       = asking_price,
        domain_name        = domain_name,
        email_preset       = body.get("email_preset") or None,
        outreach_count     = int(body.get("outreach_count", 0)),
        no_domain_no_price = not domain_name and not asking_price,
    )

    # Build strategy (or accept override goal for testing specific paths)
    strategy_report = evaluate_strategy(sig)
    strategy        = build_strategy(sig)

    # Inject any manually supplied suppressed_topics for testing
    if body.get("suppressed_topics"):
        strategy.suppressed_topics = list(body["suppressed_topics"])

    eval_result = evaluate_strategy_adherence(reply, strategy)

    # Generate suggested adjustments from failures
    suggestions: list[str] = []
    failed = eval_result.get("strategy_adherence", {}).get("failed_dimensions", [])
    adj_map = {
        "cta_style":              f"Regenerate with explicit CTA instruction: '{strategy.cta_style}'",
        "prohibited_topics":      "Prohibited topics were violated — check humanizer prohibitions list",
        "repetition_suppression": "Prior value points re-explained — increase outreach_count or add suppressed_topics",
        "persuasion_calibration": f"Persuasion level mismatch — expected level={strategy.persuasion_level}",
        "progression":            "Reply regressed — consider re_engage or stalled stage for this situation",
        "confidence_alignment":   f"Confidence mismatch — strategy confidence is low ({min(strategy.stage_confidence, strategy.buyer_confidence, strategy.goal_confidence):.2f}), check stage detection",
    }
    for dim in failed:
        if dim in adj_map:
            suggestions.append(adj_map[dim])

    return {
        "strategy":              strategy_report["strategy"],
        "prompt_brief":          strategy_report["prompt_brief"],
        "brief_tokens":          strategy_report["brief_tokens"],
        "evaluation":            eval_result.get("strategy_adherence", {}),
        "repetition_violations": eval_result.get("repetition_violations", []),
        "progression_result":    eval_result.get("progression_result", {}),
        "confidence_alignment":  eval_result.get("confidence_alignment", {}),
        "suggested_adjustments": suggestions,
    }


@app.get("/qc/analytics")
async def qc_analytics(days: int = 30):
    """
    Return strategy outcome analytics for the last N days.
    Shows which goals, buyer states, and presets perform well or poorly.
    """
    summary = strategy_analytics.get_summary(limit_days=days)
    worst   = strategy_analytics.get_worst_performing()
    return {
        "summary":            summary,
        "worst_performing":   worst,
    }


async def qc_humanize_reply(body: dict):
    """
    On-demand humanization endpoint. Scores first, then rewrites if needed.

    Body:
    {
        "reply":        str,   // required
        "situation":    str,   // what the prospect said / context
        "intent":       str,   // e.g. "negotiation" (optional)
        "model":        str,   // override model (optional)
        "domain_name":  str,   // used for factual safety check (optional)
        "asking_price": str    // used for factual safety check (optional)
    }

    Returns:
    {
        "original_reply":  str,
        "final_reply":     str,
        "rewrite_applied": bool,
        "quality_report":  dict,
        "timing_ms":       int
    }
    """
    _t0          = time.monotonic()
    reply        = (body.get("reply") or "").strip()
    situation    = (body.get("situation") or "").strip()
    intent       = body.get("intent", "general")
    model_req    = body.get("model") or MODEL
    domain_name  = body.get("domain_name") or None
    asking_price = body.get("asking_price") or None

    if not reply:
        raise HTTPException(status_code=400, detail="'reply' field is required.")

    final, quality_report = humanize_reply(
        reply        = reply,
        situation    = situation,
        intent       = intent,
        model        = model_req,
        domain_name  = domain_name,
        asking_price = asking_price,
    )

    return {
        "original_reply":  reply,
        "final_reply":     final,
        "rewrite_applied": quality_report.get("rewrite_applied", False),
        "quality_report":  quality_report,
        "timing_ms":       int((time.monotonic() - _t0) * 1000),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# BROKER MEMORY ENDPOINTS
# Lead management, outreach logging, offer tracking.
# All return 503 if broker memory is unavailable (graceful degradation).
# ─────────────────────────────────────────────────────────────────────────────

def _memory_check():
    if not _MEMORY_AVAILABLE or memory_db is None:
        from fastapi import HTTPException
        raise HTTPException(503, "Broker memory unavailable — check broker_memory.py")


@app.post("/leads")
async def create_lead(body: dict):
    """
    Create or upsert a lead.
    Body: { domain, prospect_email?, prospect_name?, notes? }
    Returns: { lead_id, created }
    """
    _memory_check()
    domain = body.get("domain", "").strip()
    if not domain:
        from fastapi import HTTPException
        raise HTTPException(422, "domain is required")
    lead_id = memory_db.upsert_lead(
        domain          = domain,
        prospect_email  = body.get("prospect_email"),
        prospect_name   = body.get("prospect_name"),
        notes           = body.get("notes"),
    )
    return {"lead_id": lead_id, "domain": domain}


@app.get("/leads")
async def get_leads(domain: Optional[str] = None):
    """List all leads, optionally filtered by domain substring."""
    _memory_check()
    return {"leads": memory_db.list_leads(domain=domain)}


@app.get("/leads/{lead_id}")
async def get_lead(lead_id: int):
    """Get a single lead with full history."""
    _memory_check()
    data = memory_db.full_history(lead_id)
    if not data["lead"]:
        from fastapi import HTTPException
        raise HTTPException(404, f"Lead {lead_id} not found")
    return data


@app.delete("/leads/{lead_id}")
async def delete_lead(lead_id: int):
    """Delete a lead and all associated history."""
    _memory_check()
    ok = memory_db.delete_lead(lead_id)
    return {"deleted": ok}


@app.post("/leads/{lead_id}/outreach")
async def log_outreach(lead_id: int, body: dict):
    """
    Log an outreach attempt for a lead.
    Body: { preset?, subject?, body? }
    Called automatically by the frontend when a generated email is saved/sent.
    """
    _memory_check()
    ok = memory_db.log_outreach(
        lead_id = lead_id,
        preset  = body.get("preset"),
        subject = body.get("subject"),
        body    = body.get("body"),
    )
    return {"logged": ok}


@app.post("/leads/{lead_id}/offer")
async def log_offer(lead_id: int, body: dict):
    """
    Log an offer or counteroffer.
    Body: { amount (number), direction ('sent'|'received'), notes? }
    """
    _memory_check()
    amount = body.get("amount")
    direction = body.get("direction", "sent")
    if amount is None:
        from fastapi import HTTPException
        raise HTTPException(422, "amount is required")
    ok = memory_db.log_offer(
        lead_id   = lead_id,
        amount    = float(amount),
        direction = direction,
        notes     = body.get("notes"),
    )
    return {"logged": ok}


@app.patch("/leads/{lead_id}/stage")
async def update_stage(lead_id: int, body: dict):
    """Update the conversation stage of a lead."""
    _memory_check()
    stage = body.get("stage", "").strip()
    if not stage:
        from fastapi import HTTPException
        raise HTTPException(422, "stage is required")
    ok = memory_db.update_lead_stage(lead_id, stage)
    return {"updated": ok}


@app.post("/leads/{lead_id}/conversation-stage")
async def get_conversation_stage(lead_id: int, body: dict = None):
    """
    Detect the conversation stage for a lead without generating a reply.

    Optional body: { "message": "...", "intent": "...", "asking_price": "..." }

    Returns:
    {
        "stage":        str,    // e.g. "negotiation"
        "explanation":  str,
        "outreach_count": int,
        "offer_count":  int,
        "latest_offer": float | null,
        "lead_stage":   str | null   // stored stage from leads table
    }
    """
    _memory_check()
    body = body or {}
    message      = body.get("message", "")
    intent       = body.get("intent", "general")
    asking_price = body.get("asking_price") or None

    lead_ctx = _get_lead_context_rich(lead_id)
    stage, reason = detect_conversation_stage(
        intent         = intent,
        message        = message,
        outreach_count = lead_ctx["outreach_count"],
        offers         = lead_ctx["offers"],
        lead_stage     = lead_ctx["lead_stage"],
        asking_price   = asking_price,
    )
    offers = lead_ctx["offers"]
    received = [o for o in offers if o.get("direction") == "received"]
    latest_offer = received[-1]["amount"] if received else None

    return {
        "stage":          stage,
        "explanation":    reason,
        "outreach_count": lead_ctx["outreach_count"],
        "offer_count":    len(offers),
        "latest_offer":   latest_offer,
        "lead_stage":     lead_ctx["lead_stage"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH + INFO
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    replies = load_replies()
    stale   = _index.is_stale(len(replies))
    try:
        from ollama_client import get_default_client
        ollama_ok = get_default_client(model=MODEL, base_url=OLLAMA_BASE_URL).health_check()
    except Exception:
        ollama_ok = False
    return {
        "status":           "ok" if ollama_ok else "degraded",
        "version":          "9.0.0",
        "model":            MODEL,
        "ollama_url":       OLLAMA_BASE_URL,
        "ollama_reachable": ollama_ok,
        "kb_size":          len(replies),
        "semantic_ready":   _index.ready and not stale,
        "retrieval_method": "semantic" if (_index.ready and not stale) else "keyword",
        "config": {
            "ENABLE_VARIATIONS": ENABLE_VARIATIONS,
            "ENABLE_AI_SCORING": ENABLE_AI_SCORING,
            "ENABLE_AI_SUBJECT": ENABLE_AI_SUBJECT,
            "ENABLE_QC_REWRITE": ENABLE_QC_REWRITE,
        },
        "note": "Set VOYAGE_API_KEY in .env and call /embed/rebuild for semantic retrieval.",
    }

@app.get("/info")
async def info():
    return {
        "name": "Domain Email Reply Generator", "version": "7.0.0",
        "backend": "Ollama (local) — no external API keys required for generation",
        "model":   MODEL,
        "new_in_v7": [
            "Ollama-first: all generation uses local qwen2.5:7b — no Anthropic key needed",
            "template mode: fully offline, zero AI calls",
            "hybrid mode: template + Ollama polish",
            "ai mode: Ollama direct generation with variations",
            "Graceful HTTP 503 if Ollama unavailable — no silent fallback",
            "Confidence scoring + routing recommendation preserved (Phase 1/2/3)",
        ],
        "endpoints": [
            "POST   /generate-reply           — ai / hybrid / template mode, 2-3 variations",
            "POST   /generate-reply/regenerate — same with reshuffled styles",
            "POST   /generate-reply/situation  — situation-based proactive generation",
            "POST   /generate-reply/stream     — pseudo-streaming via Ollama",
            "POST   /generate-reply/alternatives",
            "POST   /generate-reply/template",
            "POST   /generate-reply/template/detect-intent",
            "GET    /replies", "GET    /replies/search?q=...",
            "POST   /replies", "POST   /replies/save-generated",
            "DELETE /replies/{id}",
            "GET    /categories", "GET    /embed/status",
            "POST   /embed/rebuild  — requires VOYAGE_API_KEY for semantic search",
            "GET    /health", "GET    /info",
            "POST   /assistant/chat — general AI assistant, dynamic model selection",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASSISTANT MODE
# Completely independent from domain-reply logic.
# No template routing, no retrieval, no past_replies.json, no QC pipeline.
# Per-request model selection — does NOT use the global MODEL constant.
# ─────────────────────────────────────────────────────────────────────────────

# Models the assistant endpoint will accept
_ASSISTANT_ALLOWED_MODELS = {
    "qwen2.5:3b",
    "qwen2.5:7b",
    "groq:llama3.1-70b",
    "groq:mixtral-8x7b",
}
_ASSISTANT_DEFAULT_MODEL  = "qwen2.5:7b"

_ASSISTANT_DEFAULT_SYSTEM = (
    "You are a domain brokerage advisor with real-world experience in geo-targeted domain sales, "
    "negotiation, and local business outreach. You think practically and speak plainly.\n\n"
    "When someone asks for advice, give it directly — not as a list of considerations, "
    "but as what you'd actually tell them to do and why. When they need a draft, write it. "
    "When they ask a factual question, answer it. Match their level of detail: "
    "if they're brief, be brief. If they need depth, go deep. "
    "Never pad answers and never substitute generic guidance for a specific recommendation."
)

_ASSISTANT_ADVISOR_SYSTEM = (
    "You are a domain negotiation advisor helping a broker work through a real situation. "
    "Think of yourself as the experienced colleague they called for a second opinion.\n\n"
    "Structure your response naturally:\n"
    "First, read the situation and say what you actually see happening — "
    "what does the prospect's message signal about their position and intent?\n"
    "Then give your recommended approach, specifically — not 'consider your options' "
    "but what you'd actually do and why.\n"
    "Finally, write the reply itself. Make it sound like a real person wrote it, "
    "not like it came from a system.\n\n"
    "Be direct. Be opinionated. The broker needs a decision, not a framework."
)

# Keywords that trigger Domain Advisor Mode in the assistant
_ADVISOR_KEYWORDS: frozenset[str] = frozenset({
    "offer", "counter", "negotiate", "negotiation", "lowball", "low ball",
    "price", "pricing", "valuation", "worth", "how much", "they said",
    "they offered", "should i accept", "should i reply", "follow up",
    "follow-up", "no response", "objection", "they think", "scam",
    "not interested", "too expensive", "can't afford", "what do i say",
    "how do i respond", "help me reply", "draft", "write a reply",
    "email strategy", "sales strategy", "close the deal", "closing",
})


def _detect_advisor_mode(prompt: str) -> bool:
    """
    Returns True if the prompt contains negotiation/sales signals that
    warrant the structured Domain Advisor Mode response format.
    Pure keyword check — zero latency, no AI call.
    """
    low = prompt.lower()
    matched = sum(1 for kw in _ADVISOR_KEYWORDS if kw in low)
    return matched >= 2   # require at least 2 signals to avoid false positives
_ASSISTANT_MAX_TOKENS     = int(os.getenv("ASSISTANT_MAX_TOKENS", "1200"))
_ASSISTANT_TIMEOUT        = int(os.getenv("ASSISTANT_TIMEOUT",    "120"))


class AssistantRequest(BaseModel):
    prompt:      str
    model:       Optional[str]  = None    # None → _ASSISTANT_DEFAULT_MODEL
    system:      Optional[str]  = None    # None → _ASSISTANT_DEFAULT_SYSTEM
    temperature: Optional[float]= None    # None → 0.7
    max_tokens:  Optional[int]  = None    # None → _ASSISTANT_MAX_TOKENS

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt cannot be empty.")
        return v.strip()

    @field_validator("model")
    @classmethod
    def model_allowed(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in _ASSISTANT_ALLOWED_MODELS:
            allowed = ", ".join(sorted(_ASSISTANT_ALLOWED_MODELS))
            raise ValueError(f"model '{v}' is not supported. Allowed: {allowed}")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 4096):
            raise ValueError("max_tokens must be between 1 and 4096")
        return v


class AssistantResponse(BaseModel):
    reply:             str
    model_used:        str
    prompt_tokens_est: int    # rough estimate: len(prompt.split()) * 1.3
    timing_ms:         int


def _get_ollama_for_model(model: str) -> "OllamaClient":
    """
    Return an OllamaClient configured for the given model.
    Completely independent from get_ollama_client() — does NOT use the
    global MODEL constant or the domain-reply singleton.
    Each call may return a cached instance if the model matches.
    """
    from ollama_client import get_default_client
    return get_default_client(
        model    = model,
        base_url = OLLAMA_BASE_URL,
        timeout  = _ASSISTANT_TIMEOUT,
    )


class GroqClientWrapper:
    """
    Thin wrapper around the Groq SDK that exposes the same .generate()
    interface as OllamaClient. No other part of the codebase needs to
    know Groq exists — they just call .generate() as usual.
    """

    def __init__(self, groq_model: str):
        self._groq_model = groq_model
        self._client     = None   # lazy-init

    def _ensure_client(self):
        if self._client is None:
            try:
                from groq import Groq
                if not GROQ_API_KEY:
                    raise RuntimeError("GROQ_API_KEY is not set in environment.")
                self._client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT)
            except ImportError:
                raise RuntimeError(
                    "groq package is not installed. Run: pip install groq"
                )

    def generate(
        self,
        prompt:      str,
        system:      str  = "",
        temperature: float = 0.7,
        max_tokens:  int   = 400,
    ) -> str:
        self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model       = self._groq_model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return response.choices[0].message.content or ""


# Cache Groq wrappers so we don't re-instantiate per call
_groq_cache: dict[str, GroqClientWrapper] = {}


def _get_client_for_model(model: str):
    """
    Universal model router.
    - model starts with 'groq:' → returns a GroqClientWrapper
    - anything else             → returns an OllamaClient (existing behaviour)

    Every existing _get_ollama_for_model() call can be replaced with this
    function. The .generate() interface is identical for both backends.
    """
    if model.startswith("groq:"):
        if model not in _groq_cache:
            groq_model_name = GROQ_MODELS.get(model)
            if not groq_model_name:
                raise ValueError(f"Unknown Groq model: {model!r}. Known: {list(GROQ_MODELS)}")
            _groq_cache[model] = GroqClientWrapper(groq_model_name)
        return _groq_cache[model]
    return _get_ollama_for_model(model)


@app.post("/assistant/chat", response_model=AssistantResponse)
async def assistant_chat(req: AssistantRequest):
    """
    General-purpose AI assistant endpoint.
    Completely independent from domain-reply logic:
      - No template routing
      - No retrieval from past_replies.json
      - No intent detection or confidence scoring
      - No QC pipeline
      - No negotiation templates

    Supports dynamic model selection per request.
    Returns HTTP 503 if Ollama is unreachable.
    Returns HTTP 422 if prompt is empty or model is not in the allowed list.
    """
    _t0 = time.monotonic()

    effective_model = req.model or _ASSISTANT_DEFAULT_MODEL

    if (req.system or "").strip():
        # Caller supplied an explicit system prompt — honour it exactly
        system_prompt = req.system.strip()
        response_frame = "caller_override"
        neg_state      = "none"
    else:
        # ── Classify the request frame ────────────────────────────────────────
        response_frame = _classify_response_frame(req.prompt)
        neg_state      = _detect_negotiation_state(req.prompt, asking_price=None)

        # ── Select system prompt based on frame + negotiation state ───────────
        if neg_state in ("low_anchor_offer", "active_negotiation",
                         "hard_rejection", "soft_interest", "hesitation"):
            # Negotiation-flavoured request → advisor system
            system_prompt = _ASSISTANT_ADVISOR_SYSTEM
        elif response_frame in ("negotiation_analysis",):
            system_prompt = _ASSISTANT_ADVISOR_SYSTEM
        elif response_frame in ("educational_answer", "brainstorming"):
            # Factual / creative → default system, lower temp
            system_prompt = _ASSISTANT_DEFAULT_SYSTEM
        elif _detect_advisor_mode(req.prompt):
            # Legacy keyword check as safety net
            system_prompt = _ASSISTANT_ADVISOR_SYSTEM
        else:
            system_prompt = _ASSISTANT_DEFAULT_SYSTEM

        # ── Prepend frame-specific reasoning instruction to the prompt ─────────
        frame_prefix = {
            "strategic_advice": (
                "The user is asking for strategic advice, not a reply draft. "
                "Explain the recommended approach clearly first, "
                "then optionally provide a suggested message at the end. "
                "Label the sections: STRATEGY and SUGGESTED REPLY.\n\n"
            ),
            "educational_answer": (
                "The user is asking a factual or explanatory question. "
                "Answer it directly and completely. "
                "Keep sales framing minimal — lead with useful information.\n\n"
            ),
            "negotiation_analysis": (
                "The user wants a negotiation analysis. "
                "Cover: (1) what the offer signals about the buyer, "
                "(2) the realistic counter position, "
                "(3) risks and recommended next step. "
                "Be direct and specific.\n\n"
            ),
            "brainstorming": (
                "The user wants multiple ideas. "
                "Provide exactly 3 distinct options. "
                "Label each: Option 1, Option 2, Option 3. "
                "Keep each to 2-3 sentences.\n\n"
            ),
            "mixed_request": (
                "This message contains multiple request types. "
                "Address each part in order: "
                "information first, strategy second, any reply draft last.\n\n"
            ),
        }.get(response_frame, "")

        if frame_prefix:
            req = AssistantRequest(
                prompt      = frame_prefix + req.prompt,
                model       = req.model,
                system      = req.system,
                temperature = req.temperature,
                max_tokens  = req.max_tokens,
            )

    print(
        f"[AI_BACKEND] backend={('groq' if effective_model.startswith('groq:') else 'ollama')} model={effective_model}"
        f" label=assistant_chat"
        f" response_frame={response_frame}"
        f" neg_state={neg_state}"
    )

    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens  = req.max_tokens  if req.max_tokens  is not None else _ASSISTANT_MAX_TOKENS

    client = _get_client_for_model(effective_model)
    result = client.generate(
        prompt      = req.prompt,
        system      = system_prompt,
        temperature = temperature,
        max_tokens  = max_tokens,
    )

    if not result:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama model '{effective_model}' is unavailable or returned an empty response. "
                f"Make sure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {effective_model}`)."
            ),
        )

    timing_ms         = int((time.monotonic() - _t0) * 1000)
    prompt_tokens_est = int(len(req.prompt.split()) * 1.3)

    print(f"[TIMING] assistant_chat_ms={timing_ms} model={effective_model}")

    return AssistantResponse(
        reply             = result.strip(),
        model_used        = effective_model,
        prompt_tokens_est = prompt_tokens_est,
        timing_ms         = timing_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND
# Serves index.html at GET /  — only this file, nothing else.
# Using FileResponse instead of StaticFiles prevents the entire project
# directory (main.py, .env, past_replies.json) from being publicly accessible.
# ─────────────────────────────────────────────────────────────────────────────

_INDEX = Path(__file__).parent / "index.html"

@app.get("/", include_in_schema=False)
async def serve_frontend():
    if not _INDEX.exists():
        raise HTTPException(status_code=404, detail="index.html not found — place it in the same directory as main.py")
    return FileResponse(str(_INDEX), media_type="text/html")
