"""
Domain Email Reply Generator — FastAPI + Anthropic Claude API
=============================================================
What's new in v6 — Email Formatting, Subject Lines, Hybrid Mode, Variations, Quality Guard:
  ✓ Situation-based generation — describe context, not just prospect messages
  ✓ 4 new proactive intents: follow_up, sales_pitch, re_engagement, objection_handling
  ✓ Pitch intensity levels: low / medium / high — auto-selected per intent
  ✓ Value propositions + benefits injected per intensity level
  ✓ Soft CTAs and optional light urgency
  ✓ POST /generate-reply/situation — new dedicated endpoint
  ✓ Proper email formatting with greeting, paragraphs, closing, signature
  ✓ Subject line generation (template-based + AI)
  ✓ Hybrid mode: template → AI polish
  ✓ 2-3 reply variations per request (Safe / Persuasive / Short)
  ✓ Quality guard: auto-fix too-short, too-long, missing closing
  ✓ POST /generate-reply/regenerate — one-tap retry
  ✓ POST /replies/save-generated — save good replies for reuse
  ✓ Version 6.0.0
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional, AsyncGenerator

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from template_engine import build_template_reply, ai_polish_reply, detect_template_intent, TEMPLATE_INTENT_KEYWORDS

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL        = "claude-sonnet-4-6"
EMBED_MODEL  = "voyage-3"
MAX_TOKENS        = 700
MAX_TOKENS_MULTI  = 1200   # higher limit when generating 3 variations
MIN_REPLY_WORDS   = 30     # quality guard lower bound
MAX_REPLY_WORDS   = 200    # quality guard upper bound
DEFAULT_SENDER    = "Alex" # default signature name
TOP_K        = 4
DATA_FILE    = Path(__file__).parent / "past_replies.json"
INDEX_FILE   = Path(__file__).parent / "embeddings_index.json"
FRONTEND_DIR = Path(__file__).parent

FILLER_PHRASES = [
    "i hope this email finds you well", "trust you are doing great",
    "hope you are having a great day", "i hope you are doing well",
    "i hope all is well", "i wanted to reach out",
    "please do not hesitate", "feel free to reach out",
    "as per my last email", "going forward",
    "at the end of the day", "it is what it is",
]

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
        "Build a compelling case using all relevant angles:\n"
        "- LOCAL SEO: Exact-match domains carry strong geo-signals for 'near me' searches\n"
        "- COMPETITOR RISK: Once it's gone, a rival owns it — and gets that traffic forever\n"
        "- ZERO FRICTION: Redirect setup takes minutes. Existing site stays the same\n"
        "- PERMANENT ASSET: Unlike ads, a domain is owned forever after one purchase\n"
        "- TRUST SIGNAL: A city-specific .com builds instant local credibility\n"
        "Present 2-3 of these most relevant to the prospect's business type."
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

TONE_INSTRUCTIONS: dict[str, str] = {
    "professional and persuasive": "Write in a confident, professional tone. Use persuasive but respectful language. Lead with value, not pressure.",
    "warm and friendly":           "Write in a warm, conversational tone — as if talking to a local business owner you want to help. Be personable and genuine.",
    "firm but respectful":         "Hold your position clearly. Be polite but do not over-apologise or cave to pressure. State your case once, cleanly.",
    "concise and direct":          "Be extremely brief — 2 to 3 sentences maximum. Get straight to the point. No padding, no filler.",
    "empathetic and understanding":"Lead with understanding of the prospect's concern. Validate their feeling before pivoting to value. Never dismiss their objection.",
}

INTENT_RULES: dict[str, str] = {
    "follow_up": (
        "- Keep it to 2-3 sentences maximum\n"
        "- Do not re-pitch the full value proposition\n"
        "- Give them an easy out: 'just let me know either way'\n"
        "- Assume they're busy, not uninterested\n"
    ),
    "sales_pitch": (
        "- Open with the domain name and what it means for their specific business\n"
        "- Use concrete, visual language: 'every person in [city] searching for [service] could land on your site'\n"
        "- Mention the redirect: no new website needed\n"
        "- Address the most likely objection pre-emptively\n"
        "- End with one clear next step\n"
    ),
    "re_engagement": (
        "- Acknowledge the gap without over-explaining it\n"
        "- Re-introduce the opportunity as if for the first time\n"
        "- Offer something new: a fresh angle, updated info\n"
        "- Keep it short — earn the right to a longer conversation\n"
    ),
    "objection_handling": (
        "- Name the specific hesitation before addressing it\n"
        "- Reframe the objection as a question the prospect can answer\n"
        "- Use social proof or concrete outcomes where possible\n"
        "- End with a question that opens dialogue\n"
    ),
    "no_thanks":      "- Acknowledge their decision graciously without arguing\n- Ask one soft optional question about their objection\n- Leave the door open for future contact\n",
    "price_inquiry":  "- State the price clearly and confidently\n- Add one brief reason why the price reflects real value\n- End with an invitation to proceed or ask questions\n",
    "price_too_high": "- Validate the $10 registration point, then reframe what makes this domain premium\n- Invite a counter-offer rather than just defending the price\n",
    "negotiation":    "- Counter or ask for their best offer — don't just accept a low number\n- Give a brief reason for your floor price\n- Keep the negotiation moving — no dead ends\n",
    "trust_issue":    "- Address the concern directly — do not get defensive\n- Name a specific trust mechanism: DAN.com escrow, GoDaddy listing, Trustpilot reviews\n- Offer a verification step they can take independently right now\n",
    "have_website":   "- Confirm they do NOT need to build a new website\n- Explain the redirect in plain language — no jargon\n- Mention the competitor risk: if they don't own it, someone else will\n",
    "agreed_no_pay":  "- Reference the agreement specifically\n- Create mild urgency — the domain is publicly listed\n- Make the next step as frictionless as possible\n",
    "angry":          "- Do NOT argue or defend yourself\n- Apologise briefly and sincerely in one sentence\n- Offer to remove them from further contact immediately\n- 2 sentences total\n",
    "expired_owner":  "- Explain calmly how expired domains enter the open market\n- Mention the direct traffic the domain is already receiving\n- Offer a fair price and invite a counter-offer\n",
    "rank_well":      "- Acknowledge their strong ranking — validate it genuinely\n- Pivot to the competitor risk angle: what if a rival buys it?\n- Mention branding/geo authority as a secondary benefit\n",
    "why_buy":        "- Lead with the single strongest benefit for their specific business type\n- Use concrete language about local search traffic\n- End with a direct question to move them forward\n",
    "not_now":        "- Respect the timing — don't push back on it\n- Offer a specific follow-up date or ask when to check back\n- Mention that the domain could be gone by then — plant mild urgency without pressure\n",
}

CLOSING_TECHNIQUES: dict[str, str] = {
    "no_thanks":          "End with an open door: make it feel genuinely easy to come back later.",
    "price_inquiry":      "End with a direct CTA: give them the buy link or tell them exactly what to do next.",
    "price_too_high":     "End by inviting their number: 'What would feel fair to you?' removes the standoff.",
    "negotiation":        "End with a specific counter number and a 24-hour soft deadline.",
    "trust_issue":        "End by naming a verifiable action they can take right now.",
    "have_website":       "End with the competitor fear close: 'The only risk is someone else getting there first.'",
    "follow_up":          "End with a binary choice: interested or not — make it easy to say either.",
    "sales_pitch":        "End with the most compelling outcome for their business, then one clear next step.",
    "re_engagement":      "End with a fresh hook — something new or different from the first conversation.",
    "objection_handling": "End with a question that invites them to share their real concern.",
    "agreed_no_pay":      "End with the purchase link and a friendly deadline.",
    "angry":              "End with a genuine release: no pitch, just acknowledge and let go.",
    "why_buy":            "End with a vivid picture of the outcome: 'Every local search could end with your site.'",
    "not_now":            "End with a specific check-back date so it doesn't get forgotten.",
    "rank_well":          "End with the ownership angle: 'Owning it costs less than defending against it.'",
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
    description="FastAPI + Anthropic claude-sonnet-4-6 + situation-based proactive replies",
    version="6.0.0",
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


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    customer_message: str
    tone: Optional[str]         = "professional and persuasive"
    api_key: Optional[str]      = None
    domain_name: Optional[str]  = None
    asking_price: Optional[str] = None
    sender_name: Optional[str]  = None     # signature name, e.g. "Alex"
    prospect_name: Optional[str]= None     # greeting name, e.g. "John"
    num_variations: int         = 3        # how many reply variations to return (1-3)
    mode: Optional[str]         = "ai"     # "ai" | "hybrid" (template→AI)

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


class GenerateResponse(BaseModel):
    subject: str                           # generated subject line
    replies: list[ReplyResult]             # 2-3 variations
    detected_intent: str
    retrieval_method: str
    similar_examples_used: list[dict]
    model_used: str
    tone_applied: str


class SituationResponse(BaseModel):
    subject: str
    replies: list[ReplyResult]
    detected_intent: str
    pitch_intensity: str
    situation_interpreted: str
    model_used: str
    tone_applied: str


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

    @field_validator("customer_message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_message cannot be empty.")
        return v.strip()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_replies() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_replies(data: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def detect_intent(msg: str) -> str:
    low = msg.lower()
    for intent, phrases in INTENT_KEYWORDS.items():
        if any(p in low for p in phrases):
            return intent
    return "general"


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


def get_api_key(request_key: Optional[str]) -> str:
    key = request_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400,
            detail="Anthropic API key required. Pass 'api_key' in the request or set ANTHROPIC_API_KEY env var.")
    return key


def call_claude(client: anthropic.Anthropic, system: str, user: str) -> tuple[str, str]:
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=system, messages=[{"role": "user", "content": user}]
        )
        return msg.content[0].text.strip(), msg.model
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit hit — wait a moment and retry.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert domain name broker with years of experience selling geo-targeted "
    "and keyword-rich .com domains to local businesses. "
    "You write email replies that are clear, persuasive, and human — never robotic or template-sounding. "
    "You understand negotiation psychology: when to hold firm, when to show flexibility, "
    "when to create urgency, and when to simply listen and empathise. "
    "You adapt your language to the prospect's emotional state."
)

SCORE_SYSTEM = (
    "You are a sales email quality reviewer. "
    "You assess domain-selling email replies and return a JSON object with two fields: "
    "'score' (integer 0-100) and 'reason' (one plain English sentence explaining the score). "
    "Criteria: Does it directly address the specific situation? Is the tone appropriate? "
    "Is there a clear next step? Is it free of filler and placeholders? "
    "Does it avoid being generic? Return ONLY the JSON object, nothing else."
)


def build_reply_prompt(message, intent, examples, tone, domain_name, asking_price,
                       retrieval_method, angle_instruction=None):
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
            ex_block += f"  [{i}] {ex.get('category','general')}\n       Situation: {ex['customer_message']}\n       Reply: {ex['reply']}\n\n"

    rules   = INTENT_RULES.get(intent, "- Respond naturally and professionally.\n")
    closing = CLOSING_TECHNIQUES.get(intent, "- End with a clear next step.")
    angle_block = f"\nANGLE TO USE:\n{angle_instruction}\n" if angle_instruction else ""

    quality_gate = (
        "\nBEFORE YOU FINISH — check:\n"
        "  - Directly addresses the situation (not generic)\n"
        "  - One clear next step at the end\n"
        "  - No filler openers, no placeholders\n"
        "  - Sounds like a real person\n"
    )

    return (
        f"TONE: {tone_inst}\nDETECTED INTENT: {intent.replace('_',' ').title()}\n\n"
        f"{domain_block}{ex_block}{angle_block}"
        f"\n─────────────────────────────────────────────\n"
        f"CURRENT SITUATION:\n\"\"\"{strip_filler(message)}\"\"\"\n"
        f"─────────────────────────────────────────────\n\n"
        f"WRITING RULES:\n{rules}\n"
        f"CLOSING TECHNIQUE: {closing}\n\n"
        f"HARD RULES:\n"
        f"- Do NOT copy examples verbatim\n"
        f"- Do NOT open with filler\n"
        f"- Do NOT use placeholders like [Name] or [Link]\n"
        f"- Write ONLY the reply body\n"
        f"{quality_gate}\nWrite the reply now:"
    )


def build_situation_prompt(situation, intent, intensity, examples, tone,
                           domain_name, asking_price, retrieval_method, include_urgency=False):
    """Prompt for situation-mode — injects pitch intensity, value props, soft CTAs, urgency."""
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
            ex_block += f"  [{i}] {ex.get('category','general')}\n       Situation: {ex['customer_message']}\n       Reply: {ex['reply']}\n\n"

    rules    = INTENT_RULES.get(intent, "- Respond naturally and professionally.\n")
    closing  = CLOSING_TECHNIQUES.get(intent, "- End with a clear next step.")
    value_p  = VALUE_PROPOSITIONS.get(intensity, VALUE_PROPOSITIONS["medium"])
    cta      = SOFT_CTA.get(intensity, SOFT_CTA["medium"])
    urgency  = URGENCY_LAYER.get(intensity, "") if include_urgency else ""

    intensity_desc = {
        "low":    "subtle background mention of value — do not push",
        "medium": "clear value + suggestion — be helpful, not pushy",
        "high":   "full compelling pitch — make the strongest possible case",
    }.get(intensity, "clear value + suggestion")

    urgency_block = f"\nURGENCY GUIDANCE:\n{urgency}" if urgency else ""

    return (
        f"TONE: {tone_inst}\n"
        f"DETECTED INTENT: {intent.replace('_',' ').title()}\n"
        f"PITCH INTENSITY: {intensity.upper()} — {intensity_desc}\n\n"
        f"{domain_block}{ex_block}"
        f"\n─────────────────────────────────────────────\n"
        f"SITUATION (described by the sender — there may be no direct quote from the prospect):\n"
        f"\"\"\"{strip_filler(situation)}\"\"\"\n"
        f"─────────────────────────────────────────────\n\n"
        f"CONTEXT: Write the appropriate email on behalf of the domain broker. "
        f"This could be a follow-up, first pitch, re-engagement, or response to an indirect signal. "
        f"Infer what the right message is from the situation above.\n\n"
        f"WRITING RULES:\n{rules}\n"
        f"VALUE & BENEFITS TO INJECT:\n{value_p}\n\n"
        f"CALL TO ACTION:\n{cta}"
        f"{urgency_block}\n\n"
        f"CLOSING TECHNIQUE: {closing}\n\n"
        f"HARD RULES:\n"
        f"- No filler openers\n"
        f"- No placeholders like [Name] or [Link]\n"
        f"- Write ONLY the email body\n"
        f"- Match intensity: {intensity.upper()} means {intensity_desc}\n\n"
        f"Write the reply now:"
    )



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


def generate_subject_ai(client: anthropic.Anthropic, intent: str, reply_body: str,
                        domain_name: Optional[str] = None) -> str:
    """
    Ask Claude to write a short email subject line.
    Falls back to the template version if the API call fails.
    """
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
        msg = client.messages.create(
            model=MODEL, max_tokens=40,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip().strip('"').strip("'")
    except Exception:
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

def quality_guard(client: anthropic.Anthropic, reply: str, situation: str) -> str:
    """
    Check the reply for common quality issues and fix them:
      - Too short → expand
      - Too long → trim
      - Missing closing → fix
    Returns the (possibly corrected) reply.
    """
    words = len(reply.split())

    has_closing = bool(re.search(
        r"(best regards|kind regards|warm regards|best wishes|thanks|thank you|cheers|sincerely)",
        reply, re.IGNORECASE
    ))

    issues = []
    if words < MIN_REPLY_WORDS:
        issues.append(f"TOO SHORT ({words} words) — expand to at least {MIN_REPLY_WORDS} words")
    if words > MAX_REPLY_WORDS:
        issues.append(f"TOO LONG ({words} words) — trim to under {MAX_REPLY_WORDS} words")
    if not has_closing:
        issues.append("MISSING CLOSING — add a closing line like 'Best regards'")

    if not issues:
        return reply  # already fine — skip the extra API call

    fix_prompt = (
        f"Fix the following email reply. Issues found:\n"
        + "\n".join(f"- {i}" for i in issues)
        + f"\n\nOriginal reply:\n{reply}\n\n"
        f"Situation it was written for: {situation}\n\n"
        f"Return ONLY the corrected reply, nothing else."
    )
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": fix_prompt}]
        )
        fixed = msg.content[0].text.strip()
        print(f"[QualityGuard] Fixed: {', '.join(issues)}")
        return fixed
    except Exception as e:
        print(f"[QualityGuard] Fix failed: {e}")
        return reply  # return original if fix fails


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID MODE (Template → AI polish)
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_mode(client: anthropic.Anthropic, customer_message: str, intent: str,
                    domain_name: Optional[str], asking_price: Optional[str], tone: str) -> str:
    """
    Hybrid flow:
      1. Build a reply using the template engine (fast, no AI)
      2. Send that reply to Claude for polish — improve clarity, tone, persuasion
    Returns the AI-polished version.
    """
    # Step 1 — template reply
    template_result = build_template_reply(
        customer_message=customer_message,
        domain_name=domain_name,
        asking_price=asking_price,
        force_intent=intent,
    )
    template_reply = template_result.get("reply", "")

    # Step 2 — AI polish
    polish_prompt = (
        f"Below is a domain sales email reply that was built from templates.\n"
        f"Your job is to improve it — make it more natural, persuasive, and human.\n"
        f"Keep the same core message and intent. Do NOT change the meaning or add new facts.\n"
        f"Tone to use: {tone}\n\n"
        f"Original template reply:\n{template_reply}\n\n"
        f"Write ONLY the improved reply body. No explanation."
    )
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": polish_prompt}]
        )
        return msg.content[0].text.strip()
    except Exception:
        return template_reply  # fall back to raw template if polish fails


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-VARIATION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

VARIATION_STYLES = [
    {
        "label":       "Safe",
        "instruction": (
            "Write a balanced, professional reply. Friendly but not pushy. "
            "This is the 'safe' option — solid and reliable."
        ),
    },
    {
        "label":       "Persuasive",
        "instruction": (
            "Write a more compelling reply. Lead with the strongest value point. "
            "Be confident and direct. Add gentle urgency if it fits. "
            "This is the 'persuasive' option — more assertive."
        ),
    },
    {
        "label":       "Short",
        "instruction": (
            "Write a very brief reply — 2 to 4 sentences maximum. "
            "Get straight to the point. No padding, no build-up. "
            "This is the 'short' option — quick and easy to read on mobile."
        ),
    },
]

def generate_variations(
    client: anthropic.Anthropic,
    base_prompt: str,
    num: int,
    situation: str,
    prospect_name: Optional[str],
    sender_name: Optional[str],
    intent: str,
    domain_name: Optional[str],
) -> list[ReplyResult]:
    """
    Generate `num` reply variations (1-3), each with a different style.
    Each variation is:
      - formatted as a proper email
      - quality-checked
      - scored
    """
    num     = max(1, min(3, num))
    styles  = VARIATION_STYLES[:num]
    results = []

    for style in styles:
        # Inject the variation style instruction into the prompt
        variation_prompt = base_prompt + (
            f"\n\nVARIATION STYLE — {style['label'].upper()}:\n{style['instruction']}\n"
            f"Write this variation now:"
        )
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": variation_prompt}]
            )
            raw = msg.content[0].text.strip()
        except Exception as e:
            raw = f"[Generation failed for {style['label']}: {str(e)}]"

        # Format as proper email
        formatted = format_email_body(raw, prospect_name=prospect_name, sender_name=sender_name)
        # Quality guard
        fixed     = quality_guard(client, formatted, situation)
        # Score
        score, reason = score_reply(client, situation, fixed)

        results.append(ReplyResult(
            reply=fixed,
            label=style["label"],
            confidence_score=score,
            confidence_reason=reason,
        ))

    return results

def build_score_prompt(situation: str, reply: str) -> str:
    return (
        f"SITUATION: {situation}\n\n"
        f"REPLY TO ASSESS:\n{reply}\n\n"
        "Return a JSON object: {\"score\": <0-100>, \"reason\": \"<one sentence>\"}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_reply(client: anthropic.Anthropic, situation: str, reply_text: str) -> tuple[int, str]:
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=150, system=SCORE_SYSTEM,
            messages=[{"role": "user", "content": build_score_prompt(situation, reply_text)}]
        )
        raw  = re.sub(r"```(?:json)?", "", msg.content[0].text.strip()).strip().rstrip("```").strip()
        data = json.loads(raw)
        score  = max(0, min(100, int(data.get("score", 75))))
        reason = str(data.get("reason", "")).strip() or "Reply looks good."
        return score, reason
    except Exception as e:
        print(f"[Score] Scoring failed: {e}")
        return 75, "Could not auto-score this reply."


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/generate-reply", response_model=GenerateResponse)
async def generate_reply(req: GenerateRequest):
    """
    Standard reply — for direct prospect messages.
    Returns subject line + 2-3 formatted email variations with quality guard.
    Supports mode="hybrid" to run template→AI polish instead of pure AI.
    """
    api_key  = get_api_key(req.api_key)
    intent   = detect_intent(req.customer_message)
    tone     = req.tone or "professional and persuasive"
    replies  = load_replies()
    examples, method = retrieve(req.customer_message, replies, intent, api_key, TOP_K)
    client   = anthropic.Anthropic(api_key=api_key)

    # Hybrid mode: use template engine as the base, then AI-polish
    if req.mode == "hybrid":
        base_body = run_hybrid_mode(
            client, req.customer_message, intent,
            req.domain_name, req.asking_price, tone
        )
        # Wrap in a single variation
        formatted = format_email_body(base_body, req.prospect_name, req.sender_name)
        fixed     = quality_guard(client, formatted, req.customer_message)
        score, reason = score_reply(client, req.customer_message, fixed)
        variations = [ReplyResult(reply=fixed, label="Hybrid", confidence_score=score, confidence_reason=reason)]
    else:
        # Standard AI mode — build base prompt, generate N variations
        base_prompt = build_reply_prompt(
            req.customer_message, intent, examples, tone,
            req.domain_name, req.asking_price, method,
        )
        variations = generate_variations(
            client, base_prompt,
            num=req.num_variations,
            situation=req.customer_message,
            prospect_name=req.prospect_name,
            sender_name=req.sender_name,
            intent=intent,
            domain_name=req.domain_name,
        )

    # Generate subject line using the first variation as context
    subject = generate_subject_ai(client, intent, variations[0].reply, req.domain_name)

    return GenerateResponse(
        subject=subject,
        replies=variations,
        detected_intent=intent,
        retrieval_method=method,
        similar_examples_used=[{"category": ex.get("category",""), "snippet": ex["customer_message"][:80]} for ex in examples],
        model_used=MODEL,
        tone_applied=tone,
    )


@app.post("/generate-reply/situation", response_model=SituationResponse)
async def generate_reply_situation(req: SituationRequest):
    """
    Situation-based generation.
    Accepts natural-language description of what's happening.
    Returns subject + 2-3 formatted variations with quality guard.
    """
    api_key   = get_api_key(req.api_key)
    intent    = req.force_intent or detect_situation_intent(req.situation)
    intensity = req.force_intensity or INTENT_TO_INTENSITY.get(intent, "medium")
    tone      = req.tone or "professional and persuasive"
    replies   = load_replies()
    examples, method = retrieve(req.situation, replies, intent, api_key, TOP_K)
    client    = anthropic.Anthropic(api_key=api_key)

    base_prompt = build_situation_prompt(
        situation=req.situation, intent=intent, intensity=intensity,
        examples=examples, tone=tone,
        domain_name=req.domain_name, asking_price=req.asking_price,
        retrieval_method=method, include_urgency=req.include_urgency,
    )
    variations = generate_variations(
        client, base_prompt,
        num=req.num_variations,
        situation=req.situation,
        prospect_name=req.prospect_name,
        sender_name=req.sender_name,
        intent=intent,
        domain_name=req.domain_name,
    )
    subject = generate_subject_ai(client, intent, variations[0].reply, req.domain_name)

    return SituationResponse(
        subject=subject,
        replies=variations,
        detected_intent=intent,
        pitch_intensity=intensity,
        situation_interpreted=(
            f"Intent: {intent.replace('_', ' ').title()} · "
            f"Intensity: {intensity.title()} pitch · "
            f"Urgency: {'on' if req.include_urgency else 'off'}"
        ),
        model_used=MODEL,
        tone_applied=tone,
    )


@app.post("/generate-reply/template")
async def generate_reply_template(req: TemplateRequest):
    """Template mode: keyword matching + component assembly. No AI unless ai_polish=True."""
    result = build_template_reply(
        customer_message=req.customer_message,
        domain_name=req.domain_name,
        asking_price=req.asking_price,
        force_intent=req.force_intent,
        response_length=getattr(req, "response_length", "medium") or "medium",
        length_instructions=getattr(req, "length_instructions", None),
    )
    if not req.ai_polish:
        return result
    api_key = get_api_key(req.api_key)
    tone_with_length = req.tone
    if getattr(req, "response_length", "medium") == "long":
        tone_with_length += ". Write a detailed, multi-paragraph email — minimum 4 paragraphs with full explanation, value proposition, and clear call to action."
    elif getattr(req, "response_length", "medium") == "short":
        tone_with_length += ". Keep the reply short — maximum 3 sentences total."
    return ai_polish_reply(
        template_reply=result["reply"], customer_message=req.customer_message,
        intent=result["detected_intent"], api_key=api_key,
        domain_name=req.domain_name, asking_price=req.asking_price, tone=tone_with_length,
    )


@app.post("/generate-reply/template/detect-intent")
async def detect_intent_template(req: TemplateRequest):
    intent = req.force_intent or detect_template_intent(req.customer_message)
    return {"customer_message": req.customer_message, "detected_intent": intent,
            "available_intents": list(TEMPLATE_INTENT_KEYWORDS.keys())}


@app.post("/generate-reply/stream")
async def generate_reply_stream(req: GenerateRequest):
    """Streaming version. Reply appears word-by-word."""
    api_key  = get_api_key(req.api_key)
    intent   = detect_intent(req.customer_message)
    replies  = load_replies()
    examples, method = retrieve(req.customer_message, replies, intent, api_key, TOP_K)
    user_prompt = build_reply_prompt(
        req.customer_message, intent, examples,
        req.tone or "professional and persuasive",
        req.domain_name, req.asking_price, method,
    )

    async def stream() -> AsyncGenerator[str, None]:
        full_text = ""
        try:
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            ) as s:
                for chunk in s.text_stream:
                    full_text += chunk
                    safe_chunk = chunk.replace("\n", "\\n")
                    yield f"data: {safe_chunk}\n\n"
            score, reason = score_reply(client, req.customer_message, full_text)
            meta = json.dumps({"intent": intent, "retrieval_method": method,
                               "score": score, "reason": reason,
                               "examples": [{"category": ex.get("category",""), "snippet": ex["customer_message"][:60]} for ex in examples]})
            yield f"data: [META] {meta}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/generate-reply/alternatives", response_model=AlternativesResponse)
async def generate_alternatives(req: GenerateRequest):
    """Generate 3 alternative replies using different persuasion angles."""
    api_key  = get_api_key(req.api_key)
    intent   = detect_intent(req.customer_message)
    replies  = load_replies()
    examples, method = retrieve(req.customer_message, replies, intent, api_key, TOP_K)
    client     = anthropic.Anthropic(api_key=api_key)
    results: list[ReplyResult] = []
    model_used = MODEL
    for angle in ALTERNATIVE_ANGLES:
        user_prompt = build_reply_prompt(
            req.customer_message, intent, examples,
            req.tone or "professional and persuasive",
            req.domain_name, req.asking_price, method,
            angle_instruction=angle["instruction"],
        )
        reply_text, model_used = call_claude(client, SYSTEM_PROMPT, user_prompt)
        score, reason = score_reply(client, req.customer_message, reply_text)
        results.append(ReplyResult(reply=reply_text, confidence_score=score,
                                   confidence_reason=reason, angle=angle["label"]))
    return AlternativesResponse(alternatives=results, detected_intent=intent, model_used=model_used)



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
    key = (body or {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY","")
    if not key:
        raise HTTPException(status_code=400, detail="API key required.")
    try:
        _index.build(load_replies(), key)
        return {"message": "Index rebuilt.", "total": len(_index.entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH + INFO
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    replies = load_replies()
    stale   = _index.is_stale(len(replies))
    return {"status": "ok", "version": "6.0.0", "model": MODEL, "embed_model": EMBED_MODEL,
            "kb_size": len(replies), "semantic_ready": _index.ready and not stale,
            "retrieval_method": "semantic" if (_index.ready and not stale) else "keyword"}

@app.get("/info")
async def info():
    return {
        "name": "Domain Email Reply Generator", "version": "6.0.0",
        "new_in_v5": [
            "POST /generate-reply/situation — situation-based proactive generation",
            "4 new intents: follow_up, sales_pitch, re_engagement, objection_handling",
            "Pitch intensity: low / medium / high — auto-selected per intent",
            "Value propositions + benefits injected per intensity",
            "Soft CTAs and optional light urgency",
        ],
        "endpoints": [
            "POST   /generate-reply           — AI or hybrid mode, 2-3 variations, subject line",
            "POST   /generate-reply/regenerate — same as above with reshuffled styles",
            "POST   /generate-reply/situation  — situation-based proactive generation",
            "POST   /generate-reply/stream     — streaming version",
            "POST   /generate-reply/alternatives",
            "POST   /generate-reply/template",
            "POST   /generate-reply/template/detect-intent",
            "GET    /replies", "GET    /replies/search?q=...",
            "POST   /replies", "POST   /replies/save-generated",
            "DELETE /replies/{id}",
            "GET    /categories", "GET    /embed/status",
            "POST   /embed/rebuild", "GET    /health",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
