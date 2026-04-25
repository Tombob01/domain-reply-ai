"""
Domain Email Reply Generator — FastAPI + Anthropic Claude API
=============================================================
Stack:  FastAPI           (web framework)
        Anthropic SDK     (claude-sonnet-4-6)
        voyageai          (voyage-3 embeddings)
        Pydantic v2       (validation)
        Uvicorn           (ASGI server)

What's new in Step 4 — Quality & Alternatives:
  ✓ Confidence score (0–100) on every reply — Claude grades its own output
  ✓ Confidence explains WHY the score is what it is (1 plain-English sentence)
  ✓ /generate-reply/alternatives — generate 3 different angle replies at once
  ✓ Persuasion layer — closing_technique injected per intent
  ✓ Anti-pattern filter — strips filler phrases before sending to Claude
  ✓ Reply quality checklist — Claude checks its own reply before returning it
  ✓ Version 4.0.0

HOW TO RUN:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

ENVIRONMENT:
  ANTHROPIC_API_KEY — for Claude generation + Voyage embeddings
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
MAX_TOKENS   = 700
TOP_K        = 4
DATA_FILE    = Path(__file__).parent / "past_replies.json"
INDEX_FILE   = Path(__file__).parent / "embeddings_index.json"
FRONTEND_DIR = Path(__file__).parent

# Filler phrases that make emails sound robotic — stripped from prompts
FILLER_PHRASES = [
    "i hope this email finds you well",
    "trust you are doing great",
    "hope you are having a great day",
    "i hope you are doing well",
    "i hope all is well",
    "i wanted to reach out",
    "please do not hesitate",
    "feel free to reach out",
    "as per my last email",
    "going forward",
    "at the end of the day",
    "it is what it is",
]

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
    "professional and persuasive": (
        "Write in a confident, professional tone. Use persuasive but respectful language. Lead with value, not pressure."
    ),
    "warm and friendly": (
        "Write in a warm, conversational tone — as if talking to a local business owner you want to help. Be personable and genuine."
    ),
    "firm but respectful": (
        "Hold your position clearly. Be polite but do not over-apologise or cave to pressure. State your case once, cleanly."
    ),
    "concise and direct": (
        "Be extremely brief — 2 to 3 sentences maximum. Get straight to the point. No padding, no filler."
    ),
    "empathetic and understanding": (
        "Lead with understanding of the prospect's concern. Validate their feeling before pivoting to value. Never dismiss their objection."
    ),
}

INTENT_RULES: dict[str, str] = {
    "no_thanks": (
        "- Acknowledge their decision graciously without arguing\n"
        "- Ask one soft optional question about their objection\n"
        "- Leave the door open for future contact\n"
    ),
    "price_inquiry": (
        "- State the price clearly and confidently\n"
        "- Add one brief reason why the price reflects real value\n"
        "- End with an invitation to proceed or ask questions\n"
    ),
    "price_too_high": (
        "- Validate the $10 registration point, then reframe what makes this domain premium\n"
        "- Invite a counter-offer rather than just defending the price\n"
    ),
    "negotiation": (
        "- Counter or ask for their best offer — don't just accept a low number\n"
        "- Give a brief reason for your floor price\n"
        "- Keep the negotiation moving — no dead ends\n"
    ),
    "trust_issue": (
        "- Address the concern directly — do not get defensive\n"
        "- Name a specific trust mechanism: DAN.com escrow, GoDaddy listing, Trustpilot reviews\n"
        "- Offer a verification step they can take independently right now\n"
    ),
    "have_website": (
        "- Confirm they do NOT need to build a new website\n"
        "- Explain the redirect in plain language — no jargon\n"
        "- Mention the competitor risk: if they don't own it, someone else will\n"
    ),
    "follow_up": (
        "- Keep it to 2–3 sentences maximum\n"
        "- Do not re-pitch the full value proposition — they've already seen it\n"
        "- Give them an easy out: 'just let me know either way'\n"
    ),
    "agreed_no_pay": (
        "- Reference the agreement specifically\n"
        "- Create mild urgency — the domain is publicly listed, anyone can buy it\n"
        "- Make the next step as frictionless as possible\n"
    ),
    "angry": (
        "- Do NOT argue or defend yourself\n"
        "- Apologise briefly and sincerely in one sentence\n"
        "- Offer to remove them from further contact immediately\n"
        "- 2 sentences total — less is more\n"
    ),
    "expired_owner": (
        "- Explain calmly how expired domains enter the open market\n"
        "- Mention the direct traffic the domain is already receiving\n"
        "- Offer a fair price and invite a counter-offer\n"
    ),
    "rank_well": (
        "- Acknowledge their strong ranking — validate it genuinely\n"
        "- Pivot to the competitor risk angle: what if a rival buys it?\n"
        "- Mention branding/geo authority as a secondary benefit beyond SEO\n"
    ),
    "why_buy": (
        "- Lead with the single strongest benefit for their specific business type\n"
        "- Use concrete language: 'every person in [city] searching for [service] could land on your site'\n"
        "- End with a direct question to move them forward\n"
    ),
    "not_now": (
        "- Respect the timing — don't push back on it\n"
        "- Offer a specific follow-up date or ask when to check back\n"
        "- Mention that the domain could be gone by then — plant mild urgency without pressure\n"
    ),
}

# Closing techniques matched to intent — injected into the prompt
CLOSING_TECHNIQUES: dict[str, str] = {
    "no_thanks":      "End with an open door: make it feel genuinely easy to come back later.",
    "price_inquiry":  "End with a direct CTA: give them the buy link or tell them exactly what to do next.",
    "price_too_high": "End by inviting their number: 'What would feel fair to you?' removes the standoff.",
    "negotiation":    "End with a specific counter number and a 24-hour soft deadline.",
    "trust_issue":    "End by naming a verifiable action they can take right now — 'search DAN.com on Trustpilot'.",
    "have_website":   "End with the competitor fear close: 'The only risk is someone else getting there first.'",
    "follow_up":      "End with a binary choice: interested or not — make it easy to say either.",
    "agreed_no_pay":  "End with the purchase link and a friendly deadline to protect their spot.",
    "angry":          "End with a genuine release: no pitch, just acknowledge and let go.",
    "why_buy":        "End with a vivid picture of the outcome: 'Every local search could end with your site.'",
    "not_now":        "End with a specific check-back date so it doesn't get forgotten.",
    "rank_well":      "End with the ownership angle: 'Owning it costs less than defending against it.'",
}

# Alternative angles for the /alternatives endpoint
ALTERNATIVE_ANGLES: list[dict] = [
    {
        "label": "Lead with competitor risk",
        "instruction": (
            "Open by highlighting the risk of a competitor buying this domain first. "
            "Make that the central argument. Keep the rest supporting that fear."
        ),
    },
    {
        "label": "Lead with SEO value",
        "instruction": (
            "Open by explaining the keyword and search traffic value. "
            "Focus on how this domain could drive free organic traffic and reduce ad spend."
        ),
    },
    {
        "label": "Lead with social proof",
        "instruction": (
            "Open by referencing how other businesses in similar cities/industries have used "
            "geo-targeted domains successfully. Use the DAN.com escrow trust angle if relevant."
        ),
    },
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
    description="FastAPI + Anthropic claude-sonnet-4-6 + Voyage semantic search + confidence scoring",
    version="4.0.0",
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

    @field_validator("customer_message")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_message cannot be empty.")
        return v.strip()


class ReplyResult(BaseModel):
    """A single generated reply with its quality metadata."""
    reply: str
    confidence_score: int           # 0–100
    confidence_reason: str          # One plain-English sentence
    angle: Optional[str] = None     # e.g. "Lead with competitor risk"


class GenerateResponse(BaseModel):
    result: ReplyResult
    detected_intent: str
    retrieval_method: str
    similar_examples_used: list[dict]
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



# ── Template Mode Request ──────────────────────────────────────────────────
class TemplateRequest(BaseModel):
    customer_message: str
    domain_name: Optional[str] = None
    asking_price: Optional[str] = None
    force_intent: Optional[str] = None
    ai_polish: bool = False          # set True to run Claude improvement pass
    api_key: Optional[str] = None    # only needed when ai_polish=True
    tone: str = "professional and persuasive"

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


def strip_filler(text: str) -> str:
    """Remove known filler phrases from a prompt before sending to Claude."""
    low = text.lower()
    for phrase in FILLER_PHRASES:
        if phrase in low:
            # Remove it case-insensitively
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub("", text)
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
        raise HTTPException(
            status_code=400,
            detail="Anthropic API key required. Pass 'api_key' in the request or set ANTHROPIC_API_KEY environment variable. Get yours at https://console.anthropic.com"
        )
    return key


def call_claude(client: anthropic.Anthropic, system: str, user: str) -> tuple[str, str]:
    """Call Claude and return (text, model_name). Raises HTTPException on API errors."""
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


def build_reply_prompt(
    message: str,
    intent: str,
    examples: list[dict],
    tone: str,
    domain_name: Optional[str],
    asking_price: Optional[str],
    retrieval_method: str,
    angle_instruction: Optional[str] = None,
) -> str:
    tone_inst    = TONE_INSTRUCTIONS.get(tone, f"Tone: {tone}.")
    intent_label = intent.replace("_", " ").title()
    method_note  = "by semantic meaning" if retrieval_method == "semantic" else "by keyword"

    domain_block = ""
    if domain_name or asking_price:
        parts = []
        if domain_name:  parts.append(f"Domain being sold: {domain_name}")
        if asking_price: parts.append(f"Asking price: {asking_price}")
        domain_block = "\n".join(parts) + "\n"

    ex_block = ""
    if examples:
        ex_block = f"REFERENCE EXAMPLES (retrieved {method_note} — style references only, do NOT copy):\n\n"
        for i, ex in enumerate(examples, 1):
            ex_block += (
                f"  [{i}] {ex.get('category','general')}\n"
                f"       Situation: {ex['customer_message']}\n"
                f"       Reply: {ex['reply']}\n\n"
            )

    rules = INTENT_RULES.get(intent, "- Respond naturally and professionally.\n")
    closing = CLOSING_TECHNIQUES.get(intent, "- End with a clear next step.")

    angle_block = ""
    if angle_instruction:
        angle_block = f"\nANGLE TO USE FOR THIS REPLY:\n{angle_instruction}\n"

    # Self-check quality gate — Claude reviews its own reply before finalising
    quality_gate = (
        "\nBEFORE YOU FINISH — mentally check your reply against these:\n"
        "  □ Does it directly address what was said (not a generic response)?\n"
        "  □ Is there exactly one clear next step or question at the end?\n"
        "  □ Have you avoided all filler openers?\n"
        "  □ Is it under 120 words unless the situation genuinely needs more?\n"
        "  □ Does it sound like a real person wrote it?\n"
        "If any box is unchecked, revise before outputting.\n"
    )

    return f"""TONE: {tone_inst}
DETECTED INTENT: {intent_label}

{domain_block}{ex_block}{angle_block}
─────────────────────────────────────────────
CURRENT SITUATION:
\"\"\"{strip_filler(message)}\"\"\"
─────────────────────────────────────────────

WRITING RULES:
{rules}
CLOSING TECHNIQUE: {closing}

HARD RULES — never break these:
- Do NOT copy any example reply verbatim
- Do NOT open with filler like "I hope this email finds you well"
- Do NOT include placeholders like [Your Name] or [Link]
- Do NOT end without a clear next step or open question
- Write ONLY the reply body — no subject line, no sign-off name
{quality_gate}
Write the reply now:"""


def build_score_prompt(situation: str, reply: str) -> str:
    return (
        f"SITUATION: {situation}\n\n"
        f"REPLY TO ASSESS:\n{reply}\n\n"
        "Return a JSON object: {{\"score\": <0-100>, \"reason\": \"<one sentence>\"}}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_reply(client: anthropic.Anthropic, situation: str, reply_text: str) -> tuple[int, str]:
    """
    Ask Claude to score its own reply.

    Returns (score: int, reason: str).
    Falls back to (75, 'Auto-scored') if parsing fails — never crashes.

    Why self-scoring works:
      Claude is very good at identifying whether a reply actually addresses
      a specific situation, has filler, or is missing a next step.
      This gives the user transparent signal about reply quality.
    """
    try:
        prompt = build_score_prompt(situation, reply_text)
        msg = client.messages.create(
            model=MODEL, max_tokens=150,
            system=SCORE_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        raw  = msg.content[0].text.strip()
        # Strip markdown code fences if Claude adds them
        raw  = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
        score  = max(0, min(100, int(data.get("score", 75))))
        reason = str(data.get("reason", "")).strip() or "Reply looks good."
        return score, reason
    except Exception as e:
        print(f"[Score] Scoring failed: {e} — using fallback")
        return 75, "Could not auto-score this reply."


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/generate-reply", response_model=GenerateResponse)
async def generate_reply(req: GenerateRequest):
    """
    Generate one reply with a confidence score.

    Pipeline:
      1. Detect intent
      2. Retrieve top-4 semantically similar examples
      3. Build prompt (with quality gate + closing technique)
      4. Generate with claude-sonnet-4-6
      5. Score the reply (0–100) with a plain-English reason
      6. Return everything
    """
    api_key = get_api_key(req.api_key)
    intent  = detect_intent(req.customer_message)
    replies = load_replies()
    examples, method = retrieve(req.customer_message, replies, intent, api_key, TOP_K)

    user_prompt = build_reply_prompt(
        req.customer_message, intent, examples,
        req.tone or "professional and persuasive",
        req.domain_name, req.asking_price, method,
    )
    client = anthropic.Anthropic(api_key=api_key)
    reply_text, model_used = call_claude(client, SYSTEM_PROMPT, user_prompt)

    score, reason = score_reply(client, req.customer_message, reply_text)

    return GenerateResponse(
        result=ReplyResult(reply=reply_text, confidence_score=score, confidence_reason=reason),
        detected_intent=intent,
        retrieval_method=method,
        similar_examples_used=[
            {"category": ex.get("category",""), "snippet": ex["customer_message"][:80]}
            for ex in examples
        ],
        model_used=model_used,
        tone_applied=req.tone or "professional and persuasive",
    )


# ── TEMPLATE MODE ─────────────────────────────────────────────────────────
@app.post("/generate-reply/template")
async def generate_reply_template(req: TemplateRequest):
    """
    Template mode: generates a reply using keyword matching + component assembly.
    No AI, no API key required unless ai_polish=True.
    """
    result = build_template_reply(
        customer_message=req.customer_message,
        domain_name=req.domain_name,
        asking_price=req.asking_price,
        force_intent=req.force_intent,
    )

    if not req.ai_polish:
        return result

    # AI polish requested — run improvement pass with Claude
    api_key = get_api_key(req.api_key)  # reuses your existing helper
    polished = ai_polish_reply(
        template_reply=result["reply"],
        customer_message=req.customer_message,
        intent=result["detected_intent"],
        api_key=api_key,
        domain_name=req.domain_name,
        asking_price=req.asking_price,
        tone=req.tone,
    )
    return polished


# ── INTENT DETECTION UTILITY ──────────────────────────────────────────────
@app.post("/generate-reply/template/detect-intent")
async def detect_intent_template(req: TemplateRequest):
    """
    Returns only the detected intent without generating a reply.
    Useful for testing and debugging intent detection.
    """
    intent = req.force_intent or detect_template_intent(req.customer_message)
    return {
        "customer_message": req.customer_message,
        "detected_intent":  intent,
        "available_intents": list(TEMPLATE_INTENT_KEYWORDS.keys())
    }


@app.post("/generate-reply/stream")
async def generate_reply_stream(req: GenerateRequest):
    """
    Streaming version. Reply appears word-by-word.
    After completion sends:
      data: [META] {intent, retrieval_method, examples, score, reason}
      data: [DONE]
    """
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
                    yield f"data: {chunk.replace(chr(10), '\\n')}\n\n"

            # Score after streaming completes
            score, reason = score_reply(client, req.customer_message, full_text)

            meta = json.dumps({
                "intent":           intent,
                "retrieval_method": method,
                "score":            score,
                "reason":           reason,
                "examples": [
                    {"category": ex.get("category",""), "snippet": ex["customer_message"][:60]}
                    for ex in examples
                ],
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
    Generate 3 alternative replies using different persuasion angles:
      1. Competitor risk
      2. SEO value
      3. Social proof / trust

    Each reply gets its own confidence score.
    Useful when you want to choose the best angle for a specific prospect.
    """
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
        results.append(ReplyResult(
            reply=reply_text,
            confidence_score=score,
            confidence_reason=reason,
            angle=angle["label"],
        ))

    return AlternativesResponse(
        alternatives=results,
        detected_intent=intent,
        model_used=model_used,
    )


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
    return {
        "semantic_ready":   _index.ready and not stale,
        "index_size":       len(_index.entries),
        "kb_size":          len(replies),
        "is_stale":         stale,
        "retrieval_method": "semantic" if (_index.ready and not stale) else "keyword",
    }


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
    return {
        "status":           "ok",
        "version":          "4.0.0",
        "model":            MODEL,
        "embed_model":      EMBED_MODEL,
        "kb_size":          len(replies),
        "semantic_ready":   _index.ready and not stale,
        "retrieval_method": "semantic" if (_index.ready and not stale) else "keyword",
    }


@app.get("/info")
async def info():
    return {
        "name": "Domain Email Reply Generator", "version": "4.0.0",
        "stack": [f"FastAPI", f"Anthropic {MODEL}", f"Voyage {EMBED_MODEL}"],
        "new_in_v4": [
            "Confidence score (0-100) on every reply",
            "Plain-English confidence reason",
            "POST /generate-reply/alternatives — 3 angles at once",
            "Persuasion closing technique per intent",
            "Filler phrase filter",
            "Self-check quality gate in prompt",
        ],
        "endpoints": [
            "POST   /generate-reply",
            "POST   /generate-reply/stream",
            "POST   /generate-reply/alternatives",
            "POST   /generate-reply/template",
            "POST   /generate-reply/template/detect-intent",
            "GET    /replies",
            "GET    /replies/search?q=...",
            "POST   /replies",
            "DELETE /replies/{id}",
            "GET    /categories",
            "GET    /embed/status",
            "POST   /embed/rebuild",
            "GET    /health",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
