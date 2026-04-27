"""
template_engine.py — Modular Email Reply Generator (Template Mode)
===================================================================
No AI. No external libraries. Pure Python.

Architecture:
  - COMPONENTS: structured dict of reusable email parts
  - detect_intent(): keyword-based intent detection (reuses main.py logic)
  - build_template_reply(): selects + assembles components into a full reply
  - /generate-reply/template: FastAPI endpoint (add to main.py)

All components were extracted and compressed from 161 real past email replies.
Variations are grouped so no two calls produce identical output.
"""

import random
import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT LIBRARY
# Keys: intent name
# Values: dict with "greeting", "acknowledgment", "body", "closing"
# Each value is a LIST — random.choice() picks one per call
# ─────────────────────────────────────────────────────────────────────────────

COMPONENTS: dict[str, dict[str, list[str]]] = {

    # ── GREETINGS (shared pool, intent-neutral) ───────────────────────────────
    "_greetings": [
        "Hi,",
        "Hello,",
        "Good day,",
        "Hi there,",
        "Hello there,",
        "Dear sir or madam,",
        "Good morning,",
    ],

    # ── CLOSINGS (shared pool, intent-neutral) ────────────────────────────────
    "_closings": [
        "Best regards.",
        "Kind regards.",
        "Warm regards.",
        "Regards.",
        "Thanks for your time.",
        "Looking forward to hearing from you.",
        "I look forward to your response.",
        "Do let me know if you have any questions.",
        "Feel free to reach out if you need anything further.",
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: no_thanks
    # ─────────────────────────────────────────────────────────────────────────
    "no_thanks": {
        "acknowledgment": [
            "Thank you for taking the time to reply — I appreciate it.",
            "Thanks for getting back to me.",
            "Thank you for the honest response.",
            "I appreciate you responding.",
        ],
        "body": [
            (
                "If you don't mind me asking — what's holding you back? "
                "I ask to understand, not to push you. "
                "If it's the price, feel free to share any offer you have in mind."
            ),
            (
                "I understand completely. "
                "Just know that my pricing is negotiable, and should you reconsider, "
                "we can handle the transaction securely through an escrow service like Dan.com or GoDaddy."
            ),
            (
                "I still believe this domain could work to your advantage — "
                "I've seen over two dozen competitors in this space optimizing for these exact keywords. "
                "That said, I fully respect your decision."
            ),
            (
                "May I ask what the objection is, if you don't mind sharing? "
                "Your feedback genuinely helps me understand what matters to buyers in your industry."
            ),
        ],
        "closing": [
            "If you ever change your mind, this email is always open. Best regards.",
            "The door remains open — feel free to get in touch anytime. Kind regards.",
            "Should you reconsider down the road, don't hesitate to reach out.",
            "Wishing you all the best with your business.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: price_inquiry
    # ─────────────────────────────────────────────────────────────────────────
    "price_inquiry": {
        "acknowledgment": [
            "Thanks for your interest in this domain.",
            "Thank you for reaching out about the price.",
            "Great to hear from you.",
        ],
        "body": [
            (
                "The domain is currently listed at a competitive price. "
                "Based on GoDaddy's appraisal and comparable recent sales in this niche, "
                "this is well below retail value. "
                "Offers are welcome — just let me know your number."
            ),
            (
                "We are looking for a fee starting from $650, though this is negotiable. "
                "The domain carries strong keyword and geo value, which justifies the premium. "
                "That said, if you have a different number in mind, share it and we can work something out."
            ),
            (
                "The asking price is listed on the domain marketplace. "
                "Since we are looking to move this as part of a portfolio sale, "
                "we are open to a reasonable offer — visit the listing or reply with what works for you."
            ),
            (
                "I'm not asking for full retail value — this is a wholesale price. "
                "GoDaddy's appraisal puts similar domains well above $1,000, "
                "and I'm offering this at a fraction of that. "
                "Reply with your best offer if the listed price doesn't work."
            ),
        ],
        "closing": [
            "Visit the listing to purchase or submit an offer directly. Let me know if you have questions.",
            "For immediate ownership, click the listing link. Happy to answer any questions.",
            "Feel free to reply with a counter-offer — I'm sure we can find a number that works for both of us.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: price_too_high
    # ─────────────────────────────────────────────────────────────────────────
    "price_too_high": {
        "acknowledgment": [
            "I hear you — and I appreciate the honest feedback.",
            "Thanks for the candid response.",
            "I understand the concern about price.",
        ],
        "body": [
            (
                "You're right that a new domain registration costs around $10. "
                "The difference here is that this domain is city-specific, "
                "carries your exact business keywords, and already has SEO traction — "
                "that kind of name simply can't be hand-registered for $10."
            ),
            (
                "This isn't a standard registration — it's a premium geo-keyword domain. "
                "The value comes from its ability to rank on search engines and capture direct type-in traffic "
                "from people already searching for your service in your area. "
                "That's something a generic domain can't replicate."
            ),
            (
                "The price reflects the domain's rarity and search value — "
                "not just the cost of registration. "
                "That said, if you have a figure in mind, share it. "
                "I'd rather make a deal than leave this name sitting idle."
            ),
        ],
        "closing": [
            "What would feel like a fair price to you? Reply with your best offer.",
            "Share your number and let's see if we can meet in the middle.",
            "I'm open to a reasonable offer — just let me know what works for you.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: negotiation
    # ─────────────────────────────────────────────────────────────────────────
    "negotiation": {
        "acknowledgment": [
            "Thanks for the offer — I appreciate you engaging on this.",
            "I appreciate the response and the willingness to negotiate.",
            "Good to hear from you on this.",
        ],
        "body": [
            (
                "The offer is a bit below what I'm currently looking for. "
                "I've already passed on a higher number, so I can't go lower than my floor price. "
                "Will you be able to meet me somewhere closer to the middle?"
            ),
            (
                "I can't accept that figure, but I don't want to lose this deal either. "
                "How about we split the difference? "
                "I'll hold this price for 24 hours while we sort this out — "
                "just note the domain is publicly listed and anyone can buy it."
            ),
            (
                "I hear you on price. Here's my thinking: "
                "I need to at least cover my acquisition costs, "
                "and the name has genuine commercial value beyond what I paid. "
                "Give me your absolute best offer and I'll give you a straight yes or no."
            ),
            (
                "That's lower than I can go, but I'm open to a fair deal. "
                "What's the maximum you could stretch to? "
                "Let's be straight with each other and get this done quickly."
            ),
        ],
        "closing": [
            "Reply with your best number and I'll respond immediately.",
            "Visit the listing and submit your offer there — or just reply here.",
            "Let me know and we can wrap this up as soon as possible.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: follow_up
    # ─────────────────────────────────────────────────────────────────────────
    "follow_up": {
        "acknowledgment": [
            "Just circling back on my previous email.",
            "Following up on the domain I reached out about recently.",
            "I wanted to check in — I haven't heard back yet.",
            "Quick follow-up to make sure my earlier email didn't get buried.",
        ],
        "body": [
            (
                "If you're still considering it, I'd love to hear your thoughts. "
                "And if the price was the issue, I've since made a discount — "
                "you can now get it at a reduced rate."
            ),
            (
                "I'd love to know if you're still interested or if I should move on. "
                "Just a yes or no would be very helpful so I can plan accordingly. "
                "No pressure either way."
            ),
            (
                "I've been reaching out to other businesses in your space, "
                "and I want to give you the first right of refusal before moving on. "
                "The domain is still available, but that may not be the case for long."
            ),
            (
                "I'm not trying to be a bother — I just genuinely believe this domain "
                "could add real value to your business, and I haven't been able to reach you yet. "
                "If you'd prefer I stop reaching out, just say the word."
            ),
        ],
        "closing": [
            "Interested or not — either answer works. Just let me know. Best regards.",
            "Reply with your thoughts, or visit the listing if you're ready to move forward.",
            "If you'd rather not hear from me again, a quick reply is all it takes.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: trust_issue
    # ─────────────────────────────────────────────────────────────────────────
    "trust_issue": {
        "acknowledgment": [
            "I completely understand your concern — it's a fair one.",
            "Your caution is reasonable, and I don't take it personally.",
            "Thanks for being upfront about that.",
        ],
        "body": [
            (
                "The domain is listed on Dan.com — a leading domain marketplace. "
                "You can look them up on TrustPilot right now and read real buyer reviews. "
                "Their escrow service means your payment is held until the domain is transferred to you. "
                "You won't pay and get nothing — the whole process is automated and protected."
            ),
            (
                "This isn't a scam. The domain is also listed on GoDaddy and Afternic "
                "— two of the most trusted names in the industry. "
                "If you'd feel more comfortable buying directly through GoDaddy, I can arrange that. "
                "Just say the word."
            ),
            (
                "I can verify ownership right now. "
                "Type the domain URL into your browser — you'll see it points to a marketplace listing. "
                "You can also look up the WHOIS record. "
                "And if you'd like, I can redirect it to your website temporarily as proof I own it."
            ),
            (
                "The entire transaction is handled by a third-party escrow service — "
                "I never receive payment until you confirm the domain has been transferred to your account. "
                "There is no risk to you. "
                "The process is smooth and typically completes within a few hours."
            ),
        ],
        "closing": [
            "Let me know which platform you'd prefer — Dan.com, GoDaddy, or Epik — and I'll set it up.",
            "Search 'Dan.com Trustpilot' to see what real buyers say. Then let's proceed.",
            "Any remaining questions? I'm happy to walk you through the entire process step by step.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: have_website
    # ─────────────────────────────────────────────────────────────────────────
    "have_website": {
        "acknowledgment": [
            "That's great — and it's actually exactly why I reached out to you.",
            "Having an existing website is a plus, not a barrier.",
            "Good — that makes this even simpler.",
        ],
        "body": [
            (
                "You don't need to build anything new. "
                "All you'd do is redirect this domain to your current website. "
                "When anyone types it into their browser, it sends them straight to your existing site. "
                "It's a one-click setup in your domain admin panel — no coding needed."
            ),
            (
                "Think of it as a second front door for your business. "
                "Your current site stays exactly as it is — this domain just sends more people there. "
                "Specifically, the people already searching for your service in your city "
                "who don't know your current URL."
            ),
            (
                "You already have the infrastructure in place. "
                "This domain simply adds another channel that feeds traffic directly to what you've built. "
                "And if a competitor picks it up first, their traffic grows at your expense."
            ),
            (
                "Owning multiple relevant domains is how established businesses defend their online territory. "
                "You redirect this to your site in minutes and immediately capture type-in traffic "
                "from customers who would otherwise land on a competitor's page."
            ),
        ],
        "closing": [
            "The only risk is someone else grabbing it first. Visit the listing to secure it now.",
            "I can even do the redirect for you for free once you've purchased. Let me know.",
            "Reply with any questions — or visit the listing to get it immediately.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: rank_well
    # ─────────────────────────────────────────────────────────────────────────
    "rank_well": {
        "acknowledgment": [
            "That's genuinely impressive — ranking well takes real effort.",
            "Good to hear your SEO is solid.",
            "That's great work — strong rankings are hard to maintain.",
        ],
        "body": [
            (
                "The concern isn't your current ranking — it's what happens if a competitor buys this domain. "
                "An exact-match geo domain in the hands of a rival will put them alongside you in the results "
                "almost immediately. This domain is a defensive buy as much as an offensive one."
            ),
            (
                "Beyond SEO, there's the branding angle. "
                "A geo-keyword domain instantly communicates authority to anyone who sees it — "
                "it signals that you are the go-to business for this service in this city. "
                "That's not something your current domain can replicate."
            ),
            (
                "You've built the ranking — now protect it. "
                "Owning the exact keyword domain ensures no competitor can use it against you. "
                "It also captures direct type-in traffic from users who go straight to the URL "
                "without searching at all."
            ),
        ],
        "closing": [
            "Price is negotiable. Let me know if you'd like to make an offer.",
            "Interested? Visit the listing or reply with an offer.",
            "Happy to answer any questions about how this would work alongside your existing setup.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: how_it_works
    # ─────────────────────────────────────────────────────────────────────────
    "how_it_works": {
        "acknowledgment": [
            "Great question — let me break it down simply.",
            "Happy to explain exactly how this works.",
            "No problem — here's a plain-language explanation.",
        ],
        "body": [
            (
                "Once you purchase the domain, you redirect it to your current website URL. "
                "You do this in your domain admin panel — find the 'Forward this domain' option, "
                "paste your current website address, and save. "
                "From that point on, anyone who types or clicks the domain lands on your site. "
                "No coding or technical skills needed."
            ),
            (
                "The redirect is a simple forwarding rule — like a postal redirect for your mail. "
                "You input your current website URL in the domain settings, "
                "and all traffic to the new domain is automatically sent to your existing site. "
                "I can also do this step for you if you prefer."
            ),
            (
                "The purchase is handled by an escrow marketplace — Dan.com or GoDaddy. "
                "You click Buy Now, make payment, and the domain is transferred to your account "
                "within a few hours. "
                "Then you add a redirect to your existing website in one minute. "
                "That's the entire process."
            ),
            (
                "Here's the step-by-step: "
                "1. Go to the listing. "
                "2. Click Buy Now and complete payment. "
                "3. The domain transfers to your account within 12 hours. "
                "4. In your domain admin, set it to forward to your current website URL. "
                "Done. I can guide you through any of these steps."
            ),
        ],
        "closing": [
            "If you need me to walk you through it personally, just ask. Here's the listing link.",
            "Any other questions? I'm here to help every step of the way.",
            "Ready to proceed? Visit the listing for immediate ownership.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: why_buy
    # ─────────────────────────────────────────────────────────────────────────
    "why_buy": {
        "acknowledgment": [
            "Great question — here's the honest answer.",
            "Fair question. Let me give you the real case for this.",
            "I'm glad you asked — here's why this matters for your specific business.",
        ],
        "body": [
            (
                "Every day, people in your city type keywords like this domain directly into Google "
                "looking for a business exactly like yours. "
                "Right now, those searches don't lead to you. "
                "With this domain redirected to your site, they do."
            ),
            (
                "There are three things this domain does for you: "
                "it captures type-in traffic from people who never knew your URL, "
                "it signals geo-authority on search engines, "
                "and it stops a competitor from owning it and using it against you. "
                "Any one of those alone justifies the cost."
            ),
            (
                "The domain contains the exact keywords your customers type when they're ready to buy. "
                "According to SEO research, the number one result on Google gets 30–40% of all clicks. "
                "This domain can help put you there — faster than building organic authority from scratch."
            ),
            (
                "Benefits at a glance: guaranteed direct traffic, improved search rankings, "
                "reduced ad spend, stronger geo-authority, and competitor lockout. "
                "All from a one-time purchase and a 60-second redirect setup."
            ),
        ],
        "closing": [
            "Does that answer your question? Happy to go deeper on any of these points.",
            "If this resonates, visit the listing to get it now — or reply with any follow-up questions.",
            "Want me to explain any of these benefits in more detail? Just ask.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: not_now
    # ─────────────────────────────────────────────────────────────────────────
    "not_now": {
        "acknowledgment": [
            "Understood — timing is everything in business.",
            "That's completely fair — I won't push.",
            "I hear you. No pressure from my side.",
        ],
        "body": [
            (
                "When would be a better time to revisit this? "
                "I'm happy to check back in a month or two — "
                "just be aware the domain is publicly listed and could be gone by then."
            ),
            (
                "Noted. I'll follow up in a couple of months. "
                "Just a heads-up though — I am reaching out to other businesses in your space, "
                "so I can't guarantee the name will still be available when you're ready."
            ),
            (
                "Shall I check back in 30 days? "
                "I don't want to keep emailing if it's genuinely not the right time, "
                "but I also don't want you to miss out if you do decide you want it later."
            ),
        ],
        "closing": [
            "Let me know a good time to circle back. Best regards.",
            "I'll make a note and reach out again — unless you'd prefer I don't. Just say the word.",
            "Take care, and feel free to reach out whenever the time is right.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: partner
    # ─────────────────────────────────────────────────────────────────────────
    "partner": {
        "acknowledgment": [
            "Of course — this kind of decision often involves more than one person.",
            "That makes complete sense. Take the time you need.",
            "No problem at all.",
        ],
        "body": [
            (
                "Please do loop them in. "
                "In the meantime — is there any additional information I can provide "
                "that would make the conversation with your partner easier? "
                "I'm happy to put together a brief case for the domain's value."
            ),
            (
                "Just to help the internal conversation: the key points are "
                "the direct traffic the domain already receives, the SEO keyword value, "
                "and the competitor risk if someone else acquires it first. "
                "Price is also negotiable, which gives you flexibility."
            ),
            (
                "Any idea on a rough timeline? "
                "I'm asking only because the domain is listed publicly "
                "and I wouldn't want the decision to be made for you by another buyer."
            ),
        ],
        "closing": [
            "Do let me know once you've had a chance to discuss. Best regards.",
            "Happy to answer any questions your partner might have — just forward this email.",
            "Looking forward to hearing from you both.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: agreed_no_pay
    # ─────────────────────────────────────────────────────────────────────────
    "agreed_no_pay": {
        "acknowledgment": [
            "Just following up on our previous conversation.",
            "I wanted to check in — I thought we had an agreement in place.",
            "Circling back to make sure everything is still on track.",
        ],
        "body": [
            (
                "I haven't heard from you since we agreed on the price, "
                "and I want to make sure you haven't run into any issues. "
                "Are you experiencing any difficulties with the payment or the marketplace? "
                "I'm happy to walk you through it."
            ),
            (
                "Just a reminder that this domain is publicly listed — "
                "another buyer could come in at any time. "
                "I'd hate for you to lose the name we shook hands on. "
                "Visit the listing to complete the purchase at the price we agreed."
            ),
            (
                "I've set the listing to reflect our agreed price. "
                "Click Buy Now, follow the payment steps, "
                "and the domain will be transferred to your account within a few hours. "
                "Let me know if you hit any snags."
            ),
        ],
        "closing": [
            "Here's the listing link. Let's get this done. Best regards.",
            "Let me know if there's anything holding you up — I'm here to help.",
            "Looking forward to completing this transaction.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: payment_issue
    # ─────────────────────────────────────────────────────────────────────────
    "payment_issue": {
        "acknowledgment": [
            "Sorry to hear you're having trouble — let me sort this out for you.",
            "That shouldn't happen — let's fix it right away.",
            "Thanks for letting me know. Here's what we can do.",
        ],
        "body": [
            (
                "If the Dan.com link isn't working, I can switch to Epik or GoDaddy — "
                "both are fully trusted marketplaces with escrow services. "
                "Just let me know which you'd prefer and I'll send a new link immediately."
            ),
            (
                "Payment issues on marketplace platforms are rare but do happen. "
                "The quickest fix is to try an alternative listing — "
                "I have the domain on both Afternic (GoDaddy) and Epik as backup. "
                "Which would be easier for you?"
            ),
            (
                "Let's not let a technical issue lose this deal. "
                "I'll list it on a different platform and send you a fresh buy link. "
                "In the meantime, are you able to give me any more detail on the error you saw?"
            ),
        ],
        "closing": [
            "Reply and I'll send you the alternative link within the hour.",
            "Let me know which platform works best and I'll set it up right away.",
            "We'll get this resolved — just say the word.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: angry
    # ─────────────────────────────────────────────────────────────────────────
    "angry": {
        "acknowledgment": [
            "I sincerely apologize for the inconvenience.",
            "I'm sorry — that was not my intention at all.",
            "I hear you, and I'm sorry for the disruption.",
        ],
        "body": [
            "You will not receive any further emails from me.",
            "I'll remove you from my list immediately and won't contact you again.",
            "Consider this the last email you'll receive from me on this matter.",
        ],
        "closing": [
            "I wish you all the best.",
            "Take care.",
            "Thank you for the feedback — I'll take it on board.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: expired_owner
    # ─────────────────────────────────────────────────────────────────────────
    "expired_owner": {
        "acknowledgment": [
            "Thank you for reaching out — that's actually a really common situation.",
            "I completely understand why that's surprising.",
            "Thanks for explaining — let me give you some context.",
        ],
        "body": [
            (
                "Every domain has a one-year registration cycle. "
                "If it isn't renewed before the expiry date, it passes through a grace period "
                "and then becomes publicly available again on the open market. "
                "That's exactly how I acquired this one — through a legitimate expired auction."
            ),
            (
                "Once a domain expires and enters the open market, anyone can register it legally. "
                "I purchased it through GoDaddy's expired auctions before anyone else did. "
                "The traffic I've been receiving since then is likely from people still looking for your old site."
            ),
            (
                "This is actually good news for you — the domain is still closely associated with your business, "
                "and I'm willing to sell it back at a fair price. "
                "It will recover your lost traffic and restore your online footprint."
            ),
        ],
        "closing": [
            "Feel free to make me an offer — I'm sure we can find a price that works for both of us.",
            "Visit the listing or reply with your best offer. I'd prefer this goes back to you.",
            "I'm open to negotiation. Let me know what feels fair and we'll go from there.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: extension (owns .net or .org)
    # ─────────────────────────────────────────────────────────────────────────
    "extension": {
        "acknowledgment": [
            "That's a fair point — you already have a presence online.",
            "Good to know you already have the other extension.",
            "Understood — having a variant already is a start.",
        ],
        "body": [
            (
                "Here's the challenge: when people see your .net in marketing materials, "
                "the vast majority will type .com when they go to look you up. "
                "The more you promote the .net, the more free traffic the .com collects. "
                "Right now, that traffic goes nowhere useful — or worse, to a competitor."
            ),
            (
                ".com is the default extension in virtually every user's mind. "
                "Owning the .com version of your domain means you capture "
                "every user who types the .com out of habit. "
                "You can forward the .com directly to your .net in minutes."
            ),
            (
                "Since I've owned this domain, it's been receiving 2–5 direct visitors per day. "
                "That's real traffic from people looking for you, ending up nowhere. "
                "Redirecting the .com to your .net fixes that instantly."
            ),
        ],
        "closing": [
            "I'll lower the asking price if you're interested — just let me know.",
            "Reply with an offer. It makes sense to own both.",
            "Visit the listing or make an offer. The .com is worth protecting.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: cold_outreach (initial pitch)
    # ─────────────────────────────────────────────────────────────────────────
    "cold_outreach": {
        "acknowledgment": [
            "I'm reaching out because this domain is closely related to your business.",
            "I came across your business while researching potential buyers for a domain I own.",
            "I own a domain name that I believe could add real value to your online presence.",
        ],
        "body": [
            (
                "This domain is an exact keyword match for your service and city. "
                "When someone in your area searches for what you offer, "
                "owning this domain could put your website at the top of the results. "
                "You can redirect it to your current site — no new website needed."
            ),
            (
                "I've listed this domain for sale at a domain marketplace "
                "and I'm currently reaching out to businesses that are the best fit. "
                "Anytime someone types the related keywords into their browser, "
                "owning this domain could make your site the first they land on."
            ),
            (
                "GEO domains like this one help with brand recognition in your local market "
                "and signal authority to search engines. "
                "Your prospects are far more likely to remember a geo-targeted keyword domain "
                "than a generic brand name URL. "
                "It works alongside your current website — not instead of it."
            ),
            (
                "This is a first-come-first-served opportunity — "
                "I'm contacting several businesses in your niche today. "
                "The domain is priced competitively, and offers are welcome. "
                "Let me know if you'd like more information or if you're ready to proceed."
            ),
        ],
        "closing": [
            "Visit the listing to purchase or make an offer. Let me know if you have any questions.",
            "Reply with any questions, or head to the listing for immediate ownership.",
            "I look forward to your response. Best regards.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: sales_pitch  [SITUATION MODE]
    # First contact / proactive outreach — no prior message from prospect
    # ─────────────────────────────────────────────────────────────────────────
    "sales_pitch": {
        "acknowledgment": [
            "I'm reaching out because this domain is closely related to your business.",
            "I came across your business while researching potential buyers for a domain I own.",
            "I own a domain that could add real value to your online presence.",
        ],
        "body": [
            (
                "This domain is an exact keyword match for your service and city. "
                "Every time someone in your area searches for what you offer, "
                "owning this domain puts your website in front of them — "
                "and you can redirect it to your current site in minutes, no new website needed."
            ),
            (
                "Geo-targeted domains like this one do three things for your business: "
                "they capture direct type-in traffic from people who never knew your URL, "
                "they signal local authority to search engines, "
                "and they stop a competitor from owning the name and using it against you. "
                "Any one of those is worth the cost alone."
            ),
            (
                "I'm contacting businesses in your niche because this domain is the best fit for someone "
                "already operating in this city and service area. "
                "The asking price is competitive, offers are welcome, "
                "and the whole transaction is handled by a secure escrow marketplace."
            ),
            (
                "The domain contains the exact keywords your customers type when they're ready to buy. "
                "Redirecting it to your existing website is a one-minute setup — "
                "no coding, no rebuilding, no disruption to what you already have. "
                "It simply sends more of the right people to your door."
            ),
        ],
        "closing": [
            "Visit the listing to purchase or make an offer. Happy to answer any questions.",
            "Reply with any questions, or head to the listing for immediate ownership.",
            "Interested? Let me know and I'll walk you through the process. Best regards.",
            "This is first-come-first-served — I'm reaching out to a few businesses today. Let me know.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: re_engagement  [SITUATION MODE]
    # Prospect went cold / long gap / dormant lead
    # ─────────────────────────────────────────────────────────────────────────
    "re_engagement": {
        "acknowledgment": [
            "It's been a while since we last spoke — I wanted to reach back out.",
            "I know some time has passed since we were last in touch.",
            "Coming back to you on this after some time — hope things are going well.",
        ],
        "body": [
            (
                "The domain is still available, and I wanted to give you the first opportunity "
                "before I reach out to other businesses in your space. "
                "If the timing or price was the issue before, I'm open to a fresh conversation — "
                "just let me know where things stand."
            ),
            (
                "A lot can change in a few months — and the opportunity here hasn't. "
                "This domain is still sitting unclaimed, and it's still the best match "
                "for a business like yours in this area. "
                "If you're in a better position now to move forward, I'd love to hear from you."
            ),
            (
                "I haven't moved this to anyone else yet — and I wanted to check in "
                "before I do. If you're still interested at all, even at a different price, "
                "just reply and we can pick up where we left off."
            ),
            (
                "In case things have shifted since we last spoke: "
                "the domain can be redirected to your current website in minutes, "
                "the transaction is fully protected by escrow, "
                "and the price is still negotiable. "
                "I'd rather it go to you than to a competitor."
            ),
        ],
        "closing": [
            "Just a yes or no helps — either way I'll respect your decision. Best regards.",
            "If the timing still isn't right, no problem — just let me know. Kind regards.",
            "Reply whenever you're ready. The door is still open.",
            "Visit the listing if you'd like to move forward, or just drop me a reply.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: objection_handling  [SITUATION MODE]
    # Prospect is hesitant, unsure, or on the fence
    # ─────────────────────────────────────────────────────────────────────────
    "objection_handling": {
        "acknowledgment": [
            "I hear you — and I want to make sure I address your concern properly.",
            "That's a fair point, and I appreciate you being upfront about it.",
            "I understand the hesitation — let me try to address it directly.",
        ],
        "body": [
            (
                "The most common concern I hear is price — and it's a fair one. "
                "What I'd say is this: a domain registration costs $10 because it has no history, "
                "no keyword value, and no existing traffic. "
                "This domain has all three. That's what the premium reflects. "
                "That said, if you have a number in mind, share it — I'd rather make a deal."
            ),
            (
                "If the hesitation is about whether this will actually work for your business: "
                "the redirect takes two minutes to set up, "
                "you keep your existing website exactly as it is, "
                "and any visitor who types this domain lands on your current site automatically. "
                "There's genuinely nothing to lose in trying it."
            ),
            (
                "If you're unsure whether you need it — consider the alternative. "
                "If a competitor buys this domain, their site gets the traffic you're missing out on. "
                "That's not a scare tactic — it's just how exact-match geo domains work in local search. "
                "Owning it costs far less than losing that traffic permanently."
            ),
            (
                "Whatever the hesitation is, I'd rather you tell me than stay on the fence. "
                "Is it price? Timing? Not sure how it works? "
                "Reply with your actual concern and I'll give you a straight answer — "
                "no pressure, no pitch, just an honest response."
            ),
        ],
        "closing": [
            "What's the specific concern? Reply and I'll address it directly. Best regards.",
            "Tell me what's holding you back and we'll work through it together.",
            "Happy to answer any questions — just ask. Kind regards.",
            "No pressure at all — just let me know what would make this easier for you.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: not_interested_ask_why (specific sub-intent)
    # ─────────────────────────────────────────────────────────────────────────
    "not_interested_ask_why": {
        "acknowledgment": [
            "Thanks for the feedback.",
            "I appreciate you taking the time to respond.",
        ],
        "body": [
            (
                "May I ask why you're not interested — if you don't mind sharing? "
                "I ask to understand, not to pressure you. "
                "If it's about price, any offer is welcome. "
                "If it's something else, your feedback genuinely helps me."
            ),
            (
                "Could you share what's holding you back? "
                "Is it the price, the timing, or something about the domain itself? "
                "Knowing this helps me either address your concern or respect your decision fully."
            ),
        ],
        "closing": [
            "Looking forward to a reply from you, even if it's to confirm you're not interested.",
            "Whatever your reason — I appreciate the honesty. Best regards.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: general / fallback
    # ─────────────────────────────────────────────────────────────────────────
    "general": {
        "acknowledgment": [
            "Thanks for your message.",
            "Thank you for getting in touch.",
            "Appreciate you reaching out.",
        ],
        "body": [
            (
                "The domain is currently listed for sale at a domain marketplace. "
                "It can be redirected to your current website to boost your online visibility "
                "and capture more targeted traffic in your local area. "
                "Offers are welcome — I'm open to negotiation."
            ),
            (
                "This is a premium geo-targeted .com domain with strong keyword value. "
                "It's listed on Dan.com with full escrow protection, "
                "so the purchase process is safe and straightforward. "
                "Let me know if you'd like more information."
            ),
        ],
        "closing": [
            "Visit the listing or reply with any questions. Best regards.",
            "Feel free to ask anything — I'm happy to help. Kind regards.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: post_purchase  [NEW]
    # "I already paid — where is my domain?" / "How long does transfer take?"
    # ─────────────────────────────────────────────────────────────────────────
    "post_purchase": {
        "acknowledgment": [
            "Thank you for completing the purchase — I really appreciate it.",
            "Great — payment confirmed. Here's what happens next.",
            "Thanks for going ahead with this.",
        ],
        "body": [
            (
                "The domain transfer is handled automatically by the marketplace. "
                "You should receive a confirmation email from them shortly — "
                "please also check your spam folder just in case. "
                "The full transfer to your account typically completes within a few hours, "
                "though it can occasionally take up to 24 hours."
            ),
            (
                "Once payment is confirmed by the escrow service, "
                "the domain is released to your account at the registrar of your choice. "
                "This usually takes between 2 and 12 hours. "
                "If you haven't received a transfer email within 24 hours, "
                "contact Dan.com's support directly — they're very responsive."
            ),
            (
                "The transfer process is: payment confirmed → domain unlocked → "
                "transfer email sent to you → you accept → domain in your account. "
                "Most buyers complete this within a few hours. "
                "Let me know if anything is unclear and I'll guide you through it."
            ),
        ],
        "closing": [
            "Welcome — and let me know once the domain is in your account. Best regards.",
            "Reach out if anything doesn't arrive within 24 hours. Happy to help.",
            "Congratulations on the acquisition. Feel free to ask if you need help with the redirect setup.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: refund  [NEW]
    # "I want a refund" / "I changed my mind after paying"
    # ─────────────────────────────────────────────────────────────────────────
    "refund": {
        "acknowledgment": [
            "I'm sorry to hear the purchase didn't work out.",
            "Thank you for letting me know — I want to make this right.",
            "I understand, and I appreciate you reaching out directly.",
        ],
        "body": [
            (
                "Refund requests are handled by the marketplace escrow service — "
                "Dan.com or GoDaddy — depending on where you purchased. "
                "Please contact their support team directly and reference your order number. "
                "They have a clear buyer protection policy and will guide you through the process."
            ),
            (
                "Since the transaction was handled by a third-party escrow service, "
                "the refund process goes through them, not through me. "
                "Contact the marketplace's support team with your transaction ID "
                "and they will be able to assist you promptly."
            ),
            (
                "I want to understand what went wrong before we get to that point — "
                "is there something about the domain or the process that didn't meet your expectations? "
                "If the issue is something I can resolve on my end, I'd like to try first."
            ),
        ],
        "closing": [
            "Let me know if there's anything I can do to help on my side. Best regards.",
            "I hope we can resolve this quickly — please don't hesitate to follow up.",
            "I'm sorry for the inconvenience and want to make sure you're looked after.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: payment_method  [NEW]
    # "Can I pay by crypto / PayPal / bank transfer?"
    # ─────────────────────────────────────────────────────────────────────────
    "payment_method": {
        "acknowledgment": [
            "Good question — let me clarify what payment options are available.",
            "Happy to explain how payment works.",
            "No problem — here are your options.",
        ],
        "body": [
            (
                "The marketplace handles all payment processing securely. "
                "Dan.com accepts major credit and debit cards, bank transfer, and some crypto options. "
                "GoDaddy and Afternic also support standard card payments. "
                "Visit the listing and click Buy Now to see the full list of accepted methods at checkout."
            ),
            (
                "Payment is processed through the escrow marketplace, not directly by me — "
                "which is actually better for you, as your funds are protected until the domain is delivered. "
                "The marketplace typically accepts credit card, bank wire, and in some cases PayPal or crypto. "
                "Check the listing for the exact options at checkout."
            ),
            (
                "If you have a preferred payment method, let me know and I'll confirm whether "
                "it's supported before you proceed. "
                "I can also list the domain on a different platform if one suits your payment method better — "
                "Dan.com, Epik, and GoDaddy each have slightly different options."
            ),
        ],
        "closing": [
            "Visit the listing to see the checkout options, or reply with your preference. Best regards.",
            "Let me know which method you'd like to use and I'll confirm it's available.",
            "Happy to switch platforms if needed — just let me know.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: renewal_fees  [NEW]
    # "What are the ongoing fees after I buy?" / "Annual renewal cost?"
    # ─────────────────────────────────────────────────────────────────────────
    "renewal_fees": {
        "acknowledgment": [
            "Great question — there's only one ongoing cost to be aware of.",
            "Simple answer: the purchase is a one-time fee.",
            "Good to ask — no surprises after purchase.",
        ],
        "body": [
            (
                "The purchase price is a one-time payment — you own the domain outright after that. "
                "The only recurring cost is the annual renewal fee, which is typically $8–$12 per year "
                "depending on your registrar. "
                "This is the same renewal cost as any standard .com domain."
            ),
            (
                "Once you own it, you simply renew it each year like any other domain — "
                "usually $8 to $10 annually at registrars like GoDaddy, Namecheap, or Cloudflare. "
                "You can also set it to auto-renew so you never accidentally lose it."
            ),
            (
                "There are no hidden fees. "
                "You pay once to acquire the domain, then a small annual renewal (under $12) "
                "to keep it registered in your name. "
                "That's it — no platform fees, no commission, no monthly charges."
            ),
        ],
        "closing": [
            "Any other questions? Happy to help. Here's the listing link when you're ready.",
            "Simple and transparent — let me know if you'd like to proceed.",
            "Let me know if you have any other questions before buying.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: domain_metrics  [NEW]
    # "What is the DA / DR / authority / traffic of this domain?"
    # ─────────────────────────────────────────────────────────────────────────
    "domain_metrics": {
        "acknowledgment": [
            "Good question — let me give you the honest picture.",
            "Happy to share what I know about the domain's metrics.",
            "Fair thing to ask before buying — here's the data.",
        ],
        "body": [
            (
                "The domain's value comes primarily from its keyword composition and geo-targeting — "
                "not historical traffic data, since it was previously unregistered or expired. "
                "You can check its history on archive.org, "
                "and run a WHOIS lookup to confirm ownership details. "
                "GoDaddy's own appraisal tool also values it well above the asking price."
            ),
            (
                "You can verify the domain's stats yourself — I'd encourage it. "
                "Check archive.org for its history, use a tool like Moz or Ahrefs for authority scores, "
                "and run Google's keyword planner for monthly search volume on the exact match keywords. "
                "The numbers back up why this domain has real commercial value."
            ),
            (
                "The core value here isn't domain authority built up over years — "
                "it's the exact-match keyword advantage. "
                "Search engines treat exact-match geo domains as highly relevant for local searches, "
                "even without a history of backlinks. "
                "That's why these domains command a premium."
            ),
        ],
        "closing": [
            "Happy to provide any other details I have. Here's the listing link.",
            "Let me know if you'd like me to pull any specific data points.",
            "Any other questions before you decide? I'm happy to be transparent.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: identity  [NEW]
    # "Who are you?" / "What company is this?" / "Why are you contacting me?"
    # ─────────────────────────────────────────────────────────────────────────
    "identity": {
        "acknowledgment": [
            "Good question — let me introduce myself properly.",
            "Happy to explain exactly who I am and why I reached out.",
            "Fair enough — here's some context.",
        ],
        "body": [
            (
                "I'm a domain name investor — I purchase expired or undervalued domain names "
                "and sell them to businesses that can benefit from owning them. "
                "I reached out to you specifically because this domain closely matches "
                "your business's service and location, making you one of the most relevant potential buyers."
            ),
            (
                "I specialise in geo-targeted .com domain names — "
                "domains that contain the exact keywords people type when searching for a local service. "
                "I acquired this domain through a legitimate expired domain auction, "
                "and I'm now offering it to businesses in your industry before listing it more broadly."
            ),
            (
                "Think of me as a domain broker. "
                "I source valuable domains and connect them with businesses that can actually use them. "
                "All transactions are handled through trusted third-party escrow marketplaces "
                "like Dan.com and GoDaddy — I never handle payment directly."
            ),
        ],
        "closing": [
            "Happy to answer any other questions. I'm transparent about how this works.",
            "Let me know if that helps clarify things — and whether you'd like to know more about the domain.",
            "No obligation at all — just let me know if you're interested or have more questions.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: low_budget  [NEW]
    # "I have a limited budget" / "That's too much for a small business like mine"
    # ─────────────────────────────────────────────────────────────────────────
    "low_budget": {
        "acknowledgment": [
            "I appreciate the honesty — budget is always a real consideration.",
            "That's completely understandable — I want to work with you, not against you.",
            "Thanks for being upfront about that.",
        ],
        "body": [
            (
                "I'd rather sell this domain to someone who'll genuinely use it "
                "than leave it sitting in a portfolio. "
                "Tell me what you can realistically stretch to and I'll give you a straight answer. "
                "I can't promise I'll match every number, but I'm open to a fair conversation."
            ),
            (
                "Price is negotiable — especially for a buyer who's the right fit. "
                "Make me your best offer and I'll do my best to meet you there. "
                "A smaller sale today is better than no sale at all."
            ),
            (
                "If the full asking price isn't workable right now, "
                "reply with a number that is and we'll go from there. "
                "I also don't require immediate payment — "
                "in some cases a short payment arrangement can be discussed."
            ),
        ],
        "closing": [
            "What's your best number? Let's see if we can make it work.",
            "Reply with an offer — no judgment, just a straight yes or no from me.",
            "Let me know what you're working with and we'll figure it out.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: related_domains  [NEW]
    # "Do you have other similar domains?" / "What else do you have?"
    # ─────────────────────────────────────────────────────────────────────────
    "related_domains": {
        "acknowledgment": [
            "Great question — I do have other domains in my portfolio.",
            "Happy to help — let me see what else might suit you.",
            "Yes, I have several related names — let me give you some options.",
        ],
        "body": [
            (
                "I have a portfolio of geo-targeted keyword domains across multiple cities and service types. "
                "If this specific domain isn't the right fit, let me know what city or keywords "
                "you're targeting and I'll check what else I have available. "
                "Buying more than one domain from me also opens the door to a bundle discount."
            ),
            (
                "I can look into whether I have similar domains for nearby cities or related service keywords. "
                "Just reply with the location and service type you're most interested in "
                "and I'll come back to you with options. "
                "I'm also open to discounts on multiple purchases."
            ),
            (
                "If you're interested in expanding your online presence across multiple areas, "
                "owning several geo-targeted domains is a very cost-effective strategy. "
                "Tell me your target cities or services and I'll see what I can put together for you."
            ),
        ],
        "closing": [
            "Reply with your target area and I'll check my portfolio. Best regards.",
            "Let me know what you're looking for and I'll get back to you quickly.",
            "Happy to help you find the right fit — just send me some details.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: development  [NEW]
    # "Can I build a new website on this?" / "Can I develop it instead of redirect?"
    # ─────────────────────────────────────────────────────────────────────────
    "development": {
        "acknowledgment": [
            "Absolutely — and that's actually the highest-value use of a domain like this.",
            "Yes, you have full flexibility on what to do with it.",
            "Great thinking — developing it is a powerful option.",
        ],
        "body": [
            (
                "Once you own the domain, you can do anything with it: "
                "redirect it to your current website, build a brand new site on it, "
                "or use it for a landing page targeting a specific city or service. "
                "Most buyers redirect first for a quick win, then consider development later."
            ),
            (
                "Building a dedicated website on this domain would be even more powerful than redirecting. "
                "A site with content built around these exact keywords, "
                "hosted on this domain, would rank very strongly on its own. "
                "It's a longer-term investment, but a very strong one."
            ),
            (
                "You're not limited to a redirect. "
                "Develop it as a standalone site, a micro-site for a specific service, "
                "or a city-specific landing page. "
                "The domain is yours to use however best fits your business strategy."
            ),
        ],
        "closing": [
            "Visit the listing to get started — and let me know if you have more questions.",
            "Happy to discuss the best approach for your specific situation.",
            "The domain is yours to shape. Visit the listing for immediate ownership.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: request_info  [NEW]
    # "Can you tell me more?" / "A few questions before I decide"
    # ─────────────────────────────────────────────────────────────────────────
    "request_info": {
        "acknowledgment": [
            "Before I make a decision either way, I just wanted to get a bit more detail.",
            "Thanks for reaching out — happy to answer your questions directly.",
            "Good question — let me give you exactly what you need to decide.",
        ],
        "body": [
            (
                "The domain is listed on Dan.com with full escrow protection — "
                "your payment is held until the domain is in your account. "
                "You can verify ownership right now: type it into your browser and it redirects to the marketplace listing. "
                "Run a WHOIS lookup and you'll see the registration details."
            ),
            (
                "The transfer process is fully protected. "
                "You pay through the marketplace escrow, the domain transfers to your account within a few hours, "
                "and then you redirect it to your existing website in two minutes. "
                "No payment leaves escrow until you confirm delivery."
            ),
            (
                "Happy to answer any specific questions you have — "
                "whether that's about the process, the pricing, how the redirect works, "
                "or anything else. "
                "I'd rather you have the full picture before deciding."
            ),
        ],
        "closing": [
            "What specific questions do you have? Reply and I'll answer each one directly.",
            "Happy to share the listing link or provide any details you need.",
            "Ask away — no question is too small.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: demo_offer  [NEW]
    # "Can I see what it would look like?" / offer to show a visual
    # ─────────────────────────────────────────────────────────────────────────
    "demo_offer": {
        "acknowledgment": [
            "Rather than just telling you the domain is a good fit, I'd like to show you.",
            "I can put together a quick visual so you can see exactly what this would look like for your business.",
            "Sometimes it helps to see it rather than read about it — so let me show you.",
        ],
        "body": [
            (
                "I can put together a quick mock-up — your business name, your service, your city — "
                "so you can see how it would look in a browser bar and in a Google result. "
                "It takes me about ten minutes and costs you nothing."
            ),
            (
                "If you look at it and it doesn't move the needle, no harm done. "
                "But if it does, we can talk numbers. "
                "There's no obligation — just a visual so you can judge for yourself."
            ),
            (
                "Seeing is more convincing than reading. "
                "I'll show you the domain in context — as a browser URL, as a search result — "
                "and let the domain speak for itself."
            ),
        ],
        "closing": [
            "Interested? Just say the word and I'll have it over to you today.",
            "Reply with a yes and I'll get started on it immediately.",
            "No obligation — just let me know if you'd like to see it.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: meeting_request  [NEW]
    # "Can we get on a call?" / "Five minutes this week?"
    # ─────────────────────────────────────────────────────────────────────────
    "meeting_request": {
        "acknowledgment": [
            "I'll keep this short — I think a quick conversation would be more useful than another email.",
            "I'd rather talk you through this briefly than keep exchanging messages.",
            "Some things are easier to explain in five minutes than in five emails.",
        ],
        "body": [
            (
                "Do you have five minutes this week for a quick call? I'm flexible on timing. "
                "I'll explain the value, answer your questions, "
                "and if it still doesn't make sense after that, I'll leave you alone."
            ),
            (
                "A short call lets me understand your situation better "
                "and give you a straight answer rather than a generic pitch. "
                "Five minutes is all I need — no preparation required on your end."
            ),
            (
                "I think this domain is a better fit for your business than it might appear on paper. "
                "A call is the quickest way to find out if I'm right — "
                "and if I'm wrong, you'll know immediately."
            ),
        ],
        "closing": [
            "Let me know a time that works this week.",
            "Reply with a day and time and I'll make myself available.",
            "Five minutes. That's all I'm asking for.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: price_negotiation  [NEW]
    # "Can you come down?" / "Meet me in the middle"
    # ─────────────────────────────────────────────────────────────────────────
    "price_negotiation": {
        "acknowledgment": [
            "I appreciate the offer — and I want to work something out, which is why I'm going to be straight with you.",
            "Thanks for coming back with a number. Here's where I stand.",
            "I respect the counter — let me be equally direct.",
        ],
        "body": [
            (
                "The number you sent is below what I need to make this worthwhile, "
                "but I'm not going to counter with something unreasonable. "
                "Here's my honest position: I can come down to meet you somewhere in the middle, "
                "but I can't go as low as you've suggested."
            ),
            (
                "What if we split the difference? "
                "That's the fairest way to close this without either of us feeling like we lost. "
                "Give me your absolute best number and I'll give you a straight yes or no within the hour."
            ),
            (
                "I've already passed on a higher number, so I can't go lower than my floor. "
                "But there's room between where we both are — "
                "tell me your maximum and I'll tell you if we can make it work."
            ),
        ],
        "closing": [
            "What's your best number? Send it and I'll give you a straight answer.",
            "Reply with your maximum and let's close this today.",
            "Give me a number and I'll meet you there if I can.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: competitor_comparison  [NEW]
    # "What if a competitor buys it?" / competitive risk framing
    # ─────────────────────────────────────────────────────────────────────────
    "competitor_comparison": {
        "acknowledgment": [
            "I want to ask you something directly: do you know who your closest competitor is in this city for your service?",
            "There's a competitive angle to this that I haven't raised yet — and I think it matters.",
            "Let me frame this differently, because the most important angle here isn't just about you.",
        ],
        "body": [
            (
                "I've been reaching out to businesses in your niche today, "
                "and if you don't take this domain, the next email I send goes to your competitor. "
                "This isn't a scare tactic — it's just how it works. "
                "An exact-match geo domain in a competitor's hands means their site shows up where yours doesn't."
            ),
            (
                "Owning this domain is a defensive move as much as an offensive one. "
                "It removes a weapon from your competitor's arsenal permanently. "
                "If they own it and build on it, the traffic they gain comes directly at your expense — "
                "and that's not reversible."
            ),
            (
                "The first business to claim this locks everyone else out. "
                "That advantage compounds every month. "
                "A one-time purchase eliminates a permanent competitive risk. "
                "You don't just gain traffic — you prevent someone else from using it against you."
            ),
        ],
        "closing": [
            "Visit the listing if you'd like to move on this before I contact anyone else.",
            "You know your market better than I do. Make the call that makes sense.",
            "Reply or visit the listing — your call.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: trust_building  [NEW]
    # "Let me show you how to verify everything" / building credibility
    # ─────────────────────────────────────────────────────────────────────────
    "trust_building": {
        "acknowledgment": [
            "I want to make it easy for you to check everything before you decide.",
            "I understand the hesitation — unsolicited domain emails don't exactly scream legitimacy. So let me be transparent.",
            "Let me walk you through exactly how to verify this for yourself — independently, right now.",
        ],
        "body": [
            (
                "Type the domain into your browser — it redirects to the marketplace listing, confirming I own it. "
                "Run a WHOIS lookup and you'll see the registration details. "
                "Search 'Dan.com Trustpilot' and read what real buyers say. "
                "Three steps, all public, all verifiable — no trust required, just checking."
            ),
            (
                "I'm a domain investor. I buy domains at expired auctions — the same way GoDaddy and Afternic do — "
                "and sell them to businesses that can use them. "
                "All sales go through third-party escrow. "
                "I never handle payment directly. Your money is held until the domain lands in your account."
            ),
            (
                "The domain is listed publicly on Dan.com right now. "
                "You can see it, verify it, and read thousands of Trustpilot reviews from real buyers. "
                "I'm not asking you to trust me — I'm asking you to verify for yourself. "
                "The tools to do that are right there."
            ),
        ],
        "closing": [
            "Let me know which platform you'd prefer and I'll set it up.",
            "Search 'Dan.com Trustpilot' to see what real buyers say. Then let's proceed.",
            "Once you've checked, I think you'll feel differently. Happy to hear from you then.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: feature_explanation  [NEW]
    # "What does redirecting a domain mean?" / plain-language explanation
    # ─────────────────────────────────────────────────────────────────────────
    "feature_explanation": {
        "acknowledgment": [
            "Let me explain this in plain terms — no jargon.",
            "Happy to explain exactly what this means without the technical language.",
            "I realise I've been using technical terms without explaining them — let me fix that.",
        ],
        "body": [
            (
                "When you redirect a domain, you're setting up a forwarding rule. "
                "It works exactly like a postal redirect: anyone who types or clicks that address "
                "gets sent straight to your current website. "
                "Your existing site doesn't change. You don't need to build anything new."
            ),
            (
                "The redirect takes about two minutes to set up in your domain settings. "
                "It's a text box where you paste your current website URL and click save. "
                "No developer needed, no downtime, no special knowledge required."
            ),
            (
                "Once it's set up, every person who types this domain ends up on your current site — "
                "automatically, instantly, permanently. "
                "You get all the traffic benefits of owning a premium domain "
                "without changing a single thing about your existing website."
            ),
        ],
        "closing": [
            "Any other questions about how this works? Happy to explain anything in plain language.",
            "If you'd like me to walk you through it personally after purchase, just ask — it's free.",
            "The process is genuinely straightforward. Let me know when you're ready.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: soft_pitch  [NEW]
    # Low-pressure first contact / "just wanted to let you know"
    # ─────────────────────────────────────────────────────────────────────────
    "soft_pitch": {
        "acknowledgment": [
            "I'll keep this brief — I own a domain that closely matches your business and wanted to offer it to you first.",
            "No hard sell here — I just thought this was worth a mention.",
            "I own a domain name that might be useful for your business. I'll let you decide.",
        ],
        "body": [
            (
                "It can redirect to your existing site in minutes — nothing to build, nothing to change. "
                "If the timing is wrong or the price doesn't work, just say so and that's that. "
                "No follow-ups, no pressure."
            ),
            (
                "I'm not going to tell you your business needs this. "
                "You know your situation better than I do. "
                "But it might be worth a look — and I'd rather you have it than someone who won't put it to use."
            ),
            (
                "The domain matches the exact keywords your customers search, "
                "and redirecting it is a two-minute setup. "
                "There's a version of this that works for you, and a version that doesn't. "
                "I'd like to find out which one this is."
            ),
        ],
        "closing": [
            "Happy to answer questions or share the listing link. Just reply.",
            "No pressure — just wanted you to have the option.",
            "Let me know either way — both answers are fine with me.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: value_reminder  [NEW]
    # Re-state value before a lead goes cold / "here's the full picture"
    # ─────────────────────────────────────────────────────────────────────────
    "value_reminder": {
        "acknowledgment": [
            "Before I move on, I want to make sure I've actually made the case clearly.",
            "Let me recap the value in plain terms, because I may have buried it in earlier emails.",
            "Just to make sure you have the full picture before deciding either way.",
        ],
        "body": [
            (
                "Three things this domain does that a generic domain can't: "
                "it captures local type-in traffic from people who will never know your current URL, "
                "it signals location authority to search engines without building backlinks, "
                "and it locks a competitor out of owning it permanently. "
                "Any one of those alone justifies the cost."
            ),
            (
                "Local type-in traffic. Search authority. Competitor lockout. "
                "One-time cost, permanent advantage — those three things compound every month. "
                "This isn't about features. It's about what changes for your business "
                "if someone in your market owns this instead of you."
            ),
            (
                "The value doesn't depreciate — it compounds as more people search online for local services. "
                "Redirect to your existing site in two minutes. "
                "That's the only technical step between you and all of that."
            ),
        ],
        "closing": [
            "If none of that changes your thinking, I completely respect that. But if it does, the listing is still live.",
            "Reply or visit the listing — either way, I appreciate the consideration.",
            "Just let me know either way so I can plan accordingly.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: follow_up_no_response  [NEW — splits follow_up]
    # No reply after initial email
    # ─────────────────────────────────────────────────────────────────────────
    "follow_up_no_response": {
        "acknowledgment": [
            "I sent an email last week and haven't heard back — just wanted to make sure it didn't get buried.",
            "Quick follow-up to my previous message — no pressure, just checking in.",
            "Following up in case my last email got lost in the inbox.",
        ],
        "body": [
            (
                "The domain is still available, and I'd rather give you the first opportunity "
                "before reaching out to others in your area. "
                "A yes or a no both work for me — I just need to know which direction to go."
            ),
            (
                "I know inboxes get busy. "
                "I haven't moved forward with anyone else yet — I've been waiting to hear from you first. "
                "If the price was the sticking point, I'm open to a fresh conversation."
            ),
            (
                "I'm not trying to be a nuisance — I genuinely believe this domain could add value, "
                "and I haven't been able to reach you yet. "
                "If you'd prefer I stop reaching out, just say the word."
            ),
        ],
        "closing": [
            "Just let me know where things stand.",
            "A yes or a no both work for me. Either way I'll follow your lead.",
            "No pressure at all. Thanks for your time either way.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: follow_up_after_pricing  [NEW — splits follow_up]
    # Following up after a price quote was sent
    # ─────────────────────────────────────────────────────────────────────────
    "follow_up_after_pricing": {
        "acknowledgment": [
            "I sent over pricing information a few days ago and wanted to follow up in case you had questions.",
            "Following up on the quote I sent — just wanted to make sure it landed.",
            "Checking back in after sharing the price — happy to discuss if anything gave you pause.",
        ],
        "body": [
            (
                "If the number doesn't work, I'd genuinely like to hear your counter — "
                "I'd rather make a deal than leave this name sitting unused. "
                "If the price was fine but something else gave you pause, let me know what it was."
            ),
            (
                "The price reflects the domain's keyword value, geo-targeting, and existing traffic — "
                "not just the registration cost. "
                "That said, I'd rather talk than lose the deal over a number. "
                "Share your counter and we'll see what's possible."
            ),
            (
                "Sometimes the pricing raises questions rather than answers them — "
                "happy to address any of those directly. "
                "A quick back-and-forth is all it takes to find a number that works for both of us."
            ),
        ],
        "closing": [
            "Happy to answer anything directly. No pitch, no pressure.",
            "Reply with a counter or a question — either one moves this forward.",
            "Let me know if the price works, or share your number and we'll go from there.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: follow_up_after_interest  [NEW — splits follow_up]
    # Prospect expressed interest but then went quiet
    # ─────────────────────────────────────────────────────────────────────────
    "follow_up_after_interest": {
        "acknowledgment": [
            "You mentioned you were considering this — I just wanted to check in.",
            "Following up after your last message, where you seemed interested.",
            "You expressed interest in this a while back and I haven't heard since — I'd love to know where things landed.",
        ],
        "body": [
            (
                "I haven't moved forward with anyone else yet. "
                "I've been giving you the first shot because your business is genuinely the best fit. "
                "I can't hold it indefinitely, and I'd hate for you to miss out on something you were actually interested in."
            ),
            (
                "If something came up — price, timing, a question — just tell me and we can work through it. "
                "The domain is still available at the same terms we discussed. "
                "Nothing has changed on my end."
            ),
            (
                "If the situation on your end has shifted — budget, timing, priorities — "
                "I'm open to a fresh conversation. "
                "You were already interested. "
                "The only thing standing between you and owning it is a reply."
            ),
        ],
        "closing": [
            "If you're still thinking it over, I'm happy to wait a little longer. Just let me know.",
            "Reply and we can pick up where we left off.",
            "Hope to hear from you soon — no pressure either way.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT: general_response  [NEW — alias of general, more specific label]
    # ─────────────────────────────────────────────────────────────────────────
    "general_response": {
        "acknowledgment": [
            "Thanks for getting in touch.",
            "Thank you for your message — happy to help.",
            "Appreciate you reaching out.",
        ],
        "body": [
            (
                "The domain is currently listed for sale at a competitive price on a trusted marketplace. "
                "It can be redirected to your current website in minutes — "
                "no new site needed, no technical knowledge required. "
                "All transactions are handled through escrow, so the purchase is fully protected."
            ),
            (
                "Offers are welcome. The price is competitive, and I'm open to a reasonable discussion. "
                "The domain carries strong keyword and geo value — "
                "and it can be yours in a matter of hours."
            ),
        ],
        "closing": [
            "Let me know if you have any questions or if you'd like the listing link.",
            "Visit the listing or reply here — happy to help with whatever you need.",
            "Happy to answer anything. Just ask.",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# INTENT KEYWORDS (mirrors main.py — kept in sync)
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_INTENT_KEYWORDS: dict[str, list[str]] = {
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
    # ── NEW INTENTS ────────────────────────────────────────────────────────
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
    # ── NEW EXPANDED INTENTS ───────────────────────────────────────────────
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
    # ── SITUATION-MODE INTENTS ─────────────────────────────────────────────
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

def detect_template_intent(message: str) -> str:
    """
    Detect intent from a customer message using keyword matching.
    Returns the best-match intent key, or 'general' as fallback.
    Priority order matters — more specific intents listed first.
    """
    text = message.lower().strip()

    # Priority overrides — check these first
    priority_order = [
        "angry",
        "refund",
        "post_purchase",
        "payment_issue",
        "payment_method",
        "agreed_no_pay",
        "trust_issue",
        "trust_building",
        "expired_owner",
        "price_too_high",
        "price_negotiation",
        "extension",
        "rank_well",
        "have_website",
        "partner",
        "not_now",
        "negotiation",
        "low_budget",
        "how_it_works",
        "feature_explanation",
        "development",
        "why_buy",
        "value_reminder",
        "renewal_fees",
        "domain_metrics",
        "related_domains",
        "identity",
        "competitor_comparison",
        "objection_handling",
        "re_engagement",
        "follow_up_after_interest",
        "follow_up_after_pricing",
        "follow_up_no_response",
        "follow_up",
        "not_interested_ask_why",
        "no_thanks",
        "price_inquiry",
        "meeting_request",
        "demo_offer",
        "request_info",
        "sales_pitch",
        "soft_pitch",
        "cold_outreach",
        "general_response",
    ]

    scores: dict[str, int] = {intent: 0 for intent in priority_order}

    for intent in priority_order:
        keywords = TEMPLATE_INTENT_KEYWORDS.get(intent, [])
        for kw in keywords:
            if kw in text:
                scores[intent] += 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "general"
    return best


# ─────────────────────────────────────────────────────────────────────────────
# FILLER PHRASE CLEANER (reuses logic from main.py)
# ─────────────────────────────────────────────────────────────────────────────

_FILLER = [
    r"i hope this email finds you well[,.]?",
    r"trust you are doing (great|well)[,.]?",
    r"hope you('re| are) having a great day[,.]?",
    r"i hope you are doing well[,.]?",
    r"i hope all is well[,.]?",
    r"i wanted to reach out[,.]?",
    r"please do not hesitate[,.]?",
    r"feel free to reach out[,.]?",
    r"as per my last email[,.]?",
]

def _strip_filler(text: str) -> str:
    for pattern in _FILLER:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# REPLY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_template_reply(
    customer_message: str,
    domain_name: Optional[str] = None,
    asking_price: Optional[str] = None,
    force_intent: Optional[str] = None,
    response_length: Optional[str] = "medium",
    length_instructions: Optional[str] = None,
) -> dict:
    """
    Build a complete email reply from components.

    Args:
        customer_message:    The incoming message text.
        domain_name:         Optional domain name to mention in the reply.
        asking_price:        Optional price string to inject.
        force_intent:        Override auto-detection with a specific intent.
        response_length:     'short' | 'medium' | 'long' — controls reply depth.
        length_instructions: Custom length instruction string (from UI).

    Returns:
        dict with keys: reply, detected_intent, components_used
    """
    intent = force_intent or detect_template_intent(customer_message)
    template = COMPONENTS.get(intent, COMPONENTS["general"])

    # Pick random variation for each component
    greeting       = random.choice(COMPONENTS["_greetings"])
    acknowledgment = random.choice(template["acknowledgment"])
    closing        = random.choice(template["closing"])

    # For long mode: pick 2 body sections for richer content
    body_pool = template["body"]
    if response_length == "long" and len(body_pool) >= 2:
        selected_bodies = random.sample(body_pool, 2)
        body = "\n\n".join(selected_bodies)
    else:
        body = random.choice(body_pool)

    # Inject domain name and price placeholders if provided
    def inject(text: str) -> str:
        if domain_name:
            text = text.replace("{domain}", domain_name)
            text = re.sub(r"\bthis domain\b", f"{domain_name}", text, count=1)
        if asking_price:
            text = re.sub(r"\$[xX]+|\$xxxx|\$xxx|\$xx", asking_price, text)
        return text

    # Build parts based on length mode
    if response_length == "short":
        # Short: greeting + one key sentence from body + closing
        body_sentences = re.split(r"(?<=[.!?])\s+", inject(body))
        short_body = body_sentences[0] if body_sentences else inject(body)
        parts = [greeting, inject(acknowledgment), short_body, inject(closing)]
    elif response_length == "long":
        # Long: greeting + acknowledgment + full body (2 sections) + value prop + closing
        value_prop = _pick_value_prop(intent, domain_name)
        parts = [
            greeting,
            inject(acknowledgment),
            inject(body),
            value_prop,
            inject(closing),
        ]
    else:
        # Medium (default): greeting + acknowledgment + body + closing
        parts = [
            greeting,
            inject(acknowledgment),
            inject(body),
            inject(closing),
        ]

    # Clean and join with double newlines for proper paragraph structure
    reply = "\n\n".join(p.strip() for p in parts if p.strip())
    reply = _strip_filler(reply)

    return {
        "reply":            reply,
        "detected_intent":  intent,
        "mode":             "template",
        "response_length":  response_length or "medium",
        "components_used": {
            "greeting":      greeting,
            "acknowledgment": acknowledgment,
            "body":          body,
            "closing":       closing,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# VALUE PROPOSITION PARAGRAPHS (used in long mode)
# Adds an extra persuasive paragraph tailored to the intent
# ─────────────────────────────────────────────────────────────────────────────

_VALUE_PROPS: dict[str, list[str]] = {
    "price_inquiry": [
        "To put this in perspective: comparable geo-keyword domains in this niche have sold for well above $1,000 at public auctions. "
        "What I'm asking is already below market rate — this is a wholesale price, not a retail one. "
        "If you have a counter-offer in mind, I'm open to hearing it.",
    ],
    "price_too_high": [
        "Think of it this way: if this domain sends you just one extra customer per month, "
        "it pays for itself many times over within the first year. "
        "The one-time cost is fixed — the value compounds every single month it's in your possession.",
    ],
    "negotiation": [
        "I want to be straightforward with you: I've had interest from other parties, "
        "which is why I'm keen to resolve this quickly. "
        "Give me your best number and I'll give you an honest yes or no within the hour.",
    ],
    "sales_pitch": [
        "Here's the competitive reality: I'm reaching out to several businesses in your space today. "
        "The one who secures this domain first locks out every competitor from using it. "
        "That's not a sales tactic — it's how the domain market works. "
        "First-come-first-served is the only rule that applies.",
    ],
    "objection_handling": [
        "The way I see it, the real risk isn't in buying — it's in waiting. "
        "Every day this domain sits unclaimed is a day a competitor could walk in and claim it. "
        "Once it's gone, it's gone. And at that point, the price to buy it back from them will be significantly higher.",
    ],
    "re_engagement": [
        "Markets shift, budgets change, priorities evolve — I understand that completely. "
        "What I can tell you is that this domain hasn't moved, the value hasn't diminished, "
        "and I'd still rather see it go to you than to someone who won't put it to good use. "
        "Just let me know where things stand.",
    ],
    "trust_issue": [
        "Here's how you can verify everything before spending a single penny: "
        "type the domain into your browser and you'll see it redirects to the marketplace listing. "
        "Run a WHOIS lookup and you'll see the registration details. "
        "Search 'Dan.com Trustpilot' and read thousands of verified buyer reviews. "
        "Every step of this process is transparent and reversible.",
    ],
    "follow_up": [
        "I'll be honest — I don't want to keep emailing you if it's genuinely not a fit. "
        "But I also know that good names sell quietly, often to the person who replied last. "
        "A quick yes or no is all I need to either move forward together or move on entirely.",
    ],
    "general": [
        "This is a one-time opportunity with a one-time cost. "
        "The domain is currently listed publicly and available to anyone. "
        "If this is something that makes sense for your business, the best time to act is now — "
        "not because of pressure, but because the opportunity is real and the timing is right.",
    ],
}

def _pick_value_prop(intent: str, domain_name: Optional[str] = None) -> str:
    """Pick a value proposition paragraph for long-mode emails."""
    props = _VALUE_PROPS.get(intent, _VALUE_PROPS["general"])
    text  = random.choice(props)
    if domain_name:
        text = re.sub(r"\bthis domain\b", domain_name, text, count=1)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# AI POLISH — Optional upgrade layer using Claude
# Call this AFTER build_template_reply() to improve the assembled reply.
# Requires: anthropic SDK  |  API key passed in
# ─────────────────────────────────────────────────────────────────────────────

def ai_polish_reply(
    template_reply: str,
    customer_message: str,
    intent: str,
    api_key: str,
    domain_name: Optional[str] = None,
    asking_price: Optional[str] = None,
    tone: str = "professional and persuasive",
) -> dict:
    """
    Takes a template-generated reply and asks Claude to improve it.

    What Claude does:
      - Makes the reply sound more natural and less robotic
      - Adjusts tone to match the customer's message
      - Keeps all facts and intent intact — no hallucination
      - Does NOT add new claims not already in the template

    Args:
        template_reply:   The raw reply from build_template_reply()
        customer_message: The original customer message
        intent:           Detected intent (e.g. "trust_issue")
        api_key:          Anthropic API key
        domain_name:      Optional — for context
        asking_price:     Optional — for context
        tone:             Tone instruction string

    Returns:
        dict with: polished_reply, original_template_reply, intent, mode
    """
    try:
        import anthropic
    except ImportError:
        # If anthropic isn't installed, return the template reply unchanged
        return {
            "polished_reply":          template_reply,
            "original_template_reply": template_reply,
            "detected_intent":         intent,
            "mode":                    "template_only",
            "ai_polish":               False,
            "error":                   "anthropic SDK not installed",
        }

    context_parts = []
    if domain_name:
        context_parts.append(f"Domain name: {domain_name}")
    if asking_price:
        context_parts.append(f"Asking price: {asking_price}")
    context_str = "\n".join(context_parts) if context_parts else "Not specified"

    system_prompt = (
        "You are an expert email editor for a domain name broker. "
        "Your job is to polish draft email replies so they sound natural, human, and persuasive. "
        "RULES YOU MUST FOLLOW:\n"
        "1. Do NOT change the meaning, intent, or facts in the draft.\n"
        "2. Do NOT add claims, prices, or domain details that are not already in the draft.\n"
        "3. Do NOT use filler phrases like 'I hope this finds you well' or 'going forward'.\n"
        "4. Keep it concise — do not pad or inflate the reply.\n"
        "5. Match the requested tone exactly.\n"
        "6. Output ONLY the final email reply text. No preamble, no explanation, no quotes."
    )

    user_prompt = (
        f"CUSTOMER MESSAGE:\n{customer_message}\n\n"
        f"DETECTED INTENT: {intent}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"REQUESTED TONE: {tone}\n\n"
        f"DRAFT REPLY TO POLISH:\n{template_reply}\n\n"
        "Please improve this draft reply. Make it sound natural and human. "
        "Fix any awkward phrasing, improve flow, and match the tone. "
        "Keep all facts and the overall message intact."
    )

    try:
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 700,
            messages   = [{"role": "user", "content": user_prompt}],
            system     = system_prompt,
        )
        polished = message.content[0].text.strip()
        return {
            "polished_reply":          polished,
            "original_template_reply": template_reply,
            "detected_intent":         intent,
            "mode":                    "template_plus_ai",
            "ai_polish":               True,
        }
    except Exception as e:
        # Graceful fallback — return template reply if AI call fails
        return {
            "polished_reply":          template_reply,
            "original_template_reply": template_reply,
            "detected_intent":         intent,
            "mode":                    "template_only",
            "ai_polish":               False,
            "error":                   str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — Extract reusable components from past_replies.json
# Call this to see what greetings / bodies / closings live in your data.
# Use it to grow COMPONENTS over time without manual editing.
# ─────────────────────────────────────────────────────────────────────────────

def extract_template_components(data_file: str = None) -> dict:
    """
    Reads past_replies.json and organises reply text into:
      - greetings     : opening lines
      - body_sections : middle paragraphs
      - closings      : final sign-off lines
      - stats         : counts of what was found

    Deduplicates exact matches and removes phrases that are substrings
    of longer phrases (keeps the richest version).

    Usage:
        from template_engine import extract_template_components
        components = extract_template_components()
        for g in components['greetings']:
            print(g)
    """
    from pathlib import Path as _Path

    if data_file is None:
        data_file = _Path(__file__).parent / "past_replies.json"
    else:
        data_file = _Path(data_file)

    if not data_file.exists():
        print(f"[extract_template_components] File not found: {data_file}")
        return {}

    import json as _json
    with open(data_file, "r", encoding="utf-8") as f:
        replies = _json.load(f)

    greetings:     set = set()
    body_sections: set = set()
    closings:      set = set()

    GREETING_STARTS = ("hi,", "hello,", "good day,", "hi there,", "hello there,",
                       "dear sir", "good morning,", "hey,")
    CLOSING_STARTS  = ("best regards", "kind regards", "warm regards", "regards,",
                       "thanks for your", "thank you for", "looking forward",
                       "take care", "wishing you", "cheers")

    for entry in replies:
        raw = entry.get("reply", "").strip()
        if not raw:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\n|\n", raw) if p.strip()]
        sentences  = re.split(r"(?<=[.!?])\s+", raw)

        # Greetings — first sentence starting with a greeting word
        if sentences:
            first = sentences[0].strip()
            if any(first.lower().startswith(g) for g in GREETING_STARTS):
                greetings.add(first.rstrip(".,").rstrip() + ",")

        # Closings — last paragraph or sentence starting with a closing word
        if paragraphs:
            last_para = paragraphs[-1].strip()
            if any(last_para.lower().startswith(c) for c in CLOSING_STARTS):
                closings.add(last_para)
        if len(sentences) >= 2:
            last_sent = sentences[-1].strip()
            if any(last_sent.lower().startswith(c) for c in CLOSING_STARTS):
                closings.add(last_sent)

        # Body sections — middle paragraphs only (not first, not last)
        if len(paragraphs) >= 3:
            for para in paragraphs[1:-1]:
                if len(para.split()) >= 8:
                    body_sections.add(para)
        elif len(paragraphs) == 2 and len(paragraphs[1].split()) >= 8:
            body_sections.add(paragraphs[1])

    def dedupe_subsets(phrases: set) -> list:
        """Keep only phrases that are not substrings of a longer phrase."""
        ranked = sorted(phrases, key=len, reverse=True)
        kept = []
        for phrase in ranked:
            if not any(phrase in longer for longer in kept):
                kept.append(phrase)
        return kept

    return {
        "greetings":     dedupe_subsets(greetings),
        "body_sections": dedupe_subsets(body_sections),
        "closings":      dedupe_subsets(closings),
        "total_replies": len(replies),
        "stats": {
            "greetings_found":     len(greetings),
            "body_sections_found": len(body_sections),
            "closings_found":      len(closings),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST (run: python template_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # Original intents
        ("No thanks, not interested.", None, None),
        ("How much is the domain?", "LondonPlumber.com", "$650"),
        ("This seems like a scam.", "ChicagoRoofer.com", None),
        ("We already have a website.", None, None),
        ("The price is way too high.", None, "$495"),
        ("Stop emailing me.", None, None),
        ("How does redirection work exactly?", None, None),
        ("I need to discuss with my business partner first.", None, None),
        ("We agreed on a price but I haven't paid yet.", None, "$200"),
        ("We already own the .net version.", None, None),
        # New intents
        ("I already sent payment — where is my domain?", None, None),
        ("I want a refund please.", None, None),
        ("Can I pay by crypto or bank transfer?", None, None),
        ("What is the annual renewal fee after I buy?", None, None),
        ("What is the domain authority score?", None, None),
        ("Who are you and why are you contacting me?", None, None),
        ("I have a very tight budget as a small business.", None, None),
        ("Do you have other similar domains for nearby cities?", None, None),
        ("Can I build a new website on this domain?", None, None),
    ]

    print("=" * 70)
    print("TEMPLATE ENGINE — SAMPLE OUTPUTS")
    print("=" * 70)
    for msg, domain, price in test_cases:
        result = build_template_reply(msg, domain_name=domain, asking_price=price)
        print(f"\nINTENT: {result['detected_intent']}")
        print(f"INPUT:  {msg}")
        print(f"REPLY:\n{result['reply']}")
        print("-" * 70)
