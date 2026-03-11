"""
TAS AutoBD — Contact Quality Scorer
=====================================
Classifies and ranks extracted email addresses by decision-maker likelihood.

Philosophy
----------
Sending a personalised executive proposal to info@company.com destroys
credibility instantly. This module ensures only relevant, role-appropriate
contacts reach the sales rep — and makes it clear *who* each contact likely is.

Scoring tiers
-------------
  tier 1 — Executive / C-suite / VP / Director  → score 90-100
  tier 2 — Technical lead / IT / Engineering     → score 70-89
  tier 3 — Business / Marketing / Sales / BD     → score 50-69
  tier 4 — General / HR / Finance                → score 20-49
  tier 0 — Generic / blacklisted (filtered out)  → score 0
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ── Pattern tables ─────────────────────────────────────────────────────────────

# Local-part patterns → (role_label, score)
# Checked in order; first match wins.
_LOCAL_RULES: List[Tuple[str, str, int]] = [
    # ── Tier 0: blacklisted generics ─────────────────────────────────
    (r"^(noreply|no-reply|donotreply|do-not-reply)$",     "no-reply",       0),
    (r"^(unsubscribe|bounce|mailer-daemon|postmaster)$",  "system",         0),
    (r"^(info|information|enquir|enquiries|inquir)$",     "generic-info",   0),
    (r"^(contact|contactus|hello|hi|hey|general)$",       "generic-contact",0),
    (r"^(support|help|helpdesk|service|services)$",       "support",        0),
    (r"^(admin|administrator|webmaster|hostmaster)$",     "admin",          0),
    (r"^(mail|email|office|reception|feedback)$",         "generic",        0),
    (r"^(careers|jobs|recruitment|hr|humanresources)$",   "hr-generic",     15),

    # ── Tier 1: Executive / decision-maker ───────────────────────────
    (r"ceo|chief.?exec|president|founder|co-?founder",    "C-suite",        98),
    (r"cto|chief.?tech|chief.?digital|chief.?info",       "C-suite-Tech",   97),
    (r"coo|cfo|cmo|cpo|ciso|cdo|chief",                   "C-suite",        95),
    (r"(managing|executive).?director|md\b",              "Exec-Director",  94),
    (r"\bvp\b|vice.?president",                           "VP",             93),
    (r"svp|evp|avp",                                      "VP-Senior",      92),
    (r"(general|regional|country|group).?manager",        "GM",             90),
    (r"head.?(of|of.the)?",                               "Head-of",        88),
    (r"director",                                         "Director",       88),

    # ── Tier 2: Technical decision-influencer ─────────────────────────
    (r"cto|tech.?lead|principal.?eng|architect",          "Tech-Lead",      85),
    (r"(senior|lead|principal|staff).?(engineer|dev|developer|programmer)", "Senior-Eng", 80),
    (r"it.?(manager|head|director|lead)",                 "IT-Lead",        82),
    (r"devops|platform|infrastructure|cloud|data.?eng",   "Tech-Infra",     75),
    (r"engineer|developer|programmer|dev\b",              "Engineer",       70),

    # ── Tier 3: Business / BD / Marketing / Sales ─────────────────────
    (r"(business|corp|corporate).?dev(elopment)?|bd\b",   "BD",             68),
    (r"(head|director|vp|manager).?(sales|revenue|growth)","Sales-Leader",  67),
    (r"(sales|account).?(manager|exec|director|rep)",     "Sales",          65),
    (r"(digital|growth|product|brand).?(manager|lead)",   "Product",        63),
    (r"(marketing|communications|pr|public.?rel)",        "Marketing",      60),
    (r"procurement|purchasing|vendor|partner",            "Procurement",    58),
    (r"operations|ops\b|strategy|transformation",        "Operations",     55),
    (r"(project|program|delivery).?manager",              "PM",             52),
    (r"(consultant|advisor|specialist|analyst)",          "Analyst",        50),

    # ── Tier 4: General / Support / Finance / HR ──────────────────────
    (r"finance|financial|accounting|payroll|billing",     "Finance",        35),
    (r"legal|compliance|privacy|gdpr|security",           "Legal",          32),
    (r"hr\b|human.?res|talent|people.?ops",               "HR",             25),
    (r"press|media|journalist|newsroom",                  "Media",          22),
]

# Domain signals (bonus points if domain suggests a small/direct company)
_PERSONAL_DOMAIN_BONUS = 5   # first.last@ pattern in local part
_ROLE_PREFIX_PENALTY = -10   # e.g. "sales@" or "marketing@" (department catch-all)


@dataclass
class ScoredContact:
    email: str
    role: str
    score: int
    tier: int          # 0 = blacklisted, 1 = exec, 2 = tech, 3 = biz, 4 = general

    @property
    def is_viable(self) -> bool:
        return self.score > 0

    @property
    def tier_label(self) -> str:
        return {
            0: "Filtered",
            1: "Executive",
            2: "Technical",
            3: "Business",
            4: "General",
        }.get(self.tier, "Unknown")

    @property
    def badge(self) -> str:
        return {
            0: "🚫",
            1: "🎯",
            2: "🔧",
            3: "💼",
            4: "📋",
        }.get(self.tier, "❓")

    def __repr__(self) -> str:
        return f"ScoredContact({self.email!r}, role={self.role!r}, score={self.score}, tier={self.tier})"


def _score_local(local: str) -> Tuple[str, int]:
    """Score the local-part of an email against all rules. Returns (role, score)."""
    lc = local.lower().replace(".", "").replace("-", "").replace("_", "")
    for pattern, role, score in _LOCAL_RULES:
        if re.search(pattern, lc):
            return role, score

    # Looks like a personal email (firstname.lastname or f.lastname etc.)
    if re.match(r"^[a-z]+\.[a-z]+$", local.lower()) or re.match(r"^[a-z]\.[a-z]+$", local.lower()):
        return "Personal", 60   # personal emails are good if unclassified

    return "Unknown", 30


def _derive_tier(score: int) -> int:
    if score == 0:
        return 0
    if score >= 88:
        return 1
    if score >= 68:
        return 2
    if score >= 48:
        return 3
    return 4


def score_contacts(emails: List[str]) -> List[ScoredContact]:
    """
    Score a list of email addresses and return ScoredContact objects sorted
    by score descending (best decision-maker first).

    Blacklisted / zero-score contacts are included at the end with
    is_viable=False so the UI can display and warn about them.
    """
    seen: set = set()
    results: List[ScoredContact] = []

    for raw in emails:
        email = raw.strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)

        if "@" not in email:
            continue

        local, domain = email.rsplit("@", 1)
        role, score = _score_local(local)

        # Small bonus for personal-looking addresses (firstname.lastname@company.com)
        if "." in local and not re.search(r"\d", local):
            score = min(score + _PERSONAL_DOMAIN_BONUS, 100)

        tier = _derive_tier(score)
        results.append(ScoredContact(email=email, role=role, score=score, tier=tier))

    results.sort(key=lambda c: c.score, reverse=True)
    logger.info(
        "Contact scoring: %d total, %d viable, %d filtered",
        len(results),
        sum(1 for c in results if c.is_viable),
        sum(1 for c in results if not c.is_viable),
    )
    return results


def get_viable_emails(emails: List[str]) -> List[str]:
    """Return only viable (non-blacklisted) emails, ranked by score."""
    return [c.email for c in score_contacts(emails) if c.is_viable]


def get_best_contact(emails: List[str]) -> str | None:
    """Return the single highest-scored viable email, or None."""
    viable = [c for c in score_contacts(emails) if c.is_viable]
    return viable[0].email if viable else None
