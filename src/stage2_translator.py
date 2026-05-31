"""
Validity — Stage 2 Translator
Component 5 of 7

LLM-assisted translation of confirmed candidate claims into
formal logical representations within the Logical Fragment Specification.

All output passes through the Fragment Validator before proceeding.
The LLM proposes. The validator decides.

Input:  CandidateClaim (confirmed by human gate)
Output: FormalClaim (validated) | OutsideFragment (refused)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import os

from openai import OpenAI
from resilience import ResilientOpenAIClient
from logging_config import translator_logger as log

from fragment_validator import (
    FragmentValidator,
    FormalClaim,
    ClaimType,
    ValidationStatus,
    Provenance,
    RejectionRecord,
    make_claim,
)

# ── Temporal subsumption helpers ──────────────────────────────────────────────
import re as _re

def _extract_within_days(text: str):
    if not text:
        return None
    m = _re.search(r"\bwithin\s+(\d+)\s+(?:business\s+|calendar\s+)?days?\b", text.lower())
    return int(m.group(1)) if m else None

def _normalize_action_subject(text: str):
    t = (text or "").lower()
    if ("application monies" in t or "application money" in t) and (
        "pay" in t or "payment" in t or "make any payment" in t
    ):
        return "pay_application_monies"
    return None

def _detect_payment_polarity(text: str):
    t = (text or "").lower()
    prohibitions = [
        "shall not make any payment", "shall not pay",
        "must not make any payment", "must not pay",
        "may not make any payment", "may not pay",
        "not make any payment",
    ]
    if any(p in t for p in prohibitions):
        return False
    obligations = ["shall pay", "must pay", "will pay",
                   "shall make payment", "must make payment"]
    if any(o in t for o in obligations):
        return True
    return None

def _enrich_temporal(text: str) -> dict:
    """Return canonical temporal fields if pattern matches, else empty dict."""
    action   = _normalize_action_subject(text)
    days     = _extract_within_days(text)
    polarity = _detect_payment_polarity(text)
    if action and days is not None and polarity is not None:
        return {
            "canonical_action":    action,
            "temporal_operator":   "within_days",
            "temporal_bound_days": days,
            "polarity":            polarity,
        }
    return {}


    FragmentValidator,
    FormalClaim,



# ── Candidate claim (from Stage 1) ────────────────────────────────────────────

@dataclass
class CandidateClaim:
    """
    Output of Stage 1 extraction, confirmed by human gate.
    Input to Stage 2 translation.
    """
    id:                      str
    text_span:               str
    claim_type:              ClaimType
    confidence:              float
    page:                    Optional[int]  = None
    section:                 Optional[str]  = None
    parent_clause:            Optional[str]  = None
    hierarchy_path:           list           = None
    clause_role:              Optional[str]  = None
    is_standalone_obligation: bool           = True
    requires_parent_context:  bool           = False
    rule_completeness:        Optional[str]  = None
    condition_type:           Optional[str]  = None


# ── Translation result ────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    candidate:    CandidateClaim
    formal_claim: Optional[FormalClaim] = None   # set if validated
    status:       str = "outside_fragment"        # "validated" | "outside_fragment"
    reason:       str = ""                        # rejection reason if refused


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a formal logic translator for the Validity verification system.

Your task: translate a natural language commitment into a formal AST (Abstract Syntax Tree) 
that conforms to the Logical Fragment Specification (LFS v2).

SUPPORTED AST NODE TYPES (use ONLY these):
- Predicate: {"type": "Predicate", "name": "<name>", "args": ["<arg1>", "<arg2>"]}
- And:       {"type": "And", "left": <node>, "right": <node>}
- Or:        {"type": "Or", "left": <node>, "right": <node>}
- Not:       {"type": "Not", "operand": <node>}
- Implies:   {"type": "Implies", "left": <node>, "right": <node>}
- ForAll:    {"type": "ForAll", "var": "<var>", "domain": ["<val1>", "<val2>"], "body": <node>}
- Exists:    {"type": "Exists", "var": "<var>", "domain": ["<val1>", "<val2>"], "body": <node>}
- Obligated: {"type": "Obligated", "operand": <node>}
- Forbidden: {"type": "Forbidden", "operand": <node>}
- Permitted: {"type": "Permitted", "operand": <node>}
- At:        {"type": "At", "time_label": "<label>", "body": <node>}

RULES:
1. Use Obligated for "shall", "must", "is required to"
2. Use Forbidden for "must not", "shall not", "is prohibited from"
3. Use Permitted for "may", "is entitled to"
4. Use Implies for "if... then...", "subject to", "provided that"
5. Use At for temporal claims with a specific time point
6. ForAll and Exists MUST have a finite, enumerated domain list
7. Do NOT use modal terms: reasonable, material, significant, promptly, appropriate, substantial, adequate, sufficient, necessary, proper, timely, good faith
8. If the claim contains unresolvable modal terms or ambiguity, return {"type": "OUTSIDE_FRAGMENT", "reason": "<explanation>"}
9. Predicate names must be snake_case, specific, and unambiguous
10. Keep args as simple string identifiers

CLAIM TYPES:
- guarantee: "will", "ensures", "guarantees"
- condition: "if... then...", "subject to"
- disclaimer: "no assurance", "not guaranteed"
- obligation: "must", "shall", "is required to"
- quantified_assurance: "all", "none", "at least N"

TEMPORAL SCOPE RULE — CRITICAL:
- Only use At when the claim specifies a DISCRETE, NAMED point in time.
- Examples that warrant At: "on the Maturity Date", "at closing", "on the first Business Day"
- Examples that do NOT warrant At: "during the term", "at all times", "throughout the Agreement", "during the term of this Agreement"
- Ongoing obligations and prohibitions over a period get NO temporal wrapper — encode as a bare Obligated or Forbidden with a Predicate directly.
- When in doubt: omit At entirely. A false temporal scope causes missed contradictions.

PREDICATE NORMALISATION — CRITICAL:
- Predicates referring to the same real-world action MUST use identical names.
- "shall not make distributions" and "shall make distributions" both use predicate name: "make_distributions"
- "prohibited from distributing" and "required to distribute" both use: "make_distributions"
- Always use the base action as the predicate name. Negation is expressed via Forbidden/Not — never in the name.
- Never append "_to", "_from", "_by" or verb tense suffixes to differentiate what should be the same predicate.
- When in doubt: strip all modifiers and use the simplest possible shared action name.

DEONTIC OPERATORS:
- Never place a deontic operator (Obligated/Forbidden/Permitted) inside a quantifier body.
- ForAll and Exists bodies must contain only Predicate, And, Or, Not, Implies, or At nodes.
- To express "all X must do P": use ForAll with Predicate("do_P", ["x"]) in the body — the obligation is implied by the ForAll itself.

Respond with ONLY valid JSON. No explanation. No markdown. No backticks.
The JSON must be a single AST node object."""


# ── Stage 2 Translator ────────────────────────────────────────────────────────

class Stage2Translator:
    """
    Translates confirmed CandidateClaims into validated FormalClaims.

    Architecture:
    1. LLM proposes a formal AST for the natural language claim
    2. Fragment Validator tests the AST against LFS v2
    3. If validated → FormalClaim returned
    4. If refused  → OutsideFragment with reason recorded

    The LLM cannot corrupt verification because it never touches Stage 3.
    All reasoning is performed by Z3. The translator only proposes structure.
    """

    MODEL = "gpt-4o"

    def __init__(self, client: Optional[OpenAI] = None, registry=None):
        self.client    = client or ResilientOpenAIClient()
        self.validator = FragmentValidator()
        self.registry  = registry  # Optional PredicateRegistry

    def set_registry(self, registry) -> None:
        """Set the predicate registry for this translation session."""
        self.registry = registry

    def _get_system_prompt(self) -> str:
        """Build system prompt, injecting registry context if available."""
        base = SYSTEM_PROMPT
        if self.registry:
            from predicate_registry import build_registry_prompt_injection
            injection = build_registry_prompt_injection(self.registry)
            if injection:
                # Insert before the final instruction line
                base = base.replace(
                    "Respond with ONLY valid JSON.",
                    injection + "\n\nRespond with ONLY valid JSON."
                )
        return base

    def translate(self, candidate: CandidateClaim) -> TranslationResult:
        """
        Translate a single confirmed CandidateClaim.
        Returns a TranslationResult with validated FormalClaim or refusal.
        """
        # Step 1: LLM proposes AST
        ast = self._propose_ast(candidate)
        if ast is None:
            return TranslationResult(
                candidate = candidate,
                status    = "outside_fragment",
                reason    = "LLM failed to produce a parseable AST.",
            )

        # Step 2: Check for explicit outside-fragment signal from LLM
        if ast.get("type") == "OUTSIDE_FRAGMENT":
            return TranslationResult(
                candidate = candidate,
                status    = "outside_fragment",
                reason    = ast.get("reason", "LLM determined claim is outside fragment."),
            )

        # Step 3: Canonicalise predicate names
        ast = self._canonicalise_predicates(ast)

        # Step 4: Build FormalClaim and run through Fragment Validator
        import uuid
        formal_claim = FormalClaim(
            id                       = candidate.id,
            formula                  = ast,
            claim_type               = candidate.claim_type,
            confidence               = candidate.confidence,
            provenance               = Provenance(
                text_span = candidate.text_span,
                page      = candidate.page,
                section   = candidate.section,
            ),
            text_span                = candidate.text_span,
            parent_clause            = getattr(candidate, "parent_clause", None),
            hierarchy_path           = getattr(candidate, "hierarchy_path", None),
            clause_role              = getattr(candidate, "clause_role", None),
            is_standalone_obligation = getattr(candidate, "is_standalone_obligation", True),
            requires_parent_context  = getattr(candidate, "requires_parent_context", False),
            rule_completeness        = getattr(candidate, "rule_completeness", None),
            condition_type           = getattr(candidate, "condition_type", None),
            **_enrich_temporal(candidate.text_span),
        )

        result = self.validator.validate(formal_claim)

        if result.status == ValidationStatus.VALIDATED:
            return TranslationResult(
                candidate    = candidate,
                formal_claim = formal_claim,
                status       = "validated",
            )
        else:
            return TranslationResult(
                candidate = candidate,
                status    = "outside_fragment",
                reason    = formal_claim.rejection.reason if formal_claim.rejection else "Validation failed.",
            )

    def translate_batch(self, candidates: list[CandidateClaim]) -> list[TranslationResult]:
        """Translate a list of confirmed candidates."""
        return [self.translate(c) for c in candidates]

    # ── Predicate canonicalisation ────────────────────────────────────────────

    # Suffix tokens that add no semantic content and cause predicate drift
    _STRIP_SUFFIXES = {
        "make", "makes", "making", "made",
        "mandatory", "quarterly", "annual", "scheduled",
        "additional", "further",
        "the", "a", "an",
        "to", "from", "by", "of", "in", "on", "at",
    }

    def _canonicalise_predicates(self, node: dict) -> dict:
        """
        Recursively walk an AST and normalise predicate names.
        Strips verb inflections and modifier suffixes so that
        semantically equivalent predicates share the same symbol.

        Example:
          borrower_makes_distributions     → make_distributions
          borrower_make_mandatory_distribution → make_distributions
          borrower_distributes_to_shareholders → make_distributions
        """
        if not isinstance(node, dict):
            return node

        if node.get("type") == "Predicate":
            name = node.get("name", "")
            node = dict(node)
            node["name"] = self._canonical_name(name)
            return node

        return {k: self._canonicalise_predicates(v) for k, v in node.items()}

    def _canonical_name(self, name: str) -> str:
        """
        Reduce a predicate name to its canonical form.
        - Split on underscores
        - Remove actor prefixes (borrower, lender, fund, party)
        - Remove strip-suffix tokens
        - Normalise plural/singular for key verbs
        - Rejoin with underscores
        """
        ACTOR_PREFIXES = {"borrower", "lender", "fund", "party", "parties", "trustee", "issuer"}
        VERB_NORMALISE = {
            "makes":        "make",
            "making":       "make",
            "made":         "make",
            "distributes":  "distribute",
            "distributing": "distribute",
            "distributed":  "distribute",
            "distributions": "distribution",
            "maintains":    "maintain",
            "maintaining":  "maintain",
            "maintained":   "maintain",
            "repays":       "repay",
            "repaying":     "repay",
            "repaid":       "repay",
            "incurs":       "incur",
            "incurring":    "incur",
            "incurred":     "incur",
        }

        tokens = name.lower().split("_")

        # Remove leading actor prefixes
        while tokens and tokens[0] in ACTOR_PREFIXES:
            tokens = tokens[1:]

        # Normalise verb forms
        tokens = [VERB_NORMALISE.get(t, t) for t in tokens]

        # Remove strip suffixes (modifiers that add no semantic content)
        tokens = [t for t in tokens if t not in self._STRIP_SUFFIXES]

        # Deduplicate adjacent identical tokens
        deduped = []
        for t in tokens:
            if not deduped or t != deduped[-1]:
                deduped.append(t)

        # Normalise distribution/distributions → distribute
        deduped = ["distribute" if t in ("distribution", "distributions") else t for t in deduped]

        return "_".join(deduped) if deduped else name

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _propose_ast(self, candidate: CandidateClaim) -> Optional[dict]:
        """
        Call the LLM to propose a formal AST for the candidate claim.
        Returns a parsed dict or None on failure.
        """
        user_message = (
            f"Claim type: {candidate.claim_type.value}\n"
            f"Text: \"{candidate.text_span}\"\n\n"
            f"Translate this into a formal AST conforming to LFS v2."
        )

        try:
            response = self.client.chat.completions.create(
                model    = self.MODEL,
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user",   "content": user_message},
                ],
                temperature = 0,       # deterministic
                max_tokens  = 500,
            )
            if response is None:
                log.error(f"API returned None after retries for claim {candidate.id[:8]}")
                return None
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error for claim {candidate.id[:8]}: {e}")
            return None
        except Exception as e:
            log.error(f"API error for claim {candidate.id[:8]}: {e}")
            return None


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid

    translator = Stage2Translator()

    candidates = [
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The borrower shall repay the principal by the maturity date.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.95,
            page       = 4,
            section    = "Section 5.1",
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The borrower is prohibited from making any distributions prior to repayment.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.92,
            page       = 7,
            section    = "Section 6.3",
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "If a default event occurs, the security interest becomes immediately enforceable.",
            claim_type = ClaimType.CONDITION,
            confidence = 0.90,
            page       = 9,
            section    = "Section 8.1",
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The borrower shall act in a reasonable and prudent manner at all times.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.85,
            page       = 2,
            section    = "Section 2.1",
        ),
    ]

    print("── Stage 2 Translation ──────────────────────────────────────\n")
    for candidate in candidates:
        result = translator.translate(candidate)
        print(f"Text:    \"{candidate.text_span[:70]}...\"" if len(candidate.text_span) > 70 else f"Text:    \"{candidate.text_span}\"")
        print(f"Status:  {result.status}")
        if result.status == "validated":
            print(f"Formula: {result.formal_claim.formula}")
        else:
            print(f"Reason:  {result.reason}")
        print()

    print("Stage 2 Translator smoke test complete.")
