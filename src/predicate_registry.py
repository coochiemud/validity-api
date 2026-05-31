"""
Validity — Predicate Registry
Component 9 of 9

Sits between Stage 1 (extraction) and Stage 2 (translation).
Analyses all candidate claims together and produces a canonical
predicate name map — ensuring that semantically equivalent actions
across different claims share exactly the same predicate name.

This is the correct fix for the predicate drift problem.
The LLM that extracts claims in isolation cannot know that
"shall not make distributions" and "shall make distributions"
refer to the same underlying action. The registry resolves this
by seeing all claims simultaneously.

Input:  List[CandidateClaim]
Output: PredicateRegistry (canonical name map)
        which is passed into Stage2Translator
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import os
import re

from openai import OpenAI
from resilience import ResilientOpenAIClient
from stage2_translator import CandidateClaim


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class PredicateRegistry:
    """
    Maps semantic action descriptions to canonical predicate names.
    Passed into Stage2Translator to enforce consistent naming.
    """
    # Maps a normalised action description to its canonical predicate name
    # e.g. "make distributions to shareholders" -> "distribute"
    canonical: dict[str, str] = field(default_factory=dict)

    # Maps claim ID to its canonical predicate names
    # e.g. claim_id -> ["distribute", "repay_principal"]
    claim_predicates: dict[str, list[str]] = field(default_factory=dict)

    def lookup(self, action_description: str) -> Optional[str]:
        """
        Look up the canonical predicate name for an action description.
        Returns None if not found.
        """
        key = self._normalise_key(action_description)
        return self.canonical.get(key)

    def register(self, action_description: str, canonical_name: str):
        """Register a canonical predicate name for an action."""
        key = self._normalise_key(action_description)
        self.canonical[key] = canonical_name

    def _normalise_key(self, text: str) -> str:
        """Normalise a text key for lookup."""
        return text.lower().strip()

    def to_prompt_context(self) -> str:
        """
        Format the registry as a prompt context string for Stage 2.
        Tells the translator which predicate names to use.
        """
        if not self.canonical:
            return ""
        lines = ["CANONICAL PREDICATE NAMES — use these exactly:"]
        for action, name in self.canonical.items():
            lines.append(f"  '{action}' → {name}")
        return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

REGISTRY_SYSTEM_PROMPT = """You are a semantic predicate normaliser for the Validity formal verification system.

Your task: given a list of natural language commitments from the same document, identify all distinct real-world actions and produce a canonical predicate name for each.

RULES:
1. Group semantically equivalent actions — "shall not make distributions", "shall make distributions", "is prohibited from distributing" all refer to the same action: distribute
2. Canonical names must be snake_case, verb-first, minimal: "distribute" not "make_distributions_to_shareholders"
3. Strip all modal operators (shall, must, prohibited, forbidden) — the predicate is the underlying action only
4. Strip actor names (borrower, lender, trustee) — actors go in args, not names
5. Strip modifiers (quarterly, mandatory, material) — these are context, not the action
6. Two actions are the same if they describe the same state change in the world, regardless of who performs it or under what condition

OUTPUT FORMAT:
Return ONLY a JSON object mapping action descriptions to canonical predicate names.
No explanation. No markdown. No backticks.

Example input commitments:
- "The Borrower shall not make any distributions to shareholders"
- "The Borrower shall make a mandatory quarterly distribution"
- "The Borrower shall repay the principal on the Maturity Date"
- "Repayment of the principal amount is prohibited prior to Year 3"

Example output:
{
  "make distributions to shareholders": "distribute",
  "make a mandatory quarterly distribution": "distribute",
  "repay the principal on the Maturity Date": "repay_principal",
  "repayment of the principal amount": "repay_principal"
}"""


# ── Predicate Registry Builder ────────────────────────────────────────────────

class PredicateRegistryBuilder:
    """
    Analyses all candidate claims together and builds a PredicateRegistry.

    Uses a single LLM call with all claims visible simultaneously,
    so semantically equivalent actions across different claims
    are assigned the same canonical predicate name.
    """

    MODEL = "gpt-4o"

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or ResilientOpenAIClient()

    def build(self, candidates: list[CandidateClaim]) -> PredicateRegistry:
        """
        Build a PredicateRegistry from a list of candidate claims.
        Returns a registry with canonical predicate names for all actions.
        """
        if not candidates:
            return PredicateRegistry()

        # Extract action descriptions from each claim
        action_descriptions = self._extract_actions(candidates)

        if not action_descriptions:
            return PredicateRegistry()

        # Ask LLM to normalise all actions simultaneously
        canonical_map = self._normalise_actions(action_descriptions)

        registry = PredicateRegistry()
        for action, name in canonical_map.items():
            registry.register(action, name)

        return registry

    def _extract_actions(self, candidates: list[CandidateClaim]) -> list[str]:
        """
        Extract the core action description from each claim text.
        Uses simple heuristics — strips modal markers and actors.
        """
        actions = []
        modal_markers = [
            "the borrower shall not", "the borrower shall", "the borrower must not",
            "the borrower must", "the borrower is prohibited from", "the borrower may not",
            "the lender shall", "the lender must", "the trustee shall",
            "shall not", "shall", "must not", "must", "is prohibited from",
            "is required to", "may not", "is forbidden from",
        ]

        for candidate in candidates:
            text = candidate.text_span.strip()
            text_lower = text.lower()

            # Strip leading modal markers
            action = text
            for marker in sorted(modal_markers, key=len, reverse=True):
                if text_lower.startswith(marker):
                    action = text[len(marker):].strip()
                    break

            # Strip trailing qualifiers after common boundary words
            for boundary in [" during ", " prior to ", " on the ", " at the ", " by the "]:
                idx = action.lower().find(boundary)
                if idx > 10:  # don't strip if it's near the start
                    action = action[:idx].strip()

            # Clean punctuation
            action = action.rstrip(".,;:")

            if len(action) > 5:
                actions.append(action)

        return list(set(actions))  # deduplicate

    def _normalise_actions(self, actions: list[str]) -> dict[str, str]:
        """
        Ask the LLM to produce canonical predicate names for all actions.
        Returns a dict mapping action descriptions to canonical names.
        """
        if not actions:
            return {}

        user_message = "Commitments from this document:\n" + "\n".join(
            f"- \"{action}\"" for action in actions
        )

        try:
            response = self.client.chat.completions.create(
                model       = self.MODEL,
                messages    = [
                    {"role": "system", "content": REGISTRY_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature = 0,
                max_tokens  = 1000,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            if isinstance(result, dict):
                return {k: self._clean_name(v) for k, v in result.items()}
            return {}

        except Exception as e:
            print(f"[PredicateRegistry] API error: {e}")
            return {}

    def _clean_name(self, name: str) -> str:
        """Ensure canonical name is valid snake_case."""
        name = name.lower().strip()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        return name or "action"


# ── Integration with Stage 2 Translator ──────────────────────────────────────

def build_registry_prompt_injection(registry: PredicateRegistry) -> str:
    """
    Returns a string to inject into the Stage 2 translator system prompt
    when a registry is available.
    """
    if not registry.canonical:
        return ""
    return "\n\n" + registry.to_prompt_context()


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid
    from fragment_validator import ClaimType

    builder = PredicateRegistryBuilder()

    candidates = [
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The Borrower shall not make any distributions to shareholders during the term of this Agreement.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.95,
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The Borrower shall make a mandatory quarterly distribution to shareholders.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.93,
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The Borrower shall repay the principal amount on the Maturity Date.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.97,
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The Borrower is prohibited from incurring additional Financial Indebtedness.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.94,
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "If a Payment Default occurs, the Lender may declare all amounts immediately due.",
            claim_type = ClaimType.CONDITION,
            confidence = 0.91,
        ),
    ]

    print("Building predicate registry...\n")
    registry = builder.build(candidates)

    print("Canonical predicate map:")
    for action, name in registry.canonical.items():
        print(f"  '{action}' → {name}")

    print("\nPrompt context:")
    print(registry.to_prompt_context())

    # Check that distributions maps to same predicate
    dist_1 = registry.lookup("distributions to shareholders")
    dist_2 = registry.lookup("make a mandatory quarterly distribution to shareholders")
    print(f"\nDistribution claim 1 predicate: {dist_1}")
    print(f"Distribution claim 2 predicate: {dist_2}")
    print(f"Match: {dist_1 == dist_2 and dist_1 is not None}")

    print("\nPredicate Registry smoke test complete.")
