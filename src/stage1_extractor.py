"""
Validity — Stage 1 Extractor
Component 6 of 7

LLM-assisted extraction of candidate commitments from natural language documents.
Identifies load-bearing sentences — obligations, conditions, prohibitions,
guarantees, and quantified assurances — that may bear on satisfiability.

Output is probabilistic. Extraction quality is the primary source of system risk.
All output proceeds to the human confirmation gate (Component 7) before Stage 2.

Input:  Raw document text
Output: List[CandidateClaim]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import os
import uuid

from openai import OpenAI
from resilience import ResilientOpenAIClient
from logging_config import extractor_logger as log

from fragment_validator import ClaimType
from stage2_translator import CandidateClaim


# ── System prompt ─────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a commitment extractor for the Validity formal verification system.

Your task: read a document and identify every load-bearing commitment — sentences or clauses 
that assert an obligation, prohibition, permission, condition, guarantee, or quantified assurance 
that may affect whether the document is internally consistent.

WHAT TO EXTRACT:
- Obligations:           "shall", "must", "is required to", "will"
- Prohibitions:          "shall not", "must not", "is prohibited from", "may not"
- Permissions:           "may", "is entitled to", "is permitted to"
- Conditions:            "if... then...", "subject to", "provided that", "in the event that"
- Guarantees:            "ensures", "guarantees", "warrants", "represents"
- Quantified assurances: "all", "none", "at least N", "no more than N", "every"
- Definitions:           "means", "is defined as", "refers to" (only when they create binding scope)

WHAT NOT TO EXTRACT:
- Background narrative and recitals
- Marketing language and aspirational statements
- Procedural descriptions without commitments
- Headings and labels
- Boilerplate formalities

FOR EACH COMMITMENT, identify:
- text_span: the exact verbatim text from the document (do not paraphrase)
- claim_type: one of: guarantee, condition, disclaimer, obligation, quantified_assurance
- confidence: float 0.0-1.0 (how confident you are this is a load-bearing commitment)
- page_hint: estimated page number if discernible, else null
- section_hint: section heading if discernible, else null

CONFIDENCE GUIDE:
- 0.9-1.0: Clear, unambiguous commitment using explicit markers (shall, must, prohibited)
- 0.7-0.9: Likely commitment but with some contextual dependence
- 0.5-0.7: Possible commitment, requires human judgment
- Below 0.5: Do not include

Respond with ONLY a JSON array of commitment objects. No explanation. No markdown. No backticks.

HIERARCHY EXTRACTION (critical for correctness):
Legal documents use nested clause structures. You must identify each claim's position
in that hierarchy and its functional role. This determines whether it can be tested
as a standalone obligation or only as part of a composite rule.

Format:
[
  {
    "text_span": "verbatim text from document",
    "claim_type": "obligation",
    "confidence": 0.95,
    "page_hint": 3,
    "section_hint": "Section 4.1",
    "parent_clause": "clause (xiv)",
    "hierarchy_path": ["2.5.9", "(xiv)", "(D)"],
    "clause_role": "compliance_wrapper",
    "is_standalone_obligation": false,
    "requires_parent_context": true
  }
]

FIELD DEFINITIONS:

parent_clause: The immediate parent clause label. null if top-level.

hierarchy_path: Full ancestry from outermost clause to this clause.
  e.g. ["Section 2.5", "(xiv)", "(D)"]
  For a top-level standalone obligation, include the top-level section itself.
  Do not use [] unless the clause location is unknown.

clause_role: The functional role of this clause. Must be one of:
  - "independent_obligation"  — freestanding rule, can be tested alone
  - "composite_rule"          — parent clause made of sub-conditions
  - "condition"               — activation condition ("if X then...")
  - "exception"               — carve-out from a broader rule
  - "definition"              — defines scope or meaning
  - "consequence"             — what happens when conditions are met
  - "compliance_wrapper"      — self-referential compliance clause
  - "threshold"               — numeric or quantitative limit
  - "cross_reference"         — refers to another clause for substance

is_standalone_obligation: boolean.
  true  — this claim can be tested for contradiction independent of its parent
  false — this claim is a fragment; only meaningful as part of its parent rule

requires_parent_context: boolean.
  true  — meaning changes or is incomplete without the parent clause
  false — claim is self-contained

CLASSIFICATION RULES:
- A top-level "shall" or "must" with no parent = independent_obligation, standalone=true
- A lettered sub-paragraph (A), (B), (C) within a numbered clause = condition or consequence, standalone=false
- A compliance wrapper ("shall not... unless does not violate this clause") = compliance_wrapper, standalone=false
- A numeric limit ("no more than 90 holders") = threshold; standalone depends on context
- "Provided that..." carve-outs = exception, standalone=false unless the clause still states a complete rule
- Cross-references to other sections = cross_reference, standalone=false

CRITICAL DISTINCTION:
Conditional language does NOT make a clause non-standalone.

A clause may be a standalone obligation even if it contains:
- if
- unless
- provided that
- in the event that
- subject to
- where
- when

Mark requires_parent_context=true only when the clause is legally or grammatically incomplete without its parent clause, sibling clauses, or a referenced rule.

A standalone conditional obligation contains:
- actor
- legal modality, such as shall, must, shall not, must not
- action or prohibition
- object
- trigger condition, if any
- timing or scope, if any

Additional fields:

rule_completeness: Must be one of:
  - "complete_rule" — contains a full legal rule and can be checked by the solver
  - "fragment"      — only meaningful as part of a parent composite rule
  - "ambiguous"     — uncertain

condition_type: Must be one of:
  - "internal_trigger"             — condition inside a complete standalone rule
  - "parent_activation_condition"  — condition that only operates within a parent rule
  - "exception"                    — carve-out from a broader rule
  - "scope_modifier"               — limits scope but does not create a separate rule
  - "none"

Example standalone conditional obligation:
"The Company shall pay all application monies to applicants within 5 business days of the Closing Date in the event that no Shares are issued under the Offer."

Classification:
clause_role: independent_obligation
is_standalone_obligation: true
requires_parent_context: false
rule_completeness: complete_rule
condition_type: internal_trigger

Reason:
This clause contains a complete obligation with its own trigger condition.

Example dependent fragment:
"(D) such action would not cause the Issuer to fail to comply with this clause"

Classification:
clause_role: compliance_wrapper
is_standalone_obligation: false
requires_parent_context: true
rule_completeness: fragment
condition_type: parent_activation_condition

Reason:
This is a sub-condition inside a composite parent rule. It does not state an independent obligation.

If no extractable commitments are found, return an empty array: []"""


# ── Stage 1 Extractor ─────────────────────────────────────────────────────────

class Stage1Extractor:
    """
    Extracts candidate commitments from natural language documents.

    Uses an LLM to identify load-bearing sentences — the task LLMs
    perform well. Output is probabilistic and proceeds to the human
    confirmation gate before any formal processing.

    The extractor does not validate, translate, or verify.
    It only proposes. The human gate decides what proceeds.
    """

    MODEL             = "gpt-4o"
    LOW_CONFIDENCE_THRESHOLD = 0.80  # Raised from 0.70 — reduces noise in extraction

    def __init__(self, client: Optional[OpenAI] = None, confidence_threshold: float = 0.80):
        self.client = client or ResilientOpenAIClient()
        self.LOW_CONFIDENCE_THRESHOLD = confidence_threshold

    def extract(self, document_text: str) -> list[CandidateClaim]:
        """
        Extract candidate commitments from a document.

        Splits long documents into chunks to stay within context limits.
        Returns a deduplicated list of CandidateClaim objects.
        """
        if not document_text or not document_text.strip():
            return []

        chunks = self._chunk_document(document_text)
        all_candidates = []

        for chunk in chunks:
            candidates = self._extract_from_chunk(chunk)
            all_candidates.extend(candidates)

        return self._deduplicate(all_candidates)

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_document(self, text: str, max_chars: int = 6000) -> list[str]:
        """
        Split document into chunks at section boundaries.

        Strategy:
        1. Detect section headers using common legal document patterns
        2. Group sections into chunks up to max_chars
        3. Each chunk carries its section context
        4. Fall back to paragraph splitting if no sections detected

        max_chars is doubled from 6000 to 12000 — GPT-4o handles this
        comfortably and reduces the number of cross-chunk boundary issues.
        """
        import re

        if len(text) <= max_chars:
            return [text]

        # Detect section headers — common patterns in legal/financial documents
        section_pattern = re.compile(
            r'^(?:'
            r'(?:Section|SECTION|Article|ARTICLE|Clause|CLAUSE)\s+[\d\w.]+|'  # Section 4.1
            r'(?:\d+\.)+\s+[A-Z]|'                                             # 4.1 TITLE
            r'[IVXLCDM]+\.\s+[A-Z]|'                                           # IV. TITLE
            r'(?:SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex)\s+\w+'         # Schedule A
            r')',
            re.MULTILINE,
        )

        # Split into sections
        lines     = text.split('\n')
        sections  = []
        current   = []
        current_header = ""

        for line in lines:
            if section_pattern.match(line.strip()) and current:
                sections.append((current_header, '\n'.join(current)))
                current        = [line]
                current_header = line.strip()
            else:
                current.append(line)

        if current:
            sections.append((current_header, '\n'.join(current)))

        # If no sections detected, fall back to paragraph splitting
        if len(sections) <= 1:
            return self._chunk_by_paragraphs(text, max_chars)

        # Group sections into chunks up to max_chars
        chunks      = []
        chunk_parts = []
        chunk_len   = 0

        for header, content in sections:
            section_text = f"{header}\n{content}" if header else content
            section_len  = len(section_text)

            if chunk_len + section_len > max_chars and chunk_parts:
                chunks.append('\n\n'.join(chunk_parts))
                chunk_parts = [section_text]
                chunk_len   = section_len
            else:
                chunk_parts.append(section_text)
                chunk_len += section_len

        if chunk_parts:
            chunks.append('\n\n'.join(chunk_parts))

        return chunks

    def _chunk_by_paragraphs(self, text: str, max_chars: int) -> list[str]:
        """
        Fallback chunking by paragraph boundaries.
        Used when no section structure is detected.
        """
        paragraphs = text.split("\n\n")
        chunks     = []
        current    = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > max_chars and current:
                chunks.append("\n\n".join(current))
                current     = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _extract_from_chunk(self, chunk: str) -> list[CandidateClaim]:
        """
        Call the LLM to extract commitments from a single chunk.
        Returns a list of CandidateClaim objects.
        """
        try:
            response = self.client.chat.completions.create(
                model    = self.MODEL,
                messages = [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Extract all commitments from this document:\n\n{chunk}"},
                ],
                temperature = 0,
                max_tokens  = 4000,
            )
            if response is None:
                log.error("API returned None after retries — skipping chunk")
                return []
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)
            if not isinstance(data, list):
                return []

            return [self._parse_candidate(item) for item in data if self._is_valid_item(item)]

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error in extraction: {e}")
            return []
        except Exception as e:
            log.error(f"API error in extraction: {e}")
            return []

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_candidate(self, item: dict) -> CandidateClaim:
        """Parse a raw extraction dict into a CandidateClaim."""
        claim_type_str = item.get("claim_type", "obligation").lower()
        claim_type     = self._parse_claim_type(claim_type_str)
        confidence     = float(item.get("confidence", 0.8))

        return CandidateClaim(
            id                       = str(uuid.uuid4()),
            text_span                = item["text_span"].strip(),
            claim_type               = claim_type,
            confidence               = min(max(confidence, 0.0), 1.0),
            page                     = item.get("page_hint"),
            section                  = item.get("section_hint"),
            parent_clause            = item.get("parent_clause"),
            hierarchy_path           = item.get("hierarchy_path"),
            clause_role              = item.get("clause_role"),
            is_standalone_obligation = item.get("is_standalone_obligation", True),
            requires_parent_context  = item.get("requires_parent_context", False),
            rule_completeness        = item.get("rule_completeness"),
            condition_type           = item.get("condition_type"),
        )

    def _parse_claim_type(self, raw: str) -> ClaimType:
        mapping = {
            "guarantee":            ClaimType.GUARANTEE,
            "condition":            ClaimType.CONDITION,
            "disclaimer":           ClaimType.DISCLAIMER,
            "obligation":           ClaimType.OBLIGATION,
            "quantified_assurance": ClaimType.QUANTIFIED_ASSURANCE,
        }
        return mapping.get(raw, ClaimType.OBLIGATION)

    def _is_valid_item(self, item: dict) -> bool:
        """Check that a raw extraction item has the minimum required fields."""
        if not isinstance(item, dict):
            return False
        if "text_span" not in item or not item["text_span"].strip():
            return False
        confidence = item.get("confidence", 0)
        try:
            if float(confidence) < self.LOW_CONFIDENCE_THRESHOLD:
                return False
        except (TypeError, ValueError):
            return False
        return True

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, candidates: list[CandidateClaim]) -> list[CandidateClaim]:
        """
        Remove duplicate candidates across chunks.

        Two-pass deduplication:
        Pass 1 — Exact match on normalised text span
        Pass 2 — Near-duplicate detection using token overlap
                  (catches same commitment extracted from overlapping chunks
                   with slightly different whitespace or truncation)

        Keeps the highest-confidence version of each unique span.
        """
        if not candidates:
            return []

        # Pass 1: exact normalised match
        exact: dict[str, CandidateClaim] = {}
        for c in candidates:
            key = self._normalise_span(c.text_span)
            if key not in exact or c.confidence > exact[key].confidence:
                exact[key] = c

        deduped = list(exact.values())

        # Pass 2: near-duplicate detection via token overlap
        # Two spans are near-duplicates if their token overlap > 0.85
        final   = []
        used    = set()

        for i, c1 in enumerate(deduped):
            if i in used:
                continue
            group = [c1]
            for j, c2 in enumerate(deduped):
                if j <= i or j in used:
                    continue
                if self._token_overlap(c1.text_span, c2.text_span) > 0.85:
                    group.append(c2)
                    used.add(j)
            # Keep highest confidence from the group
            best = max(group, key=lambda c: c.confidence)
            final.append(best)
            used.add(i)

        return final

    def _normalise_span(self, text: str) -> str:
        """Normalise a text span for exact matching."""
        import re
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip('.,;:"\'')
        return text

    def _token_overlap(self, a: str, b: str) -> float:
        """
        Compute token overlap between two strings.
        Returns a float in [0, 1] — 1.0 means identical token sets.
        """
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union        = tokens_a | tokens_b
        return len(intersection) / len(union)

    # ── Confidence flagging ───────────────────────────────────────────────────

    def flag_low_confidence(
        self,
        candidates: list[CandidateClaim],
        threshold:  float = 0.85,
    ) -> tuple[list[CandidateClaim], list[CandidateClaim]]:
        """
        Split candidates into high-confidence and flagged-for-review.
        Returns (high_confidence, flagged).
        """
        high    = [c for c in candidates if c.confidence >= threshold]
        flagged = [c for c in candidates if c.confidence <  threshold]
        return high, flagged


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = Stage1Extractor()

    sample_document = """
FACILITY AGREEMENT

Section 4 — Borrower Obligations

The Borrower shall repay the principal amount of the Facility in full on the Maturity Date.

The Borrower must maintain a debt service reserve account with a minimum balance equal 
to six months of scheduled debt service payments at all times during the term of the Facility.

The Borrower shall not make any distributions to shareholders prior to the repayment 
of all outstanding amounts under this Agreement.

Section 5 — Events of Default

If a Payment Default occurs, the Lender may declare the entire outstanding principal 
amount immediately due and payable.

The Borrower is prohibited from incurring any additional Financial Indebtedness 
without the prior written consent of the Lender.

Section 6 — Representations

The Borrower represents and warrants that all financial statements provided are true 
and accurate in all material respects as at the date of this Agreement.

Section 7 — General

This Agreement shall be governed by the laws of New South Wales.

The parties agree to act in good faith in the performance of their obligations.
"""

    print("── Stage 1 Extraction ───────────────────────────────────────\n")
    candidates = extractor.extract(sample_document)

    print(f"Extracted {len(candidates)} candidate commitment(s):\n")
    for i, c in enumerate(candidates, 1):
        flag = " ⚑ LOW CONFIDENCE" if c.confidence < 0.85 else ""
        print(f"{i}. [{c.claim_type.value}] confidence={c.confidence:.2f}{flag}")
        print(f"   \"{c.text_span}\"")
        if c.section:
            print(f"   Section: {c.section}")
        print()

    high, flagged = extractor.flag_low_confidence(candidates)
    print(f"High confidence: {len(high)}  |  Flagged for review: {len(flagged)}")
    print("\nStage 1 Extractor smoke test complete.")
