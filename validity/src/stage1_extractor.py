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

Format:
[
  {
    "text_span": "verbatim text from document",
    "claim_type": "obligation",
    "confidence": 0.95,
    "page_hint": 3,
    "section_hint": "Section 4.1"
  }
]

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
    LOW_CONFIDENCE_THRESHOLD = 0.70  # Claims below this are flagged for review

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
        Split document into chunks at paragraph boundaries.
        Keeps chunks under max_chars to stay within context limits.
        """
        if len(text) <= max_chars:
            return [text]

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
                max_tokens  = 2000,
            )
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
            print(f"[Extractor] JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"[Extractor] API error: {e}")
            return []

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_candidate(self, item: dict) -> CandidateClaim:
        """Parse a raw extraction dict into a CandidateClaim."""
        claim_type_str = item.get("claim_type", "obligation").lower()
        claim_type     = self._parse_claim_type(claim_type_str)
        confidence     = float(item.get("confidence", 0.8))

        return CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = item["text_span"].strip(),
            claim_type = claim_type,
            confidence = min(max(confidence, 0.0), 1.0),
            page       = item.get("page_hint"),
            section    = item.get("section_hint"),
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
        Remove duplicate candidates by text_span.
        Keeps the highest-confidence version of each unique span.
        """
        seen:   dict[str, CandidateClaim] = {}
        for candidate in candidates:
            key = candidate.text_span.strip().lower()
            if key not in seen or candidate.confidence > seen[key].confidence:
                seen[key] = candidate
        return list(seen.values())

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
