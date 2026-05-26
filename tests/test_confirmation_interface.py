"""
Validity — Confirmation Interface Test Suite
tests/test_confirmation_interface.py

Run with: python3 -m pytest tests/test_confirmation_interface.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import uuid
import pytest

from fragment_validator import ClaimType
from stage2_translator import CandidateClaim
from confirmation_interface import (
    ConfirmationInterface,
    ConfirmationSession,
    ReviewRecord,
    Decision,
)

interface = ConfirmationInterface()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_candidate(
    text_span:  str,
    claim_type: ClaimType = ClaimType.OBLIGATION,
    confidence: float     = 0.95,
    page:       int       = 1,
    section:    str       = "Section 1",
) -> CandidateClaim:
    return CandidateClaim(
        id         = str(uuid.uuid4()),
        text_span  = text_span,
        claim_type = claim_type,
        confidence = confidence,
        page       = page,
        section    = section,
    )

def confirm(candidate: CandidateClaim) -> dict:
    return {"claim_id": candidate.id, "decision": "confirmed"}

def reject(candidate: CandidateClaim, note: str = "") -> dict:
    return {"claim_id": candidate.id, "decision": "rejected", "note": note}

def annotate(candidate: CandidateClaim, text: str) -> dict:
    return {"claim_id": candidate.id, "decision": "annotated", "annotation": text}


# ── Session structure ─────────────────────────────────────────────────────────

class TestSessionStructure:
    def test_session_is_confirmation_session(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert isinstance(session, ConfirmationSession)

    def test_records_count_matches_candidates(self):
        candidates = [make_candidate(f"Claim {i}.") for i in range(4)]
        decisions  = [confirm(c) for c in candidates]
        session    = interface.review_programmatic(candidates, decisions)
        assert len(session.records) == 4

    def test_empty_candidates_empty_session(self):
        session = interface.review_programmatic([], [])
        assert len(session.records) == 0
        assert session.confirmed == []
        assert session.rejected  == []


# ── Confirmed decisions ───────────────────────────────────────────────────────

class TestConfirmedDecisions:
    def test_confirmed_claim_in_confirmed(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert len(session.confirmed) == 1

    def test_confirmed_claim_text_span_preserved(self):
        c = make_candidate("The borrower shall repay the principal.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert session.confirmed[0].text_span == "The borrower shall repay the principal."

    def test_confirmed_claim_id_preserved(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert session.confirmed[0].id == c.id

    def test_confirmed_count_correct(self):
        candidates = [make_candidate(f"Claim {i}.") for i in range(3)]
        decisions  = [confirm(c) for c in candidates]
        session    = interface.review_programmatic(candidates, decisions)
        assert session.confirmed_count == 3

    def test_confirmed_not_in_rejected(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert c not in session.rejected


# ── Rejected decisions ────────────────────────────────────────────────────────

class TestRejectedDecisions:
    def test_rejected_claim_in_rejected(self):
        c = make_candidate("The borrower shall act reasonably.")
        session = interface.review_programmatic([c], [reject(c)])
        assert len(session.rejected) == 1

    def test_rejected_claim_not_in_confirmed(self):
        c = make_candidate("The borrower shall act reasonably.")
        session = interface.review_programmatic([c], [reject(c)])
        assert len(session.confirmed) == 0

    def test_rejected_count_correct(self):
        candidates = [make_candidate(f"Claim {i}.") for i in range(3)]
        decisions  = [reject(c) for c in candidates]
        session    = interface.review_programmatic(candidates, decisions)
        assert session.rejected_count == 3

    def test_rejected_note_preserved(self):
        c = make_candidate("The borrower shall act reasonably.")
        session = interface.review_programmatic([c], [reject(c, note="Modal term")])
        record  = next(r for r in session.records if r.decision == Decision.REJECTED)
        assert record.note == "Modal term"

    def test_mixed_confirmed_and_rejected(self):
        c1 = make_candidate("The borrower shall repay.")
        c2 = make_candidate("The borrower shall act reasonably.")
        c3 = make_candidate("The borrower shall maintain insurance.")
        session = interface.review_programmatic(
            [c1, c2, c3],
            [confirm(c1), reject(c2), confirm(c3)],
        )
        assert session.confirmed_count == 2
        assert session.rejected_count  == 1


# ── Annotated decisions ───────────────────────────────────────────────────────

class TestAnnotatedDecisions:
    def test_annotated_claim_in_confirmed(self):
        c = make_candidate("The borrower shall repay principal.")
        session = interface.review_programmatic(
            [c], [annotate(c, "The borrower shall repay the principal on the Maturity Date.")]
        )
        assert len(session.confirmed) == 1

    def test_annotated_text_replaces_original(self):
        c           = make_candidate("The borrower shall repay principal.")
        corrected   = "The borrower shall repay the principal on the Maturity Date."
        session     = interface.review_programmatic([c], [annotate(c, corrected)])
        assert session.confirmed[0].text_span == corrected

    def test_annotated_original_text_unchanged(self):
        original    = "The borrower shall repay principal."
        c           = make_candidate(original)
        corrected   = "The borrower shall repay the principal on the Maturity Date."
        session     = interface.review_programmatic([c], [annotate(c, corrected)])
        # Original candidate unchanged
        assert c.text_span == original

    def test_annotated_id_preserved(self):
        c = make_candidate("The borrower shall repay principal.")
        session = interface.review_programmatic(
            [c], [annotate(c, "The borrower shall repay the principal.")]
        )
        assert session.confirmed[0].id == c.id

    def test_annotated_appears_in_annotated_list(self):
        c = make_candidate("The borrower shall repay principal.")
        session = interface.review_programmatic(
            [c], [annotate(c, "The borrower shall repay the principal.")]
        )
        assert len(session.annotated) == 1

    def test_annotated_record_has_annotation(self):
        c           = make_candidate("The borrower shall repay principal.")
        corrected   = "The borrower shall repay the principal."
        session     = interface.review_programmatic([c], [annotate(c, corrected)])
        record      = session.annotated[0]
        assert record.annotation == corrected

    def test_empty_annotation_uses_original(self):
        c = make_candidate("The borrower shall repay.")
        decisions = [{"claim_id": c.id, "decision": "annotated", "annotation": ""}]
        session   = interface.review_programmatic([c], decisions)
        assert session.confirmed[0].text_span == c.text_span


# ── Batch confirm mode ────────────────────────────────────────────────────────

class TestBatchConfirmMode:
    def test_all_high_confidence_confirmed(self):
        candidates = [make_candidate(f"Claim {i}.", confidence=0.95) for i in range(3)]
        session    = interface.review_batch_confirm(candidates, confidence_threshold=0.85)
        assert session.confirmed_count == 3
        assert session.rejected_count  == 0

    def test_low_confidence_still_confirmed_with_flag(self):
        c       = make_candidate("The borrower may act.", confidence=0.75)
        session = interface.review_batch_confirm([c], confidence_threshold=0.85)
        # Low confidence confirmed but flagged
        assert session.confirmed_count == 1
        record = session.records[0]
        assert record.note is not None
        assert "Low confidence" in record.note

    def test_batch_confirm_empty_list(self):
        session = interface.review_batch_confirm([])
        assert session.confirmed_count == 0

    def test_batch_confirm_all_above_threshold(self):
        candidates = [
            make_candidate("Claim A.", confidence=0.95),
            make_candidate("Claim B.", confidence=0.90),
            make_candidate("Claim C.", confidence=0.88),
        ]
        session = interface.review_batch_confirm(candidates, confidence_threshold=0.85)
        assert len(session.confirmed) == 3


# ── Programmatic mode edge cases ──────────────────────────────────────────────

class TestProgrammaticEdgeCases:
    def test_no_decision_provided_auto_confirmed(self):
        c       = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [])
        assert session.confirmed_count == 1
        record  = session.records[0]
        assert "Auto-confirmed" in (record.note or "")

    def test_invalid_decision_string_defaults_to_confirmed(self):
        c = make_candidate("The borrower shall repay.")
        decisions = [{"claim_id": c.id, "decision": "maybe"}]
        session   = interface.review_programmatic([c], decisions)
        assert session.confirmed_count == 1

    def test_unknown_claim_id_in_decisions_ignored(self):
        c = make_candidate("The borrower shall repay.")
        decisions = [
            confirm(c),
            {"claim_id": "nonexistent-id", "decision": "confirmed"},
        ]
        session = interface.review_programmatic([c], decisions)
        assert len(session.records) == 1


# ── Session summary ───────────────────────────────────────────────────────────

class TestSessionSummary:
    def test_summary_is_string(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert isinstance(session.summary(), str)

    def test_summary_contains_counts(self):
        c1 = make_candidate("Claim 1.")
        c2 = make_candidate("Claim 2.")
        c3 = make_candidate("Claim 3.")
        session = interface.review_programmatic(
            [c1, c2, c3],
            [confirm(c1), reject(c2), annotate(c3, "Corrected claim 3.")],
        )
        summary = session.summary()
        assert "3" in summary   # total
        assert "2" in summary   # confirmed (c1 + c3 annotated)
        assert "1" in summary   # rejected


# ── Review records ────────────────────────────────────────────────────────────

class TestReviewRecords:
    def test_record_has_candidate(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert session.records[0].candidate is c

    def test_record_has_decision(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert session.records[0].decision == Decision.CONFIRMED

    def test_confirmed_record_decision(self):
        c = make_candidate("The borrower shall repay.")
        session = interface.review_programmatic([c], [confirm(c)])
        assert session.records[0].decision == Decision.CONFIRMED

    def test_rejected_record_decision(self):
        c = make_candidate("The borrower shall act reasonably.")
        session = interface.review_programmatic([c], [reject(c)])
        assert session.records[0].decision == Decision.REJECTED

    def test_annotated_record_decision(self):
        c = make_candidate("The borrower shall repay principal.")
        session = interface.review_programmatic(
            [c], [annotate(c, "The borrower shall repay the principal.")]
        )
        assert session.records[0].decision == Decision.ANNOTATED
