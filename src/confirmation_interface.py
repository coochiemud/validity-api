"""
Validity — Confirmation Interface
Component 7 of 7

Human-in-the-loop gate between Stage 1 (extraction) and Stage 2 (translation).

Presents each candidate claim to the user for review.
The user may confirm, reject, or annotate each claim.
Low-confidence claims are flagged explicitly.

Only confirmed claims proceed to Stage 2 translation.
Rejected claims are recorded but do not proceed.
Annotated claims carry the human's correction into Stage 2.

This gate is not a UX nicety.
It is the primary quality control mechanism for the probabilistic extraction stage.

Input:  List[CandidateClaim] from Stage 1
Output: ConfirmationSession (confirmed + rejected + annotated records)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from fragment_validator import ClaimType
from stage2_translator import CandidateClaim


# ── Decision types ────────────────────────────────────────────────────────────

class Decision(Enum):
    CONFIRMED = "confirmed"
    REJECTED  = "rejected"
    ANNOTATED = "annotated"   # confirmed with human correction


# ── Review record ─────────────────────────────────────────────────────────────

@dataclass
class ReviewRecord:
    """
    The result of a human reviewing a single candidate claim.
    """
    candidate:   CandidateClaim
    decision:    Decision
    annotation:  Optional[str] = None   # human correction to text_span
    note:        Optional[str] = None   # free-form reviewer note


@dataclass
class ConfirmationSession:
    """
    The complete output of a confirmation session.
    Contains all review records and the confirmed claim set ready for Stage 2.
    """
    records:    list[ReviewRecord] = field(default_factory=list)

    @property
    def confirmed(self) -> list[CandidateClaim]:
        """Claims confirmed or annotated — ready for Stage 2."""
        result = []
        for r in self.records:
            if r.decision in (Decision.CONFIRMED, Decision.ANNOTATED):
                if r.decision == Decision.ANNOTATED and r.annotation:
                    # Return a copy with the human's corrected text_span
                    import copy
                    corrected          = copy.copy(r.candidate)
                    corrected.text_span = r.annotation
                    result.append(corrected)
                else:
                    result.append(r.candidate)
        return result

    @property
    def rejected(self) -> list[CandidateClaim]:
        return [r.candidate for r in self.records if r.decision == Decision.REJECTED]

    @property
    def annotated(self) -> list[ReviewRecord]:
        return [r for r in self.records if r.decision == Decision.ANNOTATED]

    @property
    def confirmed_count(self) -> int:
        return len([r for r in self.records if r.decision in (Decision.CONFIRMED, Decision.ANNOTATED)])

    @property
    def rejected_count(self) -> int:
        return len([r for r in self.records if r.decision == Decision.REJECTED])

    def summary(self) -> str:
        total = len(self.records)
        return (
            f"Reviewed {total} claim(s): "
            f"{self.confirmed_count} confirmed, "
            f"{self.rejected_count} rejected, "
            f"{len(self.annotated)} annotated."
        )


# ── Confirmation Interface ────────────────────────────────────────────────────

class ConfirmationInterface:
    """
    Presents candidate claims to a human reviewer for confirmation.

    Modes:
    - interactive: presents each claim in the terminal, waits for input
    - batch_confirm: confirms all claims above a confidence threshold (automated)
    - programmatic: accepts a pre-built decision list (for testing/API use)

    The interface displays three items simultaneously for each claim:
    1. The verbatim source text span
    2. A plain-language summary of what was extracted
    3. The claim type assigned by the extractor
    """

    LOW_CONFIDENCE_THRESHOLD = 0.85

    # ── Interactive mode ──────────────────────────────────────────────────────

    def review_interactive(
        self,
        candidates: list[CandidateClaim],
    ) -> ConfirmationSession:
        """
        Present each candidate claim interactively in the terminal.
        The user types c (confirm), r (reject), or a (annotate) for each.
        """
        session = ConfirmationSession()

        if not candidates:
            print("No candidate claims to review.")
            return session

        print("\n" + "━" * 60)
        print("  VALIDITY — CLAIM CONFIRMATION")
        print("━" * 60)
        print(f"  {len(candidates)} candidate claim(s) require review.")
        print("  For each claim: [c] confirm  [r] reject  [a] annotate")
        print("━" * 60 + "\n")

        for i, candidate in enumerate(candidates, 1):
            record = self._review_one_interactive(candidate, i, len(candidates))
            session.records.append(record)

        print("\n" + "━" * 60)
        print(f"  {session.summary()}")
        print("━" * 60 + "\n")

        return session

    def _review_one_interactive(
        self,
        candidate: CandidateClaim,
        index:     int,
        total:     int,
    ) -> ReviewRecord:
        """Present a single claim and collect the reviewer's decision."""
        low_confidence = candidate.confidence < self.LOW_CONFIDENCE_THRESHOLD
        flag           = " ⚑ LOW CONFIDENCE — review carefully" if low_confidence else ""

        print(f"Claim {index} of {total}{flag}")
        print(f"  Type:       {candidate.claim_type.value}")
        print(f"  Confidence: {candidate.confidence:.0%}")
        if candidate.page:
            location = f"page {candidate.page}"
            if candidate.section:
                location += f" · {candidate.section}"
            print(f"  Location:   {location}")
        print(f"  Text:       \"{candidate.text_span}\"")
        print(f"  Meaning:    {self._plain_summary(candidate)}")
        print()

        while True:
            choice = input("  Decision [c/r/a]: ").strip().lower()

            if choice == "c":
                print(f"  ✓ Confirmed\n")
                return ReviewRecord(candidate=candidate, decision=Decision.CONFIRMED)

            elif choice == "r":
                note = input("  Note (optional): ").strip() or None
                print(f"  ✗ Rejected\n")
                return ReviewRecord(candidate=candidate, decision=Decision.REJECTED, note=note)

            elif choice == "a":
                print(f"  Current text: \"{candidate.text_span}\"")
                annotation = input("  Corrected text: ").strip()
                if not annotation:
                    annotation = candidate.text_span
                print(f"  ✓ Annotated\n")
                return ReviewRecord(
                    candidate  = candidate,
                    decision   = Decision.ANNOTATED,
                    annotation = annotation,
                )
            else:
                print("  Please enter c, r, or a.")

    # ── Batch confirm mode ────────────────────────────────────────────────────

    def review_batch_confirm(
        self,
        candidates:           list[CandidateClaim],
        confidence_threshold: float = 0.85,
    ) -> ConfirmationSession:
        """
        Automatically confirm claims above the confidence threshold.
        Flag and confirm low-confidence claims with a warning.
        Used for high-volume processing where human review is deferred.
        """
        session = ConfirmationSession()

        for candidate in candidates:
            if candidate.confidence >= confidence_threshold:
                session.records.append(ReviewRecord(
                    candidate = candidate,
                    decision  = Decision.CONFIRMED,
                ))
            else:
                # Still confirm but flag for downstream attention
                session.records.append(ReviewRecord(
                    candidate = candidate,
                    decision  = Decision.CONFIRMED,
                    note      = f"Low confidence ({candidate.confidence:.0%}) — flagged for review",
                ))

        return session

    # ── Programmatic mode ─────────────────────────────────────────────────────

    def review_programmatic(
        self,
        candidates: list[CandidateClaim],
        decisions:  list[dict],
    ) -> ConfirmationSession:
        """
        Accept pre-built decisions programmatically.
        Used for testing, API integration, and automated pipelines.

        decisions: list of dicts with keys:
          - claim_id:   str (must match a candidate)
          - decision:   "confirmed" | "rejected" | "annotated"
          - annotation: str (required if decision == "annotated")
          - note:       str (optional)
        """
        session    = ConfirmationSession()
        index      = {c.id: c for c in candidates}
        decision_map = {d["claim_id"]: d for d in decisions}

        for candidate in candidates:
            decision_data = decision_map.get(candidate.id)

            if decision_data is None:
                # No decision provided — default to confirmed
                session.records.append(ReviewRecord(
                    candidate = candidate,
                    decision  = Decision.CONFIRMED,
                    note      = "Auto-confirmed (no decision provided)",
                ))
                continue

            decision_str = decision_data.get("decision", "confirmed").lower()
            try:
                decision = Decision(decision_str)
            except ValueError:
                decision = Decision.CONFIRMED

            session.records.append(ReviewRecord(
                candidate  = candidate,
                decision   = decision,
                annotation = decision_data.get("annotation"),
                note       = decision_data.get("note"),
            ))

        return session

    # ── Plain-language summary ────────────────────────────────────────────────

    def _plain_summary(self, candidate: CandidateClaim) -> str:
        """
        Generate a plain-language summary of what was extracted.
        Displayed alongside verbatim text to aid reviewer judgment.
        """
        ct = candidate.claim_type
        text = candidate.text_span.lower()

        if ct == ClaimType.OBLIGATION:
            return "A party is required to do something."
        elif ct == ClaimType.DISCLAIMER:
            return "A party disclaims or limits a commitment."
        elif ct == ClaimType.GUARANTEE:
            return "A party asserts or warrants something is true."
        elif ct == ClaimType.CONDITION:
            return "A conditional commitment: if X then Y."
        elif ct == ClaimType.QUANTIFIED_ASSURANCE:
            return "A commitment about all, none, or a specific quantity."
        return "A commitment that may affect document consistency."


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid
    from fragment_validator import ClaimType

    interface = ConfirmationInterface()

    # Build sample candidates
    candidates = [
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The Borrower shall repay the principal amount on the Maturity Date.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.95,
            page       = 4,
            section    = "Section 4.1",
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The Borrower is prohibited from making distributions prior to repayment.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.92,
            page       = 7,
            section    = "Section 6.3",
        ),
        CandidateClaim(
            id         = str(uuid.uuid4()),
            text_span  = "The Borrower shall act in a reasonable and prudent manner.",
            claim_type = ClaimType.OBLIGATION,
            confidence = 0.80,
            page       = 2,
            section    = "Section 2.1",
        ),
    ]

    print("── Batch Confirm Mode ───────────────────────────────────────\n")
    session = interface.review_batch_confirm(candidates, confidence_threshold=0.85)
    print(session.summary())
    print(f"Confirmed claims: {len(session.confirmed)}")
    for c in session.confirmed:
        print(f"  · \"{c.text_span[:60]}...\"" if len(c.text_span) > 60 else f"  · \"{c.text_span}\"")

    print("\n── Programmatic Mode ────────────────────────────────────────\n")
    decisions = [
        {"claim_id": candidates[0].id, "decision": "confirmed"},
        {"claim_id": candidates[1].id, "decision": "confirmed"},
        {"claim_id": candidates[2].id, "decision": "rejected", "note": "Modal term — will be refused by validator anyway"},
    ]
    session2 = interface.review_programmatic(candidates, decisions)
    print(session2.summary())
    print(f"Confirmed: {len(session2.confirmed)}  Rejected: {len(session2.rejected)}")

    print("\n── Annotated Example ────────────────────────────────────────\n")
    decisions3 = [
        {
            "claim_id":   candidates[0].id,
            "decision":   "annotated",
            "annotation": "The Borrower shall repay the principal on or before the Maturity Date.",
            "note":       "Corrected to match exact clause wording",
        },
    ]
    session3 = interface.review_programmatic([candidates[0]], decisions3)
    print(session3.summary())
    confirmed = session3.confirmed
    print(f"Annotated text: \"{confirmed[0].text_span}\"")

    print("\nConfirmation Interface smoke test complete.")
