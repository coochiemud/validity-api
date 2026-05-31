"""
Validity — Resource Constraint Encoder v2
resource_constraint_encoder.py

Extends the Z3 encoder with linear arithmetic reasoning over shared resources.
Detects payment priority conflicts that propositional logic cannot find.

v2 changes vs v1:
- Models write-down offset mechanics: Return Amount payments correspond to
  reductions in Class Principal Balance, so P and O are not independent.
- Three-signal detection: priority clause + maturity obligation + write-down mechanic.
- If write-down mechanic is present: encodes adjusted_principal = O - W.
  The contradiction only holds when W = 0 (no write-downs) AND assets are depleted.
- If no write-down mechanic detected: falls back to v1 behaviour.
- Clean single-class implementation — no appended methods.

Verdicts:
  UNSAT — proven contradiction (no write-down offset present)
  CONDITIONAL — stress exposure exists but document may resolve via write-downs
  SAT — no conflict detected
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid as _uuid

try:
    from z3 import (
        Real, Solver, sat, unsat,
        And, Implies, RealVal,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ── Signal lists ──────────────────────────────────────────────────────────────

PRIORITY_SIGNALS = [
    "before allocating",
    "before paying",
    "in priority to",
    "senior to",
    "before any amounts owed",
    "before being allocated to",
    "paid first",
    "allocated first",
]

OBLIGATION_SIGNALS = [
    "will pay 100%",
    "shall pay 100%",
    "pay 100% of the outstanding",
    "pay 100% of",
    "pay in full",
    "full repayment",
    "entire outstanding",
    "maturity date the trust will pay",
    "maturity date, the trust will pay",
]

OBLIGATION_DISQUALIFIERS = [
    "return reimbursement amount",
    "transfer amount",
    "reimbursement amount, if any",
    "if any, on the business day",
    "at our option",
    "in its discretion",
    "sole discretion",
    "subject to available funds",
    "to the extent available",
]

RESOURCE_SIGNALS = [
    "eligible investments",
    "distribution account",
    "proceeds",
    "assets",
    "funds",
    "monies",
    "amounts on deposit",
]

# Write-down mechanic: signals that the principal balance is reduced by
# the same events that trigger the priority payment. If present, P and O
# are linked — paying P reduces O by a corresponding amount.
WRITEDOWN_SIGNALS = [
    "tranche write-down amount",
    "write-down amount",
    "class principal balance",
    "reduction of the class principal balance",
    "reduces the class principal balance",
    "reduced by",
    "write-down",
    "writedown",
    "principal balance reduction",
    "corresponding reduction",
]

# If the document contains these signals near the write-down language,
# it suggests the write-down offsets the obligation.
WRITEDOWN_OFFSET_SIGNALS = [
    "corresponding",
    "reduces",
    "reduction",
    "offset",
    "net",
    "adjusted",
    "as reduced",
    "as so reduced",
]


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class PaymentPriorityPair:
    priority_claim_id:     str
    priority_claim_text:   str
    obligation_claim_id:   str
    obligation_claim_text: str
    writedown_detected:    bool = False
    writedown_text:        str  = ""
    resource_name:         str  = "eligible_investments"


@dataclass
class ResourceConstraintResult:
    verdict:           str   # "unsat" | "conditional" | "sat" | "skip"
    pairs_detected:    list  = field(default_factory=list)
    contradiction:     Optional[PaymentPriorityPair] = None
    formal_proof:      str   = ""
    explanation:       str   = ""


# ── Encoder ───────────────────────────────────────────────────────────────────

class ResourceConstraintEncoder:
    """
    Detects and proves payment priority conflicts using Z3 linear arithmetic.

    Called after the main propositional solver returns SAT. Checks whether
    the SAT result hides an arithmetic contradiction in shared resource
    allocation — taking write-down offset mechanics into account.
    """

    # ── Public entry points ───────────────────────────────────────────────────

    def analyse(self, claims: list) -> ResourceConstraintResult:
        """
        Analyse a list of FormalClaim objects for payment priority conflicts.
        Uses extracted claims from the pipeline.
        """
        if not Z3_AVAILABLE:
            return ResourceConstraintResult(verdict="skip", explanation="Z3 not available.")

        pairs = self._detect_pairs_from_claims(claims)
        if not pairs:
            return ResourceConstraintResult(verdict="sat", explanation="No payment priority pairs detected.")

        for pair in pairs:
            result = self._check_pair(pair)
            if result.verdict in ("unsat", "conditional"):
                result.pairs_detected = pairs
                return result

        return ResourceConstraintResult(
            verdict        = "sat",
            pairs_detected = pairs,
            explanation    = f"{len(pairs)} payment priority pair(s) detected. No conflict proven.",
        )

    def check_document_text(self, document_text: str) -> Optional[ResourceConstraintResult]:
        """
        Search document text directly for payment priority conflicts.
        Used as a fallback when the extractor doesn't surface both claims.
        Detects write-down mechanics and models them in the proof.
        """
        if not Z3_AVAILABLE:
            return None

        text_lower = document_text.lower()

        # ── Find priority clause ──────────────────────────────────────────────
        priority_text = self._find_clause(document_text, text_lower, PRIORITY_SIGNALS, RESOURCE_SIGNALS, require_resource=True)
        if not priority_text:
            return None

        # ── Find absolute obligation clause ───────────────────────────────────
        obligation_text = self._find_obligation_clause(document_text, text_lower)
        if not obligation_text:
            return None

        # ── Detect write-down mechanic ────────────────────────────────────────
        writedown_detected, writedown_text = self._detect_writedown(document_text, text_lower)

        pair = PaymentPriorityPair(
            priority_claim_id     = str(_uuid.uuid4()),
            priority_claim_text   = priority_text[:500],
            obligation_claim_id   = str(_uuid.uuid4()),
            obligation_claim_text = obligation_text[:500],
            writedown_detected    = writedown_detected,
            writedown_text        = writedown_text[:300] if writedown_text else "",
        )

        result = self._check_pair(pair)
        if result.verdict in ("unsat", "conditional"):
            return result
        return None

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _find_clause(self, document_text: str, text_lower: str, signals: list,
                     require_also: list = None, require_resource: bool = False) -> Optional[str]:
        """Find the first clause matching any signal, optionally requiring a resource signal nearby."""
        for signal in signals:
            idx = text_lower.find(signal)
            if idx < 0:
                continue
            start = max(0, text_lower.rfind(".", 0, idx) + 1)
            end   = text_lower.find(".", idx)
            if end < 0:
                end = idx + 400
            candidate = document_text[start:end + 1].strip()
            if require_resource and not any(r in candidate.lower() for r in RESOURCE_SIGNALS):
                continue
            if require_also and not any(r in candidate.lower() for r in require_also):
                continue
            return candidate
        return None

    def _find_obligation_clause(self, document_text: str, text_lower: str) -> Optional[str]:
        """Find an absolute obligation clause, excluding disqualified claims."""
        for signal in ["will pay 100%", "shall pay 100%", "pay 100% of the outstanding", "pay 100% of"]:
            idx = text_lower.find(signal)
            if idx < 0:
                continue
            start = max(0, text_lower.rfind(".", 0, idx) + 1)
            end   = text_lower.find(".", idx)
            if end < 0:
                end = idx + 400
            candidate = document_text[start:end + 1].strip()
            # Disqualify conditional / non-absolute obligations
            if any(d in candidate.lower() for d in OBLIGATION_DISQUALIFIERS):
                continue
            return candidate
        return None

    def _detect_writedown(self, document_text: str, text_lower: str) -> tuple[bool, str]:
        """
        Detect whether the document contains a write-down mechanic that links
        the priority payment to a reduction in the principal balance.

        Returns (detected: bool, excerpt: str).
        """
        for signal in WRITEDOWN_SIGNALS:
            idx = text_lower.find(signal)
            if idx < 0:
                continue
            # Extract context around the write-down signal
            start   = max(0, text_lower.rfind(".", 0, idx) + 1)
            end     = text_lower.find(".", idx)
            if end < 0:
                end = idx + 400
            context = document_text[start:end + 1].strip()
            context_lower = context.lower()
            # Confirm the write-down is associated with an offset/reduction
            if any(s in context_lower for s in WRITEDOWN_OFFSET_SIGNALS):
                # Also confirm it's linked to a principal balance
                if "principal balance" in context_lower or "class principal" in context_lower:
                    return True, context
        return False, ""

    def _detect_pairs_from_claims(self, claims: list) -> list:
        """Detect priority/obligation pairs from extracted FormalClaim objects."""
        priority_claims   = []
        obligation_claims = []

        for claim in claims:
            text       = getattr(claim, "text_span", "") or ""
            text_lower = text.lower()

            is_priority     = any(sig in text_lower for sig in PRIORITY_SIGNALS)
            is_obligation   = any(sig in text_lower for sig in OBLIGATION_SIGNALS)
            is_disqualified = any(sig in text_lower for sig in OBLIGATION_DISQUALIFIERS)
            has_resource    = any(sig in text_lower for sig in RESOURCE_SIGNALS)
            has_100pct      = any(s in text_lower for s in ["100%", "in full", "entire outstanding", "full amount"])

            if is_priority and has_resource:
                priority_claims.append(claim)
            if is_obligation and not is_disqualified and has_100pct:
                obligation_claims.append(claim)

        pairs = []
        for pc in priority_claims:
            for oc in obligation_claims:
                if pc.id == oc.id:
                    continue
                oc_lower = (getattr(oc, "text_span", "") or "").lower()
                if not any(s in oc_lower for s in ["100%", "in full", "entire", "maturity"]):
                    continue
                pairs.append(PaymentPriorityPair(
                    priority_claim_id    = pc.id,
                    priority_claim_text  = getattr(pc, "text_span", ""),
                    obligation_claim_id  = oc.id,
                    obligation_claim_text = getattr(oc, "text_span", ""),
                    writedown_detected   = False,
                ))
        return pairs

    # ── Z3 proof ──────────────────────────────────────────────────────────────

    def _check_pair(self, pair: PaymentPriorityPair) -> ResourceConstraintResult:
        """
        Prove or characterise the conflict between a priority clause and an
        absolute obligation clause, modelling write-down offsets if detected.

        Case A — No write-down mechanic detected:
            P and O are independent obligations on the same pool.
            Under stress (A < P + O), paying P in full leaves < O for the obligation.
            Result: UNSAT (proven contradiction).

        Case B — Write-down mechanic detected:
            Paying P corresponds to a write-down W that reduces O.
            Adjusted obligation = O - W.
            Under complete write-down (W = P), adjusted obligation = O - P.
            remaining after priority payment = A - P.
            paid_to_obligation = O - P <= A - P  =>  O <= A (satisfiable when A >= O).
            Result: CONDITIONAL — conflict only arises when W < P (partial write-down)
            and assets are simultaneously depleted.
        """
        solver = Solver()

        A  = Real("total_assets")
        P  = Real("priority_amount")
        O  = Real("original_obligation")
        W  = Real("write_down_amount")
        R  = Real("remaining_assets")
        PO = Real("paid_to_obligation")
        PP = Real("paid_to_priority")
        AO = Real("adjusted_obligation")

        solver.add(A >= RealVal(0))
        solver.add(P >  RealVal(0))
        solver.add(O >  RealVal(0))
        solver.add(W >= RealVal(0))

        # Stress: assets insufficient for both original obligation and priority
        solver.add(A < P + O)

        # Priority paid first, in full
        solver.add(PP == P)
        solver.add(PP <= A)

        # Remaining pool
        solver.add(R == A - PP)
        solver.add(R >= RealVal(0))

        if pair.writedown_detected:
            # Write-down reduces the obligation
            solver.add(W <= O)       # write-down cannot exceed original obligation
            solver.add(W <= P)       # write-down bounded by priority payment
            solver.add(AO == O - W)  # adjusted obligation
            solver.add(PO == AO)
            solver.add(PO <= R)

            result = solver.check()

            if result == unsat:
                # Even with write-downs, system is UNSAT
                proof = self._build_proof(pair, with_writedown=True, writedown_resolves=False)
                return ResourceConstraintResult(
                    verdict       = "unsat",
                    contradiction = pair,
                    formal_proof  = proof,
                    explanation   = (
                        "Even accounting for write-down mechanics, the priority payment and "
                        "adjusted maturity obligation cannot simultaneously be satisfied under stress. "
                        "The write-down offset does not fully resolve the conflict."
                    ),
                )
            else:
                # Write-downs may resolve the conflict under normal conditions,
                # but stress exposure exists when W < P
                proof = self._build_proof(pair, with_writedown=True, writedown_resolves=True)
                return ResourceConstraintResult(
                    verdict       = "conditional",
                    contradiction = pair,
                    formal_proof  = proof,
                    explanation   = (
                        "The document contains a write-down mechanic that links the priority payment "
                        "to a reduction in the principal balance. Under complete write-down (W = P), "
                        "the conflict may be resolved. However, stress exposure exists when write-downs "
                        "are partial (W < P) and the asset pool is simultaneously depleted. "
                        "This is a conditional stress scenario, not a proven contradiction."
                    ),
                )
        else:
            # No write-down detected — model as independent obligations
            solver.add(PO == O)
            solver.add(PO <= R)

            result = solver.check()

            if result == unsat:
                proof = self._build_proof(pair, with_writedown=False, writedown_resolves=False)
                return ResourceConstraintResult(
                    verdict       = "unsat",
                    contradiction = pair,
                    formal_proof  = proof,
                    explanation   = (
                        "Under stress conditions where available assets are insufficient to satisfy "
                        "both the priority payment and the absolute obligation, the two commitments "
                        "cannot simultaneously hold. No write-down mechanic was detected to offset "
                        "the obligation. The document does not resolve this competition."
                    ),
                )
            return ResourceConstraintResult(verdict="sat", explanation="No arithmetic contradiction proven.")

    # ── Proof builder ─────────────────────────────────────────────────────────

    def _build_proof(self, pair: PaymentPriorityPair,
                     with_writedown: bool, writedown_resolves: bool) -> str:
        lines = ["RESOURCE CONSTRAINT PROOF", "=" * 44, ""]

        lines += [
            "Variables:",
            "  A  = available assets (Eligible Investments)",
            "  P  = priority amount (Return Amount owed)",
            "  O  = original obligation (Class Principal Balance)",
        ]
        if with_writedown:
            lines += [
                "  W  = write-down amount (reduction to principal balance)",
                "  AO = adjusted obligation = O - W",
            ]
        lines += [""]

        lines += ["Constraints:", "  1. A >= 0, P > 0, O > 0"]
        if with_writedown:
            lines += ["  2. W >= 0, W <= min(P, O)"]
            lines += ["  3. Stress: A < P + O"]
            lines += ["  4. Priority: paid_to_priority = P, P <= A"]
            lines += ["  5. Remaining: R = A - P"]
            lines += ["  6. Write-down offset: AO = O - W"]
            lines += ["  7. Maturity obligation: paid_to_obligation = AO, AO <= R"]
        else:
            lines += ["  2. Stress: A < P + O"]
            lines += ["  3. Priority: paid_to_priority = P, P <= A"]
            lines += ["  4. Remaining: R = A - P"]
            lines += ["  5. Maturity obligation: paid_to_obligation = O, O <= R"]
        lines += [""]

        if with_writedown and writedown_resolves:
            lines += [
                "Write-down mechanic detected:",
                "  When W = P (complete write-down): AO = O - P",
                "  R = A - P",
                "  AO <= R  =>  O - P <= A - P  =>  O <= A",
                "  Satisfiable when A >= O.",
                "",
                "Conditional stress scenario:",
                "  When W < P (partial write-down): AO = O - W > O - P",
                "  R = A - P",
                "  AO <= R  =>  O - W <= A - P  =>  A >= O - W + P",
                "  Under simultaneous asset depletion (A < P + O) and partial",
                "  write-down (W < P), the adjusted obligation may not be satisfied.",
                "",
                "Verdict: CONDITIONAL — stress exposure exists but document may",
                "resolve via write-down mechanics under normal conditions.",
            ]
        elif with_writedown and not writedown_resolves:
            lines += [
                "Write-down mechanic detected but does not resolve conflict:",
                "  Z3 verdict: unsat",
                "  Even with W = O (maximum write-down), constraints are unsatisfiable.",
                "",
                "⊥  Contradiction. Cannot simultaneously hold under stress.",
            ]
        else:
            lines += [
                "No write-down mechanic detected.",
                "From (3): R = A - P",
                "From (5): O <= R  =>  O <= A - P  =>  A >= P + O",
                "From (2): A < P + O",
                "",
                "⊥  Contradiction. These constraints cannot simultaneously hold.",
            ]

        lines += [
            "", "=" * 44, "",
            "Priority commitment:",
            f'  "{pair.priority_claim_text[:200]}..."',
            "",
            "Maturity obligation:",
            f'  "{pair.obligation_claim_text[:200]}..."',
        ]
        if with_writedown and pair.writedown_text:
            lines += [
                "",
                "Write-down mechanic:",
                f'  "{pair.writedown_text[:200]}..."',
            ]
        lines += ["", f"Z3 verdict: {'conditional' if with_writedown and writedown_resolves else 'unsat'}"]

        return "\n".join(lines)


# ── Integration helper ────────────────────────────────────────────────────────

def run_resource_constraint_check(claims: list) -> Optional[ResourceConstraintResult]:
    """
    Run the resource constraint encoder against extracted validated claims.
    Returns a result if a conflict is found, None if clean.
    Called after the main Z3 solver returns SAT.
    """
    encoder = ResourceConstraintEncoder()
    result  = encoder.analyse(claims)
    if result.verdict in ("unsat", "conditional"):
        return result
    return None


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from fragment_validator import FormalClaim, ClaimType, Provenance, ValidationStatus
    import uuid

    def make_claim(text, claim_type=ClaimType.OBLIGATION):
        c = FormalClaim.__new__(FormalClaim)
        c.id = str(uuid.uuid4())
        c.text_span = text
        c.claim_type = claim_type
        c.confidence = 0.95
        c.formula = {}
        c.status = ValidationStatus.VALIDATED
        c.rejection = None
        c.provenance = Provenance(text_span=text)
        return c

    encoder = ResourceConstraintEncoder()

    print("=" * 60)
    print("Test 1: No write-down — should return UNSAT")
    print("=" * 60)
    doc_no_writedown = (
        "On each Payment Date the Trust will allocate proceeds of the Eligible Investments "
        "to the Return Amount before allocating any proceeds to pay amounts owed on the Notes. "
        "On the Maturity Date the Trust will pay 100% of the outstanding Class Principal Balance."
    )
    result = encoder.check_document_text(doc_no_writedown)
    print(f"Verdict: {result.verdict if result else 'None — no conflict detected'}")
    if result:
        print(f"Write-down detected: {result.contradiction.writedown_detected}")
        print()
        print(result.formal_proof)

    print()
    print("=" * 60)
    print("Test 2: With write-down mechanic — should return CONDITIONAL")
    print("=" * 60)
    doc_with_writedown = (
        "On each Payment Date the Trust will allocate proceeds of the Eligible Investments "
        "to the Return Amount before allocating any proceeds to pay amounts owed on the Notes. "
        "The Return Amount coincides with Tranche Write-down Amounts and a corresponding "
        "reduction of the Class Principal Balance of the affected Notes. "
        "On the Maturity Date the Trust will pay 100% of the outstanding Class Principal Balance "
        "as of such date, as reduced by any Tranche Write-down Amounts."
    )
    result2 = encoder.check_document_text(doc_with_writedown)
    print(f"Verdict: {result2.verdict if result2 else 'None — no conflict detected'}")
    if result2:
        print(f"Write-down detected: {result2.contradiction.writedown_detected}")
        print()
        print(result2.formal_proof)

    print()
    print("=" * 60)
    print("Test 3: No priority clause — should return None")
    print("=" * 60)
    doc_no_priority = (
        "The Borrower shall not make distributions while any default is continuing. "
        "The Borrower shall repay the loan in full on the maturity date."
    )
    result3 = encoder.check_document_text(doc_no_priority)
    print(f"Verdict: {result3.verdict if result3 else 'None — no conflict detected (correct)'}")

    print()
    print("Smoke test complete.")
