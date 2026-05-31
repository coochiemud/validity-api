"""
Validity — Solver Interface
Component 3 of 7

Submits encoded constraints to Z3.
Returns sat/unsat and, on unsat, the minimal unsatisfiable core.

Input:  One or more EncodedClaim objects
Output: SolverResult (verdict + minimal core on contradiction)

Fully deterministic. No probabilistic element. No LLM involvement.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import z3

from z3_encoder import EncodedClaim


# ── Verdict ───────────────────────────────────────────────────────────────────

class Verdict(Enum):
    SAT   = "sat"    # All claims simultaneously satisfiable. No contradiction proven.
    UNSAT = "unsat"  # Contradiction exists. Minimal core returned.


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class CoreEntry:
    """A single claim identified as part of the minimal unsatisfiable core."""
    claim_id:   str
    provenance: str   # verbatim source text span
    constraint: str   # string representation of the Z3 constraint


@dataclass
class SolverResult:
    verdict:      Verdict
    claims_count: int                    # total claims submitted
    core:         list[CoreEntry] = field(default_factory=list)  # populated on unsat
    note:         str = ""


# ── Solver Interface ──────────────────────────────────────────────────────────

class SolverInterface:
    """
    Submits a set of EncodedClaims to Z3 and returns a SolverResult.

    On sat:   Returns Verdict.SAT with no core.
    On unsat: Returns Verdict.UNSAT with the minimal unsatisfiable core
              mapped back to claim IDs and source text spans.

    Uses Z3's Optimize/Solver with named assertions to enable
    unsat core extraction via track_for_unsat_core.
    """

    def _temporal_subsumption_conflict(self, a, b) -> bool:
        """
        Detect temporal subsumption contradictions that Z3 cannot infer without
        a bridge axiom. A required payment within a shorter window contradicts
        a prohibition on payment within a longer overlapping window.
        e.g. "shall pay within 5 days" + "shall not pay within 30 days" = UNSAT
        """
        same_action = (
            getattr(a, "canonical_action", None) is not None
            and getattr(a, "canonical_action", None) == getattr(b, "canonical_action", None)
        )
        if not same_action:
            return False
        if getattr(a, "temporal_operator", None) != "within_days":
            return False
        if getattr(b, "temporal_operator", None) != "within_days":
            return False
        a_days = getattr(a, "temporal_bound_days", None)
        b_days = getattr(b, "temporal_bound_days", None)
        if a_days is None or b_days is None:
            return False
        a_pol = getattr(a, "polarity", None)
        b_pol = getattr(b, "polarity", None)
        if a_pol is None or b_pol is None:
            return False
        if a_pol is True and b_pol is False and a_days <= b_days:
            return True
        if b_pol is True and a_pol is False and b_days <= a_days:
            return True
        return False

    def _is_complete_rule(self, claim) -> bool:
        """Determine if a claim is a complete standalone rule."""
        if getattr(claim, "rule_completeness", None) == "complete_rule":
            return True

        text = (
            getattr(claim, "text_span", None)
            or getattr(claim, "provenance", None)
            or ""
        ).lower()

        has_modality = any(m in text for m in [
            "shall", "shall not", "must", "must not", "will", "may not"
        ])

        hierarchy_path = getattr(claim, "hierarchy_path", None) or []
        parent_clause  = getattr(claim, "parent_clause", None)
        is_top_level   = parent_clause is None and len(hierarchy_path) <= 1

        return is_top_level and has_modality and len(text.split()) >= 8

    def _should_check_pair(self, a, b) -> bool:
        """Determine whether two claims should be checked for contradiction."""
        a_complete = self._is_complete_rule(a)
        b_complete = self._is_complete_rule(b)

        # Complete top-level rules are always checked against each other.
        if a_complete and b_complete:
            return True

        same_parent = (
            getattr(a, "parent_clause", None) is not None
            and getattr(a, "parent_clause", None) == getattr(b, "parent_clause", None)
        )

        # Suppress sibling fragments inside the same composite parent.
        if same_parent and not (a_complete and b_complete):
            return False

        # Do not check incomplete fragments directly.
        if getattr(a, "requires_parent_context", False) and not a_complete:
            return False
        if getattr(b, "requires_parent_context", False) and not b_complete:
            return False

        return True

    def _filter_same_parent_claims(self, encoded_claims):
        """Pass-through — filtering now happens at pair level in solve()."""
        return encoded_claims

    def solve(self, encoded_claims: list[EncodedClaim]) -> SolverResult:
        """
        Main entry point.
        Accepts a list of EncodedClaim objects.
        Returns a SolverResult.
        """
        if not encoded_claims:
            return SolverResult(
                verdict      = Verdict.SAT,
                claims_count = 0,
                note         = "No claims submitted.",
            )

        # Build index: assertion_name → EncodedClaim
        # Each claim's constraints are grouped under a named Boolean tracker.
        # Pair-level filtering: only add claims that have at least one valid pair.
        solver      = z3.Solver()
        claim_index = {}   # tracker_name → EncodedClaim

        # Pre-pass: check for temporal subsumption contradictions
        for i, a in enumerate(encoded_claims):
            for b in encoded_claims[i + 1:]:
                if self._temporal_subsumption_conflict(a, b):
                    import logging
                    logging.getLogger("validity.pipeline").info(
                        f"Temporal subsumption conflict detected: "
                        f"{getattr(a, 'canonical_action', '?')}, "
                        f"{getattr(a, 'temporal_bound_days', '?')}d vs "
                        f"{getattr(b, 'temporal_bound_days', '?')}d"
                    )
                    return SolverResult(
                        verdict      = Verdict.UNSAT,
                        claims_count = len(encoded_claims),
                        core         = [a, b],
                        note         = "Temporal subsumption conflict: a required payment within a shorter window contradicts a prohibition within a longer overlapping window.",
                    )

        # Determine which claims participate in at least one valid pair
        participating = set()
        for i, a in enumerate(encoded_claims):
            for b in encoded_claims[i + 1:]:
                if self._should_check_pair(a, b):
                    participating.add(a.claim_id)
                    participating.add(b.claim_id)

        # If no valid pairs, fall back to all claims (avoid empty solver)
        if not participating:
            participating = {ec.claim_id for ec in encoded_claims}

        for ec in encoded_claims:
            if ec.claim_id not in participating:
                continue
            tracker_name = f"claim__{ec.claim_id}"
            tracker      = z3.Bool(tracker_name)
            claim_index[tracker_name] = ec

            # Assert each constraint under the tracker
            for constraint in ec.constraints:
                solver.assert_and_track(constraint, tracker)

        result = solver.check()

        if result == z3.sat:
            return SolverResult(
                verdict      = Verdict.SAT,
                claims_count = len(encoded_claims),
                note         = "No contradiction provable within analysed fragment.",
            )

        if result == z3.unsat:
            core_entries = self._extract_core(solver, claim_index)
            return SolverResult(
                verdict      = Verdict.UNSAT,
                claims_count = len(encoded_claims),
                core         = core_entries,
                note         = f"Contradiction proven. Minimal core: {len(core_entries)} claim(s).",
            )

        # z3.unknown — solver could not determine satisfiability
        return SolverResult(
            verdict      = Verdict.SAT,
            claims_count = len(encoded_claims),
            note         = "Solver returned unknown. Verification inconclusive.",
        )

    def _extract_core(
        self,
        solver:      z3.Solver,
        claim_index: dict[str, EncodedClaim],
    ) -> list[CoreEntry]:
        """
        Extract the minimal unsatisfiable core from the solver.
        Maps each core element back to its EncodedClaim.
        Returns a list of CoreEntry objects.
        """
        core_entries = []

        for tracker in solver.unsat_core():
            tracker_name = str(tracker)
            ec = claim_index.get(tracker_name)
            if ec is None:
                continue
            core_entries.append(CoreEntry(
                claim_id   = ec.claim_id,
                provenance = ec.provenance,
                constraint = " ∧ ".join(str(c) for c in ec.constraints),
            ))

        return core_entries


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fragment_validator import FragmentValidator, make_claim, ClaimType
    from z3_encoder import Z3Encoder

    validator = FragmentValidator()
    encoder   = Z3Encoder()
    solver_if = SolverInterface()

    def prepare(text_span, formula, claim_type=ClaimType.OBLIGATION):
        claim = make_claim(text_span=text_span, formula=formula, claim_type=claim_type)
        validator.validate(claim)
        return encoder.encode(claim)

    print("── Test 1: No contradiction ─────────────────────────────────")
    e1 = prepare(
        "The borrower shall repay the principal.",
        {"type": "Obligated", "operand": {"type": "Predicate", "name": "repay_principal", "args": ["borrower"]}},
    )
    e2 = prepare(
        "The borrower shall maintain insurance.",
        {"type": "Obligated", "operand": {"type": "Predicate", "name": "maintain_insurance", "args": ["borrower"]}},
    )
    result = solver_if.solve([e1, e2])
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Note:    {result.note}")

    print("\n── Test 2: Deontic contradiction ────────────────────────────")
    e3 = prepare(
        "The borrower shall maintain a reserve fund.",
        {"type": "Obligated", "operand": {"type": "Predicate", "name": "maintain_reserve", "args": ["borrower"]}},
    )
    e4 = prepare(
        "The borrower is prohibited from maintaining a reserve fund.",
        {"type": "Forbidden", "operand": {"type": "Predicate", "name": "maintain_reserve", "args": ["borrower"]}},
    )
    result = solver_if.solve([e3, e4])
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Note:    {result.note}")
    print(f"  Core ({len(result.core)} claim(s)):")
    for entry in result.core:
        print(f"    [{entry.claim_id[:8]}...]  \"{entry.provenance}\"")
        print(f"    Constraint: {entry.constraint}")

    print("\n── Test 3: Implication chain contradiction ───────────────────")
    e5 = prepare(
        "If a default occurs, the reserve is triggered.",
        {"type": "Implies",
         "left":  {"type": "Predicate", "name": "default",           "args": []},
         "right": {"type": "Predicate", "name": "reserve_triggered", "args": []}},
        ClaimType.CONDITION,
    )
    e6 = prepare(
        "A default has occurred.",
        {"type": "Predicate", "name": "default", "args": []},
        ClaimType.GUARANTEE,
    )
    e7 = prepare(
        "The reserve has not been triggered.",
        {"type": "Not", "operand": {"type": "Predicate", "name": "reserve_triggered", "args": []}},
        ClaimType.DISCLAIMER,
    )
    result = solver_if.solve([e5, e6, e7])
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Note:    {result.note}")
    print(f"  Core ({len(result.core)} claim(s)):")
    for entry in result.core:
        print(f"    [{entry.claim_id[:8]}...]  \"{entry.provenance}\"")

    print("\n── Test 4: Empty input ──────────────────────────────────────")
    result = solver_if.solve([])
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Note:    {result.note}")

    print("\nSolver Interface smoke test complete.")
