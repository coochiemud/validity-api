"""
Validity — Solver Interface Test Suite
tests/test_solver_interface.py

Run with: python3 -m pytest tests/test_solver_interface.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import z3

from fragment_validator import FragmentValidator, make_claim, ClaimType, ValidationStatus
from z3_encoder import Z3Encoder, EncodedClaim
from solver_interface import SolverInterface, Verdict, SolverResult, CoreEntry

validator = FragmentValidator()
encoder   = Z3Encoder()
solver_if = SolverInterface()


# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare(text_span, formula, claim_type=ClaimType.OBLIGATION) -> EncodedClaim:
    claim = make_claim(text_span=text_span, formula=formula, claim_type=claim_type)
    validator.validate(claim)
    assert claim.status == ValidationStatus.VALIDATED, f"Claim outside fragment: {claim.rejection}"
    return encoder.encode(claim)

def predicate(name, args=None):
    return {"type": "Predicate", "name": name, "args": args or []}

def obligated(op):
    return {"type": "Obligated", "operand": op}

def forbidden(op):
    return {"type": "Forbidden", "operand": op}

def not_(op):
    return {"type": "Not", "operand": op}

def implies(l, r):
    return {"type": "Implies", "left": l, "right": r}

def and_(l, r):
    return {"type": "And", "left": l, "right": r}

def at(time_label, body):
    return {"type": "At", "time_label": time_label, "body": body}

def forall(var, domain, body):
    return {"type": "ForAll", "var": var, "domain": domain, "body": body}


# ── Empty input ───────────────────────────────────────────────────────────────

class TestEmptyInput:
    def test_empty_list_returns_sat(self):
        result = solver_if.solve([])
        assert result.verdict == Verdict.SAT

    def test_empty_list_zero_claims(self):
        result = solver_if.solve([])
        assert result.claims_count == 0

    def test_empty_list_empty_core(self):
        result = solver_if.solve([])
        assert result.core == []


# ── Single claim ──────────────────────────────────────────────────────────────

class TestSingleClaim:
    def test_single_obligation_sat(self):
        e = prepare(
            "The borrower shall repay.",
            obligated(predicate("repay", ["borrower"])),
        )
        result = solver_if.solve([e])
        assert result.verdict == Verdict.SAT

    def test_single_forbidden_sat(self):
        e = prepare(
            "The borrower is prohibited from withdrawing.",
            forbidden(predicate("withdraw", ["borrower"])),
        )
        result = solver_if.solve([e])
        assert result.verdict == Verdict.SAT

    def test_single_self_contradiction_unsat(self):
        """P ∧ ¬P in a single claim → unsat."""
        e = prepare(
            "The fund distributes and does not distribute profits.",
            and_(
                predicate("distributes_profits", ["fund"]),
                not_(predicate("distributes_profits", ["fund"])),
            ),
        )
        result = solver_if.solve([e])
        assert result.verdict == Verdict.UNSAT


# ── Sat verdicts ──────────────────────────────────────────────────────────────

class TestSatVerdicts:
    def test_two_compatible_obligations_sat(self):
        e1 = prepare(
            "The borrower shall repay the principal.",
            obligated(predicate("repay_principal", ["borrower"])),
        )
        e2 = prepare(
            "The borrower shall maintain insurance.",
            obligated(predicate("maintain_insurance", ["borrower"])),
        )
        result = solver_if.solve([e1, e2])
        assert result.verdict == Verdict.SAT

    def test_sat_result_has_empty_core(self):
        e1 = prepare(
            "The borrower shall repay.",
            obligated(predicate("repay", ["borrower"])),
        )
        result = solver_if.solve([e1])
        assert result.core == []

    def test_sat_claims_count_correct(self):
        claims = [
            prepare("Claim one.", obligated(predicate("P"))),
            prepare("Claim two.", obligated(predicate("Q"))),
            prepare("Claim three.", obligated(predicate("R"))),
        ]
        result = solver_if.solve(claims)
        assert result.claims_count == 3
        assert result.verdict == Verdict.SAT


# ── Unsat verdicts ────────────────────────────────────────────────────────────

class TestUnsatVerdicts:
    def test_deontic_contradiction_unsat(self):
        e1 = prepare(
            "The borrower shall maintain a reserve fund.",
            obligated(predicate("maintain_reserve", ["borrower"])),
        )
        e2 = prepare(
            "The borrower is prohibited from maintaining a reserve fund.",
            forbidden(predicate("maintain_reserve", ["borrower"])),
        )
        result = solver_if.solve([e1, e2])
        assert result.verdict == Verdict.UNSAT

    def test_unsat_core_non_empty(self):
        e1 = prepare(
            "The borrower shall maintain a reserve fund.",
            obligated(predicate("maintain_reserve", ["borrower"])),
        )
        e2 = prepare(
            "The borrower is prohibited from maintaining a reserve fund.",
            forbidden(predicate("maintain_reserve", ["borrower"])),
        )
        result = solver_if.solve([e1, e2])
        assert len(result.core) > 0

    def test_unsat_core_contains_core_entries(self):
        e1 = prepare(
            "The borrower shall maintain a reserve fund.",
            obligated(predicate("maintain_reserve", ["borrower"])),
        )
        e2 = prepare(
            "The borrower is prohibited from maintaining a reserve fund.",
            forbidden(predicate("maintain_reserve", ["borrower"])),
        )
        result = solver_if.solve([e1, e2])
        for entry in result.core:
            assert isinstance(entry, CoreEntry)

    def test_unsat_core_entries_have_provenance(self):
        e1 = prepare(
            "The borrower shall maintain a reserve fund.",
            obligated(predicate("maintain_reserve", ["borrower"])),
        )
        e2 = prepare(
            "The borrower is prohibited from maintaining a reserve fund.",
            forbidden(predicate("maintain_reserve", ["borrower"])),
        )
        result = solver_if.solve([e1, e2])
        for entry in result.core:
            assert entry.provenance
            assert isinstance(entry.provenance, str)

    def test_unsat_core_claim_ids_match_inputs(self):
        e1 = prepare(
            "The borrower shall maintain a reserve fund.",
            obligated(predicate("maintain_reserve", ["borrower"])),
        )
        e2 = prepare(
            "The borrower is prohibited from maintaining a reserve fund.",
            forbidden(predicate("maintain_reserve", ["borrower"])),
        )
        input_ids = {e1.claim_id, e2.claim_id}
        result    = solver_if.solve([e1, e2])
        for entry in result.core:
            assert entry.claim_id in input_ids

    def test_unsat_claims_count_correct(self):
        e1 = prepare(
            "The borrower shall maintain a reserve fund.",
            obligated(predicate("maintain_reserve", ["borrower"])),
        )
        e2 = prepare(
            "The borrower is prohibited from maintaining a reserve fund.",
            forbidden(predicate("maintain_reserve", ["borrower"])),
        )
        result = solver_if.solve([e1, e2])
        assert result.claims_count == 2


# ── Minimal core ──────────────────────────────────────────────────────────────

class TestMinimalCore:
    def test_core_is_minimal_not_all_claims(self):
        """
        Submit three claims: two that contradict, one that is irrelevant.
        The core should not include the irrelevant claim.
        """
        e1 = prepare(
            "The borrower shall maintain a reserve fund.",
            obligated(predicate("maintain_reserve", ["borrower"])),
        )
        e2 = prepare(
            "The borrower is prohibited from maintaining a reserve fund.",
            forbidden(predicate("maintain_reserve", ["borrower"])),
        )
        e3 = prepare(
            "The borrower shall maintain insurance.",
            obligated(predicate("maintain_insurance", ["borrower"])),
        )
        result = solver_if.solve([e1, e2, e3])
        assert result.verdict == Verdict.UNSAT
        core_ids = {entry.claim_id for entry in result.core}
        assert e3.claim_id not in core_ids

    def test_implication_chain_contradiction(self):
        """
        If default → reserve_triggered
        default is true
        reserve_triggered is false
        → unsat, all three in core
        """
        e1 = prepare(
            "If a default occurs, the reserve is triggered.",
            implies(predicate("default"), predicate("reserve_triggered")),
            ClaimType.CONDITION,
        )
        e2 = prepare(
            "A default has occurred.",
            predicate("default"),
            ClaimType.GUARANTEE,
        )
        e3 = prepare(
            "The reserve has not been triggered.",
            not_(predicate("reserve_triggered")),
            ClaimType.DISCLAIMER,
        )
        result = solver_if.solve([e1, e2, e3])
        assert result.verdict == Verdict.UNSAT
        assert len(result.core) == 3


# ── Temporal contradictions ───────────────────────────────────────────────────

class TestTemporalContradictions:
    def test_same_time_contradiction(self):
        """
        At(maturity, loan_repaid) ∧ At(maturity, ¬loan_repaid)
        Forces at__maturity = True externally to trigger contradiction.
        """
        e1 = prepare(
            "At maturity, the loan is repaid.",
            at("maturity", predicate("loan_repaid")),
            ClaimType.GUARANTEE,
        )
        e2 = prepare(
            "At maturity, the loan is not repaid.",
            at("maturity", not_(predicate("loan_repaid"))),
            ClaimType.DISCLAIMER,
        )
        # Force the time sentinel active
        sentinel = EncodedClaim(
            claim_id     = "sentinel__maturity",
            constraints  = [z3.Bool("at__maturity")],
            declarations = {},
            provenance   = "Time point: maturity is active.",
            claim_type   = ClaimType.GUARANTEE,
        )
        result = solver_if.solve([e1, e2, sentinel])
        assert result.verdict == Verdict.UNSAT

    def test_different_time_points_sat(self):
        e1 = prepare(
            "At maturity, the loan is repaid.",
            at("maturity", predicate("loan_repaid")),
            ClaimType.GUARANTEE,
        )
        e2 = prepare(
            "At year_3, the loan is not repaid.",
            at("year_3", not_(predicate("loan_repaid"))),
            ClaimType.DISCLAIMER,
        )
        result = solver_if.solve([e1, e2])
        assert result.verdict == Verdict.SAT


# ── Quantifier contradictions ─────────────────────────────────────────────────

class TestQuantifierContradictions:
    def test_forall_and_forbidden_instance_unsat(self):
        """
        ForAll x in [A, B]: receives(x)
        Forbidden(receives(A))
        → unsat
        """
        e1 = prepare(
            "All investors receive distributions.",
            forall("x", ["investor_A", "investor_B"], predicate("receives_distribution", ["x"])),
            ClaimType.QUANTIFIED_ASSURANCE,
        )
        e2 = prepare(
            "Investor A is prohibited from receiving distributions.",
            forbidden(predicate("receives_distribution", ["investor_A"])),
        )
        result = solver_if.solve([e1, e2])
        assert result.verdict == Verdict.UNSAT


# ── Result structure ──────────────────────────────────────────────────────────

class TestResultStructure:
    def test_sat_result_has_note(self):
        e = prepare("The borrower shall repay.", obligated(predicate("repay")))
        result = solver_if.solve([e])
        assert result.note
        assert isinstance(result.note, str)

    def test_unsat_result_has_note(self):
        e1 = prepare("Obligated P.", obligated(predicate("P")))
        e2 = prepare("Forbidden P.", forbidden(predicate("P")))
        result = solver_if.solve([e1, e2])
        assert result.note
        assert isinstance(result.note, str)

    def test_core_entry_has_constraint_string(self):
        e1 = prepare("Obligated P.", obligated(predicate("P")))
        e2 = prepare("Forbidden P.", forbidden(predicate("P")))
        result = solver_if.solve([e1, e2])
        for entry in result.core:
            assert isinstance(entry.constraint, str)
            assert len(entry.constraint) > 0
