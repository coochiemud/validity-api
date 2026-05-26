"""
Validity — Z3 Encoder Test Suite
tests/test_z3_encoder.py

Run with: python3 -m pytest tests/test_z3_encoder.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import z3

from fragment_validator import (
    FragmentValidator,
    FormalClaim,
    ClaimType,
    ValidationStatus,
    make_claim,
)
from z3_encoder import Z3Encoder, EncodedClaim, EncodingError

validator = FragmentValidator()
encoder   = Z3Encoder()


# ── Helpers ───────────────────────────────────────────────────────────────────

def validated(claim: FormalClaim) -> FormalClaim:
    validator.validate(claim)
    assert claim.status == ValidationStatus.VALIDATED, (
        f"Claim unexpectedly outside fragment: {claim.rejection}"
    )
    return claim

def predicate(name, args=None):
    return {"type": "Predicate", "name": name, "args": args or []}

def obligated(operand):
    return {"type": "Obligated", "operand": operand}

def forbidden(operand):
    return {"type": "Forbidden", "operand": operand}

def permitted(operand):
    return {"type": "Permitted", "operand": operand}

def and_(l, r):
    return {"type": "And", "left": l, "right": r}

def or_(l, r):
    return {"type": "Or", "left": l, "right": r}

def not_(o):
    return {"type": "Not", "operand": o}

def implies(l, r):
    return {"type": "Implies", "left": l, "right": r}

def forall(var, domain, body):
    return {"type": "ForAll", "var": var, "domain": domain, "body": body}

def exists(var, domain, body):
    return {"type": "Exists", "var": var, "domain": domain, "body": body}

def at(time_label, body):
    return {"type": "At", "time_label": time_label, "body": body}

def solve(*encoded_claims) -> z3.CheckSatResult:
    s = z3.Solver()
    for ec in encoded_claims:
        s.add(*ec.constraints)
    return s.check()


# ── Encoding errors ───────────────────────────────────────────────────────────

class TestEncodingGuards:
    def test_non_validated_claim_raises(self):
        claim = make_claim(
            text_span = "The borrower shall act reasonably.",
            formula   = obligated(predicate("act", ["borrower"])),
        )
        validator.validate(claim)
        assert claim.status == ValidationStatus.OUTSIDE_FRAGMENT
        with pytest.raises(ValueError, match="not validated"):
            encoder.encode(claim)

    def test_validated_claim_returns_encoded_claim(self):
        claim = validated(make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        result = encoder.encode(claim)
        assert isinstance(result, EncodedClaim)

    def test_encoded_claim_has_correct_id(self):
        claim = validated(make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        result = encoder.encode(claim)
        assert result.claim_id == claim.id

    def test_encoded_claim_preserves_provenance(self):
        text = "The borrower shall repay the principal."
        claim = validated(make_claim(
            text_span = text,
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        result = encoder.encode(claim)
        assert result.provenance == text


# ── Predicate encoding ────────────────────────────────────────────────────────

class TestPredicateEncoding:
    def test_predicate_produces_bool(self):
        claim = validated(make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        result = encoder.encode(claim)
        assert len(result.constraints) == 1
        assert z3.is_bool(result.constraints[0])

    def test_same_predicate_same_symbol(self):
        """Two claims with identical predicates should share the same Z3 symbol."""
        c1 = validated(make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        c2 = validated(make_claim(
            text_span = "The borrower shall repay the loan.",
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        e1 = encoder.encode(c1)
        e2 = encoder.encode(c2)
        # Both should encode to the same symbol name
        assert str(e1.constraints[0]) == str(e2.constraints[0])

    def test_different_args_different_symbols(self):
        c1 = validated(make_claim(
            text_span = "Borrower A shall repay.",
            formula   = obligated(predicate("repay", ["borrower_A"])),
        ))
        c2 = validated(make_claim(
            text_span = "Borrower B shall repay.",
            formula   = obligated(predicate("repay", ["borrower_B"])),
        ))
        e1 = encoder.encode(c1)
        e2 = encoder.encode(c2)
        assert str(e1.constraints[0]) != str(e2.constraints[0])


# ── Deontic encoding ──────────────────────────────────────────────────────────

class TestDeonticEncoding:
    def test_obligated_encodes_to_positive(self):
        """Obligated(P) → P must be true."""
        claim = validated(make_claim(
            text_span = "The borrower shall maintain reserves.",
            formula   = obligated(predicate("maintain_reserves", ["borrower"])),
        ))
        result = encoder.encode(claim)
        s = z3.Solver()
        s.add(*result.constraints)
        assert s.check() == z3.sat

    def test_forbidden_encodes_to_negation(self):
        """Forbidden(P) → Not(P) must be true."""
        claim = validated(make_claim(
            text_span = "The borrower is prohibited from maintaining reserves.",
            formula   = forbidden(predicate("maintain_reserves", ["borrower"])),
        ))
        result = encoder.encode(claim)
        # Not(P) — satisfiable on its own
        s = z3.Solver()
        s.add(*result.constraints)
        assert s.check() == z3.sat

    def test_obligated_and_forbidden_same_predicate_is_unsat(self):
        """Obligated(P) ∧ Forbidden(P) → contradiction."""
        c1 = validated(make_claim(
            text_span = "The borrower shall maintain a reserve fund.",
            formula   = obligated(predicate("maintain_reserve", ["borrower"])),
        ))
        c2 = validated(make_claim(
            text_span = "The borrower is prohibited from maintaining a reserve fund.",
            formula   = forbidden(predicate("maintain_reserve", ["borrower"])),
        ))
        assert solve(encoder.encode(c1), encoder.encode(c2)) == z3.unsat

    def test_two_obligations_compatible(self):
        """Obligated(P) ∧ Obligated(Q) where P ≠ Q → satisfiable."""
        c1 = validated(make_claim(
            text_span = "The borrower shall repay the principal.",
            formula   = obligated(predicate("repay_principal", ["borrower"])),
        ))
        c2 = validated(make_claim(
            text_span = "The borrower shall maintain insurance.",
            formula   = obligated(predicate("maintain_insurance", ["borrower"])),
        ))
        assert solve(encoder.encode(c1), encoder.encode(c2)) == z3.sat


# ── Boolean encoding ──────────────────────────────────────────────────────────

class TestBooleanEncoding:
    def test_and_satisfiable(self):
        claim = validated(make_claim(
            text_span = "A and B.",
            formula   = and_(predicate("A"), predicate("B")),
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.sat

    def test_and_contradiction(self):
        """P ∧ ¬P → unsat."""
        claim = validated(make_claim(
            text_span = "The fund distributes profits.",
            formula   = and_(
                predicate("distributes_profits", ["fund"]),
                not_(predicate("distributes_profits", ["fund"])),
            ),
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.unsat

    def test_or_satisfiable(self):
        claim = validated(make_claim(
            text_span = "A or B.",
            formula   = or_(predicate("A"), predicate("B")),
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.sat

    def test_implies_satisfiable(self):
        claim = validated(make_claim(
            text_span = "If A then B.",
            formula   = implies(predicate("A"), predicate("B")),
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.sat

    def test_implies_contradiction(self):
        """P → Q, P, ¬Q → unsat."""
        c1 = validated(make_claim(
            text_span = "If default occurs then reserve is triggered.",
            formula   = implies(predicate("default"), predicate("reserve_triggered")),
            claim_type = ClaimType.CONDITION,
        ))
        c2 = validated(make_claim(
            text_span = "A default has occurred.",
            formula   = predicate("default"),
            claim_type = ClaimType.GUARANTEE,
        ))
        c3 = validated(make_claim(
            text_span = "The reserve has not been triggered.",
            formula   = not_(predicate("reserve_triggered")),
            claim_type = ClaimType.DISCLAIMER,
        ))
        assert solve(
            encoder.encode(c1),
            encoder.encode(c2),
            encoder.encode(c3),
        ) == z3.unsat


# ── Quantifier unrolling ──────────────────────────────────────────────────────

class TestQuantifierEncoding:
    def test_forall_unrolled_sat(self):
        claim = validated(make_claim(
            text_span  = "All class A investors receive distributions.",
            formula    = forall(
                "x", ["investor_A", "investor_B"],
                predicate("receives_distribution", ["x"]),
            ),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.sat

    def test_forall_contradiction_with_forbidden(self):
        """
        ForAll x in [A,B]: receives(x)
        Forbidden(receives(investor_A))
        → unsat: investor_A must both receive and not receive.
        """
        c1 = validated(make_claim(
            text_span  = "All investors receive distributions.",
            formula    = forall(
                "x", ["investor_A", "investor_B"],
                predicate("receives_distribution", ["x"]),
            ),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        ))
        c2 = validated(make_claim(
            text_span  = "Investor A is prohibited from receiving distributions.",
            formula    = forbidden(predicate("receives_distribution", ["investor_A"])),
            claim_type = ClaimType.OBLIGATION,
        ))
        assert solve(encoder.encode(c1), encoder.encode(c2)) == z3.unsat

    def test_exists_satisfiable(self):
        claim = validated(make_claim(
            text_span  = "At least one investor qualifies.",
            formula    = exists(
                "x", ["investor_A", "investor_B"],
                predicate("qualifies", ["x"]),
            ),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.sat

    def test_single_domain_element_forall(self):
        claim = validated(make_claim(
            text_span  = "The sole investor receives distributions.",
            formula    = forall(
                "x", ["sole_investor"],
                predicate("receives_distribution", ["x"]),
            ),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.sat


# ── Temporal encoding ─────────────────────────────────────────────────────────

class TestTemporalEncoding:
    def test_temporal_sat(self):
        claim = validated(make_claim(
            text_span  = "At maturity, the loan is repaid.",
            formula    = at("maturity", predicate("loan_repaid", [])),
            claim_type = ClaimType.GUARANTEE,
        ))
        result = encoder.encode(claim)
        assert solve(result) == z3.sat

    def test_temporal_contradiction(self):
        """
        At(maturity, loan_repaid) ∧ At(maturity, ¬loan_repaid)
        When at__maturity is true, both loan_repaid and ¬loan_repaid must hold.
        """
        c1 = validated(make_claim(
            text_span  = "At maturity, the loan is repaid.",
            formula    = at("maturity", predicate("loan_repaid", [])),
            claim_type = ClaimType.GUARANTEE,
        ))
        c2 = validated(make_claim(
            text_span  = "At maturity, the loan is not repaid.",
            formula    = at("maturity", not_(predicate("loan_repaid", []))),
            claim_type = ClaimType.DISCLAIMER,
        ))
        # Force at__maturity = True to trigger the contradiction
        e1 = encoder.encode(c1)
        e2 = encoder.encode(c2)
        s = z3.Solver()
        s.add(*e1.constraints)
        s.add(*e2.constraints)
        s.add(z3.Bool("at__maturity"))   # assert the time point is active
        assert s.check() == z3.unsat

    def test_different_time_labels_independent(self):
        """Claims at different time points are independent."""
        c1 = validated(make_claim(
            text_span  = "At maturity, the loan is repaid.",
            formula    = at("maturity", predicate("loan_repaid", [])),
            claim_type = ClaimType.GUARANTEE,
        ))
        c2 = validated(make_claim(
            text_span  = "At year_3, the loan is not repaid.",
            formula    = at("year_3", not_(predicate("loan_repaid", []))),
            claim_type = ClaimType.DISCLAIMER,
        ))
        assert solve(encoder.encode(c1), encoder.encode(c2)) == z3.sat


# ── Declarations ──────────────────────────────────────────────────────────────

class TestDeclarations:
    def test_declarations_populated(self):
        claim = validated(make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        result = encoder.encode(claim)
        assert len(result.declarations) > 0

    def test_declaration_keys_are_strings(self):
        claim = validated(make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated(predicate("repay", ["borrower"])),
        ))
        result = encoder.encode(claim)
        for key in result.declarations:
            assert isinstance(key, str)
