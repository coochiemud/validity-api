"""
Validity — Fragment Validator Test Suite
tests/test_fragment_validator.py

Run with: pytest tests/test_fragment_validator.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from fragment_validator import (
    FragmentValidator,
    FormalClaim,
    ClaimType,
    ValidationStatus,
    make_claim,
)

validator = FragmentValidator()


# ── Helpers ───────────────────────────────────────────────────────────────────

def valid_predicate(name="repay", args=None):
    return {
        "type": "Predicate",
        "name": name,
        "args": args or ["borrower"],
    }

def obligated(operand):
    return {"type": "Obligated", "operand": operand}

def forbidden(operand):
    return {"type": "Forbidden", "operand": operand}

def and_(left, right):
    return {"type": "And", "left": left, "right": right}

def or_(left, right):
    return {"type": "Or", "left": left, "right": right}

def not_(operand):
    return {"type": "Not", "operand": operand}

def implies(left, right):
    return {"type": "Implies", "left": left, "right": right}

def forall(var, domain, body):
    return {"type": "ForAll", "var": var, "domain": domain, "body": body}

def exists(var, domain, body):
    return {"type": "Exists", "var": var, "domain": domain, "body": body}

def at(time_label, body):
    return {"type": "At", "time_label": time_label, "body": body}


# ── R01: Claim type ───────────────────────────────────────────────────────────

class TestR01ClaimType:
    def test_valid_claim_types(self):
        for ct in ClaimType:
            claim = make_claim(
                text_span  = "The borrower shall repay.",
                formula    = obligated(valid_predicate()),
                claim_type = ct,
            )
            result = validator.validate(claim)
            assert result.status == ValidationStatus.VALIDATED, f"Expected VALIDATED for {ct}"

    def test_invalid_claim_type_string(self):
        claim = make_claim(
            text_span  = "The borrower shall repay.",
            formula    = obligated(valid_predicate()),
            claim_type = ClaimType.OBLIGATION,
        )
        claim.claim_type = "not_a_real_type"
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R01"


# ── R02: Formula structure ────────────────────────────────────────────────────

class TestR02FormulaStructure:
    def test_empty_formula_rejected(self):
        claim = make_claim(text_span="Valid text.", formula={})
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R02"

    def test_formula_missing_type_key(self):
        claim = make_claim(text_span="Valid text.", formula={"name": "foo"})
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R02"

    def test_formula_not_dict(self):
        claim = make_claim(text_span="Valid text.", formula={"type": "Predicate", "name": "p", "args": []})
        claim.formula = "not a dict"
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R02"


# ── R03: Node types ───────────────────────────────────────────────────────────

class TestR03NodeTypes:
    def test_probabilistic_node_rejected(self):
        claim = make_claim(
            text_span = "The fund will likely outperform.",
            formula   = {"type": "Probabilistic", "p": 0.8},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R03"

    def test_causal_node_rejected(self):
        claim = make_claim(
            text_span = "Growth will cause returns.",
            formula   = {"type": "Causes", "left": valid_predicate(), "right": valid_predicate()},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R03"

    def test_modal_node_rejected(self):
        claim = make_claim(
            text_span = "Returns might improve.",
            formula   = {"type": "Possibly", "operand": valid_predicate()},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R03"


# ── R04: Predicates ───────────────────────────────────────────────────────────

class TestR04Predicates:
    def test_valid_predicate(self):
        claim = make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated(valid_predicate("repay", ["borrower"])),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED

    def test_predicate_missing_name(self):
        claim = make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated({"type": "Predicate", "args": ["borrower"]}),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R04"

    def test_predicate_missing_args(self):
        claim = make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated({"type": "Predicate", "name": "repay"}),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R04"

    def test_predicate_empty_name(self):
        claim = make_claim(
            text_span = "The borrower shall repay.",
            formula   = obligated({"type": "Predicate", "name": "", "args": []}),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R04"


# ── R05: Boolean arity ────────────────────────────────────────────────────────

class TestR05BooleanArity:
    def test_valid_and(self):
        claim = make_claim(
            text_span = "A and B.",
            formula   = and_(valid_predicate("A"), valid_predicate("B")),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED

    def test_and_missing_right(self):
        claim = make_claim(
            text_span = "A and B.",
            formula   = {"type": "And", "left": valid_predicate("A")},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R05"

    def test_not_missing_operand(self):
        claim = make_claim(
            text_span = "Not A.",
            formula   = {"type": "Not"},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R05"

    def test_implies_missing_left(self):
        claim = make_claim(
            text_span = "If A then B.",
            formula   = {"type": "Implies", "right": valid_predicate("B")},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R05"


# ── R06: Bounded quantifiers ──────────────────────────────────────────────────

class TestR06BoundedQuantifiers:
    def test_valid_bounded_forall(self):
        claim = make_claim(
            text_span  = "All investors in class A receive distributions.",
            formula    = forall("x", ["investor_A", "investor_B"], valid_predicate("receives", ["x"])),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED

    def test_forall_missing_domain(self):
        claim = make_claim(
            text_span  = "All investors receive distributions.",
            formula    = {"type": "ForAll", "var": "x", "body": valid_predicate("receives", ["x"])},
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R06"

    def test_forall_empty_domain(self):
        claim = make_claim(
            text_span  = "All investors receive distributions.",
            formula    = forall("x", [], valid_predicate("receives", ["x"])),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R06"

    def test_exists_valid(self):
        claim = make_claim(
            text_span  = "At least one investor qualifies.",
            formula    = exists("x", ["investor_A", "investor_B"], valid_predicate("qualifies", ["x"])),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED


# ── R07: Quantifier depth ─────────────────────────────────────────────────────

class TestR07QuantifierDepth:
    def test_single_quantifier_passes(self):
        claim = make_claim(
            text_span  = "All class A investors receive distributions.",
            formula    = forall("x", ["A", "B"], valid_predicate("receives", ["x"])),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED

    def test_nested_quantifier_rejected(self):
        claim = make_claim(
            text_span  = "For all x, for all y, P(x,y).",
            formula    = forall(
                "x", ["A", "B"],
                forall("y", ["C", "D"], valid_predicate("P", ["x", "y"]))
            ),
            claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R07"


# ── R08: Deontic/quantifier nesting ──────────────────────────────────────────

class TestR08DeonticNesting:
    def test_quantifier_inside_deontic_rejected(self):
        claim = make_claim(
            text_span = "All borrowers must repay.",
            formula   = obligated(
                forall("x", ["borrower_A", "borrower_B"], valid_predicate("repay", ["x"]))
            ),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R08"

    def test_predicate_inside_deontic_passes(self):
        claim = make_claim(
            text_span = "The borrower must repay.",
            formula   = obligated(valid_predicate("repay", ["borrower"])),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED


# ── R09: Temporal nodes ───────────────────────────────────────────────────────

class TestR09Temporal:
    def test_valid_temporal(self):
        claim = make_claim(
            text_span = "At maturity, the loan expires.",
            formula   = at("maturity", valid_predicate("loan_expires", [])),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED

    def test_temporal_missing_time_label(self):
        claim = make_claim(
            text_span = "At some point, the loan expires.",
            formula   = {"type": "At", "body": valid_predicate("loan_expires", [])},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R09"

    def test_temporal_empty_time_label(self):
        claim = make_claim(
            text_span = "At some point, the loan expires.",
            formula   = {"type": "At", "time_label": "", "body": valid_predicate("loan_expires", [])},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R09"

    def test_temporal_missing_body(self):
        claim = make_claim(
            text_span = "At maturity.",
            formula   = {"type": "At", "time_label": "maturity"},
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R09"


# ── R10: Modal terms ──────────────────────────────────────────────────────────

class TestR10ModalTerms:
    def test_reasonable_rejected(self):
        claim = make_claim(
            text_span = "The borrower shall act in a reasonable manner.",
            formula   = obligated(valid_predicate("act", ["borrower"])),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R10"

    def test_material_rejected(self):
        claim = make_claim(
            text_span = "Any material change must be disclosed.",
            formula   = obligated(valid_predicate("disclose", ["change"])),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R10"

    def test_promptly_rejected(self):
        claim = make_claim(
            text_span = "The borrower must promptly notify the lender.",
            formula   = obligated(valid_predicate("notify", ["borrower", "lender"])),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R10"

    def test_clean_text_passes(self):
        claim = make_claim(
            text_span = "The borrower shall repay the principal by maturity date.",
            formula   = obligated(valid_predicate("repay", ["borrower", "principal"])),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED


# ── R11: Confidence range ─────────────────────────────────────────────────────

class TestR11Confidence:
    def test_valid_confidence(self):
        claim = make_claim(
            text_span  = "The borrower shall repay.",
            formula    = obligated(valid_predicate()),
            confidence = 0.95,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.VALIDATED

    def test_confidence_above_one(self):
        claim = make_claim(
            text_span  = "The borrower shall repay.",
            formula    = obligated(valid_predicate()),
            confidence = 1.5,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R11"

    def test_confidence_below_zero(self):
        claim = make_claim(
            text_span  = "The borrower shall repay.",
            formula    = obligated(valid_predicate()),
            confidence = -0.1,
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R11"


# ── R12: Provenance ───────────────────────────────────────────────────────────

class TestR12Provenance:
    def test_empty_text_span_rejected(self):
        claim = make_claim(
            text_span = "",
            formula   = obligated(valid_predicate()),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R12"

    def test_whitespace_text_span_rejected(self):
        claim = make_claim(
            text_span = "   ",
            formula   = obligated(valid_predicate()),
        )
        result = validator.validate(claim)
        assert result.status == ValidationStatus.OUTSIDE_FRAGMENT
        assert result.rejection.rule_violated == "R12"


# ── Status mutation ───────────────────────────────────────────────────────────

class TestStatusMutation:
    def test_validated_claim_has_no_rejection(self):
        claim = make_claim(
            text_span = "The borrower shall repay the principal.",
            formula   = obligated(valid_predicate("repay", ["borrower"])),
        )
        validator.validate(claim)
        assert claim.status    == ValidationStatus.VALIDATED
        assert claim.rejection is None

    def test_rejected_claim_has_rejection_record(self):
        claim = make_claim(
            text_span = "The borrower shall act reasonably.",
            formula   = obligated(valid_predicate("act", ["borrower"])),
        )
        validator.validate(claim)
        assert claim.status               == ValidationStatus.OUTSIDE_FRAGMENT
        assert claim.rejection            is not None
        assert claim.rejection.rule_violated == "R10"
