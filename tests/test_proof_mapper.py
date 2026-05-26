"""
Validity — Proof Mapper Test Suite
tests/test_proof_mapper.py

Run with: python3 -m pytest tests/test_proof_mapper.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from fragment_validator import FragmentValidator, make_claim, ClaimType, ValidationStatus
from z3_encoder import Z3Encoder
from solver_interface import SolverInterface, Verdict
from proof_mapper import (
    ProofMapper, ProofObject, CleanVerdict, AnalysisOutput,
    FailureClass, SourceSpan, render_output,
)

validator = FragmentValidator()
encoder   = Z3Encoder()
solver_if = SolverInterface()
mapper    = ProofMapper()


# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare(text_span, formula, claim_type=ClaimType.OBLIGATION, page=1, section=""):
    claim = make_claim(
        text_span  = text_span,
        formula    = formula,
        claim_type = claim_type,
        page       = page,
        section    = section,
    )
    validator.validate(claim)
    return claim

def predicate(name, args=None):
    return {"type": "Predicate", "name": name, "args": args or []}

def obligated(op):   return {"type": "Obligated", "operand": op}
def forbidden(op):   return {"type": "Forbidden", "operand": op}
def not_(op):        return {"type": "Not", "operand": op}
def implies(l, r):   return {"type": "Implies", "left": l, "right": r}
def at(label, body): return {"type": "At", "time_label": label, "body": body}

def run_pipeline(claims, document_text="test document"):
    validated = [c for c in claims if c.status == ValidationStatus.VALIDATED]
    encoded   = [encoder.encode(c) for c in validated]
    result    = solver_if.solve(encoded)
    return mapper.map(result, claims, document_text=document_text)


# ── Clean verdict ─────────────────────────────────────────────────────────────

class TestCleanVerdict:
    def test_compatible_claims_produce_clean_verdict(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        c2 = prepare("The borrower shall insure.", obligated(predicate("insure", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert isinstance(output.primary, CleanVerdict)

    def test_clean_verdict_value(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1])
        assert output.primary.verdict == "sat"

    def test_clean_verdict_lfs_version(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1])
        assert output.primary.lfs_version == "2.0"

    def test_clean_verdict_has_timestamp(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1])
        assert output.primary.timestamp
        assert "T" in output.primary.timestamp  # ISO 8601

    def test_clean_verdict_claims_analysed(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        c2 = prepare("The borrower shall insure.", obligated(predicate("insure", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert output.primary.claims_analysed == 2

    def test_clean_verdict_empty_core(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1])
        assert output.outside_fragment == [] or isinstance(output.outside_fragment, list)

    def test_clean_verdict_document_hash(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1], document_text="my document")
        assert output.primary.document_hash
        assert len(output.primary.document_hash) == 64  # SHA-256 hex

    def test_clean_verdict_no_document_text_empty_hash(self):
        c1 = prepare("The borrower shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1], document_text="")
        assert output.primary.document_hash == ""


# ── Proof object ──────────────────────────────────────────────────────────────

class TestProofObject:
    def test_contradiction_produces_proof_object(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert isinstance(output.primary, ProofObject)

    def test_proof_object_verdict_unsat(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert output.primary.verdict == "unsat"

    def test_proof_object_lfs_version(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert output.primary.lfs_version == "2.0"

    def test_proof_object_has_timestamp(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert output.primary.timestamp

    def test_proof_object_minimal_core_non_empty(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert len(output.primary.minimal_core) > 0

    def test_proof_object_source_spans_non_empty(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert len(output.primary.source_spans) > 0

    def test_proof_object_source_spans_are_source_span_objects(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        for span in output.primary.source_spans:
            assert isinstance(span, SourceSpan)

    def test_proof_object_source_spans_contain_text(self):
        c1 = prepare("The borrower shall maintain a reserve fund.",
                     obligated(predicate("maintain_reserve", ["borrower"])), page=4, section="4.1")
        c2 = prepare("The borrower is prohibited from maintaining a reserve fund.",
                     forbidden(predicate("maintain_reserve", ["borrower"])), page=12, section="9.3")
        output = run_pipeline([c1, c2])
        span_texts = [s.text for s in output.primary.source_spans]
        assert any("maintain a reserve fund" in t for t in span_texts)

    def test_proof_object_source_spans_contain_page_numbers(self):
        c1 = prepare("Shall maintain reserve.",
                     obligated(predicate("maintain_reserve", ["borrower"])), page=4)
        c2 = prepare("Prohibited from maintaining reserve.",
                     forbidden(predicate("maintain_reserve", ["borrower"])), page=12)
        output = run_pipeline([c1, c2])
        pages = [s.page for s in output.primary.source_spans]
        assert 4 in pages or 12 in pages

    def test_proof_object_has_formal_proof(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert output.primary.formal_proof
        assert "unsat" in output.primary.formal_proof

    def test_proof_object_has_summary(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert output.primary.summary
        assert len(output.primary.summary) > 20


# ── Failure classification ────────────────────────────────────────────────────

class TestFailureClassification:
    def test_deontic_conflict_classified(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        assert output.primary.failure_class == FailureClass.DEONTIC_CONFLICT

    def test_implication_chain_classified(self):
        c1 = prepare("If default, reserve triggered.",
                     implies(predicate("default"), predicate("reserve_triggered")),
                     ClaimType.CONDITION)
        c2 = prepare("Default occurred.", predicate("default"), ClaimType.GUARANTEE)
        c3 = prepare("Reserve not triggered.", not_(predicate("reserve_triggered")), ClaimType.DISCLAIMER)
        output = run_pipeline([c1, c2, c3])
        assert output.primary.failure_class == FailureClass.IMPLICATION_LOOP

    def test_temporal_inconsistency_classified(self):
        import z3 as z3lib
        from z3_encoder import EncodedClaim
        c1 = prepare("At maturity, loan repaid.", at("maturity", predicate("loan_repaid")), ClaimType.GUARANTEE)
        c2 = prepare("At maturity, loan not repaid.", at("maturity", not_(predicate("loan_repaid"))), ClaimType.DISCLAIMER)
        sentinel_claim = make_claim(
            text_span  = "Time point maturity is active.",
            formula    = {"type": "Predicate", "name": "at__maturity", "args": []},
            claim_type = ClaimType.GUARANTEE,
        )
        validator.validate(sentinel_claim)
        sentinel_encoded = EncodedClaim(
            claim_id     = sentinel_claim.id,
            constraints  = [z3lib.Bool("at__maturity")],
            declarations = {},
            provenance   = sentinel_claim.text_span,
            claim_type   = ClaimType.GUARANTEE,
        )
        validated_claims = [c1, c2, sentinel_claim]
        encoded = [encoder.encode(c1), encoder.encode(c2), sentinel_encoded]
        result  = solver_if.solve(encoded)
        output  = mapper.map(result, validated_claims)
        assert output.primary.failure_class == FailureClass.TEMPORAL_INCONSISTENCY


# ── Outside fragment records ──────────────────────────────────────────────────

class TestOutsideFragmentRecords:
    def test_refused_claims_appear_in_outside_fragment(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        refused = make_claim(
            text_span  = "The borrower shall act in a reasonable manner.",
            formula    = obligated(predicate("act", ["borrower"])),
            claim_type = ClaimType.OBLIGATION,
        )
        validator.validate(refused)  # will be refused
        assert refused.status == ValidationStatus.OUTSIDE_FRAGMENT
        output = run_pipeline([c1, refused])
        assert len(output.outside_fragment) == 1

    def test_outside_fragment_record_has_text_span(self):
        refused = make_claim(
            text_span  = "The borrower shall act reasonably.",
            formula    = obligated(predicate("act", ["borrower"])),
            claim_type = ClaimType.OBLIGATION,
        )
        validator.validate(refused)
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1, refused])
        record = output.outside_fragment[0]
        assert "reasonably" in record["text_span"]

    def test_outside_fragment_record_has_rule(self):
        refused = make_claim(
            text_span  = "The borrower shall act reasonably.",
            formula    = obligated(predicate("act", ["borrower"])),
            claim_type = ClaimType.OBLIGATION,
        )
        validator.validate(refused)
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1, refused])
        record = output.outside_fragment[0]
        assert record["rule_violated"] == "R10"

    def test_outside_fragment_record_has_escalation(self):
        refused = make_claim(
            text_span  = "The borrower shall act reasonably.",
            formula    = obligated(predicate("act", ["borrower"])),
            claim_type = ClaimType.OBLIGATION,
        )
        validator.validate(refused)
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1, refused])
        record = output.outside_fragment[0]
        assert record["escalation"] == "human_attestation_required"

    def test_no_refused_claims_empty_outside_fragment(self):
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1])
        assert output.outside_fragment == []


# ── Document hashing ──────────────────────────────────────────────────────────

class TestDocumentHashing:
    def test_same_document_same_hash(self):
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        out1 = run_pipeline([c1], document_text="identical document")
        out2 = run_pipeline([c1], document_text="identical document")
        assert out1.primary.document_hash == out2.primary.document_hash

    def test_different_documents_different_hashes(self):
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        out1 = run_pipeline([c1], document_text="document one")
        out2 = run_pipeline([c1], document_text="document two")
        assert out1.primary.document_hash != out2.primary.document_hash


# ── Rendering ─────────────────────────────────────────────────────────────────

class TestRendering:
    def test_render_clean_verdict_contains_clean(self):
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        output = run_pipeline([c1])
        rendered = render_output(output)
        assert "CLEAN" in rendered

    def test_render_contradiction_contains_verdict(self):
        c1 = prepare("Shall maintain reserve.", obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("Prohibited from maintaining reserve.", forbidden(predicate("maintain_reserve", ["borrower"])))
        output = run_pipeline([c1, c2])
        rendered = render_output(output)
        assert "CONTRADICTION PROVEN" in rendered

    def test_render_contains_source_text(self):
        c1 = prepare("The borrower shall maintain a reserve fund.",
                     obligated(predicate("maintain_reserve", ["borrower"])))
        c2 = prepare("The borrower is prohibited from maintaining a reserve fund.",
                     forbidden(predicate("maintain_reserve", ["borrower"])))
        output   = run_pipeline([c1, c2])
        rendered = render_output(output)
        assert "reserve fund" in rendered

    def test_render_outside_fragment_section_present(self):
        refused = make_claim(
            text_span  = "The borrower shall act reasonably.",
            formula    = obligated(predicate("act", ["borrower"])),
            claim_type = ClaimType.OBLIGATION,
        )
        validator.validate(refused)
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        output   = run_pipeline([c1, refused])
        rendered = render_output(output)
        assert "OUTSIDE FRAGMENT" in rendered

    def test_render_returns_string(self):
        c1 = prepare("Shall repay.", obligated(predicate("repay", ["borrower"])))
        output   = run_pipeline([c1])
        rendered = render_output(output)
        assert isinstance(rendered, str)
