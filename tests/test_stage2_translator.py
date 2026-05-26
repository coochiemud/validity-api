"""
Validity — Stage 2 Translator Test Suite
tests/test_stage2_translator.py

Uses a mock LLM client to keep tests fast, deterministic, and free.
The mock returns pre-defined AST responses without calling the API.

Run with: python3 -m pytest tests/test_stage2_translator.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import uuid
import pytest
from unittest.mock import MagicMock

from fragment_validator import ClaimType, ValidationStatus
from stage2_translator import Stage2Translator, CandidateClaim, TranslationResult


# ── Mock LLM client ───────────────────────────────────────────────────────────

def make_mock_client(response_json: str):
    """
    Build a mock OpenAI client that returns a fixed JSON string.
    Mimics the structure of openai.chat.completions.create().
    """
    mock_message    = MagicMock()
    mock_message.content = response_json

    mock_choice     = MagicMock()
    mock_choice.message = mock_message

    mock_response   = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client     = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


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


# ── Valid translations ────────────────────────────────────────────────────────

class TestValidTranslations:
    def test_simple_obligation_validated(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay_principal", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay the principal.")
        result     = translator.translate(candidate)
        assert result.status == "validated"
        assert result.formal_claim is not None

    def test_forbidden_validated(self):
        ast = '{"type": "Forbidden", "operand": {"type": "Predicate", "name": "make_distributions", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower is prohibited from making distributions.", ClaimType.OBLIGATION)
        result     = translator.translate(candidate)
        assert result.status == "validated"

    def test_condition_validated(self):
        ast = '{"type": "Implies", "left": {"type": "Predicate", "name": "default_occurs", "args": []}, "right": {"type": "Predicate", "name": "reserve_triggered", "args": []}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("If a default occurs, the reserve is triggered.", ClaimType.CONDITION)
        result     = translator.translate(candidate)
        assert result.status == "validated"

    def test_temporal_validated(self):
        ast = '{"type": "At", "time_label": "maturity", "body": {"type": "Predicate", "name": "loan_repaid", "args": []}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("At maturity, the loan is repaid.", ClaimType.GUARANTEE)
        result     = translator.translate(candidate)
        assert result.status == "validated"

    def test_bounded_forall_validated(self):
        ast = '{"type": "ForAll", "var": "x", "domain": ["investor_A", "investor_B"], "body": {"type": "Predicate", "name": "receives_distribution", "args": ["x"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("All investors receive distributions.", ClaimType.QUANTIFIED_ASSURANCE)
        result     = translator.translate(candidate)
        assert result.status == "validated"

    def test_validated_result_has_formal_claim(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.formal_claim is not None

    def test_validated_claim_id_matches_candidate(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.formal_claim.id == candidate.id

    def test_validated_claim_preserves_text_span(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        text       = "The borrower shall repay the principal."
        candidate  = make_candidate(text)
        result     = translator.translate(candidate)
        assert result.formal_claim.text_span == text

    def test_validated_claim_preserves_page(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.", page=7)
        result     = translator.translate(candidate)
        assert result.formal_claim.provenance.page == 7

    def test_validated_claim_preserves_claim_type(self):
        ast = '{"type": "Forbidden", "operand": {"type": "Predicate", "name": "distribute", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall not distribute.", ClaimType.OBLIGATION)
        result     = translator.translate(candidate)
        assert result.formal_claim.claim_type == ClaimType.OBLIGATION

    def test_validated_claim_status_is_validated(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.formal_claim.status == ValidationStatus.VALIDATED


# ── Outside fragment — LLM signals refusal ────────────────────────────────────

class TestLLMSignalsRefusal:
    def test_outside_fragment_signal_refused(self):
        ast = '{"type": "OUTSIDE_FRAGMENT", "reason": "Contains unresolvable modal term: reasonable"}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall act in a reasonable manner.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"

    def test_outside_fragment_reason_preserved(self):
        reason = "Contains unresolvable modal term: reasonable"
        ast    = f'{{"type": "OUTSIDE_FRAGMENT", "reason": "{reason}"}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall act in a reasonable manner.")
        result     = translator.translate(candidate)
        assert reason in result.reason

    def test_outside_fragment_no_formal_claim(self):
        ast = '{"type": "OUTSIDE_FRAGMENT", "reason": "Modal term present."}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall act reasonably.")
        result     = translator.translate(candidate)
        assert result.formal_claim is None


# ── Outside fragment — validator catches invalid AST ──────────────────────────

class TestValidatorCatchesInvalidAST:
    def test_unsupported_node_type_refused(self):
        ast = '{"type": "Probabilistic", "p": 0.8}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The fund will likely outperform.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"

    def test_modal_term_in_predicate_refused(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "act_reasonably", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        # Modal term in text_span triggers R10
        candidate  = make_candidate("The borrower shall act in a reasonable manner.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"

    def test_unbounded_quantifier_refused(self):
        ast = '{"type": "ForAll", "var": "x", "body": {"type": "Predicate", "name": "receives", "args": ["x"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("All investors receive distributions.", ClaimType.QUANTIFIED_ASSURANCE)
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"

    def test_nested_quantifier_refused(self):
        ast = '{"type": "ForAll", "var": "x", "domain": ["A", "B"], "body": {"type": "ForAll", "var": "y", "domain": ["C", "D"], "body": {"type": "Predicate", "name": "P", "args": ["x", "y"]}}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("For all x for all y P(x,y).", ClaimType.QUANTIFIED_ASSURANCE)
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"

    def test_missing_predicate_name_refused(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"

    def test_deontic_with_quantifier_refused(self):
        ast = '{"type": "Obligated", "operand": {"type": "ForAll", "var": "x", "domain": ["A", "B"], "body": {"type": "Predicate", "name": "repay", "args": ["x"]}}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("All borrowers must repay.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"


# ── API failures ──────────────────────────────────────────────────────────────

class TestAPIFailures:
    def test_invalid_json_response_refused(self):
        translator = Stage2Translator(client=make_mock_client("not valid json {{"))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"
        assert result.formal_claim is None

    def test_empty_response_refused(self):
        translator = Stage2Translator(client=make_mock_client(""))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"

    def test_api_exception_refused(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API timeout")
        translator = Stage2Translator(client=mock_client)
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.status == "outside_fragment"
        assert result.formal_claim is None


# ── Batch translation ─────────────────────────────────────────────────────────

class TestBatchTranslation:
    def test_batch_returns_correct_count(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator  = Stage2Translator(client=make_mock_client(ast))
        candidates  = [make_candidate(f"Claim {i}.") for i in range(4)]
        results     = translator.translate_batch(candidates)
        assert len(results) == 4

    def test_batch_returns_translation_results(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator  = Stage2Translator(client=make_mock_client(ast))
        candidates  = [make_candidate("The borrower shall repay.")]
        results     = translator.translate_batch(candidates)
        assert isinstance(results[0], TranslationResult)

    def test_batch_preserves_candidate_reference(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.")
        results    = translator.translate_batch([candidate])
        assert results[0].candidate is candidate


# ── Result structure ──────────────────────────────────────────────────────────

class TestResultStructure:
    def test_result_has_candidate(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.candidate is candidate

    def test_refused_result_has_reason(self):
        ast = '{"type": "OUTSIDE_FRAGMENT", "reason": "Modal term present."}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall act reasonably.")
        result     = translator.translate(candidate)
        assert result.reason
        assert isinstance(result.reason, str)

    def test_validated_result_reason_empty(self):
        ast = '{"type": "Obligated", "operand": {"type": "Predicate", "name": "repay", "args": ["borrower"]}}'
        translator = Stage2Translator(client=make_mock_client(ast))
        candidate  = make_candidate("The borrower shall repay.")
        result     = translator.translate(candidate)
        assert result.status == "validated"
        # reason may be empty for validated claims
        assert isinstance(result.reason, str)
