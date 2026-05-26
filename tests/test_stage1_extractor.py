"""
Validity — Stage 1 Extractor Test Suite
tests/test_stage1_extractor.py

Uses a mock LLM client to keep tests fast, deterministic, and free.

Run with: python3 -m pytest tests/test_stage1_extractor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import pytest
from unittest.mock import MagicMock

from fragment_validator import ClaimType
from stage2_translator import CandidateClaim
from stage1_extractor import Stage1Extractor


# ── Mock LLM client ───────────────────────────────────────────────────────────

def make_mock_client(response_json: str):
    mock_message          = MagicMock()
    mock_message.content  = response_json
    mock_choice           = MagicMock()
    mock_choice.message   = mock_message
    mock_response         = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client           = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def make_extraction_response(items: list[dict]) -> str:
    return json.dumps(items)


SIMPLE_OBLIGATION = make_extraction_response([{
    "text_span":    "The Borrower shall repay the principal on the Maturity Date.",
    "claim_type":   "obligation",
    "confidence":   0.95,
    "page_hint":    4,
    "section_hint": "Section 4.1",
}])

TWO_CLAIMS = make_extraction_response([
    {
        "text_span":    "The Borrower shall repay the principal.",
        "claim_type":   "obligation",
        "confidence":   0.95,
        "page_hint":    4,
        "section_hint": "Section 4.1",
    },
    {
        "text_span":    "The Borrower is prohibited from making distributions.",
        "claim_type":   "obligation",
        "confidence":   0.92,
        "page_hint":    7,
        "section_hint": "Section 6.3",
    },
])

MIXED_CONFIDENCE = make_extraction_response([
    {
        "text_span":  "The Borrower shall repay.",
        "claim_type": "obligation",
        "confidence": 0.95,
        "page_hint":  1,
    },
    {
        "text_span":  "The parties may discuss amendments.",
        "claim_type": "permission",
        "confidence": 0.60,   # below threshold — should be filtered
        "page_hint":  2,
    },
])

DUPLICATE_CLAIMS = make_extraction_response([
    {
        "text_span":  "The Borrower shall repay the principal.",
        "claim_type": "obligation",
        "confidence": 0.95,
        "page_hint":  4,
    },
    {
        "text_span":  "The Borrower shall repay the principal.",
        "claim_type": "obligation",
        "confidence": 0.85,
        "page_hint":  4,
    },
])

ALL_CLAIM_TYPES = make_extraction_response([
    {"text_span": "Guarantee claim.",          "claim_type": "guarantee",            "confidence": 0.90},
    {"text_span": "Condition claim.",          "claim_type": "condition",            "confidence": 0.90},
    {"text_span": "Disclaimer claim.",         "claim_type": "disclaimer",           "confidence": 0.90},
    {"text_span": "Obligation claim.",         "claim_type": "obligation",           "confidence": 0.90},
    {"text_span": "Quantified assurance.",     "claim_type": "quantified_assurance", "confidence": 0.90},
])


# ── Empty and edge cases ──────────────────────────────────────────────────────

class TestEmptyAndEdgeCases:
    def test_empty_document_returns_empty(self):
        extractor = Stage1Extractor(client=make_mock_client("[]"))
        result    = extractor.extract("")
        assert result == []

    def test_whitespace_document_returns_empty(self):
        extractor = Stage1Extractor(client=make_mock_client("[]"))
        result    = extractor.extract("   \n\n  ")
        assert result == []

    def test_no_commitments_returns_empty(self):
        extractor = Stage1Extractor(client=make_mock_client("[]"))
        result    = extractor.extract("This is a document with no commitments.")
        assert result == []

    def test_api_returns_empty_array(self):
        extractor = Stage1Extractor(client=make_mock_client("[]"))
        result    = extractor.extract("Some document text.")
        assert isinstance(result, list)
        assert len(result) == 0


# ── Extraction results ────────────────────────────────────────────────────────

class TestExtractionResults:
    def test_single_claim_extracted(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("The Borrower shall repay the principal on the Maturity Date.")
        assert len(result) == 1

    def test_result_is_list_of_candidate_claims(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("Some document.")
        for item in result:
            assert isinstance(item, CandidateClaim)

    def test_text_span_preserved(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("Some document.")
        assert result[0].text_span == "The Borrower shall repay the principal on the Maturity Date."

    def test_claim_type_parsed(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("Some document.")
        assert result[0].claim_type == ClaimType.OBLIGATION

    def test_confidence_preserved(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("Some document.")
        assert result[0].confidence == 0.95

    def test_page_hint_preserved(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("Some document.")
        assert result[0].page == 4

    def test_section_hint_preserved(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("Some document.")
        assert result[0].section == "Section 4.1"

    def test_candidate_has_uuid(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        result    = extractor.extract("Some document.")
        assert result[0].id
        assert len(result[0].id) == 36  # UUID format

    def test_two_claims_extracted(self):
        extractor = Stage1Extractor(client=make_mock_client(TWO_CLAIMS))
        result    = extractor.extract("Some document.")
        assert len(result) == 2


# ── Claim type parsing ────────────────────────────────────────────────────────

class TestClaimTypeParsing:
    def test_all_claim_types_parsed(self):
        extractor = Stage1Extractor(client=make_mock_client(ALL_CLAIM_TYPES))
        result    = extractor.extract("Some document.")
        types     = {c.claim_type for c in result}
        assert ClaimType.GUARANTEE            in types
        assert ClaimType.CONDITION            in types
        assert ClaimType.DISCLAIMER           in types
        assert ClaimType.OBLIGATION           in types
        assert ClaimType.QUANTIFIED_ASSURANCE in types

    def test_unknown_claim_type_defaults_to_obligation(self):
        response  = make_extraction_response([{
            "text_span":  "Some commitment.",
            "claim_type": "unknown_type",
            "confidence": 0.90,
        }])
        extractor = Stage1Extractor(client=make_mock_client(response))
        result    = extractor.extract("Some document.")
        assert result[0].claim_type == ClaimType.OBLIGATION


# ── Confidence filtering ──────────────────────────────────────────────────────

class TestConfidenceFiltering:
    def test_low_confidence_claims_filtered(self):
        extractor = Stage1Extractor(client=make_mock_client(MIXED_CONFIDENCE))
        result    = extractor.extract("Some document.")
        # Only the 0.95 claim should pass; 0.60 is below threshold
        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_confidence_clamped_to_one(self):
        response  = make_extraction_response([{
            "text_span":  "The Borrower shall repay.",
            "claim_type": "obligation",
            "confidence": 1.5,   # above 1.0
        }])
        extractor = Stage1Extractor(client=make_mock_client(response))
        result    = extractor.extract("Some document.")
        assert result[0].confidence <= 1.0

    def test_missing_text_span_filtered(self):
        response  = make_extraction_response([{
            "claim_type": "obligation",
            "confidence": 0.95,
        }])
        extractor = Stage1Extractor(client=make_mock_client(response))
        result    = extractor.extract("Some document.")
        assert len(result) == 0

    def test_empty_text_span_filtered(self):
        response  = make_extraction_response([{
            "text_span":  "",
            "claim_type": "obligation",
            "confidence": 0.95,
        }])
        extractor = Stage1Extractor(client=make_mock_client(response))
        result    = extractor.extract("Some document.")
        assert len(result) == 0


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_duplicate_spans_deduplicated(self):
        extractor = Stage1Extractor(client=make_mock_client(DUPLICATE_CLAIMS))
        result    = extractor.extract("Some document.")
        assert len(result) == 1

    def test_deduplication_keeps_highest_confidence(self):
        extractor = Stage1Extractor(client=make_mock_client(DUPLICATE_CLAIMS))
        result    = extractor.extract("Some document.")
        assert result[0].confidence == 0.95


# ── Confidence flagging ───────────────────────────────────────────────────────

class TestConfidenceFlagging:
    def test_flag_low_confidence_splits_correctly(self):
        extractor  = Stage1Extractor(client=make_mock_client(TWO_CLAIMS))
        candidates = extractor.extract("Some document.")
        high, flagged = extractor.flag_low_confidence(candidates, threshold=0.94)
        # 0.95 → high, 0.92 → flagged
        assert len(high)   == 1
        assert len(flagged) == 1

    def test_flag_all_high_confidence(self):
        extractor  = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        candidates = extractor.extract("Some document.")
        high, flagged = extractor.flag_low_confidence(candidates, threshold=0.80)
        assert len(high)    == 1
        assert len(flagged) == 0

    def test_flag_all_low_confidence(self):
        extractor  = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        candidates = extractor.extract("Some document.")
        high, flagged = extractor.flag_low_confidence(candidates, threshold=1.0)
        assert len(high)    == 0
        assert len(flagged) == 1


# ── API failure handling ──────────────────────────────────────────────────────

class TestAPIFailures:
    def test_invalid_json_returns_empty(self):
        extractor = Stage1Extractor(client=make_mock_client("not valid json {{"))
        result    = extractor.extract("Some document.")
        assert result == []

    def test_non_array_response_returns_empty(self):
        extractor = Stage1Extractor(client=make_mock_client('{"error": "bad"}'))
        result    = extractor.extract("Some document.")
        assert result == []

    def test_api_exception_returns_empty(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API timeout")
        extractor = Stage1Extractor(client=mock_client)
        result    = extractor.extract("Some document.")
        assert result == []

    def test_markdown_fences_stripped(self):
        response  = "```json\n" + SIMPLE_OBLIGATION + "\n```"
        extractor = Stage1Extractor(client=make_mock_client(response))
        result    = extractor.extract("Some document.")
        assert len(result) == 1


# ── Chunking ──────────────────────────────────────────────────────────────────

class TestChunking:
    def test_short_document_single_chunk(self):
        extractor = Stage1Extractor(client=make_mock_client(SIMPLE_OBLIGATION))
        chunks    = extractor._chunk_document("Short document.", max_chars=6000)
        assert len(chunks) == 1

    def test_long_document_multiple_chunks(self):
        extractor  = Stage1Extractor(client=make_mock_client("[]"))
        long_text  = "\n\n".join([f"Paragraph {i} " + "x" * 200 for i in range(50)])
        chunks     = extractor._chunk_document(long_text, max_chars=1000)
        assert len(chunks) > 1

    def test_chunks_contain_all_content(self):
        extractor  = Stage1Extractor(client=make_mock_client("[]"))
        paragraphs = [f"Para {i}." for i in range(10)]
        long_text  = "\n\n".join(paragraphs)
        chunks     = extractor._chunk_document(long_text, max_chars=50)
        rejoined   = "\n\n".join(chunks)
        for para in paragraphs:
            assert para in rejoined
