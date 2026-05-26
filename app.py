"""
Validity API
Flask wrapper around the verification pipeline for Railway deployment.
"""

import os
import sys
import json
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'validity', 'src'))

from stage1_extractor       import Stage1Extractor
from confirmation_interface import ConfirmationInterface
from stage2_translator      import Stage2Translator
from fragment_validator     import FragmentValidator, ValidationStatus, FormalClaim, Provenance, RejectionRecord
from z3_encoder             import Z3Encoder
from solver_interface       import SolverInterface
from proof_mapper           import ProofMapper
from report_renderer        import ReportRenderer

app = Flask(__name__)
CORS(app)

# Initialise pipeline components once at startup
extractor    = Stage1Extractor()
gate         = ConfirmationInterface()
translator   = Stage2Translator()
validator    = FragmentValidator()
encoder      = Z3Encoder()
solver_if    = SolverInterface()
mapper       = ProofMapper()
renderer     = ReportRenderer()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "validity-api"})


@app.route("/verify", methods=["POST"])
def verify():
    """
    POST /verify
    Body: { "text": "document text here" }
    Returns: verification result as JSON
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    document_text = data["text"].strip()
    if not document_text:
        return jsonify({"error": "Document text is empty"}), 400

    if len(document_text) > 500_000:
        return jsonify({"error": "Document too large. Maximum 500,000 characters."}), 400

    try:
        # Stage 1: Extract
        candidates = extractor.extract(document_text)
        if not candidates:
            return jsonify({
                "verdict": "sat",
                "outcome": "No extractable commitments found.",
                "claims_analysed": 0,
                "claims_refused": 0,
                "core": [],
                "outside_fragment": [],
                "document_hash": "",
                "timestamp": "",
                "lfs_version": "2.0",
            })

        # Gate: auto-confirm all
        session   = gate.review_batch_confirm(candidates)
        confirmed = session.confirmed

        if not confirmed:
            return jsonify({"error": "No claims confirmed."}), 500

        # Stage 2: Translate
        translation_results = translator.translate_batch(confirmed)
        validated_claims = []
        refused_claims   = []

        for result in translation_results:
            if result.status == "validated":
                validated_claims.append(result.formal_claim)
            else:
                refused = FormalClaim(
                    id         = result.candidate.id,
                    formula    = {},
                    claim_type = result.candidate.claim_type,
                    confidence = result.candidate.confidence,
                    provenance = Provenance(
                        text_span = result.candidate.text_span,
                        page      = result.candidate.page,
                        section   = result.candidate.section,
                    ),
                    text_span  = result.candidate.text_span,
                    status     = ValidationStatus.OUTSIDE_FRAGMENT,
                    rejection  = RejectionRecord(
                        reason        = result.reason,
                        rule_violated = "TRANSLATION",
                    ),
                )
                refused_claims.append(refused)

        all_claims = validated_claims + refused_claims

        if not validated_claims:
            return jsonify({
                "verdict": "sat",
                "outcome": "All claims outside fragment.",
                "claims_analysed": 0,
                "claims_refused": len(refused_claims),
                "core": [],
                "outside_fragment": _format_refused(refused_claims),
                "document_hash": "",
                "timestamp": "",
                "lfs_version": "2.0",
            })

        # Stage 3: Encode and verify
        encoded = [encoder.encode(c) for c in validated_claims]
        result  = solver_if.solve(encoded)
        output  = mapper.map(result, all_claims, document_text=document_text)

        return jsonify(_format_output(output))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _format_output(output) -> dict:
    from proof_mapper import ProofObject, CleanVerdict

    primary  = output.primary
    is_unsat = isinstance(primary, ProofObject)

    core = []
    if is_unsat and primary.source_spans:
        for span in primary.source_spans:
            core.append({
                "text":       span.text,
                "page":       span.page,
                "section":    span.section,
                "claim_id":   span.claim_id,
            })

    outside = []
    for record in output.outside_fragment:
        outside.append({
            "text":   record.get("text_span", ""),
            "reason": record.get("reason", ""),
            "rule":   record.get("rule_violated", ""),
        })

    return {
        "verdict":         primary.verdict if hasattr(primary, "verdict") else "sat",
        "outcome":         primary.failure_class.value if is_unsat else "No contradiction proven.",
        "failure_class":   primary.failure_class.value if is_unsat else None,
        "claims_analysed": output.claims_analysed,
        "claims_refused":  output.claims_refused,
        "core":            core,
        "outside_fragment": outside,
        "document_hash":   getattr(primary, "document_hash", ""),
        "timestamp":       getattr(primary, "timestamp", ""),
        "lfs_version":     getattr(primary, "lfs_version", "2.0"),
        "formal_proof":    getattr(primary, "formal_proof", ""),
    }


def _format_refused(refused_claims) -> list:
    result = []
    for c in refused_claims:
        result.append({
            "text":   c.text_span,
            "reason": c.rejection.reason if c.rejection else "",
            "rule":   c.rejection.rule_violated if c.rejection else "",
        })
    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
