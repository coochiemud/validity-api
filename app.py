"""
Validity API
Flask wrapper around the verification pipeline for Railway deployment.
Accepts both JSON text and multipart file uploads.
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
from predicate_registry     import PredicateRegistryBuilder
from fragment_validator     import FragmentValidator, ValidationStatus, FormalClaim, Provenance, RejectionRecord
from z3_encoder             import Z3Encoder
from solver_interface       import SolverInterface
from proof_mapper           import ProofMapper
from document_loader        import DocumentLoader, LoadError

app = Flask(__name__)
CORS(app)

# Initialise pipeline components once at startup
extractor        = Stage1Extractor()
gate             = ConfirmationInterface()
translator       = Stage2Translator()
registry_builder = PredicateRegistryBuilder()
encoder          = Z3Encoder()
solver_if        = SolverInterface()
mapper           = ProofMapper()
loader           = DocumentLoader()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "validity-api"})


@app.route("/verify", methods=["POST"])
def verify():
    """
    POST /verify

    Accepts either:
    A) JSON body:       { "text": "document text here" }
    B) Multipart form:  file field named "file" (PDF, DOCX, TXT)

    Returns: verification result as JSON
    """

    document_text = None

    # ── A: File upload ────────────────────────────────────────────────────────
    if request.files and 'file' in request.files:
        f = request.files['file']
        filename = f.filename or 'document'
        ext = os.path.splitext(filename)[-1].lower()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = loader.load_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        if isinstance(result, LoadError):
            return jsonify({"error": result.reason}), 400

        document_text = result.text

    # ── B: JSON text ──────────────────────────────────────────────────────────
    else:
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Provide either a file upload or JSON with 'text' field"}), 400
        document_text = data["text"].strip()

    if not document_text:
        return jsonify({"error": "Document text is empty"}), 400

    if len(document_text) > 500_000:
        return jsonify({"error": "Document too large. Maximum 500,000 characters."}), 400

    try:
        return jsonify(_run_pipeline(document_text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_pipeline(document_text: str) -> dict:
    """Run the full verification pipeline and return a result dict."""

    # Stage 1: Extract
    candidates = extractor.extract(document_text)
    if not candidates:
        return _empty_result("No extractable commitments found.")

    # Gate: auto-confirm all
    session   = gate.review_batch_confirm(candidates)
    confirmed = session.confirmed
    if not confirmed:
        return _empty_result("No claims confirmed.")

    # Stage 2: Registry + Translation
    registry = registry_builder.build(confirmed)
    translator.set_registry(registry)
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
        return {
            "verdict":          "sat",
            "outcome":          "All claims outside fragment.",
            "claims_analysed":  0,
            "claims_refused":   len(refused_claims),
            "core":             [],
            "outside_fragment": _format_refused(refused_claims),
            "document_hash":    "",
            "timestamp":        "",
            "lfs_version":      "2.0",
            "formal_proof":     "",
        }

    # Stage 3: Encode and verify
    encoded = [encoder.encode(c) for c in validated_claims]
    result  = solver_if.solve(encoded)
    output  = mapper.map(result, all_claims, document_text=document_text)

    return _format_output(output)


def _empty_result(note: str) -> dict:
    return {
        "verdict":          "sat",
        "outcome":          note,
        "claims_analysed":  0,
        "claims_refused":   0,
        "core":             [],
        "outside_fragment": [],
        "document_hash":    "",
        "timestamp":        "",
        "lfs_version":      "2.0",
        "formal_proof":     "",
    }


def _format_output(output) -> dict:
    from proof_mapper import ProofObject

    primary  = output.primary
    is_unsat = isinstance(primary, ProofObject)

    core = []
    if is_unsat and primary.source_spans:
        for span in primary.source_spans:
            core.append({
                "text":    span.text,
                "page":    span.page,
                "section": span.section,
            })

    outside = []
    for record in output.outside_fragment:
        outside.append({
            "text":   record.get("text_span", ""),
            "reason": record.get("reason", ""),
            "rule":   record.get("rule_violated", ""),
        })

    return {
        "verdict":          primary.verdict if hasattr(primary, "verdict") else "sat",
        "outcome":          primary.failure_class.value if is_unsat else "No contradiction proven.",
        "failure_class":    primary.failure_class.value if is_unsat else None,
        "claims_analysed":  output.claims_analysed,
        "claims_refused":   output.claims_refused,
        "core":             core,
        "outside_fragment": outside,
        "document_hash":    getattr(primary, "document_hash", ""),
        "timestamp":        getattr(primary, "timestamp", ""),
        "lfs_version":      getattr(primary, "lfs_version", "2.0"),
        "formal_proof":     getattr(primary, "formal_proof", ""),
    }


def _format_refused(refused_claims) -> list:
    return [{
        "text":   c.text_span,
        "reason": c.rejection.reason if c.rejection else "",
        "rule":   c.rejection.rule_violated if c.rejection else "",
    } for c in refused_claims]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
