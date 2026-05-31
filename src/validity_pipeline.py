"""
Validity — End-to-End Pipeline
validity_pipeline.py

Wires all seven components together.
Document in. Proof object out.

Usage:
    python3 validity_pipeline.py --file path/to/document.txt
    python3 validity_pipeline.py --demo
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stage1_extractor       import Stage1Extractor
from confirmation_interface import ConfirmationInterface
from stage2_translator      import Stage2Translator
from predicate_registry     import PredicateRegistryBuilder
from fragment_validator     import FragmentValidator, ValidationStatus
from z3_encoder             import Z3Encoder
from solver_interface       import SolverInterface
from proof_mapper           import ProofMapper, render_output
from report_renderer        import ReportRenderer
from document_loader        import DocumentLoader, LoadError
from logging_config         import setup_logging, pipeline_logger as log


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ValidityPipeline:
    """
    Full Validity verification pipeline.

    Stage 1: Extract candidate commitments from document (LLM)
    Gate:    Human confirmation of extracted claims
    Stage 2: Translate confirmed claims to formal AST (LLM + validator)
    Stage 3: Encode to Z3 constraints and verify (deterministic)
    Output:  Proof object or clean verdict
    """

    def __init__(self, auto_confirm: bool = False, confidence_threshold: float = 0.80):
        self.extractor        = Stage1Extractor(confidence_threshold=confidence_threshold)
        self.gate             = ConfirmationInterface()
        self.translator       = Stage2Translator()
        self.registry_builder = PredicateRegistryBuilder()
        self.validator        = FragmentValidator()
        self.encoder          = Z3Encoder()
        self.solver           = SolverInterface()
        self.mapper           = ProofMapper()
        self.auto_confirm     = auto_confirm

    def run(self, document_text: str) -> tuple:
        """
        Run the full pipeline on a document.
        Returns (AnalysisOutput, terminal_report_string).
        """

        log.info("Pipeline started")

        # ── Stage 1: Extraction ───────────────────────────────────────────────
        log.info("Stage 1 — extracting candidate commitments")
        candidates = self.extractor.extract(document_text)
        log.info(f"Stage 1 — {len(candidates)} candidate(s) identified")

        if not candidates:
            print("No extractable commitments found. Analysis complete.")
            from proof_mapper import CleanVerdict, AnalysisOutput
            empty = AnalysisOutput(primary=CleanVerdict(), outside_fragment=[], claims_analysed=0, claims_refused=0)
            return empty, "No extractable commitments found."

        # ── Confirmation gate ─────────────────────────────────────────────────
        if self.auto_confirm:
            session = self.gate.review_batch_confirm(candidates)
        else:
            session = self.gate.review_interactive(candidates)

        confirmed = session.confirmed
        log.info(f"Gate — {session.summary()}")

        if not confirmed:
            print("No claims confirmed. Analysis complete.")
            from proof_mapper import CleanVerdict, AnalysisOutput
            empty = AnalysisOutput(primary=CleanVerdict(), outside_fragment=[], claims_analysed=0, claims_refused=0)
            return empty, "No claims confirmed for analysis."

        # ── Stage 2: Translation ──────────────────────────────────────────────
        log.info("Stage 2 — building predicate registry")
        registry = self.registry_builder.build(confirmed)
        self.translator.set_registry(registry)
        log.info(f"Stage 2 — {len(registry.canonical)} canonical predicates registered")

        log.info("Stage 2 — translating confirmed claims")
        translation_results = self.translator.translate_batch(confirmed)

        validated_claims = []
        refused_claims   = []

        for result in translation_results:
            if result.status == "validated":
                validated_claims.append(result.formal_claim)
            else:
                from fragment_validator import FormalClaim, Provenance, ClaimType
                from fragment_validator import ValidationStatus as VS
                from fragment_validator import RejectionRecord
                import uuid
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
                    status     = VS.OUTSIDE_FRAGMENT,
                    rejection  = RejectionRecord(
                        reason        = result.reason,
                        rule_violated = "TRANSLATION",
                    ),
                )
                refused_claims.append(refused)

        all_claims = validated_claims + refused_claims
        log.info(f"Stage 2 — {len(validated_claims)} validated, {len(refused_claims)} outside fragment")

        if not validated_claims:
            print("No claims survived translation and validation.")
            from proof_mapper import CleanVerdict, AnalysisOutput
            output = AnalysisOutput(
                primary         = CleanVerdict(
                    claims_analysed = 0,
                    claims_refused  = len(refused_claims),
                    note            = "All claims were outside the supported logical fragment.",
                ),
                outside_fragment = [
                    {
                        "claim_id":      c.id,
                        "text_span":     c.text_span,
                        "reason":        c.rejection.reason if c.rejection else "",
                        "rule_violated": c.rejection.rule_violated if c.rejection else "",
                        "escalation":    "human_attestation_required",
                    }
                    for c in refused_claims
                ],
                claims_analysed  = 0,
                claims_refused   = len(refused_claims),
            )
            return output, render_output(output)

        # ── Stage 3: Encoding and verification ────────────────────────────────
        log.info("Stage 3 — encoding and running Z3 solver")
        encoded = [self.encoder.encode(claim) for claim in validated_claims]
        result  = self.solver.solve(encoded)
        log.info(f"Stage 3 — Z3 verdict: {result.verdict.value}")

        # ── Proof mapping ─────────────────────────────────────────────────────
        output = self.mapper.map(result, all_claims, document_text=document_text)

        return output, render_output(output)


# ── Demo document ─────────────────────────────────────────────────────────────

DEMO_DOCUMENT = """
FACILITY AGREEMENT — EXCERPTS

Section 4 — Borrower Obligations

4.1  The Borrower shall repay the principal amount of the Facility in full 
     on the Maturity Date.

4.2  The Borrower must maintain a debt service reserve account with a minimum 
     balance equal to six months of scheduled debt service payments at all 
     times during the term of the Facility.

4.3  The Borrower shall not make any distributions to shareholders during 
     the term of this Agreement.

Section 5 — Events of Default

5.1  If a Payment Default occurs, the Lender may declare the entire outstanding 
     principal amount immediately due and payable.

5.2  The Borrower is prohibited from incurring any additional Financial 
     Indebtedness without the prior written consent of the Lender.

Section 6 — Conflicting Provisions

6.1  The Borrower shall make distributions to shareholders 
     during the term of this Agreement.

Section 7 — Representations

7.1  The Borrower represents and warrants that all financial statements 
     provided are true and accurate in all material respects as at the 
     date of this Agreement.
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    setup_logging(level=os.environ.get("VALIDITY_LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Validity — Formal Verification Pipeline")
    parser.add_argument("--file",   type=str, help="Path to document file to analyse")
    parser.add_argument("--demo",   action="store_true", help="Run against built-in demo document")
    parser.add_argument("--auto",   action="store_true", help="Auto-confirm all claims (skip interactive gate)")
    parser.add_argument("--report",     type=str,   help="Path to write HTML report (e.g. reports/my-report.html)")
    parser.add_argument("--name",       type=str,   help="Document name for the report title")
    parser.add_argument("--type",       type=str,   help="Document type (e.g. 'Facility agreement')")
    parser.add_argument("--confidence", type=float, default=0.80, help="Minimum confidence threshold for claim extraction (default: 0.80)")
    args = parser.parse_args()

    if not args.file and not args.demo:
        parser.print_help()
        sys.exit(1)

    if args.demo:
        document_text = DEMO_DOCUMENT
        log.info("Running demo document — deliberate contradiction present")
    else:
        loader = DocumentLoader()
        result = loader.load_file(args.file)
        if isinstance(result, LoadError):
            log.error(f"Document load failed: {result.reason}")
            sys.exit(1)
        document_text = result.text
        log.info(f"Loaded: {result.filename} ({result.char_count:,} chars{f', {result.page_count} pages' if result.page_count else ''})")

    pipeline = ValidityPipeline(auto_confirm=args.auto or args.demo, confidence_threshold=args.confidence)
    output, terminal_report = pipeline.run(document_text)
    print(terminal_report)

    # Generate HTML report if --report flag set or if file provided
    if args.report or args.file:
        document_name = args.name or (
            os.path.splitext(os.path.basename(args.file))[0].replace("-", " ").replace("_", " ").title()
            if args.file else "Demo Document"
        )
        report_path = args.report or (
            os.path.splitext(args.file)[0] + "-report.html"
            if args.file else "demo-report.html"
        )
        renderer = ReportRenderer()
        renderer.write(
            output        = output,
            document_name = document_name,
            output_path   = report_path,
            document_type = args.type or "Document",
        )
        log.info(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
