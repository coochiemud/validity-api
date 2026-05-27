"""
Validity — Proof Mapper
Component 4 of 7

Takes the minimal unsatisfiable core from the Solver Interface
and maps it back to exact source text spans, producing the
formal proof object.

This is the output that goes in front of the responsible party.

Input:  SolverResult (verdict + minimal core)
        List[FormalClaim] (full claim set with provenance)
Output: ProofObject | CleanVerdict | OutsideFragmentRecord
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import hashlib
import json

from solver_interface import SolverResult, Verdict
from fragment_validator import FormalClaim, ValidationStatus


# ── Failure classes (per LFS v2 Section 3.4) ─────────────────────────────────

class FailureClass(Enum):
    CONTRADICTION          = "Contradiction"           # φ ∧ ¬φ
    DEONTIC_CONFLICT       = "Deontic Conflict"        # Obligated(φ) ∧ Forbidden(φ)
    QUANTIFIER_CLASH       = "Quantifier Clash"        # ∀x P(x) ∧ ∃y ¬P(y)
    TEMPORAL_INCONSISTENCY = "Temporal Inconsistency"  # At(t, φ) ∧ At(t, ¬φ)
    IMPLICATION_LOOP       = "Implication Loop"        # φ→ψ ∧ ψ→φ (no base)
    RESOURCE_CONFLICT      = "Resource Conflict"       # Priority + absolute obligation on shared pool
    STRESS_EXPOSURE        = "Conditional Stress Exposure"  # Priority conflict resolved by write-downs under normal conditions but exposed under stress
    GENERAL                = "Logical Contradiction"   # catch-all


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class SourceSpan:
    """A verbatim text span anchored to its location in the source document."""
    text:        str
    page:        Optional[int]
    section:     Optional[str]
    char_offset: Optional[int]
    claim_id:    str


@dataclass
class ProofObject:
    """
    Produced when a contradiction is proven.
    Independently verifiable: any party with the formal claims and Z3
    can reproduce this result deterministically.
    """
    verdict:        str = "unsat"
    lfs_version:    str = "2.0"
    document_hash:  str = ""
    timestamp:      str = ""
    failure_class:  FailureClass = FailureClass.GENERAL
    minimal_core:   list[str] = field(default_factory=list)    # claim IDs
    source_spans:   list[SourceSpan] = field(default_factory=list)
    formal_proof:   str = ""    # SMT-LIB2 certificate (constraint strings)
    summary:        str = ""    # plain-language description


@dataclass
class CleanVerdict:
    """Produced when no contradiction is provable."""
    verdict:          str = "sat"
    lfs_version:      str = "2.0"
    document_hash:    str = ""
    timestamp:        str = ""
    claims_analysed:  int = 0
    claims_refused:   int = 0
    note:             str = "No contradiction provable within analysed fragment."


@dataclass
class AnalysisOutput:
    """
    Complete output of a Validity analysis run.
    Always produced regardless of primary verdict.
    Contains the primary result plus outside-fragment records.
    """
    primary:          ProofObject | CleanVerdict
    outside_fragment: list[dict] = field(default_factory=list)  # refused claims
    claims_analysed:  int = 0
    claims_refused:   int = 0


# ── Proof Mapper ──────────────────────────────────────────────────────────────

class ProofMapper:
    """
    Maps a SolverResult to a structured ProofObject or CleanVerdict.

    Responsibilities:
    - Identify failure class from the minimal core structure
    - Map core claim IDs back to source text spans via FormalClaim provenance
    - Produce a human-readable summary of the contradiction
    - Compile the formal proof certificate
    - Collect outside-fragment records from refused claims
    - Hash the document content for independent verifiability
    """

    LFS_VERSION = "2.0"

    def map(
        self,
        solver_result:  SolverResult,
        all_claims:     list[FormalClaim],
        document_text:  str = "",
    ) -> AnalysisOutput:
        """
        Main entry point.

        solver_result:  Output from SolverInterface.solve()
        all_claims:     All FormalClaim objects (validated + refused)
        document_text:  Raw document text for hashing (optional)
        """
        timestamp     = datetime.now(timezone.utc).isoformat()
        document_hash = self._hash_document(document_text)

        # Build claim index by ID
        claim_index = {c.id: c for c in all_claims}

        # Separate validated from refused
        validated_claims = [c for c in all_claims if c.status == ValidationStatus.VALIDATED]
        refused_claims   = [c for c in all_claims if c.status == ValidationStatus.OUTSIDE_FRAGMENT]

        outside_fragment_records = [
            self._outside_fragment_record(c) for c in refused_claims
        ]

        if solver_result.verdict == Verdict.SAT:
            primary = CleanVerdict(
                lfs_version     = self.LFS_VERSION,
                document_hash   = document_hash,
                timestamp       = timestamp,
                claims_analysed = len(validated_claims),
                claims_refused  = len(refused_claims),
            )
        else:
            # UNSAT — build proof object
            core_claim_ids = [entry.claim_id for entry in solver_result.core]
            source_spans   = self._extract_source_spans(core_claim_ids, claim_index)
            failure_class  = self._classify_failure(core_claim_ids, claim_index)
            formal_proof   = self._compile_formal_proof(solver_result)
            summary        = self._build_summary(source_spans, failure_class)

            primary = ProofObject(
                lfs_version   = self.LFS_VERSION,
                document_hash = document_hash,
                timestamp     = timestamp,
                failure_class = failure_class,
                minimal_core  = core_claim_ids,
                source_spans  = source_spans,
                formal_proof  = formal_proof,
                summary       = summary,
            )

        return AnalysisOutput(
            primary          = primary,
            outside_fragment = outside_fragment_records,
            claims_analysed  = len(validated_claims),
            claims_refused   = len(refused_claims),
        )

    # ── Core mapping ──────────────────────────────────────────────────────────

    def _extract_source_spans(
        self,
        core_claim_ids: list[str],
        claim_index:    dict[str, FormalClaim],
    ) -> list[SourceSpan]:
        spans = []
        for claim_id in core_claim_ids:
            claim = claim_index.get(claim_id)
            if claim is None:
                continue
            spans.append(SourceSpan(
                text        = claim.text_span,
                page        = claim.provenance.page,
                section     = claim.provenance.section,
                char_offset = claim.provenance.char_offset,
                claim_id    = claim_id,
            ))
        return spans

    # ── Failure classification ────────────────────────────────────────────────

    def _classify_failure(
        self,
        core_claim_ids: list[str],
        claim_index:    dict[str, FormalClaim],
    ) -> FailureClass:
        """
        Infer the failure class from the structure of the core claims.
        Uses formula structure heuristics — deterministic, no LLM.
        """
        core_claims = [claim_index[cid] for cid in core_claim_ids if cid in claim_index]
        formulas    = [c.formula for c in core_claims]

        # Check for deontic conflict: Obligated(P) ∧ Forbidden(P)
        obligated_predicates = set()
        forbidden_predicates = set()
        temporal_labels      = set()

        for f in formulas:
            ftype = f.get("type", "")
            if ftype == "Obligated":
                obligated_predicates.add(self._formula_key(f.get("operand", {})))
            elif ftype == "Forbidden":
                forbidden_predicates.add(self._formula_key(f.get("operand", {})))
            elif ftype == "At":
                temporal_labels.add(f.get("time_label", ""))

        if obligated_predicates & forbidden_predicates:
            return FailureClass.DEONTIC_CONFLICT

        if len(temporal_labels) == 1 and len(formulas) > 1:
            return FailureClass.TEMPORAL_INCONSISTENCY

        # Check for direct contradiction: P and Not(P) in same formula
        for f in formulas:
            if f.get("type") == "And":
                left  = f.get("left", {})
                right = f.get("right", {})
                if right.get("type") == "Not":
                    if self._formula_key(left) == self._formula_key(right.get("operand", {})):
                        return FailureClass.CONTRADICTION

        # Check for implication chain (Implies present in core)
        has_implies = any(f.get("type") == "Implies" for f in formulas)
        if has_implies and len(formulas) >= 3:
            return FailureClass.IMPLICATION_LOOP

        return FailureClass.GENERAL

    def _formula_key(self, formula: dict) -> str:
        """Stable string key for a formula node, used for matching."""
        return json.dumps(formula, sort_keys=True)

    # ── Formal proof certificate ──────────────────────────────────────────────

    def _compile_formal_proof(self, solver_result: SolverResult) -> str:
        """
        Compile a human-readable formal proof certificate.
        Lists each constraint in the minimal core.
        """
        lines = ["MINIMAL UNSATISFIABLE CORE"]
        lines.append("=" * 40)
        for i, entry in enumerate(solver_result.core, 1):
            lines.append(f"{i}. [{entry.claim_id[:8]}...]")
            lines.append(f"   Source: \"{entry.provenance}\"")
            lines.append(f"   Constraint: {entry.constraint}")
        lines.append("=" * 40)
        lines.append("Z3 verdict: unsat")
        lines.append("These constraints cannot simultaneously hold.")
        return "\n".join(lines)

    # ── Plain-language summary ────────────────────────────────────────────────

    def _build_summary(
        self,
        source_spans:  list[SourceSpan],
        failure_class: FailureClass,
    ) -> str:
        count = len(source_spans)
        intro = f"A {failure_class.value} was detected involving {count} commitment(s)."

        span_lines = []
        for i, span in enumerate(source_spans, 1):
            location = ""
            if span.page:
                location += f" (page {span.page}"
                if span.section:
                    location += f", {span.section}"
                location += ")"
            span_lines.append(f"  {i}. \"{span.text}\"{location}")

        body = "\n".join(span_lines)
        closing = (
            "These commitments cannot simultaneously hold. "
            "The contradiction is formally proven and independently verifiable."
        )
        return f"{intro}\n\n{body}\n\n{closing}"

    # ── Outside fragment records ──────────────────────────────────────────────

    def _outside_fragment_record(self, claim: FormalClaim) -> dict:
        return {
            "claim_id":      claim.id,
            "text_span":     claim.text_span,
            "reason":        claim.rejection.reason if claim.rejection else "Unknown",
            "rule_violated": claim.rejection.rule_violated if claim.rejection else "",
            "escalation":    "human_attestation_required",
        }

    # ── Document hashing ──────────────────────────────────────────────────────

    def _hash_document(self, document_text: str) -> str:
        if not document_text:
            return ""
        return hashlib.sha256(document_text.encode("utf-8")).hexdigest()


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_output(output: AnalysisOutput) -> str:
    """
    Render an AnalysisOutput as a clean terminal report.
    This is the output that goes in front of the responsible party.
    """
    lines = []
    lines.append("")
    lines.append("━" * 60)
    lines.append("  VALIDITY ANALYSIS REPORT")
    lines.append("━" * 60)

    primary = output.primary

    if isinstance(primary, CleanVerdict):
        lines.append(f"  Verdict:          ✓ CLEAN — No contradiction proven")
        lines.append(f"  Claims analysed:  {primary.claims_analysed}")
        lines.append(f"  Claims refused:   {primary.claims_refused}")
        lines.append(f"  LFS version:      {primary.lfs_version}")
        lines.append(f"  Timestamp:        {primary.timestamp}")
        if primary.document_hash:
            lines.append(f"  Document hash:    {primary.document_hash[:16]}...")
        lines.append("")
        lines.append(f"  {primary.note}")

    elif isinstance(primary, ProofObject):
        is_conditional = primary.verdict == "conditional"
        if is_conditional:
            lines.append(f"  Verdict:          ⚠ CONDITIONAL — Stress exposure detected")
        else:
            lines.append(f"  Verdict:          ✗ CONTRADICTION PROVEN")
        lines.append(f"  Failure class:    {primary.failure_class.value}")
        lines.append(f"  Claims in core:   {len(primary.source_spans)}")
        lines.append(f"  LFS version:      {primary.lfs_version}")
        lines.append(f"  Timestamp:        {primary.timestamp}")
        if primary.document_hash:
            lines.append(f"  Document hash:    {primary.document_hash[:16]}...")
        lines.append("")
        if is_conditional:
            lines.append("  RELATED COMMITMENTS")
        else:
            lines.append("  CONTRADICTING COMMITMENTS")
        lines.append("  " + "─" * 56)
        for i, span in enumerate(primary.source_spans, 1):
            location = ""
            if span.page:
                location = f"  [page {span.page}"
                if span.section:
                    location += f" · {span.section}"
                location += "]"
            lines.append(f"  {i}.{location}")
            lines.append(f"     \"{span.text}\"")
        lines.append("")
        lines.append("  FORMAL PROOF")
        lines.append("  " + "─" * 56)
        for proof_line in primary.formal_proof.split("\n"):
            lines.append(f"  {proof_line}")

    lines.append("")

    if output.outside_fragment:
        lines.append("━" * 60)
        lines.append(f"  OUTSIDE FRAGMENT  ({len(output.outside_fragment)} claim(s) refused)")
        lines.append("━" * 60)
        for record in output.outside_fragment:
            lines.append(f"  \"{record['text_span']}\"")
            lines.append(f"  Rule: {record['rule_violated']}  Reason: {record['reason']}")
            lines.append(f"  Escalation: {record['escalation']}")
            lines.append("")

    lines.append("━" * 60)
    return "\n".join(lines)


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fragment_validator import FragmentValidator, make_claim, ClaimType, Provenance
    from z3_encoder import Z3Encoder
    from solver_interface import SolverInterface

    validator = FragmentValidator()
    encoder   = Z3Encoder()
    solver_if = SolverInterface()
    mapper    = ProofMapper()

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

    print("── Test 1: Clean verdict ─────────────────────────────────────")
    c1 = prepare("The borrower shall repay the principal.", {
        "type": "Obligated",
        "operand": {"type": "Predicate", "name": "repay_principal", "args": ["borrower"]},
    }, page=3, section="Section 4.1")
    c2 = prepare("The borrower shall maintain insurance.", {
        "type": "Obligated",
        "operand": {"type": "Predicate", "name": "maintain_insurance", "args": ["borrower"]},
    }, page=7, section="Section 6.2")

    encoded   = [encoder.encode(c) for c in [c1, c2]]
    result    = solver_if.solve(encoded)
    output    = mapper.map(result, [c1, c2], document_text="sample document text")
    print(render_output(output))

    print("── Test 2: Deontic contradiction ─────────────────────────────")
    c3 = prepare("The borrower shall maintain a reserve fund.", {
        "type": "Obligated",
        "operand": {"type": "Predicate", "name": "maintain_reserve", "args": ["borrower"]},
    }, page=4, section="Section 5.1")
    c4 = prepare("The borrower is prohibited from maintaining a reserve fund.", {
        "type": "Forbidden",
        "operand": {"type": "Predicate", "name": "maintain_reserve", "args": ["borrower"]},
    }, page=12, section="Section 9.3")

    # Add a refused claim
    c5 = make_claim(
        text_span  = "The borrower shall act in a reasonable manner.",
        formula    = {"type": "Obligated", "operand": {"type": "Predicate", "name": "act", "args": ["borrower"]}},
        claim_type = ClaimType.OBLIGATION,
        page       = 2,
        section    = "Section 2.1",
    )
    validator.validate(c5)  # will be refused (modal term)

    encoded   = [encoder.encode(c) for c in [c3, c4]]
    result    = solver_if.solve(encoded)
    output    = mapper.map(result, [c3, c4, c5], document_text="sample document text")
    print(render_output(output))
